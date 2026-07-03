/**
 * RAG pipeline for the AI assistant bar.
 *
 * 1. Embed the user's query.
 * 2. Search three sources in parallel (limit each to top-K).
 * 3. Build a numbered context block — each source gets an index [1], [2], ...
 *    the LLM is told to cite sources by these indices inline.
 * 4. (Optional) Re-rank with a cross-encoder for better retrieval quality.
 * 5. Compute a confidence score from source quality metrics.
 * 6. Send to the LLM with a system prompt that enforces citation + honesty,
 *    prepending any conversation history for multi-turn context.
 * 7. Return { answer, sources[], confidence, model } — sources carry
 *    id + title + url + snippet so the frontend can render them as inline cards.
 *
 * Sources searched:
 *   - FAQ (yaksha_faq_faqs)
 *   - Community posts (yaksha_faq_communityposts, answered+unanswered)
 *   - TranscriptKnowledge (Zoom meeting Q&A extractions)
 */

import mongoose from 'mongoose';
import { generateQueryEmbedding } from '../../utils/ai/embeddings.js';
import { resolveProviderAsync } from '../../utils/ai/aiProvider.js';
import { searchKnowledge } from '../knowledge/knowledge-base.service.js';
import { logger } from '../../utils/http/logger.js';

// ─── Public interfaces ────────────────────────────────────────────────────────

export interface RagSource {
  /** Stable id — the client uses this as a React key and to link out. */
  id: string;
  /** "faq" | "community" | "knowledge" */
  type: 'faq' | 'community' | 'knowledge';
  /** Display title. */
  title: string;
  /** Snippet shown in the source card (truncated body). */
  snippet: string;
  /** URL the client can deep-link to. */
  url: string;
  /** Confidence in [0, 1] (vector cosine + keyword overlap). */
  score: number;
}

export interface RagResult {
  answer: string;
  sources: RagSource[];
  /** The model that produced the answer (e.g. "gpt-4o-mini"). */
  model: string;
  /** Confidence score in [0, 1] derived from source quality metrics. */
  confidence: number;
}

/** Conversation turn sent from the frontend for multi-turn context. */
export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const TOP_K_PER_SOURCE = 4;
const MAX_CONTEXT_CHARS = 14000; // leave headroom under typical 16k context windows

/** Max conversation history messages (sliding window). 6 = 3 exchanges.
 *  Enough for pronoun resolution in any FAQ session. ≈400 tokens — negligible
 *  vs Gemini 1M limit. Trivially adjustable if needed. */
const MAX_HISTORY_MESSAGES = 6;

/** HuggingFace model for cross-encoder re-ranking (stage 2 retrieval). */
const CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2';
const CROSS_ENCODER_API_URL = `https://api-inference.huggingface.co/models/${CROSS_ENCODER_MODEL}`;

// ─── Internal hit types ───────────────────────────────────────────────────────

interface FaqHit { _id: unknown; question: string; answer: string; category?: string; trustLevel?: string; score: number }
interface PostHit { _id: unknown; title: string; body: string; status: string; score: number }

// ─── Source search functions (unchanged) ──────────────────────────────────────

/**
 * Search FAQs via Atlas vector search + native text search, merged with RRF.
 * Reuses the helper from the search controller (no behavior change — same
 * ordering + thresholds users already see on the FAQ page).
 */
async function searchFaqs(embedding: number[], query: string, limit: number): Promise<FaqHit[]> {
  const db = mongoose.connection.db;
  if (!db) return [];

  const [vec, txt] = await Promise.all([
    db.collection('yaksha_faq_faqs')
      .aggregate([
        { $vectorSearch: {
          index: 'vector_index',
          path: 'embedding',
          queryVector: embedding,
          numCandidates: limit * 10,
          limit,
        } },
        { $project: {
          _id: 1, question: 1, answer: 1, category: 1, trustLevel: 1,
          score: { $meta: 'vectorSearchScore' },
        } },
        // Match the trust-level boost the search endpoint uses
        { $addFields: {
          score: {
            $add: [
              { $meta: 'vectorSearchScore' },
              { $switch: {
                branches: [
                  { case: { $eq: ['$trustLevel', 'high'] },   then: 0.15 },
                  { case: { $eq: ['$trustLevel', 'expert'] }, then: 0.07 },
                  { case: { $eq: ['$trustLevel', 'medium'] }, then: 0.02 },
                ],
                default: 0,
              } },
            ],
          },
        } },
      ]).toArray().catch((err) => {
        logger.warn(`[rag] searchFaqs aggregate vector search failed: ${(err as Error).message}`);
        return [];
      }),
    db.collection('yaksha_faq_faqs').find(
      { $text: { $search: query } },
      { projection: { score: { $meta: 'textScore' }, question: 1, answer: 1, category: 1, trustLevel: 1 } }
    ).sort({ score: { $meta: 'textScore' } }).limit(limit).toArray().catch((err) => {
      logger.warn(`[rag] searchFaqs text search failed: ${(err as Error).message}`);
      return [];
    }),
  ]);

  // Reciprocal Rank Fusion — same formula as the search controller.
  const rrf = (k: number) => 1 / (60 + k);
  const scoreMap = new Map<string, number>();
  const docs = new Map<string, Record<string, unknown>>();
  vec.forEach((d, i) => {
    const id = String(d._id);
    scoreMap.set(id, (scoreMap.get(id) ?? 0) + rrf(i));
    docs.set(id, d as Record<string, unknown>);
  });
  txt.forEach((d, i) => {
    const id = String(d._id);
    scoreMap.set(id, (scoreMap.get(id) ?? 0) + rrf(i));
    docs.set(id, d as Record<string, unknown>);
  });
  return [...scoreMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([id, s]) => {
      const d = docs.get(id)!;
      return {
        _id: d._id,
        question: String(d.question ?? ''),
        answer: String(d.answer ?? ''),
        category: d.category as string | undefined,
        trustLevel: d.trustLevel as string | undefined,
        score: s,
      };
    });
}

async function searchCommunity(embedding: number[], query: string, limit: number): Promise<PostHit[]> {
  const db = mongoose.connection.db;
  if (!db) return [];

  const [vec, txt] = await Promise.all([
    db.collection('yaksha_faq_communityposts')
      .aggregate([
        { $vectorSearch: {
          index: 'vector_index',
          path: 'embedding',
          queryVector: embedding,
          numCandidates: limit * 10,
          limit,
        } },
        { $project: { _id: 1, title: 1, body: 1, status: 1, score: { $meta: 'vectorSearchScore' } } },
      ]).toArray().catch((err) => {
        logger.warn(`[rag] searchCommunity aggregate vector search failed: ${(err as Error).message}`);
        return [];
      }),
    db.collection('yaksha_faq_communityposts').find(
      { $text: { $search: query } },
      { projection: { score: { $meta: 'textScore' }, title: 1, body: 1, status: 1 } }
    ).sort({ score: { $meta: 'textScore' } }).limit(limit).toArray().catch((err) => {
      logger.warn(`[rag] searchCommunity text search failed: ${(err as Error).message}`);
      return [];
    }),
  ]);

  const rrf = (k: number) => 1 / (60 + k);
  const scoreMap = new Map<string, number>();
  const docs = new Map<string, Record<string, unknown>>();
  vec.forEach((d, i) => {
    const id = String(d._id);
    scoreMap.set(id, (scoreMap.get(id) ?? 0) + rrf(i));
    docs.set(id, d as Record<string, unknown>);
  });
  txt.forEach((d, i) => {
    const id = String(d._id);
    scoreMap.set(id, (scoreMap.get(id) ?? 0) + rrf(i));
    docs.set(id, d as Record<string, unknown>);
  });
  return [...scoreMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([id, s]) => {
      const d = docs.get(id)!;
      return {
        _id: d._id,
        title: String(d.title ?? ''),
        body: String(d.body ?? ''),
        status: String(d.status ?? ''),
        score: s,
      };
    });
}

// ─── Context + prompt builders ────────────────────────────────────────────────

/** Render source snippets — the LLM only sees these, plus the question. */
function buildContext(sources: RagSource[]): string {
  const blocks: string[] = [];
  let total = 0;
  for (let i = 0; i < sources.length; i++) {
    const s = sources[i];
    const block = `[${i + 1}] (${s.type}) ${s.title}\n${s.snippet}`;
    if (total + block.length > MAX_CONTEXT_CHARS) break;
    blocks.push(block);
    total += block.length;
  }
  return blocks.join('\n\n');
}

/**
 * System prompt — instructions + sources. Separated from the user message
 * so conversation history interleaves cleanly between system and user turns.
 * Matches the pattern used in auto-answer.controller.ts (line 183).
 */
function buildSystemPrompt(context: string): string {
  return `You are the Yaksha FAQ assistant. Answer the user's question using ONLY the sources provided below. Be honest about uncertainty — if the sources don't contain the answer, say so plainly and suggest they ask the community.

Cite sources inline by their bracketed index, e.g. "The NOC is required by your HOD before you sign it [1][3]." Use one citation per fact, multiple citations are fine when sources agree.

Keep the answer under 8 sentences unless the user explicitly asks for a longer explanation. Be specific. Use the user's tone (English / Hinglish mix is fine).

SOURCES
${context}`;
}

/** User message — just the question. Attachment notes are appended by the caller. */
function buildUserMessage(question: string): string {
  return question;
}

// ─── Cross-encoder re-ranking (PR 2) ─────────────────────────────────────────

/**
 * Re-rank sources using a cross-encoder model via HuggingFace Inference API.
 * Cross-encoders apply cross-attention between query and document, capturing
 * nuance that bi-encoder cosine scores miss (e.g. "how long NOC?" ranks
 * "NOC duration" higher than "NOC definition").
 *
 * Falls back to the original ordering if:
 *   - HUGGINGFACE_API_KEY is not set
 *   - The API call fails (timeout, rate-limit, model cold-start)
 *   - No sources to re-rank
 */
async function crossEncoderRerank(
  query: string,
  sources: RagSource[],
): Promise<RagSource[]> {
  if (sources.length <= 1) return sources;

  const hfKey = process.env['HUGGINGFACE_API_KEY'];
  if (!hfKey) {
    logger.info('[rag] cross-encoder skipped — HUGGINGFACE_API_KEY not set');
    return sources;
  }

  try {
    // Build query–document pairs for the cross-encoder.
    // Use title + snippet (first 200 chars) to keep input compact.
    const passages = sources.map((s) => `${s.title} ${s.snippet.slice(0, 200)}`);

    const res = await fetch(CROSS_ENCODER_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${hfKey}`,
      },
      body: JSON.stringify({
        inputs: {
          source_sentence: query,
          sentences: passages,
        },
      }),
      signal: AbortSignal.timeout(5000), // 5s timeout — don't block the pipeline
    });

    if (!res.ok) {
      logger.warn(`[rag] cross-encoder API error: ${res.status}`);
      return sources;
    }

    const scores = (await res.json()) as number[];
    if (!Array.isArray(scores) || scores.length !== sources.length) {
      logger.warn('[rag] cross-encoder returned unexpected shape, falling back');
      return sources;
    }

    // Merge cross-encoder scores and re-sort. Replace the original score
    // with the cross-encoder score so downstream (confidence calc, context
    // ordering) uses the higher-quality signal.
    const reranked = sources
      .map((s, i) => ({ ...s, score: scores[i] }))
      .sort((a, b) => b.score - a.score);

    logger.info('[rag] cross-encoder rerank done', {
      topBefore: sources[0]?.title?.slice(0, 40),
      topAfter: reranked[0]?.title?.slice(0, 40),
    });

    return reranked;
  } catch (err) {
    // Graceful fallback — cross-encoder is an optimisation, not a requirement.
    logger.warn(`[rag] cross-encoder failed, using bi-encoder order: ${(err as Error).message}`);
    return sources;
  }
}

// ─── Confidence scoring (PR 3) ────────────────────────────────────────────────

/**
 * Compute a confidence score in [0, 1] from source quality metrics.
 * Pure math — no LLM call. Formula:
 *
 *   confidence = avg(top3 scores) × log(sourceCount + 1) / log(13)
 *
 * - avg(top3):      strong retrieval → high base confidence
 * - log(count+1):   multiple agreeing sources → boost
 * - / log(13):      normalise so 12 sources (3×4 top-K) yields factor ~1.0
 *
 * Returns 0 when no sources exist.
 */
function computeConfidence(sources: RagSource[]): number {
  if (sources.length === 0) return 0;

  // Top-3 source scores (or fewer if we have < 3 sources).
  const topScores = sources
    .slice(0, 3)
    .map((s) => s.score);
  const avgTop = topScores.reduce((sum, s) => sum + s, 0) / topScores.length;

  // Source count factor: log(n+1) / log(13)
  const countFactor = Math.log(sources.length + 1) / Math.log(13);

  // Clamp to [0, 1].
  return Math.min(1, Math.max(0, avgTop * countFactor));
}

// ─── File attachment type ─────────────────────────────────────────────────────

/** File/image attachment passed in from the controller. */
export interface RagAttachment {
  /** Image (vision-capable) or text (read as part of context). */
  kind: 'image' | 'text';
  mimeType: string;
  /** For images: base64-encoded data. For text: UTF-8 string content. */
  data: string;
  /** Original filename, shown to the model. */
  filename: string;
}

// ─── Main RAG entry point ─────────────────────────────────────────────────────

/**
 * Main entry — runs the full RAG pipeline. Returns the answer + the
 * sources the LLM saw, in citation order. The caller (controller) just
 * forwards this as JSON.
 *
 * When `attachments` is provided, text files have their content inlined
 * into the prompt and images are sent as multi-part content (vision input)
 * to the LLM. Both Anthropic and OpenAI support this.
 *
 * When `history` is provided, previous conversation turns are injected
 * into the LLM messages array for multi-turn context (pronoun resolution,
 * follow-up questions). The frontend manages the sliding window.
 */
export async function runRag(
  question: string,
  attachments: RagAttachment[] = [],
  history: HistoryMessage[] = [],
): Promise<RagResult> {
  const t0 = Date.now();
  const embedding = await generateQueryEmbedding(question);
  logger.info('rag.embedding.done', { ms: Date.now() - t0 });

  // Fan out — 3 sources, top-K each, in parallel.
  const [faqHits, postHits, knowledgeHits] = await Promise.all([
    searchFaqs(embedding, question, TOP_K_PER_SOURCE).catch((e) => {
      logger.warn('rag.faq.search.failed', { error: (e as Error).message });
      return [] as FaqHit[];
    }),
    searchCommunity(embedding, question, TOP_K_PER_SOURCE).catch((e) => {
      logger.warn('rag.community.search.failed', { error: (e as Error).message });
      return [] as PostHit[];
    }),
    searchKnowledge(question, TOP_K_PER_SOURCE).catch((e) => {
      logger.warn('rag.knowledge.search.failed', { error: (e as Error).message });
      return [] as Awaited<ReturnType<typeof searchKnowledge>>;
    }),
  ]);

  // Normalize each source into the common shape.
  let sources: RagSource[] = [
    ...faqHits.map((h) => ({
      id: `faq:${String(h._id)}`,
      type: 'faq' as const,
      title: h.question,
      snippet: h.answer.slice(0, 600),
      url: `/faq/${String(h._id)}`,
      score: h.score,
    })),
    ...postHits.map((h) => ({
      id: `community:${String(h._id)}`,
      type: 'community' as const,
      title: h.title,
      snippet: h.body.slice(0, 600),
      url: `/community?post=${String(h._id)}`,
      score: h.score,
    })),
    ...knowledgeHits.map((h) => ({
      id: `knowledge:${h._id}`,
      type: 'knowledge' as const,
      title: h.question,
      snippet: h.answer.slice(0, 600),
      // Knowledge isn't a public page yet — link to the post that sourced it,
      // or to the admin KB if we know the meeting. For now, a stable
      // deep-link to a future /knowledge/:id is best-effort.
      url: `/community?post=${h._id}`,
      score: h.score,
    })),
  ];

  // Stage 1: Re-rank by bi-encoder score.
  sources.sort((a, b) => b.score - a.score);

  // Stage 2: Cross-encoder re-ranking for better retrieval quality.
  // Uses HuggingFace Inference API — falls back silently if unavailable.
  sources = await crossEncoderRerank(question, sources);

  // If we found nothing at all, skip the LLM call — just say "no answer".
  if (sources.length === 0) {
    return {
      answer: "I couldn't find anything relevant in the FAQ, community, or your team's Zoom knowledge base. Try rephrasing, or post a new question to the community.",
      sources: [],
      model: 'none',
      confidence: 0,
    };
  }

  // Compute confidence from source quality (before LLM call — pure math).
  const confidence = computeConfidence(sources);

  const context = buildContext(sources);
  const systemContent = buildSystemPrompt(context);

  // Build the user-message content. When there are attachments we send a
  // multi-part content array (text + image parts) instead of a plain string.
  // Text-file attachments are inlined into the user message so the LLM sees
  // them as part of the question context.
  const attachmentNote = attachments.length > 0
    ? `\n\n[Attached files (${attachments.length}): ${attachments.map((a) => a.filename).join(', ')}]`
    : '';
  const textAttachments = attachments
    .filter((a) => a.kind === 'text')
    .map((a) => `\n\n--- Attached file: ${a.filename} ---\n${a.data}\n--- end ---`)
    .join('');
  const userContent = buildUserMessage(question) + attachmentNote + textAttachments;
  const imageAttachments = attachments.filter((a) => a.kind === 'image');

  // Call the LLM. We use the same provider resolution as duplicate detection
  // and knowledge extraction so the same AI key chain powers the assistant.
  // If the AI fails (provider down / 403 / rate-limited), we still return
  // the sources so the frontend can show the top snippet as a fallback.
  let answer = '';
  let model = 'fallback';
  try {
    const cfg = await resolveProviderAsync();
    const t1 = Date.now();
    answer = await chatCompletion(cfg, systemContent, userContent, imageAttachments, history);
    model = cfg.model;
    logger.info('rag.completion.done', { ms: Date.now() - t1, model: cfg.model, sources: sources.length, attachments: attachments.length });
  } catch (llmErr) {
    logger.warn('rag.completion.failed', { error: (llmErr as Error).message });
    answer = sources[0]?.snippet ?? '';
  }

  return { answer, sources, model, confidence };
}

// ─── LLM chat completion helper ───────────────────────────────────────────────

/**
 * Chat completion helper with system/user message separation and
 * conversation history support.
 *
 * The system prompt (instructions + sources) is sent as role:"system".
 * Previous conversation turns are injected between system and the current
 * user message. This matches the pattern in auto-answer.controller.ts
 * and is the standard way all LLM APIs implement conversation memory.
 *
 * When `images` is non-empty, the user message is sent as a multi-part
 * content array (text + image parts). The exact shape depends on the
 * provider: Anthropic uses `{type:'image', source:{type:'base64',...}}`,
 * OpenAI-compatible uses `{type:'image_url', image_url:{url:'data:...'}}`
 */
async function chatCompletion(
  cfg: { apiKey: string; baseURL: string; model: string; provider: string; needsAnthropicVersion: boolean; authHeader: 'x-api-key' | 'Authorization' },
  systemContent: string,
  userContent: string,
  images: RagAttachment[] = [],
  history: HistoryMessage[] = [],
): Promise<string> {
  const authValue = cfg.provider === 'anthropic' ? cfg.apiKey : `Bearer ${cfg.apiKey}`;
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    [cfg.authHeader]: authValue,
  };

  // Sanitize history: drop empty messages (Anthropic returns 400 on empty
  // content strings) and enforce the sliding window.
  const safeHistory = history
    .filter((m) => m.content.trim() !== '')
    .slice(-MAX_HISTORY_MESSAGES);

  // Build the user message content. If no images, send as a plain string
  // (cheaper, works with every model). If images are present, send a
  // content array — the text becomes the first part.
  const buildContent = (): unknown => {
    if (images.length === 0) return userContent;
    if (cfg.provider === 'anthropic') {
      return [
        { type: 'text', text: userContent },
        ...images.map((img) => ({
          type: 'image' as const,
          source: { type: 'base64' as const, media_type: img.mimeType, data: img.data },
        })),
      ];
    }
    // OpenAI-compatible (openai, xai, minimax) — all use image_url with a data URI.
    return [
      { type: 'text', text: userContent },
      ...images.map((img) => ({
        type: 'image_url' as const,
        image_url: { url: `data:${img.mimeType};base64,${img.data}` },
      })),
    ];
  };

  if (cfg.needsAnthropicVersion) {
    headers['anthropic-version'] = '2023-06-01';
    const res = await fetch(`${cfg.baseURL}/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: cfg.model,
        system: systemContent,
        messages: [
          ...safeHistory,
          { role: 'user', content: buildContent() },
        ],
        max_tokens: 800,
      }),
    });
    if (!res.ok) throw new Error(`Anthropic error: ${res.status} ${await res.text()}`);
    const data = (await res.json()) as { content?: Array<{ text?: string }> };
    return data.content?.[0]?.text ?? '';
  }

  const res = await fetch(`${cfg.baseURL}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: cfg.model,
      messages: [
        { role: 'system', content: systemContent },
        ...safeHistory,
        { role: 'user', content: buildContent() },
      ],
      max_tokens: 800,
      temperature: 0.2,
    }),
  });
  if (!res.ok) throw new Error(`${cfg.provider} error: ${res.status} ${await res.text()}`);
  const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
  return data.choices?.[0]?.message?.content ?? '';
}