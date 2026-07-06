# CSFAQ Contributions — Team Submission

This document details the contributions made by our team to the **Crowd Source FAQ (Yaksha FAQ Portal / Samagama)** project.

**Upstream repository:** https://github.com/vicharanashala/crowd-source-faq

---

## Team & Contributions

| Name | Contribution |
|---|---|
| **Samridhi Goyal** | Team Lead |
| **Dhananjay Dinesh K** | Maintaining `PRODUCT.md` |
| **Patan Jamsheer** | Code cleanup changes |
| **Ranjan Gupta** | Code cleanup changes |
| **Jandhyala Uma Sri Lekha** | Maintaining project report |
| **Lovepreet** | Security features (sent via email — see [Section 3](#3-security-features)) |
| **Kashish Gupta** | AI features |
| **Aman Sagar** | Project guidance |
| **Mohammad Hamdan** | Project discussion |
| **Sunaina** | Project discussion |
| **Vishal Kumar** | Project discussion |

---

## 1. AI Features — Conversation Memory for the Ask-AI Assistant

### Problem

The Yaksha FAQ Assistant (`/ask-ai`) is a RAG-powered chatbot that answers student queries using MongoDB Vector Search combined with Reciprocal Rank Fusion and LLM synthesis, drawing on 145 FAQs across 14 categories. The feature is fully built but shipped hidden behind an admin feature flag (`askAiChatbot.defaultEnabled: false`).

**Core bug — no conversational memory:** every question was sent to the LLM as a single, isolated message with no prior context. As a result, the assistant could not resolve follow-up questions.

> Turn 1 — User: *"What is a NOC?"* → Assistant explains correctly.
> Turn 2 — User: *"How long does it take?"* → Assistant has no idea "it" refers to the NOC, and returns an unrelated answer.

**Root cause:** in `apps/backend/src/modules/ai/rag.service.ts`, both the Anthropic and OpenAI/Gemini branches of `chatCompletion()` sent a single `{ role: "user", content: buildContent() }` message with no history, and the request never carried a history payload from `knowledge.controller.ts` or the frontend (`AskAIButton.tsx`).

**Secondary issue:** the system prompt (instructions + retrieved sources) was concatenated directly into the *user* message inside `buildPrompt()`, instead of being sent as a separate `system` message — inconsistent with the pattern already used elsewhere in the codebase (`auto-answer.controller.ts`), and something that would make clean history interleaving impossible if left unfixed.

### Solution

LLM APIs are stateless — "memory" is simply passing prior turns back in the `messages` array on every call. The fix implements a **6-message sliding window** (the last 3 user/assistant exchanges, ≈400 tokens), held entirely in frontend React state and never persisted server-side, so the backend remains stateless and a page refresh correctly resets the session.

Changes were scoped to **3 files, ~50 lines, 0 new dependencies, 0 new environment variables, 0 new infrastructure**:

1. **`apps/backend/src/modules/ai/rag.service.ts`**
   - `runRag()` and `chatCompletion()` now accept an optional `history: { role, content }[]` parameter.
   - `buildPrompt()` was split into `buildSystemPrompt(context)` and `buildUserMessage(question)`, so the system instructions and sources are sent as a proper `system` message, separate from the user's question.
   - History is sanitized (empty-content messages filtered out) and trimmed to the last 6 messages before being injected into both the Anthropic and OpenAI/Gemini message arrays.

2. **`apps/backend/src/modules/knowledge/knowledge.controller.ts`**
   - The request body type now accepts an optional `history` field, parsed as either a JSON array (`application/json` requests) or a JSON string (`multipart/form-data` requests with file attachments, since `multer` flattens all fields to strings).
   - `runRag()` is called with the parsed history; the parameter defaults to `[]`, so existing callers are unaffected.

3. **`apps/frontend/src/components/askai/AskAIButton.tsx`**
   - The `send()` callback now builds a `history` array from the existing `messages` state, filtering out in-flight, error, and attachment-only turns, and slicing to the last 6 entries.
   - History is sent in both the JSON path (`{ question, history }`) and the `FormData` path (`fd.append("history", JSON.stringify(history))`).

### Impact

- The LLM now receives full conversation context and resolves pronouns/follow-ups natively — no query rewriting, no Redis session store, no extra LLM calls.
- System/user separation is now consistent with the rest of the codebase.
- Fully backward-compatible: all new parameters default to `[]`, so turn-1 behavior is identical to before the change.

### Verification performed

- **Type check:** `npx tsc --noEmit`
- **Conversation test:** enabled `askAiChatbot` flag at `/admin/features`; ran a 3-turn chain ("What is a NOC?" → "How long does it take?" → "Who approves it?") and confirmed each follow-up correctly referenced the NOC context; confirmed the `history` array grows each turn via DevTools → Network.
- **Attachment test:** confirmed a file-attached turn followed by a text follow-up works with no 400 error, and that `history` is correctly carried as stringified JSON in the `FormData` request.
- **Edge cases:** verified turn-1 behavior is unchanged (`history = []`), verified `.slice(-6)` correctly trims long sessions, and verified error turns are excluded from the history sent to the model.

### Proposed follow-up AI/ML work (not yet implemented, scoped for future PRs)

- **Cross-encoder re-ranking:** re-rank the top-12 retrieved candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` (via the HuggingFace Inference API, already integrated) after the initial bi-encoder cosine-similarity retrieval, to improve relevance ordering before the LLM sees the sources (~40 lines, no new infrastructure).
- **Confidence-calibrated responses:** compute a confidence score from the top-3 source scores, source count, and score variance, and surface it to the user as a "High confidence" / "Uncertain — ask community" badge, so users know when to trust an answer versus escalate.
- Explicitly scoped *out*: semantic duplicate detection (already implemented elsewhere via vector similarity) and query rewriting / a Redis session store (unnecessary overhead for a 145-FAQ domain where the LLM already resolves references natively from history).

---

## 2. Code Cleanup

A code cleanup pass was carried out on the FAQ pagination logic and the community page, verified directly against the commit merged into the working fork:

**Fork:** https://github.com/samridhigoyal-00/crowd-source-faq (forked from `vicharanashala/crowd-source-faq`)
**Merged as:** PR #1, 2 files changed, 26 insertions / 25 deletions

1. **`apps/backend/src/modules/faq/faq.controller.ts`**
   - The base64 cursor-decoding logic for keyset pagination was duplicated across two functions (`getAllFAQs` and `getPaginatedFAQs`).
   - This was extracted into a single shared `decodeCursor()` helper, which both functions now call — removing the duplication and centralizing the cursor-parsing/error handling in one place.

2. **`apps/frontend/src/pages/CommunityPage.tsx`**
   - `visible`, `displayedPosts`, `answeredCount`, and `unansweredCount` were previously recomputed on every render.
   - These were wrapped in `useMemo()` with the correct dependency arrays (`search`, `searchResults`, `posts`, `sort`, `filter`, `visible`), so they now only recompute when their underlying dependencies actually change.

Together, these changes reduce code duplication in the backend pagination logic and improve rendering efficiency on the community page, without altering existing functionality or API behavior.

---

## 3. Security Features

A set of security enhancements has been implemented for the platform.

**These changes are not included in this public repository or PR.** Given the sensitive nature of security-related code (e.g., details that could expose attack surfaces if published openly before a fix is deployed), a detailed report describing each security feature/fix, along with the corresponding code files, has been **sent separately via email** to the maintainers/reviewers rather than raised as a public pull request.

Reviewers should refer to that email submission for the full write-up and code.

---

## Summary

This submission brings together three distinct contributions to the CSFAQ project: an AI feature fix restoring conversational memory to the Ask-AI assistant, a code cleanup pass improving code quality and maintainability across the codebase, and a set of security enhancements shared separately via email due to their sensitive nature. Alongside the code contributions, the team maintained the supporting product and project documentation throughout, under overall team leadership and project guidance.
