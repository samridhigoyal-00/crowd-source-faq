import { Request, Response } from 'express';
import { Types } from 'mongoose';
import CommunityPost, { ICommunityPost, IComment } from '../models/CommunityPost.js';
import { generateEmbedding } from '../utils/embeddings.js';
import User, { IUser } from '../models/User.js';
import { invalidateCache } from '../utils/cache.js';

// Extend Express Request to include user (same pattern as auth middleware)
declare global {
  namespace Express {
    interface Request {
      user?: IUser;
    }
  }
}

/** Build a nested comment tree from a flat comments array */
function buildCommentTree(flat: IComment[]): IComment[] {
  const map = new Map<string, IComment>();
  const roots: IComment[] = [];

  // Clone each comment so we can mutate safely
  for (const c of flat) {
    map.set(c._id.toString(), { ...c, replies: [] });
  }

  for (const c of flat) {
    const node = map.get(c._id.toString())!;
    if (node.parentId) {
      const parent = map.get(node.parentId.toString());
      if (parent) {
        parent.replies.push(node);
      } else {
        roots.push(node); // Orphaned reply — treat as root
      }
    } else {
      roots.push(node);
    }
  }

  return roots;
}

// GET /api/community — All posts (paginated)
export const getAllPosts = async (req: Request, res: Response): Promise<void> => {
  try {
    const page = Math.max(1, parseInt(req.query.page as string) || 1);
    const limit = Math.max(0, parseInt(req.query.limit as string) || 20); // default 20 per page
    const skip = (page - 1) * limit;

    const total = await CommunityPost.countDocuments();

    const posts = await CommunityPost.find({})
      .select('-embedding')
      .populate('author', 'name')
      .populate('comments.author', 'name')
      .populate('comments.upvotes', 'name')
      .populate('comments.downvotes', 'name')
      .populate('comments.replies.upvotes', 'name')
      .populate('comments.replies.downvotes', 'name')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit);

    res.json({
      posts,
      total,
      page,
      limit,
      pages: Math.ceil(total / limit),
      hasMore: skip + posts.length < total,
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// GET /api/community/:id — Single post with nested comment tree
export const getPostById = async (req: Request, res: Response): Promise<void> => {
  try {
    const post = await CommunityPost.findById(req.params.id)
      .select('-embedding')
      .populate('author', 'name')
      .populate('comments.author', 'name')
      .populate('comments.upvotes', 'name')
      .populate('comments.downvotes', 'name')
      .populate('comments.replies.upvotes', 'name')
      .populate('comments.replies.downvotes', 'name');

    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // Attach nested replies tree to the response
    const postObj = post.toObject();
    (postObj as any).comments = buildCommentTree(post.comments as unknown as IComment[]);

    res.json(postObj);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// GET /api/community/answers/list — Paginated list of posts with an official expert answer
export const getAnswersList = async (req: Request, res: Response): Promise<void> => {
  try {
    const page = Math.max(1, parseInt(req.query.page as string) || 1);
    const limit = Math.min(50, Math.max(0, parseInt(req.query.limit as string) || 20));
    const skip = (page - 1) * limit;

    const filter = { status: 'answered' };

    const total = await CommunityPost.countDocuments(filter);

    const posts = await CommunityPost.find(filter)
      .select('-embedding')
      .populate('author', 'name')
      .sort({ updatedAt: -1 })
      .skip(skip)
      .limit(limit);

    res.json({
      posts,
      total,
      page,
      limit,
      pages: Math.ceil(total / limit),
      hasMore: skip + posts.length < total,
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community — Create a new post (protected)
export const createPost = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: 'Not authorized' }); return; }
  try {
    const { title, body } = req.body as { title?: string; body?: string };

    // Validate inputs
    if (!title || !body) {
      res.status(400).json({ message: 'Title and body are required.' });
      return;
    }

    // ── Server-side duplicate check ──────────────────────────────────────────
    const words = title.trim().split(' ').filter((w) => w.length >= 3);
    const isShortQuery = words.length < 3;
    const matches = await checkDuplicate(title, isShortQuery);
    if (matches.length > 0) {
      res.status(409).json({
        message: 'This question has already been asked by the universe. Try searching first.',
        matches,
        isDuplicate: true,
      });
      return;
    }

    // Generate vector embedding for semantic search
    let embedding: number[] | undefined;
    try {
      embedding = await generateEmbedding(`Question: ${title}. Description: ${body}`);
    } catch (err) {
      console.warn('Failed to generate embedding for post:', (err as Error).message);
    }

    // Create post linked to the authenticated user with a default 'unanswered' status
    const post = await CommunityPost.create({
      title,
      body,
      author: req.user!._id,
      status: 'unanswered',
      embedding,
    });

    // Hydrate the author field before sending back the response
    await post.populate('author', 'name');

    // Invalidate search cache so new post appears in community search immediately
    await invalidateCache().catch(() => {});

    res.status(201).json({ post });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/upvote — Toggle upvote
export const toggleUpvote = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    const userId = req.user!._id.toString();

    // Check if the user has already upvoted the post
    const alreadyUpvoted = post.upvotes.map((u: Types.ObjectId) => u.toString()).includes(userId);

    // Toggle logic: remove user ID if already upvoted, otherwise push it to the array
    if (alreadyUpvoted) {
      post.upvotes = post.upvotes.filter((u: Types.ObjectId) => u.toString() !== userId);
    } else {
      post.upvotes.push(req.user!._id);
    }

    await post.save();
    res.json({ upvotes: post.upvotes.length, upvotedByMe: !alreadyUpvoted });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/comments — Add a comment or reply to another comment
// Query param: ?parentId=<commentId> to reply to a specific comment
export const addComment = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const { body } = req.body as { body?: string };
    const { parentId } = req.query as { parentId?: string };

    if (!body || !body.trim()) {
      res.status(400).json({ message: 'Comment body is required.' });
      return;
    }

    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // Resolve parent comment if this is a reply
    let resolvedParent: any = null;
    if (parentId) {
      resolvedParent = (post.comments as any).id(parentId);
      if (!resolvedParent) {
        res.status(404).json({ message: 'Parent comment not found.' });
        return;
      }
      if (resolvedParent.depth >= 3) {
        res.status(400).json({ message: 'Maximum reply depth (3) reached. Cannot nest deeper.' });
        return;
      }
    }

    // Build comment object with parentId and depth for replies
    const commentObj: Record<string, unknown> = { author: req.user!._id, body: body.trim() };
    if (resolvedParent) {
      commentObj.parentId = new Types.ObjectId(parentId);
      commentObj.depth = resolvedParent.depth + 1;
    } else {
      commentObj.parentId = null;
      commentObj.depth = 0;
    }

    post.comments.push(commentObj as any);
    await post.save();

    await post.populate('comments.author', 'name');
    const newComment = post.comments[post.comments.length - 1];

    // Notify post author
    if (post.author.toString() !== req.user!._id.toString()) {
      import('./notificationController.js').then(n =>
        n.createNotification({
          recipient: post.author,
          type: 'comment_replied',
          title: 'New comment on your post',
          message: `${req.user!.name} commented on "${post.title}": "${body.trim().slice(0, 80)}${body.trim().length > 80 ? '…' : ''}"`,
          link: `/community?post=${post._id}`,
        })
      ).catch(() => {});
    }

    // Notify parent comment author
    if (resolvedParent && resolvedParent.author.toString() !== req.user!._id.toString()) {
      import('./notificationController.js').then(n =>
        n.createNotification({
          recipient: resolvedParent.author,
          type: 'comment_replied',
          title: 'Someone replied to your comment',
          message: `${req.user!.name} replied: "${body.trim().slice(0, 80)}${body.trim().length > 80 ? '…' : ''}"`,
          link: `/community?post=${post._id}`,
        })
      ).catch(() => {});
    }

    res.status(201).json({ comment: newComment, total: post.comments.length });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// PATCH /api/community/:id/resolve — Mark as answered (admin/moderator only)
// POST /api/community/solved — Get recently resolved posts (for "Top Solved Today" widget)
// Query params: limit (default 4), since (hours, default 24)
export const getSolvedPosts = async (req: Request, res: Response): Promise<void> => {
  try {
    const limit = Math.min(parseInt(req.query.limit as string) || 4, 10);
    const hours = parseInt(req.query.hours as string) || 24;

    const since = new Date(Date.now() - hours * 60 * 60 * 1000);

    const posts = await CommunityPost.find({
      status: 'answered',
      updatedAt: { $gte: since },
    })
      .sort({ updatedAt: -1 })
      .limit(limit)
      .populate('author', 'name')
      .lean();

    res.json({ posts });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/resolve — Mark a community post as resolved (admin/mod only)
// When resolved, the post author is notified via the notification system
export const resolvePost = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const { answer } = req.body as { answer?: string };

    if (!answer || !answer.trim()) {
      res.status(400).json({ message: 'Answer text is required to resolve.' });
      return;
    }

    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    post.status = 'answered';
    post.answer = answer.trim();
    // Set answerIsExpert flag when a moderator or admin resolves the post
    if (req.user?.role === 'moderator' || req.user?.role === 'admin' || req.user?.role === 'expert') {
      post.answerIsExpert = true;
    }
    await post.save();

    // Invalidate search cache so resolved answer reflects immediately
    await invalidateCache().catch(() => {});

    // Notify the post author that their question was resolved
    await import('./notificationController.js').then(n => 
      n.createNotification({
        recipient: post.author,
        type: 'post_resolved',
        title: 'Your question was resolved!',
        message: `An admin resolved your question "${post.title}" — tap to see the answer.`,
        link: `/community?post=${post._id}`,
      })
    ).catch(() => {}); // non-critical — don't fail the resolve if notification creation fails

    res.json({ message: 'Post resolved.', post });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/request-expert — Request expert help on an unanswered post (protected)
// Notifies all moderators and admins
export const requestExpertHelp = async (req: Request, res: Response): Promise<void> => {
  try {
    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    if (post.status === 'answered') {
      res.status(400).json({ message: 'This post is already answered.' });
      return;
    }

    // Find all moderators and admins
    const moderatorsAndAdmins = await User.find({
      role: { $in: ['moderator', 'admin', 'expert'] },
    }).select('_id');

    // Create notifications for each moderator/admin
    const notificationPromises = moderatorsAndAdmins.map((mod) =>
      import('./notificationController.js').then((n) =>
        n.createNotification({
          recipient: mod._id,
          type: 'expert_request',
          title: 'Expert help requested!',
          message: `A student is waiting for help: "${post.title}"`,
          link: `/community?post=${post._id}`,
        })
      ).catch(() => {})
    );

    await Promise.all(notificationPromises);

    res.json({ message: 'Expert help requested. Moderators have been notified.' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// DELETE /api/community/:id — Delete a community post (Admin/Moderator only)
export const deletePost = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const post = await CommunityPost.findByIdAndDelete(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // Invalidate search cache so deleted post is removed from results
    await invalidateCache().catch(() => {});

    res.json({ message: 'Post deleted successfully.' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/comments/:commentId/upvote — Toggle upvote on a comment
export const toggleCommentUpvote = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // post.comments is a Mongoose subdocument array; .id() is a valid method on the array
    // but TypeScript's IComment[] type doesn't reflect it. Cast through 'any' to access runtime method.
    const comment = (post.comments as any).id(req.params.commentId);
    if (!comment) {
      res.status(404).json({ message: 'Comment not found.' });
      return;
    }

    const userId = req.user!._id.toString();
    const alreadyUpvoted = comment.upvotes.map((u: Types.ObjectId) => u.toString()).includes(userId);

    if (alreadyUpvoted) {
      // Remove upvote
      comment.upvotes = comment.upvotes.filter((u: Types.ObjectId) => u.toString() !== userId);
    } else {
      // Add upvote, remove from downvotes if present
      comment.upvotes.push(req.user!._id);
      comment.downvotes = comment.downvotes.filter((u: Types.ObjectId) => u.toString() !== userId);
    }

    await post.save();

    const netScore = comment.upvotes.length - comment.downvotes.length;
    res.json({
      upvotes: comment.upvotes.length,
      downvotes: comment.downvotes.length,
      netScore,
      upvotedByMe: !alreadyUpvoted,
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/comments/:commentId/downvote — Toggle downvote on a comment
// When net score reaches -5, the comment is auto-deleted and { deleted: true } is returned
export const toggleCommentDownvote = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // post.comments is a Mongoose subdocument array; .id() is a valid method on the array
    // but TypeScript's IComment[] type doesn't reflect it. Cast through 'any' to access runtime method.
    const comment = (post.comments as any).id(req.params.commentId);
    if (!comment) {
      res.status(404).json({ message: 'Comment not found.' });
      return;
    }

    const userId = req.user!._id.toString();
    const alreadyDownvoted = comment.downvotes.map((u: Types.ObjectId) => u.toString()).includes(userId);

    if (alreadyDownvoted) {
      // Remove downvote
      comment.downvotes = comment.downvotes.filter((u: Types.ObjectId) => u.toString() !== userId);
    } else {
      // Add downvote, remove from upvotes if present
      comment.downvotes.push(req.user!._id);
      comment.upvotes = comment.upvotes.filter((u: Types.ObjectId) => u.toString() !== userId);
    }

    const netScore = comment.upvotes.length - comment.downvotes.length;

    // Auto-delete comment if net score reaches -5
    if (netScore <= -5) {
      comment.deleteOne();
      await post.save();
      res.json({
        deleted: true,
        message: 'Comment obliterated.',
      });
      return;
    }

    await post.save();

    res.json({
      upvotes: comment.upvotes.length,
      downvotes: comment.downvotes.length,
      netScore,
      downvotedByMe: !alreadyDownvoted,
      deleted: false,
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// PATCH /api/community/:id/comments/:commentId/verify — Mark a comment as verified top answer
export const verifyComment = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // post.comments is a Mongoose subdocument array; .id() is a valid method on the array
    // but TypeScript's IComment[] type doesn't reflect it. Cast through 'any' to access runtime method.
    const comment = (post.comments as any).id(req.params.commentId);
    if (!comment) {
      res.status(404).json({ message: 'Comment not found.' });
      return;
    }

    // Toggle verified status
    comment.verified = !comment.verified;
    await post.save();

    res.json({ verified: comment.verified, commentId: req.params.commentId });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// POST /api/community/:id/report — Report a community post
export const reportPost = async (req: Request<{ id: string }, {}, { reason: string }>, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: 'Not authorized' }); return; }
  try {
    const { reason } = req.body;
    if (!reason || !reason.trim()) {
      res.status(400).json({ message: 'Reason is required.' });
      return;
    }

    const post = await CommunityPost.findById(req.params.id);
    if (!post) {
      res.status(404).json({ message: 'Post not found.' });
      return;
    }

    // Prevent duplicate reports by the same user
    const alreadyReported = post.reports.some(
      (r) => r.reportedBy.toString() === req.user!._id.toString()
    );
    if (alreadyReported) {
      res.status(409).json({ message: 'You have already reported this post.' });
      return;
    }

    post.reports.push({ reportedBy: req.user!._id, reason: reason.trim() });
    await post.save();

    res.json({ message: 'Report submitted. Thank you.' });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: (error as Error).message });
  }
};

// ─── Duplicate Detection ──────────────────────────────────────────────────────

const DUPLICATE_VECTOR_THRESHOLD = 0.78;
const DUPLICATE_TEXT_THRESHOLD = 0.35;
const DUPLICATE_SHORT_QUERY_THRESHOLD = 0.88; // higher bar for <3-word queries

export interface DuplicateMatch {
  _id: string;
  title: string;
  question?: string;
  answer?: string;
  body?: string;
  score: number;
  source: 'faq' | 'community';
  matchType: 'vector' | 'text';
}

/**
 * Hybrid duplicate check against FAQs and community posts.
 * Uses semantic similarity (vector) + keyword text matching.
 */
export async function checkDuplicate(
  query: string,
  isShortQuery: boolean
): Promise<DuplicateMatch[]> {
  const matches: DuplicateMatch[] = [];
  const lower = query.toLowerCase().trim();

  // ── 1. FAQ vector + keyword search ─────────────────────────────────────────
  try {
    const { generateEmbedding } = await import('../utils/embeddings.js');
    const { default: FAQ } = await import('../models/FAQ.js');

    const queryEmbedding = await generateEmbedding(query).catch(() => null);
    if (!queryEmbedding) throw new Error('Embedding generation failed');

    const vectorThreshold = isShortQuery ? DUPLICATE_SHORT_QUERY_THRESHOLD : DUPLICATE_VECTOR_THRESHOLD;

    // Run vector + text search in parallel
    const [vectorResults, textResults] = await Promise.all([
      FAQ.find({
        embedding: { $exists: true, $ne: null },
        status: 'approved',
      })
        .select('_id question answer category embedding')
        .lean()
        .then(async (faqs) => {
          // Compute cosine similarity in JS using dot product of normalized vectors
          const scored = faqs
            .map((f) => {
              const dot =
                f.embedding!.reduce((s: number, v: number, i: number) => s + v * queryEmbedding[i], 0);
              return { faq: f, similarity: dot };
            })
            .filter((x) => x.similarity >= vectorThreshold)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 5);
          return scored.map((x) => ({
            _id: (x.faq._id as unknown as Types.ObjectId).toString(),
            title: x.faq.question,
            question: x.faq.question,
            answer: x.faq.answer,
            category: x.faq.category,
            score: x.similarity,
            matchType: 'vector' as const,
          }));
        }),

      FAQ.find({
        status: 'approved',
        $or: [
          { question: { $regex: escapeRegex(lower), $options: 'i' } },
          { answer: { $regex: escapeRegex(lower), $options: 'i' } },
        ],
      })
        .select('_id question answer category')
        .lean()
        .then((faqs) => {
          const scored = faqs
            .map((f) => {
              const qlen = lower.split(' ').filter(Boolean).length;
              const qlenNorm = qlen / Math.max(1, lower.length);
              const qScore = Math.min(1, qlenNorm * 8);
              return { faq: f, score: qScore };
            })
            .filter((x) => x.score >= DUPLICATE_TEXT_THRESHOLD)
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);
          return scored.map((x) => ({
            _id: (x.faq._id as unknown as Types.ObjectId).toString(),
            title: x.faq.question,
            question: x.faq.question,
            answer: x.faq.answer,
            category: x.faq.category,
            score: x.score,
            matchType: 'text' as const,
          }));
        }),
    ]);

    const seenFaq = new Set<string>();
    for (const r of [...vectorResults, ...textResults]) {
      if (!seenFaq.has(r._id)) {
        seenFaq.add(r._id);
        matches.push({ ...r, source: 'faq' });
      }
    }
  } catch (err) {
    console.warn('FAQ duplicate check failed:', (err as Error).message);
  }

  // ── 2. Community post keyword search (title match) ──────────────────────────
  try {
    const { default: CommunityPost } = await import('../models/CommunityPost.js');

    const words = lower.split(' ').filter((w) => w.length >= 3);
    if (words.length > 0) {
      const textResults = await CommunityPost.find({
        $or: [
          { title: { $regex: escapeRegex(lower), $options: 'i' } },
          ...words.map((w) => ({ title: { $regex: `\\b${escapeRegex(w)}\\b`, $options: 'i' } })),
        ],
      })
        .select('_id title body status')
        .lean()
        .then((posts) => {
          const scored = posts
            .map((p) => {
              const tLower = p.title.toLowerCase();
              let matchWords = 0;
              for (const w of words) {
                if (tLower.includes(w)) matchWords++;
              }
              const score = words.length > 0 ? matchWords / words.length : 0;
              return { post: p, score };
            })
            .filter((x) => x.score >= DUPLICATE_TEXT_THRESHOLD)
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);
          return scored.map((x) => ({
            _id: (x.post._id as unknown as Types.ObjectId).toString(),
            title: x.post.title,
            body: x.post.body,
            status: x.post.status,
            score: x.score,
            matchType: 'text' as const,
          }));
        });

      const seenComm = new Set<string>();
      for (const r of textResults) {
        if (!seenComm.has(r._id)) {
          seenComm.add(r._id);
          matches.push({ ...r, source: 'community' });
        }
      }
    }
  } catch (err) {
    console.warn('Community duplicate check failed:', (err as Error).message);
  }

  // Sort by score descending, deduplicate
  return matches.sort((a, b) => b.score - a.score).slice(0, 5);
}

// POST /api/community/check-duplicate
export const checkDuplicateController = async (req: Request, res: Response): Promise<void> => {
  if (!req.user) { res.status(401).json({ message: "Not authorized" }); return; }
  try {
    const { query } = req.body as { query?: string };
    if (!query?.trim()) {
      res.json({ isDuplicate: false, matches: [] });
      return;
    }

    const words = query.trim().split(' ').filter((w) => w.length >= 3);
    const isShortQuery = words.length < 3;

    const matches = await checkDuplicate(query, isShortQuery);

    res.json({
      isDuplicate: matches.length > 0,
      matches,
      matchCount: matches.length,
    });
  } catch (error) {
    res.status(500).json({ message: 'Duplicate check failed', error: (error as Error).message });
  }
};

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}