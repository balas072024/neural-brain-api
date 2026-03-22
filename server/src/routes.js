const { Router } = require('express');
const { body, param, validationResult } = require('express-validator');
const bcrypt = require('bcryptjs');
const axios = require('axios');
const { getDb } = require('./db');
const { generateToken, authMiddleware } = require('./auth');

const router = Router();

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------
function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
router.get('/health', (_req, res) => {
  const db = getDb();
  const userCount = db.prepare('SELECT COUNT(*) as count FROM users').get().count;
  res.json({
    status: 'healthy',
    service: 'neural-brain-api',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    stats: { users: userCount }
  });
});

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
router.post(
  '/auth/login',
  body('username').isString().trim().notEmpty(),
  body('password').isString().notEmpty(),
  validate,
  (req, res) => {
    const db = getDb();
    const { username, password } = req.body;

    const user = db.prepare('SELECT * FROM users WHERE username = ?').get(username);
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });

    const valid = bcrypt.compareSync(password, user.password_hash);
    if (!valid) return res.status(401).json({ error: 'Invalid credentials' });

    const token = generateToken(user);
    res.json({
      token,
      user: { id: user.id, username: user.username, email: user.email, role: user.role }
    });
  }
);

router.get('/auth/me', authMiddleware, (req, res) => {
  const db = getDb();
  const user = db.prepare(
    'SELECT id, username, email, role, total_messages, total_tokens, created_at FROM users WHERE id = ?'
  ).get(req.user.id);
  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json({ user });
});

// ---------------------------------------------------------------------------
// MiniMax AI helper  (Anthropic-compatible endpoint)
// ---------------------------------------------------------------------------
const MINIMAX_URL = 'https://api.minimax.io/anthropic/v1/messages';
const MINIMAX_MODEL = 'MiniMax-M2.5';

function estimateTokens(text) {
  return Math.ceil((text || '').length / 4);
}

async function callMiniMax(messages, maxTokens = 1024) {
  const apiKey = process.env.MINIMAX_API_KEY;
  if (!apiKey) return null; // caller must handle 503

  const response = await axios.post(
    MINIMAX_URL,
    { model: MINIMAX_MODEL, messages, max_tokens: maxTokens },
    {
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      timeout: 30000
    }
  );

  const content = response.data.content;
  const text = Array.isArray(content)
    ? content.map(c => c.text || '').join('')
    : (typeof content === 'string' ? content : '');
  const tokens = (response.data.usage && response.data.usage.output_tokens) || estimateTokens(text);
  return { text, tokens };
}

// ---------------------------------------------------------------------------
// POST /chat  –  send message to AI
// ---------------------------------------------------------------------------
router.post(
  '/chat',
  authMiddleware,
  body('conversation_id').isInt(),
  body('message').isString().trim().notEmpty(),
  validate,
  async (req, res) => {
    const apiKey = process.env.MINIMAX_API_KEY;
    if (!apiKey) {
      return res.status(503).json({ error: 'AI service unavailable – MINIMAX_API_KEY not configured' });
    }

    try {
      const db = getDb();
      const { conversation_id, message } = req.body;

      const convo = db.prepare('SELECT * FROM conversations WHERE id = ? AND user_id = ?').get(conversation_id, req.user.id);
      if (!convo) return res.status(404).json({ error: 'Conversation not found' });

      // Save user message
      db.prepare('INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)').run(conversation_id, 'user', message);

      // Build context
      const history = db.prepare(
        'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 20'
      ).all(conversation_id).reverse();

      const aiMessages = [
        { role: 'user', content: history.map(m => `${m.role}: ${m.content}`).join('\n') }
      ];

      const result = await callMiniMax(aiMessages);
      const replyText = result.text;
      const tokens = result.tokens;

      // Save assistant reply
      db.prepare('INSERT INTO messages (conversation_id, role, content, tokens_used) VALUES (?, ?, ?, ?)').run(
        conversation_id, 'assistant', replyText, tokens
      );

      // Update stats
      db.prepare('UPDATE users SET total_messages = total_messages + 2, total_tokens = total_tokens + ? WHERE id = ?').run(tokens, req.user.id);
      db.prepare("UPDATE conversations SET updated_at = datetime('now') WHERE id = ?").run(conversation_id);

      res.json({ reply: replyText, tokens_used: tokens, conversation_id });
    } catch (err) {
      res.status(500).json({ error: 'AI processing failed', details: err.message });
    }
  }
);

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------
router.get('/conversations', authMiddleware, (req, res) => {
  const db = getDb();
  const conversations = db.prepare(
    'SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC'
  ).all(req.user.id);
  res.json({ conversations });
});

router.post(
  '/conversations',
  authMiddleware,
  body('title').isString().trim().notEmpty(),
  validate,
  (req, res) => {
    const db = getDb();
    const { title } = req.body;
    const model = req.body.model || 'minimax';

    const result = db.prepare('INSERT INTO conversations (user_id, title, model) VALUES (?, ?, ?)').run(req.user.id, title, model);
    const convo = db.prepare('SELECT * FROM conversations WHERE id = ?').get(result.lastInsertRowid);
    res.status(201).json({ conversation: convo });
  }
);

router.get(
  '/conversations/:id/messages',
  authMiddleware,
  param('id').isInt(),
  validate,
  (req, res) => {
    const db = getDb();
    const convo = db.prepare('SELECT * FROM conversations WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!convo) return res.status(404).json({ error: 'Conversation not found' });

    const messages = db.prepare('SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC').all(convo.id);
    res.json({ messages });
  }
);

router.delete(
  '/conversations/:id',
  authMiddleware,
  param('id').isInt(),
  validate,
  (req, res) => {
    const db = getDb();
    const convo = db.prepare('SELECT * FROM conversations WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!convo) return res.status(404).json({ error: 'Conversation not found' });

    db.prepare('DELETE FROM conversations WHERE id = ?').run(convo.id);
    res.json({ message: 'Conversation deleted' });
  }
);

router.get(
  '/conversations/:id/export',
  authMiddleware,
  param('id').isInt(),
  validate,
  (req, res) => {
    const db = getDb();
    const convo = db.prepare('SELECT * FROM conversations WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!convo) return res.status(404).json({ error: 'Conversation not found' });

    const messages = db.prepare('SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC').all(convo.id);

    const exportData = {
      conversation: convo,
      messages,
      exported_at: new Date().toISOString()
    };

    const filename = `conversation-${convo.id}-${Date.now()}.json`;
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.json(exportData);
  }
);

// ---------------------------------------------------------------------------
// Analyze – summarize
// ---------------------------------------------------------------------------
router.post(
  '/analyze/summarize',
  authMiddleware,
  body('text').isString().trim().notEmpty(),
  body('max_length').optional().isInt({ min: 20, max: 2000 }),
  validate,
  async (req, res) => {
    try {
      const { text, max_length } = req.body;
      const maxLen = max_length || 200;

      // Try AI if available, otherwise simple extractive summary
      let summary;
      let tokens = 0;

      const apiKey = process.env.MINIMAX_API_KEY;
      if (apiKey) {
        const aiMessages = [
          { role: 'user', content: `Summarize the following text in at most ${maxLen} characters. Be concise:\n\n${text}` }
        ];
        const result = await callMiniMax(aiMessages, 512);
        summary = result.text;
        tokens = result.tokens;
      } else {
        // Simple extractive summary: first N characters
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        summary = sentences.slice(0, 3).join('. ').trim();
        if (summary.length > maxLen) summary = summary.slice(0, maxLen) + '...';
        if (!summary) summary = text.slice(0, maxLen);
        tokens = estimateTokens(summary);
      }

      const db = getDb();
      db.prepare('UPDATE users SET total_tokens = total_tokens + ? WHERE id = ?').run(tokens, req.user.id);

      res.json({ summary, tokens_used: tokens });
    } catch (err) {
      res.status(500).json({ error: 'Summarization failed', details: err.message });
    }
  }
);

// ---------------------------------------------------------------------------
// Analyze – translate
// ---------------------------------------------------------------------------
router.post(
  '/analyze/translate',
  authMiddleware,
  body('text').isString().trim().notEmpty(),
  body('target_language').isString().trim().notEmpty(),
  body('source_language').optional().isString().trim(),
  validate,
  async (req, res) => {
    try {
      const { text, target_language, source_language } = req.body;
      const srcLang = source_language || 'auto-detect';

      let translation;
      let tokens = 0;

      const apiKey = process.env.MINIMAX_API_KEY;
      if (apiKey) {
        const aiMessages = [
          { role: 'user', content: `Translate the following text from ${srcLang} to ${target_language}. Return only the translated text:\n\n${text}` }
        ];
        const result = await callMiniMax(aiMessages, 1024);
        translation = result.text;
        tokens = result.tokens;
      } else {
        // Simple fallback – just echo with note
        translation = `[${target_language}] ${text}`;
        tokens = estimateTokens(translation);
      }

      const db = getDb();
      db.prepare('UPDATE users SET total_tokens = total_tokens + ? WHERE id = ?').run(tokens, req.user.id);

      res.json({ translation, source_language: srcLang, target_language, tokens_used: tokens });
    } catch (err) {
      res.status(500).json({ error: 'Translation failed', details: err.message });
    }
  }
);

// ---------------------------------------------------------------------------
// Analyze – sentiment
// ---------------------------------------------------------------------------
router.post(
  '/analyze/sentiment',
  authMiddleware,
  body('text').isString().trim().notEmpty(),
  validate,
  (req, res) => {
    try {
      const { text } = req.body;

      const positiveWords = ['good', 'great', 'excellent', 'amazing', 'love', 'happy', 'wonderful', 'fantastic', 'awesome', 'best', 'beautiful', 'joy', 'pleased', 'brilliant'];
      const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'ugly', 'angry', 'sad', 'disappointing', 'poor', 'fail', 'broken', 'annoying'];

      const words = text.toLowerCase().split(/\W+/);
      let score = 0;
      let posCount = 0;
      let negCount = 0;
      for (const w of words) {
        if (positiveWords.includes(w)) { score += 1; posCount++; }
        if (negativeWords.includes(w)) { score -= 1; negCount++; }
      }

      let sentiment, confidence;
      const total = posCount + negCount;
      if (total === 0) {
        sentiment = 'neutral';
        confidence = 0.5;
      } else if (score > 0) {
        sentiment = 'positive';
        confidence = Math.min(0.95, 0.5 + posCount / (total * 2));
      } else if (score < 0) {
        sentiment = 'negative';
        confidence = Math.min(0.95, 0.5 + negCount / (total * 2));
      } else {
        sentiment = 'neutral';
        confidence = 0.4;
      }

      res.json({
        sentiment,
        confidence: Math.round(confidence * 100) / 100,
        score,
        positive_words: posCount,
        negative_words: negCount
      });
    } catch (err) {
      res.status(500).json({ error: 'Sentiment analysis failed', details: err.message });
    }
  }
);

// ---------------------------------------------------------------------------
// Prompts library
// ---------------------------------------------------------------------------
router.get('/prompts', authMiddleware, (req, res) => {
  const db = getDb();
  const category = req.query.category;
  let prompts;
  if (category) {
    prompts = db.prepare(
      'SELECT * FROM prompts_library WHERE (user_id = ? OR is_public = 1) AND category = ? ORDER BY usage_count DESC'
    ).all(req.user.id, category);
  } else {
    prompts = db.prepare(
      'SELECT * FROM prompts_library WHERE user_id = ? OR is_public = 1 ORDER BY usage_count DESC'
    ).all(req.user.id);
  }
  res.json({ prompts });
});

router.post(
  '/prompts',
  authMiddleware,
  body('title').isString().trim().notEmpty(),
  body('content').isString().trim().notEmpty(),
  body('category').optional().isString().trim(),
  validate,
  (req, res) => {
    const db = getDb();
    const { title, content, category } = req.body;
    const result = db.prepare(
      'INSERT INTO prompts_library (user_id, title, content, category) VALUES (?, ?, ?, ?)'
    ).run(req.user.id, title, content, category || 'general');
    const prompt = db.prepare('SELECT * FROM prompts_library WHERE id = ?').get(result.lastInsertRowid);
    res.status(201).json({ prompt });
  }
);

router.post(
  '/prompts/:id/use',
  authMiddleware,
  param('id').isInt(),
  validate,
  (req, res) => {
    const db = getDb();
    const prompt = db.prepare('SELECT * FROM prompts_library WHERE id = ? AND (user_id = ? OR is_public = 1)').get(req.params.id, req.user.id);
    if (!prompt) return res.status(404).json({ error: 'Prompt not found' });

    db.prepare('UPDATE prompts_library SET usage_count = usage_count + 1 WHERE id = ?').run(prompt.id);
    res.json({ prompt: { ...prompt, usage_count: prompt.usage_count + 1 } });
  }
);

router.delete(
  '/prompts/:id',
  authMiddleware,
  param('id').isInt(),
  validate,
  (req, res) => {
    const db = getDb();
    const prompt = db.prepare('SELECT * FROM prompts_library WHERE id = ? AND user_id = ?').get(req.params.id, req.user.id);
    if (!prompt) return res.status(404).json({ error: 'Prompt not found' });

    db.prepare('DELETE FROM prompts_library WHERE id = ?').run(prompt.id);
    res.json({ message: 'Prompt deleted' });
  }
);

// ---------------------------------------------------------------------------
// Usage stats
// ---------------------------------------------------------------------------
router.get('/usage/stats', authMiddleware, (req, res) => {
  const db = getDb();
  const user = db.prepare('SELECT total_messages, total_tokens FROM users WHERE id = ?').get(req.user.id);
  if (!user) return res.status(404).json({ error: 'User not found' });

  const conversationCount = db.prepare('SELECT COUNT(*) as count FROM conversations WHERE user_id = ?').get(req.user.id).count;
  const promptCount = db.prepare('SELECT COUNT(*) as count FROM prompts_library WHERE user_id = ?').get(req.user.id).count;

  const daily = db.prepare(`
    SELECT date(m.created_at) as day, COUNT(*) as count
    FROM messages m
    JOIN conversations c ON c.id = m.conversation_id
    WHERE c.user_id = ? AND m.created_at >= datetime('now', '-7 days')
    GROUP BY day ORDER BY day
  `).all(req.user.id);

  res.json({
    total_messages: user.total_messages,
    total_tokens: user.total_tokens,
    conversations: conversationCount,
    saved_prompts: promptCount,
    daily_activity: daily
  });
});

module.exports = router;
