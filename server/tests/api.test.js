const request = require('supertest');
const bcrypt = require('bcryptjs');
const path = require('path');
const fs = require('fs');

// Use a separate test database
const TEST_DB = path.join(__dirname, '..', '..', 'data', 'test-neural-brain.db');
process.env.DB_PATH = TEST_DB;
process.env.JWT_SECRET = 'test-secret-key';
// Ensure no MINIMAX_API_KEY so chat returns 503
delete process.env.MINIMAX_API_KEY;

const { app } = require('../src/index');
const { getDb, closeDb } = require('../src/db');

let token;
let userId;

beforeAll(() => {
  const db = getDb();
  // Create test user
  const hash = bcrypt.hashSync('testpass', 10);
  const result = db.prepare(
    'INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)'
  ).run('testuser', 'test@arivu.ai', hash, 'user');
  userId = Number(result.lastInsertRowid);

  // Create admin user with Neural@2024 password
  const adminHash = bcrypt.hashSync('Neural@2024', 10);
  db.prepare(
    'INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)'
  ).run('admin', 'admin@arivu.ai', adminHash, 'admin');
});

afterAll(() => {
  closeDb();
  try { fs.unlinkSync(TEST_DB); } catch (_e) { /* ignore */ }
  try { fs.unlinkSync(TEST_DB + '-wal'); } catch (_e) { /* ignore */ }
  try { fs.unlinkSync(TEST_DB + '-shm'); } catch (_e) { /* ignore */ }
});

// ---------------------------------------------------------------------------
// 1. Health
// ---------------------------------------------------------------------------
describe('Health', () => {
  test('GET /api/health returns healthy status', async () => {
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toBe('healthy');
    expect(res.body.service).toBe('neural-brain-api');
    expect(res.body.version).toBe('1.0.0');
    expect(res.body.timestamp).toBeDefined();
    expect(res.body.stats.users).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// 2-6. Auth
// ---------------------------------------------------------------------------
describe('Auth', () => {
  test('POST /api/auth/login with valid credentials returns token', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({ username: 'testuser', password: 'testpass' });
    expect(res.status).toBe(200);
    expect(res.body.token).toBeDefined();
    expect(res.body.user.username).toBe('testuser');
    token = res.body.token;
  });

  test('POST /api/auth/login with admin credentials works', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({ username: 'admin', password: 'Neural@2024' });
    expect(res.status).toBe(200);
    expect(res.body.token).toBeDefined();
    expect(res.body.user.role).toBe('admin');
  });

  test('POST /api/auth/login with wrong password returns 401', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({ username: 'testuser', password: 'wrong' });
    expect(res.status).toBe(401);
    expect(res.body.error).toBe('Invalid credentials');
  });

  test('POST /api/auth/login with missing fields returns 400', async () => {
    const res = await request(app).post('/api/auth/login').send({});
    expect(res.status).toBe(400);
  });

  test('GET /api/auth/me returns current user', async () => {
    const res = await request(app)
      .get('/api/auth/me')
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(res.body.user.username).toBe('testuser');
    expect(res.body.user.email).toBe('test@arivu.ai');
  });

  test('GET /api/auth/me without token returns 401', async () => {
    const res = await request(app).get('/api/auth/me');
    expect(res.status).toBe(401);
  });
});

// ---------------------------------------------------------------------------
// 7-11. Conversations
// ---------------------------------------------------------------------------
let conversationId;

describe('Conversations', () => {
  test('POST /api/conversations creates a new conversation', async () => {
    const res = await request(app)
      .post('/api/conversations')
      .set('Authorization', `Bearer ${token}`)
      .send({ title: 'Test Conversation' });
    expect(res.status).toBe(201);
    expect(res.body.conversation.title).toBe('Test Conversation');
    conversationId = res.body.conversation.id;
  });

  test('GET /api/conversations lists user conversations', async () => {
    const res = await request(app)
      .get('/api/conversations')
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body.conversations)).toBe(true);
    expect(res.body.conversations.length).toBeGreaterThanOrEqual(1);
  });

  test('GET /api/conversations/:id/messages returns messages for conversation', async () => {
    const res = await request(app)
      .get(`/api/conversations/${conversationId}/messages`)
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body.messages)).toBe(true);
  });

  test('GET /api/conversations/:id/messages with invalid id returns 404', async () => {
    const res = await request(app)
      .get('/api/conversations/99999/messages')
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(404);
  });

  test('DELETE /api/conversations/:id deletes conversation', async () => {
    const createRes = await request(app)
      .post('/api/conversations')
      .set('Authorization', `Bearer ${token}`)
      .send({ title: 'To Delete' });
    const deleteId = createRes.body.conversation.id;

    const res = await request(app)
      .delete(`/api/conversations/${deleteId}`)
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(res.body.message).toBe('Conversation deleted');
  });
});

// ---------------------------------------------------------------------------
// 12-13. Chat – POST /api/chat (503 when no API key)
// ---------------------------------------------------------------------------
describe('Chat', () => {
  test('POST /api/chat returns 503 when MINIMAX_API_KEY is not set', async () => {
    const res = await request(app)
      .post('/api/chat')
      .set('Authorization', `Bearer ${token}`)
      .send({ conversation_id: conversationId, message: 'Hello, how are you?' });
    expect(res.status).toBe(503);
    expect(res.body.error).toMatch(/unavailable|MINIMAX_API_KEY/i);
  });

  test('POST /api/chat without auth returns 401', async () => {
    const res = await request(app)
      .post('/api/chat')
      .send({ conversation_id: conversationId, message: 'Hello' });
    expect(res.status).toBe(401);
  });
});

// ---------------------------------------------------------------------------
// 14-15. Analyze - summarize
// ---------------------------------------------------------------------------
describe('Analyze - summarize', () => {
  test('POST /api/analyze/summarize returns summary', async () => {
    const res = await request(app)
      .post('/api/analyze/summarize')
      .set('Authorization', `Bearer ${token}`)
      .send({
        text: 'Artificial intelligence is transforming the world. It is used in healthcare, finance, education, and many other industries. The technology continues to advance rapidly.'
      });
    expect(res.status).toBe(200);
    expect(res.body.summary).toBeDefined();
    expect(typeof res.body.summary).toBe('string');
    expect(res.body.tokens_used).toBeDefined();
  });

  test('POST /api/analyze/summarize with empty text returns 400', async () => {
    const res = await request(app)
      .post('/api/analyze/summarize')
      .set('Authorization', `Bearer ${token}`)
      .send({ text: '' });
    expect(res.status).toBe(400);
  });
});

// ---------------------------------------------------------------------------
// 16-18. Analyze - sentiment
// ---------------------------------------------------------------------------
describe('Analyze - sentiment', () => {
  test('POST /api/analyze/sentiment detects positive sentiment', async () => {
    const res = await request(app)
      .post('/api/analyze/sentiment')
      .set('Authorization', `Bearer ${token}`)
      .send({ text: 'This product is amazing and wonderful! I love it so much.' });
    expect(res.status).toBe(200);
    expect(res.body.sentiment).toBe('positive');
    expect(res.body.confidence).toBeGreaterThan(0);
    expect(res.body.positive_words).toBeGreaterThan(0);
  });

  test('POST /api/analyze/sentiment detects negative sentiment', async () => {
    const res = await request(app)
      .post('/api/analyze/sentiment')
      .set('Authorization', `Bearer ${token}`)
      .send({ text: 'This is terrible and awful. I hate it. The worst experience ever.' });
    expect(res.status).toBe(200);
    expect(res.body.sentiment).toBe('negative');
    expect(res.body.negative_words).toBeGreaterThan(0);
    expect(res.body.score).toBeLessThan(0);
  });

  test('POST /api/analyze/sentiment detects neutral sentiment', async () => {
    const res = await request(app)
      .post('/api/analyze/sentiment')
      .set('Authorization', `Bearer ${token}`)
      .send({ text: 'The meeting is scheduled for Tuesday at 3pm in room 204.' });
    expect(res.status).toBe(200);
    expect(res.body.sentiment).toBe('neutral');
  });
});

// ---------------------------------------------------------------------------
// 19. Analyze - translate
// ---------------------------------------------------------------------------
describe('Analyze - translate', () => {
  test('POST /api/analyze/translate returns translation', async () => {
    const res = await request(app)
      .post('/api/analyze/translate')
      .set('Authorization', `Bearer ${token}`)
      .send({ text: 'Hello world', target_language: 'Spanish' });
    expect(res.status).toBe(200);
    expect(res.body.translation).toBeDefined();
    expect(res.body.target_language).toBe('Spanish');
  });
});

// ---------------------------------------------------------------------------
// 20-23. Prompts library
// ---------------------------------------------------------------------------
let promptId;

describe('Prompts library', () => {
  test('POST /api/prompts creates a new prompt', async () => {
    const res = await request(app)
      .post('/api/prompts')
      .set('Authorization', `Bearer ${token}`)
      .send({ title: 'Test Prompt', content: 'Do something useful with: {{input}}', category: 'testing' });
    expect(res.status).toBe(201);
    expect(res.body.prompt.title).toBe('Test Prompt');
    expect(res.body.prompt.category).toBe('testing');
    promptId = res.body.prompt.id;
  });

  test('GET /api/prompts lists prompts', async () => {
    const res = await request(app)
      .get('/api/prompts')
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body.prompts)).toBe(true);
    expect(res.body.prompts.length).toBeGreaterThanOrEqual(1);
  });

  test('DELETE /api/prompts/:id deletes prompt', async () => {
    const res = await request(app)
      .delete(`/api/prompts/${promptId}`)
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(res.body.message).toBe('Prompt deleted');
  });

  test('DELETE /api/prompts/:id with invalid id returns 404', async () => {
    const res = await request(app)
      .delete('/api/prompts/99999')
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(404);
  });
});

// ---------------------------------------------------------------------------
// 24. Usage stats
// ---------------------------------------------------------------------------
describe('Usage stats', () => {
  test('GET /api/usage/stats returns user statistics', async () => {
    const res = await request(app)
      .get('/api/usage/stats')
      .set('Authorization', `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(typeof res.body.total_messages).toBe('number');
    expect(typeof res.body.total_tokens).toBe('number');
    expect(typeof res.body.conversations).toBe('number');
    expect(Array.isArray(res.body.daily_activity)).toBe(true);
  });
});
