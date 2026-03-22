require('dotenv').config();

const bcrypt = require('bcryptjs');
const { getDb, closeDb } = require('./db');

function seed() {
  const db = getDb();

  console.log('Seeding Neural Brain database...');

  // ------- Users -------
  const hash = (pw) => bcrypt.hashSync(pw, 10);

  const insertUser = db.prepare(`
    INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)
  `);

  insertUser.run('admin', 'admin@arivu.ai', hash('Neural@2024'), 'admin');
  insertUser.run('demo', 'demo@arivu.ai', hash('demo123'), 'user');
  insertUser.run('alice', 'alice@arivu.ai', hash('alice123'), 'user');

  // ------- Conversations -------
  const insertConvo = db.prepare(`
    INSERT OR IGNORE INTO conversations (user_id, title, model) VALUES (?, ?, ?)
  `);

  insertConvo.run(2, 'Getting Started', 'minimax');
  insertConvo.run(2, 'Code Review Help', 'minimax');
  insertConvo.run(3, 'Translation Tasks', 'minimax');

  // ------- Messages -------
  const insertMsg = db.prepare(`
    INSERT INTO messages (conversation_id, role, content, tokens_used) VALUES (?, ?, ?, ?)
  `);

  insertMsg.run(1, 'user', 'Hello, what can you do?', 0);
  insertMsg.run(1, 'assistant', 'Hello! I am Neural Brain, your AI assistant. I can help with chat, text summarization, translation, sentiment analysis, and more. How can I assist you today?', 42);
  insertMsg.run(1, 'user', 'Can you summarize long texts?', 0);
  insertMsg.run(1, 'assistant', 'Absolutely! Send me any text and I will provide a concise summary capturing the key points. You can also use the dedicated /api/analysis/summarize endpoint for batch processing.', 48);

  insertMsg.run(2, 'user', 'Please review this function for potential bugs.', 0);
  insertMsg.run(2, 'assistant', 'I would be happy to help review your code. Please share the function and I will analyze it for potential bugs, performance issues, and best practices.', 38);

  // Update user stats
  db.prepare('UPDATE users SET total_messages = 6, total_tokens = 128 WHERE id = 2').run();

  // ------- Prompts library -------
  const insertPrompt = db.prepare(`
    INSERT OR IGNORE INTO prompts_library (user_id, title, content, category, is_public, usage_count) VALUES (?, ?, ?, ?, ?, ?)
  `);

  insertPrompt.run(1, 'Code Review', 'Review the following code for bugs, security issues, and performance improvements. Provide specific suggestions.', 'development', 1, 15);
  insertPrompt.run(1, 'Summarize Article', 'Summarize the following article in 3-5 bullet points, focusing on the key takeaways.', 'writing', 1, 22);
  insertPrompt.run(1, 'Explain Like I\'m 5', 'Explain the following concept in simple terms that a 5-year-old could understand.', 'education', 1, 18);
  insertPrompt.run(2, 'Debug Helper', 'I have the following error. Help me debug it step by step:\n\nError: {{error}}\nCode: {{code}}', 'development', 0, 5);
  insertPrompt.run(2, 'Email Writer', 'Write a professional email about the following topic. Keep it concise and polite.\n\nTopic: {{topic}}', 'writing', 0, 3);

  console.log('Seed complete!');
  console.log('  Users: admin/Neural@2024, demo/demo123, alice/alice123');
  closeDb();
}

seed();
