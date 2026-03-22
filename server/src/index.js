require('dotenv').config();

const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const path = require('path');

const routes = require('./routes');
const { getDb } = require('./db');

const app = express();
const PORT = process.env.PORT || 8200;

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.set('trust proxy', 1);

app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
app.use(cors());
app.use(express.json({ limit: '2mb' }));

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 300,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later.' }
});
app.use('/api/', limiter);

// ---------------------------------------------------------------------------
// Static files
// ---------------------------------------------------------------------------
app.use(express.static(path.join(__dirname, '..', '..', 'public')));

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------
app.use('/api', routes);

// ---------------------------------------------------------------------------
// Fallback for SPA
// ---------------------------------------------------------------------------
app.use((req, res, next) => {
  if (req.method === 'GET' && !req.path.startsWith('/api')) {
    res.sendFile(path.join(__dirname, '..', '..', 'public', 'index.html'));
  } else {
    next();
  }
});

// ---------------------------------------------------------------------------
// Error handler
// ---------------------------------------------------------------------------
app.use((err, _req, res, _next) => {
  console.error('[ERROR]', err);
  res.status(err.status || 500).json({ error: err.message || 'Internal server error' });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
let server;
if (require.main === module) {
  // Ensure DB is initialized
  getDb();
  server = app.listen(PORT, () => {
    console.log(`Neural Brain API running on http://localhost:${PORT}`);
  });
}

module.exports = { app, server };
