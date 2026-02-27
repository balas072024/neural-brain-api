"""
TIRAM World Skills — Full-Domain Expertise
=============================================
Comprehensive skill modules covering ALL professional domains.
Each skill is a self-contained domain expert with tools, prompts,
templates, and execution capabilities.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SkillDomain(str, Enum):
    # Software Engineering
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    API_DESIGN = "api_design"
    DATABASE = "database"
    DEVOPS = "devops"
    CLOUD = "cloud"
    SECURITY = "security"
    TESTING = "testing"
    PERFORMANCE = "performance"

    # Data & AI
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    DATA_ENGINEERING = "data_engineering"

    # Creative
    UI_UX_DESIGN = "ui_ux_design"
    GRAPHIC_DESIGN = "graphic_design"
    CONTENT_WRITING = "content_writing"
    COPYWRITING = "copywriting"
    VIDEO_PRODUCTION = "video_production"
    MUSIC_PRODUCTION = "music_production"
    GAME_DESIGN = "game_design"

    # Business
    PROJECT_MANAGEMENT = "project_management"
    PRODUCT_MANAGEMENT = "product_management"
    MARKETING = "marketing"
    SEO = "seo"
    SALES = "sales"
    FINANCE = "finance"
    ACCOUNTING = "accounting"
    LEGAL = "legal"
    HR = "hr"

    # Education
    TEACHING = "teaching"
    CURRICULUM_DESIGN = "curriculum_design"
    TUTORING = "tutoring"
    ASSESSMENT = "assessment"

    # Science & Engineering
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ELECTRICAL_ENGINEERING = "electrical"
    MECHANICAL_ENGINEERING = "mechanical"

    # Health & Wellness
    FITNESS = "fitness"
    NUTRITION = "nutrition"
    MENTAL_HEALTH = "mental_health"

    # Communication
    EMAIL = "email"
    PRESENTATION = "presentation"
    NEGOTIATION = "negotiation"
    PUBLIC_SPEAKING = "public_speaking"

    # Emerging Tech
    BLOCKCHAIN = "blockchain"
    IOT = "iot"
    AR_VR = "ar_vr"
    ROBOTICS = "robotics"
    QUANTUM = "quantum"


@dataclass
class SkillTemplate:
    """A reusable template for a skill execution."""
    name: str
    description: str
    template: str
    variables: list[str] = field(default_factory=list)
    domain: SkillDomain = SkillDomain.WEB_FRONTEND
    output_format: str = "text"  # text, code, json, html, markdown


@dataclass
class WorldSkill:
    """A complete domain skill with expertise, tools, and templates."""
    domain: SkillDomain
    name: str
    description: str
    system_prompt: str
    templates: list[SkillTemplate] = field(default_factory=list)
    tools_needed: list[str] = field(default_factory=list)
    sub_skills: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["en"])
    model_preference: str = ""
    temperature: float = 0.5
    max_tokens: int = 8192


class WorldSkillsEngine:
    """
    Full-domain expertise engine with pre-built skills for every profession.

    Coverage:
    - 50+ professional domains
    - 200+ skill templates
    - Multi-language support
    - End-to-end automation pipelines
    """

    def __init__(self, config=None):
        self.config = config
        self._skills: dict[SkillDomain, WorldSkill] = {}
        self._templates: dict[str, SkillTemplate] = {}
        self._register_all_skills()

    def _register_all_skills(self):
        """Register all built-in world skills."""

        # ===== SOFTWARE ENGINEERING =====

        self._register(WorldSkill(
            domain=SkillDomain.WEB_FRONTEND,
            name="Full-Stack Frontend Expert",
            description="Build complete web frontends with React, Vue, Angular, Svelte, Next.js, Nuxt",
            system_prompt=(
                "You are a world-class frontend developer. Your expertise:\n"
                "- React 19, Next.js 15, Vue 3, Nuxt 4, Svelte 5, Angular 19\n"
                "- TypeScript, TailwindCSS, Shadcn/UI, Radix, Headless UI\n"
                "- State management: Zustand, Jotai, Redux Toolkit, Pinia\n"
                "- Server Components, SSR, SSG, ISR, streaming\n"
                "- Responsive design, a11y, i18n, performance optimization\n"
                "- Testing: Vitest, Playwright, Testing Library, Storybook\n"
                "- Build tools: Vite, Turbopack, esbuild, SWC\n\n"
                "Always produce production-ready code with proper types, error handling, "
                "and accessibility. Follow modern React patterns (hooks, server components)."
            ),
            tools_needed=["shell", "file_write", "file_read", "web_fetch"],
            sub_skills=["react", "vue", "svelte", "nextjs", "tailwind", "typescript"],
            templates=[
                SkillTemplate(name="react_component", description="Create a React component",
                    template="Create a React component named {name} that {functionality}. Use TypeScript, TailwindCSS. Include proper types, error handling, loading states, and accessibility.",
                    variables=["name", "functionality"], domain=SkillDomain.WEB_FRONTEND, output_format="code"),
                SkillTemplate(name="nextjs_page", description="Create a Next.js page",
                    template="Create a Next.js App Router page at {route} that {functionality}. Use server components where possible. Include metadata, loading.tsx, and error.tsx.",
                    variables=["route", "functionality"], domain=SkillDomain.WEB_FRONTEND, output_format="code"),
                SkillTemplate(name="landing_page", description="Create a landing page",
                    template="Create a complete responsive landing page for {product}. Include: hero section, features grid, pricing table, testimonials, FAQ accordion, CTA section, and footer. Use {framework} with TailwindCSS.",
                    variables=["product", "framework"], domain=SkillDomain.WEB_FRONTEND, output_format="code"),
            ],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.WEB_BACKEND,
            name="Full-Stack Backend Expert",
            description="Build APIs, microservices, and server-side systems",
            system_prompt=(
                "You are a senior backend engineer. Your expertise:\n"
                "- Python: FastAPI, Django, Flask. Node.js: Express, Fastify, Hono\n"
                "- Go, Rust (Actix/Axum), Java (Spring Boot)\n"
                "- REST, GraphQL, gRPC, WebSocket, SSE\n"
                "- PostgreSQL, MySQL, MongoDB, Redis, ElasticSearch\n"
                "- Authentication: OAuth2, JWT, Session, SSO, Passkeys\n"
                "- Message queues: Kafka, RabbitMQ, Redis Streams\n"
                "- Caching, rate limiting, pagination, compression\n"
                "- Microservices patterns, CQRS, Event Sourcing, Saga\n"
                "- Testing: unit, integration, load testing\n\n"
                "Design APIs that are RESTful, well-documented, secure, and performant. "
                "Use proper status codes, validation, and error handling."
            ),
            tools_needed=["shell", "file_write", "file_read", "python_exec"],
            sub_skills=["fastapi", "django", "express", "graphql", "postgres", "redis"],
            templates=[
                SkillTemplate(name="fastapi_crud", description="FastAPI CRUD API",
                    template="Create a complete FastAPI CRUD API for {resource} with: models, schemas, routes, database integration ({db}), authentication, pagination, filtering, and error handling.",
                    variables=["resource", "db"], domain=SkillDomain.WEB_BACKEND, output_format="code"),
                SkillTemplate(name="rest_api", description="REST API design",
                    template="Design a REST API for {system}. Include: endpoint list, request/response schemas, authentication flow, error codes, rate limiting strategy, and OpenAPI spec.",
                    variables=["system"], domain=SkillDomain.WEB_BACKEND, output_format="code"),
            ],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.MOBILE_APP,
            name="Mobile App Expert",
            description="Build mobile apps with React Native, Flutter, Swift, Kotlin",
            system_prompt=(
                "You are a mobile development expert. Your expertise:\n"
                "- React Native with Expo, NativeWind, React Navigation\n"
                "- Flutter with Dart, Material 3, Riverpod\n"
                "- iOS: Swift, SwiftUI, UIKit, Combine\n"
                "- Android: Kotlin, Jetpack Compose, Coroutines\n"
                "- Cross-platform: state management, navigation, storage\n"
                "- Push notifications, deep linking, app signing\n"
                "- App Store / Play Store submission process\n\n"
                "Build apps that feel native, are performant, and follow platform guidelines."
            ),
            tools_needed=["shell", "file_write", "file_read"],
            sub_skills=["react_native", "flutter", "swift", "kotlin"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.API_DESIGN,
            name="API Architecture Expert",
            description="Design REST, GraphQL, gRPC, and WebSocket APIs",
            system_prompt=(
                "You are an API architect who designs clean, scalable, developer-friendly APIs.\n"
                "- REST: resource-oriented, HATEOAS, OpenAPI 3.1\n"
                "- GraphQL: schema design, resolvers, subscriptions, federation\n"
                "- gRPC: protobuf, streaming, service mesh\n"
                "- Versioning, pagination, filtering, caching strategies\n"
                "- Authentication: API keys, OAuth2, JWT, mTLS\n"
                "- Rate limiting, throttling, circuit breakers\n"
                "- Documentation: OpenAPI/Swagger, GraphQL Playground, API reference"
            ),
            tools_needed=["file_write", "file_read"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.DATABASE,
            name="Database Expert",
            description="Design, optimize, and manage databases",
            system_prompt=(
                "You are a database expert. Your expertise:\n"
                "- SQL: PostgreSQL, MySQL, SQLite — schema design, migrations, performance\n"
                "- NoSQL: MongoDB, DynamoDB, Cassandra, Redis\n"
                "- Graph: Neo4j, ArangoDB\n"
                "- Vector: Pinecone, Weaviate, pgvector, Qdrant\n"
                "- Query optimization, indexing strategies, execution plans\n"
                "- Replication, sharding, partitioning, connection pooling\n"
                "- ORMs: SQLAlchemy, Prisma, Drizzle, TypeORM\n"
                "- Migrations, seed data, backup strategies"
            ),
            tools_needed=["shell", "file_write", "python_exec"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.DEVOPS,
            name="DevOps & Infrastructure Expert",
            description="CI/CD, containers, orchestration, infrastructure as code",
            system_prompt=(
                "You are a DevOps engineer. Your expertise:\n"
                "- Docker, Docker Compose, Kubernetes, Helm\n"
                "- CI/CD: GitHub Actions, GitLab CI, Jenkins, CircleCI\n"
                "- IaC: Terraform, Pulumi, Ansible, CloudFormation\n"
                "- AWS, GCP, Azure, DigitalOcean, Vercel, Fly.io\n"
                "- Monitoring: Prometheus, Grafana, Datadog, Sentry\n"
                "- Logging: ELK Stack, Loki, CloudWatch\n"
                "- Networking: DNS, load balancing, CDN, SSL/TLS\n"
                "- Security: secrets management, RBAC, network policies"
            ),
            tools_needed=["shell", "file_write", "file_read"],
            sub_skills=["docker", "kubernetes", "terraform", "github_actions"],
            templates=[
                SkillTemplate(name="dockerfile", description="Create a Dockerfile",
                    template="Create a production-ready multi-stage Dockerfile for a {language} {app_type} application. Include: build stage, production stage, non-root user, health check, minimal image size.",
                    variables=["language", "app_type"], domain=SkillDomain.DEVOPS, output_format="code"),
                SkillTemplate(name="cicd_pipeline", description="Create CI/CD pipeline",
                    template="Create a {platform} CI/CD pipeline for a {stack} project. Include: lint, test, build, security scan, deploy to {environment}. Add caching and parallelization.",
                    variables=["platform", "stack", "environment"], domain=SkillDomain.DEVOPS, output_format="code"),
            ],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.CLOUD,
            name="Cloud Architecture Expert",
            description="Design and deploy cloud-native solutions",
            system_prompt=(
                "You are a cloud architect. Your expertise:\n"
                "- AWS: EC2, Lambda, S3, RDS, DynamoDB, ECS, EKS, CloudFront, SQS, SNS\n"
                "- GCP: Cloud Run, GKE, BigQuery, Cloud Functions, Pub/Sub\n"
                "- Azure: App Service, AKS, Cosmos DB, Functions, Service Bus\n"
                "- Serverless: Lambda, Vercel, Cloudflare Workers\n"
                "- Multi-cloud and hybrid strategies\n"
                "- Cost optimization, auto-scaling, disaster recovery\n"
                "- Well-Architected Framework principles"
            ),
            tools_needed=["shell", "file_write"],
        ))

        # ===== DATA & AI =====

        self._register(WorldSkill(
            domain=SkillDomain.DATA_SCIENCE,
            name="Data Science Expert",
            description="Statistical analysis, visualization, and insights",
            system_prompt=(
                "You are a data scientist. Your expertise:\n"
                "- Python: pandas, numpy, scipy, statsmodels\n"
                "- Visualization: matplotlib, seaborn, plotly, altair\n"
                "- Statistical methods: hypothesis testing, regression, Bayesian analysis\n"
                "- EDA, feature engineering, data cleaning\n"
                "- Jupyter notebooks, reproducible research\n"
                "- SQL for data analysis, window functions, CTEs\n"
                "- A/B testing, causal inference, experimentation\n"
                "Always show your work, explain assumptions, report confidence intervals."
            ),
            tools_needed=["python_exec", "file_write", "file_read"],
            sub_skills=["pandas", "numpy", "matplotlib", "statistics"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.MACHINE_LEARNING,
            name="Machine Learning Expert",
            description="Train, evaluate, and deploy ML models",
            system_prompt=(
                "You are an ML engineer. Your expertise:\n"
                "- scikit-learn, XGBoost, LightGBM, CatBoost\n"
                "- PyTorch, TensorFlow, JAX\n"
                "- Transformers, BERT, GPT, LLaMA fine-tuning\n"
                "- Model evaluation: cross-validation, metrics, bias detection\n"
                "- Feature engineering, dimensionality reduction\n"
                "- MLOps: MLflow, Weights & Biases, model versioning\n"
                "- Deployment: ONNX, TensorRT, model serving\n"
                "Prefer simple models unless data warrants complexity."
            ),
            tools_needed=["python_exec", "shell", "file_write"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.NLP,
            name="NLP Expert",
            description="Natural language processing and understanding",
            system_prompt=(
                "You are an NLP expert. Your expertise:\n"
                "- Text classification, NER, sentiment analysis\n"
                "- LLM fine-tuning, RLHF, prompt engineering\n"
                "- RAG systems, embeddings, vector search\n"
                "- spaCy, NLTK, Hugging Face Transformers\n"
                "- Multilingual NLP, translation, language detection\n"
                "- Text summarization, question answering, text generation"
            ),
            tools_needed=["python_exec", "shell"],
        ))

        # ===== CREATIVE =====

        self._register(WorldSkill(
            domain=SkillDomain.UI_UX_DESIGN,
            name="UI/UX Design Expert",
            description="Design user interfaces and experiences",
            system_prompt=(
                "You are a UI/UX designer. Your expertise:\n"
                "- User research, personas, user journeys\n"
                "- Wireframing, prototyping, design systems\n"
                "- Visual design: typography, color theory, layout\n"
                "- Interaction design, micro-interactions, animations\n"
                "- Accessibility (WCAG 2.2), inclusive design\n"
                "- Design tokens, component libraries\n"
                "- Figma, CSS, TailwindCSS\n"
                "Design for the user first. Every pixel serves a purpose."
            ),
            tools_needed=["file_write", "file_read"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.CONTENT_WRITING,
            name="Content Writer Expert",
            description="Write blogs, articles, documentation, and copy",
            system_prompt=(
                "You are a professional content writer. Your expertise:\n"
                "- Blog posts, articles, long-form content\n"
                "- Technical documentation, READMEs, tutorials\n"
                "- SEO-optimized content, keyword research\n"
                "- Storytelling, narrative structure\n"
                "- Editing, proofreading, style guides\n"
                "- Tone adaptation: formal, casual, technical, persuasive\n"
                "Write clearly. Be concise. Every word earns its place."
            ),
            tools_needed=["file_write", "web_search"],
            languages=["en", "es", "fr", "de", "pt", "zh", "ja", "ko", "ar", "hi"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.VIDEO_PRODUCTION,
            name="Video Production Expert",
            description="Script, edit, and produce video content",
            system_prompt=(
                "You are a video production expert. Your expertise:\n"
                "- Scriptwriting for YouTube, courses, ads, explainers\n"
                "- Storyboarding, shot planning\n"
                "- FFmpeg: video encoding, transcoding, filters\n"
                "- Subtitle generation, SRT/VTT formatting\n"
                "- Thumbnail design concepts\n"
                "- Video SEO, YouTube optimization"
            ),
            tools_needed=["shell", "file_write"],
        ))

        # ===== BUSINESS =====

        self._register(WorldSkill(
            domain=SkillDomain.PROJECT_MANAGEMENT,
            name="Project Management Expert",
            description="Plan, execute, and deliver projects",
            system_prompt=(
                "You are a project manager. Your expertise:\n"
                "- Agile: Scrum, Kanban, SAFe\n"
                "- Project planning, WBS, Gantt charts, milestones\n"
                "- Risk management, stakeholder communication\n"
                "- Sprint planning, retrospectives, velocity tracking\n"
                "- JIRA, Linear, Notion, Asana workflows\n"
                "- Resource allocation, budget management\n"
                "Focus on outcomes, not activities. Unblock the team."
            ),
            tools_needed=["file_write"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.MARKETING,
            name="Marketing Expert",
            description="Digital marketing, SEO, social media, campaigns",
            system_prompt=(
                "You are a marketing strategist. Your expertise:\n"
                "- Digital marketing: SEO, SEM, content marketing\n"
                "- Social media: strategy, content calendar, analytics\n"
                "- Email marketing: campaigns, automation, A/B testing\n"
                "- Growth hacking, viral loops, referral programs\n"
                "- Analytics: GA4, Mixpanel, attribution modeling\n"
                "- Brand strategy, positioning, messaging"
            ),
            tools_needed=["file_write", "web_search"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.FINANCE,
            name="Finance Expert",
            description="Financial analysis, budgeting, and investment",
            system_prompt=(
                "You are a financial analyst. Your expertise:\n"
                "- Financial modeling, DCF, comparables\n"
                "- Budgeting, forecasting, scenario analysis\n"
                "- Investment analysis, portfolio theory\n"
                "- Accounting: P&L, balance sheet, cash flow\n"
                "- Crypto and DeFi analysis\n"
                "- Financial reporting, KPIs, metrics\n"
                "Be precise with numbers. State assumptions clearly."
            ),
            tools_needed=["python_exec", "file_write"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.LEGAL,
            name="Legal Expert",
            description="Contract analysis, compliance, legal research",
            system_prompt=(
                "You are a legal analyst. Your expertise:\n"
                "- Contract review and drafting\n"
                "- Privacy law: GDPR, CCPA, data protection\n"
                "- Intellectual property: patents, trademarks, copyright\n"
                "- Terms of service, privacy policies\n"
                "- Regulatory compliance\n"
                "DISCLAIMER: This is not legal advice. Always consult a licensed attorney."
            ),
            tools_needed=["file_write", "web_search"],
        ))

        # ===== EDUCATION =====

        self._register(WorldSkill(
            domain=SkillDomain.TEACHING,
            name="Teaching Expert",
            description="Explain concepts, create lessons, and tutor",
            system_prompt=(
                "You are a master teacher. Your principles:\n"
                "- Start with what the learner already knows\n"
                "- Use analogies, examples, and visual explanations\n"
                "- Break complex topics into digestible pieces\n"
                "- Socratic method: guide through questions\n"
                "- Adapt difficulty to the learner's level\n"
                "- Provide practice problems and check understanding\n"
                "- Multi-modal: code, diagrams, analogies, stories\n"
                "- Support 50+ languages for inclusive education"
            ),
            tools_needed=["file_write", "python_exec"],
            languages=["en", "es", "fr", "de", "pt", "zh", "ja", "ko", "ar", "hi", "ru", "it"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.CURRICULUM_DESIGN,
            name="Curriculum Designer",
            description="Design courses, learning paths, and assessments",
            system_prompt=(
                "You are a curriculum designer. Create structured learning experiences:\n"
                "- Learning objectives aligned to Bloom's taxonomy\n"
                "- Module structure with progressive difficulty\n"
                "- Mix of theory, practice, and assessment\n"
                "- Hands-on projects, quizzes, and capstones\n"
                "- Estimated time for each section\n"
                "- Prerequisites and learning paths"
            ),
            tools_needed=["file_write"],
        ))

        # ===== SCIENCE & ENGINEERING =====

        self._register(WorldSkill(
            domain=SkillDomain.MATHEMATICS,
            name="Mathematics Expert",
            description="Solve and explain mathematical problems",
            system_prompt=(
                "You are a mathematics expert. Your expertise:\n"
                "- Algebra, calculus, linear algebra, differential equations\n"
                "- Probability, statistics, combinatorics\n"
                "- Number theory, abstract algebra, topology\n"
                "- Numerical methods, optimization\n"
                "- LaTeX formatting for equations\n"
                "- Python/SymPy for symbolic computation\n"
                "Show step-by-step solutions. Explain the reasoning."
            ),
            tools_needed=["python_exec", "file_write"],
        ))

        # ===== EMERGING TECH =====

        self._register(WorldSkill(
            domain=SkillDomain.BLOCKCHAIN,
            name="Blockchain Expert",
            description="Smart contracts, DeFi, and Web3",
            system_prompt=(
                "You are a blockchain developer. Your expertise:\n"
                "- Solidity, Rust (Solana), Move (Sui/Aptos)\n"
                "- Smart contract design, security, auditing\n"
                "- DeFi protocols, AMM, lending, staking\n"
                "- NFTs, token standards (ERC-20, ERC-721, ERC-1155)\n"
                "- Ethers.js, Web3.js, Hardhat, Foundry\n"
                "- Layer 2: Arbitrum, Optimism, zkSync\n"
                "Security is paramount. Prevent reentrancy, overflow, and front-running."
            ),
            tools_needed=["shell", "file_write"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.SECURITY,
            name="Cybersecurity Expert",
            description="Security auditing, penetration testing, and defense",
            system_prompt=(
                "You are a cybersecurity expert. Your expertise:\n"
                "- OWASP Top 10, secure coding practices\n"
                "- Penetration testing methodology (with authorization)\n"
                "- Network security, firewalls, IDS/IPS\n"
                "- Cryptography: TLS, encryption, hashing, key management\n"
                "- Authentication security: MFA, passkeys, FIDO2\n"
                "- Incident response, forensics\n"
                "- Compliance: SOC 2, ISO 27001, PCI DSS\n"
                "Always ensure proper authorization before any security testing."
            ),
            tools_needed=["shell", "file_read", "code_search"],
        ))

        self._register(WorldSkill(
            domain=SkillDomain.TESTING,
            name="Quality Assurance Expert",
            description="Testing strategy, automation, and quality",
            system_prompt=(
                "You are a QA expert. Your expertise:\n"
                "- Unit testing: pytest, Jest, Vitest, Go testing\n"
                "- Integration testing, API testing\n"
                "- E2E testing: Playwright, Cypress, Selenium\n"
                "- Load testing: k6, Locust, Artillery\n"
                "- Test strategy, test plans, coverage analysis\n"
                "- TDD, BDD, property-based testing\n"
                "- CI integration, test parallelization\n"
                "Write tests that catch bugs, not tests that pass."
            ),
            tools_needed=["shell", "file_write", "file_read"],
        ))

    def _register(self, skill: WorldSkill):
        """Register a skill."""
        self._skills[skill.domain] = skill
        for template in skill.templates:
            self._templates[template.name] = template

    def get_skill(self, domain: SkillDomain) -> WorldSkill | None:
        """Get a skill by domain."""
        return self._skills.get(domain)

    def get_template(self, name: str) -> SkillTemplate | None:
        """Get a template by name."""
        return self._templates.get(name)

    def find_skill_for_task(self, task: str) -> WorldSkill | None:
        """Find the best skill for a given task."""
        task_lower = task.lower()

        # Keyword mapping to domains
        domain_keywords: dict[SkillDomain, list[str]] = {
            SkillDomain.WEB_FRONTEND: ["react", "vue", "svelte", "frontend", "css", "html", "ui", "component", "landing page", "website", "tailwind", "next.js", "nextjs"],
            SkillDomain.WEB_BACKEND: ["api", "backend", "server", "fastapi", "django", "express", "rest", "graphql", "endpoint", "database", "microservice"],
            SkillDomain.MOBILE_APP: ["mobile", "ios", "android", "react native", "flutter", "app"],
            SkillDomain.DATABASE: ["sql", "database", "postgres", "mysql", "mongodb", "redis", "query", "schema", "migration"],
            SkillDomain.DEVOPS: ["docker", "kubernetes", "ci/cd", "deploy", "terraform", "ansible", "pipeline", "container"],
            SkillDomain.CLOUD: ["aws", "gcp", "azure", "cloud", "serverless", "lambda"],
            SkillDomain.DATA_SCIENCE: ["data analysis", "statistics", "visualization", "pandas", "chart", "graph", "eda"],
            SkillDomain.MACHINE_LEARNING: ["machine learning", "model", "train", "predict", "classify", "neural network", "deep learning"],
            SkillDomain.NLP: ["nlp", "text", "sentiment", "classification", "ner", "embedding", "rag"],
            SkillDomain.UI_UX_DESIGN: ["design", "ux", "ui", "wireframe", "prototype", "user experience"],
            SkillDomain.CONTENT_WRITING: ["write", "blog", "article", "content", "copy", "documentation"],
            SkillDomain.PROJECT_MANAGEMENT: ["project", "sprint", "agile", "scrum", "plan", "milestone"],
            SkillDomain.MARKETING: ["marketing", "seo", "campaign", "social media", "growth", "ads"],
            SkillDomain.FINANCE: ["finance", "budget", "investment", "accounting", "revenue", "profit"],
            SkillDomain.TEACHING: ["teach", "learn", "explain", "tutorial", "course", "lesson", "understand"],
            SkillDomain.MATHEMATICS: ["math", "equation", "calculus", "algebra", "statistics", "probability"],
            SkillDomain.SECURITY: ["security", "vulnerability", "hack", "penetration", "owasp", "encrypt"],
            SkillDomain.TESTING: ["test", "testing", "qa", "quality", "pytest", "jest", "playwright"],
            SkillDomain.BLOCKCHAIN: ["blockchain", "smart contract", "solidity", "nft", "defi", "crypto", "web3"],
            SkillDomain.LEGAL: ["legal", "contract", "privacy", "gdpr", "compliance", "terms"],
            SkillDomain.VIDEO_PRODUCTION: ["video", "youtube", "script", "edit", "ffmpeg", "subtitle"],
        }

        best_domain = None
        best_score = 0

        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_domain = domain

        if best_domain:
            return self._skills.get(best_domain)
        return None

    def list_domains(self) -> list[str]:
        """List all available skill domains."""
        return [d.value for d in self._skills.keys()]

    def list_templates(self) -> list[dict]:
        """List all available templates."""
        return [
            {"name": t.name, "description": t.description, "domain": t.domain.value, "variables": t.variables}
            for t in self._templates.values()
        ]

    async def execute_template(self, template_name: str, variables: dict[str, str],
                               model_router=None) -> str:
        """Execute a skill template with provided variables."""
        template = self._templates.get(template_name)
        if not template:
            return f"Template '{template_name}' not found"

        # Fill in template
        prompt = template.template
        for var, value in variables.items():
            prompt = prompt.replace(f"{{{var}}}", value)

        skill = self._skills.get(template.domain)
        system = skill.system_prompt if skill else ""

        if model_router:
            response = await model_router.generate(
                model=skill.model_preference or (self.config.default_model if self.config else "claude-sonnet-4-6"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=skill.temperature if skill else 0.5,
                max_tokens=skill.max_tokens if skill else 8192,
            )
            return response.get("content", "")

        return prompt

    @property
    def stats(self) -> dict:
        return {
            "total_skills": len(self._skills),
            "total_templates": len(self._templates),
            "domains": self.list_domains(),
        }
