"""
TIRAM End-to-End Automation Pipelines
=======================================
Complete automation for building websites, apps, systems — from idea to deployment.

Inspired by:
- Bolt.new, Blink.new (AI app builders)
- Firebase Studio (Google's full-stack AI workspace)
- v0.dev (Vercel's AI UI builder)

Pipelines:
1. Website Builder  → Idea → Design → Frontend → Backend → Deploy
2. App Builder      → Spec → UI → Logic → API → Database → Test → Deploy
3. API Builder      → Schema → Routes → Validation → Auth → Docs → Deploy
4. DevOps Pipeline  → Docker → CI/CD → Cloud → Monitor → Scale
5. Data Pipeline    → Collect → Clean → Analyze → Visualize → Report
6. ML Pipeline      → Data → Features → Train → Evaluate → Deploy → Monitor
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PipelineType(str, Enum):
    WEBSITE = "website"
    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    API = "api"
    DEVOPS = "devops"
    DATA = "data"
    ML = "ml"
    LANDING_PAGE = "landing_page"
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    PORTFOLIO = "portfolio"
    BLOG = "blog"
    DASHBOARD = "dashboard"
    CRM = "crm"
    DOCUMENTATION = "documentation"


class PipelineStage(str, Enum):
    PLANNING = "planning"
    DESIGN = "design"
    SCAFFOLDING = "scaffolding"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    API_INTEGRATION = "api_integration"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class PipelineStep:
    """A single step in an automation pipeline."""
    stage: PipelineStage
    name: str
    description: str
    prompt_template: str
    tools_needed: list[str] = field(default_factory=list)
    files_generated: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    estimated_duration_ms: float = 5000
    status: str = "pending"
    result: str = ""


@dataclass
class PipelineSpec:
    """Specification for an automation pipeline."""
    pipeline_type: PipelineType
    name: str
    description: str
    steps: list[PipelineStep] = field(default_factory=list)
    tech_stack: dict[str, str] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    target_platform: str = "web"
    output_dir: str = "./output"


@dataclass
class PipelineResult:
    """Result of an automation pipeline execution."""
    success: bool = True
    files_created: list[str] = field(default_factory=list)
    total_duration_ms: float = 0
    steps_completed: int = 0
    steps_total: int = 0
    errors: list[str] = field(default_factory=list)
    deploy_url: str = ""
    summary: str = ""


class AutomationEngine:
    """
    End-to-end automation engine that can build complete projects
    from natural language descriptions.

    Usage:
        engine = AutomationEngine(config)
        result = await engine.build("Build a SaaS dashboard with auth, payments, and analytics")
    """

    def __init__(self, config=None):
        self.config = config
        self._pipelines: dict[str, PipelineSpec] = {}
        self._register_default_pipelines()

    def _register_default_pipelines(self):
        """Register all default pipeline templates."""

        # ===== Full-Stack Website =====
        self._pipelines["website"] = PipelineSpec(
            pipeline_type=PipelineType.WEBSITE,
            name="Full-Stack Website",
            description="Build a complete website from idea to deployment",
            tech_stack={"frontend": "Next.js 15", "css": "TailwindCSS", "backend": "FastAPI", "database": "PostgreSQL", "hosting": "Vercel + Railway"},
            steps=[
                PipelineStep(stage=PipelineStage.PLANNING, name="Requirements Analysis",
                    description="Analyze requirements and create project plan",
                    prompt_template="Analyze this website idea and create a detailed project plan with pages, features, and tech decisions: {description}",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.DESIGN, name="UI/UX Design",
                    description="Create wireframes and design system",
                    prompt_template="Design the UI/UX for: {description}. Create component hierarchy, color palette, typography, and responsive layout specs.",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.SCAFFOLDING, name="Project Scaffolding",
                    description="Initialize project with framework boilerplate",
                    prompt_template="Generate the complete project scaffold: package.json, tsconfig, tailwind.config, app directory structure, layout.tsx, globals.css for: {description}",
                    tools_needed=["shell", "file_write"], depends_on=["Requirements Analysis"]),
                PipelineStep(stage=PipelineStage.FRONTEND, name="Frontend Implementation",
                    description="Build all pages and components",
                    prompt_template="Implement all frontend pages and components for: {description}. Use Next.js App Router, TypeScript, TailwindCSS, Shadcn/UI. Include loading states, error boundaries, responsive design.",
                    tools_needed=["file_write", "file_read"], depends_on=["Project Scaffolding"]),
                PipelineStep(stage=PipelineStage.BACKEND, name="Backend API",
                    description="Build the backend API",
                    prompt_template="Build the complete backend API for: {description}. Use FastAPI with SQLAlchemy, Pydantic schemas, CRUD operations, authentication middleware.",
                    tools_needed=["file_write", "python_exec"], depends_on=["Requirements Analysis"]),
                PipelineStep(stage=PipelineStage.DATABASE, name="Database Schema",
                    description="Design and create database schema",
                    prompt_template="Design the complete database schema for: {description}. Include tables, relationships, indexes, migrations, and seed data.",
                    tools_needed=["file_write"], depends_on=["Backend API"]),
                PipelineStep(stage=PipelineStage.AUTHENTICATION, name="Auth System",
                    description="Implement authentication and authorization",
                    prompt_template="Implement a complete auth system for: {description}. Include signup, login, password reset, OAuth (Google/GitHub), session management, role-based access.",
                    tools_needed=["file_write"], depends_on=["Backend API"]),
                PipelineStep(stage=PipelineStage.TESTING, name="Test Suite",
                    description="Write comprehensive tests",
                    prompt_template="Write complete test suites for: {description}. Include unit tests (pytest/vitest), integration tests, and E2E tests (playwright).",
                    tools_needed=["file_write", "shell"], depends_on=["Frontend Implementation", "Backend API"]),
                PipelineStep(stage=PipelineStage.OPTIMIZATION, name="Performance Optimization",
                    description="Optimize performance and SEO",
                    prompt_template="Optimize the website for: performance (Core Web Vitals), SEO (meta tags, sitemap, robots.txt), accessibility (WCAG 2.2), and security headers.",
                    tools_needed=["file_write"], depends_on=["Test Suite"]),
                PipelineStep(stage=PipelineStage.DOCUMENTATION, name="Documentation",
                    description="Generate project documentation",
                    prompt_template="Generate complete documentation: README.md, API docs, environment setup guide, deployment instructions for: {description}",
                    tools_needed=["file_write"], depends_on=["Test Suite"]),
                PipelineStep(stage=PipelineStage.DEPLOYMENT, name="Deployment Config",
                    description="Configure deployment",
                    prompt_template="Create deployment configs: Dockerfile, docker-compose.yml, Vercel config, Railway config, GitHub Actions CI/CD pipeline for: {description}",
                    tools_needed=["file_write", "shell"], depends_on=["Test Suite"]),
            ],
        )

        # ===== SaaS Application =====
        self._pipelines["saas"] = PipelineSpec(
            pipeline_type=PipelineType.SAAS,
            name="SaaS Application",
            description="Build a complete SaaS product",
            tech_stack={"frontend": "Next.js 15", "css": "TailwindCSS + Shadcn/UI", "backend": "FastAPI", "database": "PostgreSQL", "auth": "Better Auth", "payments": "Stripe", "hosting": "Vercel + Railway"},
            steps=[
                PipelineStep(stage=PipelineStage.PLANNING, name="SaaS Architecture",
                    description="Plan multi-tenant SaaS architecture",
                    prompt_template="Design a SaaS architecture for: {description}. Include: multi-tenancy strategy, pricing tiers, feature flags, user roles, onboarding flow."),
                PipelineStep(stage=PipelineStage.SCAFFOLDING, name="SaaS Scaffold",
                    description="Initialize SaaS project with all integrations",
                    prompt_template="Scaffold a full SaaS project for: {description}. Include: auth, Stripe payments, dashboard layout, settings page, admin panel.",
                    tools_needed=["shell", "file_write"]),
                PipelineStep(stage=PipelineStage.AUTHENTICATION, name="Multi-tenant Auth",
                    description="Build multi-tenant authentication",
                    prompt_template="Implement multi-tenant auth with team management, invitations, roles (admin/member/viewer), SSO support for: {description}",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.FRONTEND, name="Dashboard & Features",
                    description="Build the SaaS dashboard and core features",
                    prompt_template="Build the complete SaaS dashboard for: {description}. Include: data tables, charts, filters, CRUD operations, real-time updates, notifications.",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.BACKEND, name="SaaS API & Billing",
                    description="Build API with billing integration",
                    prompt_template="Build the SaaS backend with Stripe integration for: {description}. Include: subscription management, usage metering, webhooks, invoicing.",
                    tools_needed=["file_write", "python_exec"]),
                PipelineStep(stage=PipelineStage.TESTING, name="SaaS Tests",
                    description="Comprehensive SaaS testing",
                    prompt_template="Write tests covering: auth flows, payment flows, multi-tenancy isolation, API rate limiting, webhook handling for: {description}",
                    tools_needed=["file_write", "shell"]),
                PipelineStep(stage=PipelineStage.DEPLOYMENT, name="SaaS Deploy",
                    description="Production deployment with monitoring",
                    prompt_template="Deploy SaaS to production: Docker, CI/CD, database migrations, Stripe webhooks, monitoring (Sentry), analytics, email (Resend) for: {description}",
                    tools_needed=["file_write", "shell"]),
            ],
        )

        # ===== REST API =====
        self._pipelines["api"] = PipelineSpec(
            pipeline_type=PipelineType.API,
            name="REST API",
            description="Build a production-ready REST API",
            tech_stack={"framework": "FastAPI", "database": "PostgreSQL", "orm": "SQLAlchemy", "auth": "JWT + OAuth2"},
            steps=[
                PipelineStep(stage=PipelineStage.PLANNING, name="API Design",
                    description="Design API endpoints and data model",
                    prompt_template="Design a REST API for: {description}. Include: endpoint list, request/response schemas, data model, auth strategy, error handling."),
                PipelineStep(stage=PipelineStage.SCAFFOLDING, name="API Scaffold",
                    description="Initialize API project",
                    prompt_template="Scaffold a FastAPI project for: {description}. Include: project structure, database config, auth middleware, CORS, rate limiting.",
                    tools_needed=["shell", "file_write"]),
                PipelineStep(stage=PipelineStage.DATABASE, name="Models & Migrations",
                    description="Create database models",
                    prompt_template="Create SQLAlchemy models, Alembic migrations, and seed data for: {description}",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.BACKEND, name="API Routes",
                    description="Implement all API endpoints",
                    prompt_template="Implement all CRUD endpoints with validation, pagination, filtering, search, and error handling for: {description}",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.AUTHENTICATION, name="API Auth",
                    description="Implement API authentication",
                    prompt_template="Implement JWT + OAuth2 authentication with: registration, login, refresh tokens, password reset, API keys for: {description}",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.TESTING, name="API Tests",
                    description="Write API tests",
                    prompt_template="Write comprehensive pytest tests: unit tests, integration tests, auth tests, edge cases for: {description}",
                    tools_needed=["file_write", "shell"]),
                PipelineStep(stage=PipelineStage.DOCUMENTATION, name="API Docs",
                    description="Generate API documentation",
                    prompt_template="Generate OpenAPI spec, Postman collection, and developer guide for: {description}",
                    tools_needed=["file_write"]),
            ],
        )

        # ===== ML Pipeline =====
        self._pipelines["ml"] = PipelineSpec(
            pipeline_type=PipelineType.ML,
            name="ML Pipeline",
            description="Build a complete machine learning pipeline",
            tech_stack={"framework": "PyTorch/scikit-learn", "tracking": "MLflow", "serving": "FastAPI", "data": "pandas"},
            steps=[
                PipelineStep(stage=PipelineStage.PLANNING, name="ML Problem Definition",
                    description="Define the ML problem and approach",
                    prompt_template="Define the ML problem, metrics, and approach for: {description}. Include: problem type, target variable, features, evaluation metrics, baseline."),
                PipelineStep(stage=PipelineStage.DATABASE, name="Data Pipeline",
                    description="Build data ingestion and preprocessing",
                    prompt_template="Build a data pipeline for: {description}. Include: data loading, cleaning, EDA, feature engineering, train/val/test split.",
                    tools_needed=["python_exec", "file_write"]),
                PipelineStep(stage=PipelineStage.BACKEND, name="Model Training",
                    description="Train and evaluate models",
                    prompt_template="Train ML models for: {description}. Include: baseline model, hyperparameter tuning, cross-validation, model comparison, feature importance.",
                    tools_needed=["python_exec", "file_write"]),
                PipelineStep(stage=PipelineStage.TESTING, name="Model Evaluation",
                    description="Evaluate model performance",
                    prompt_template="Comprehensive model evaluation for: {description}. Include: metrics, confusion matrix, ROC/PR curves, bias detection, fairness analysis.",
                    tools_needed=["python_exec", "file_write"]),
                PipelineStep(stage=PipelineStage.DEPLOYMENT, name="Model Serving",
                    description="Deploy model as API",
                    prompt_template="Deploy the ML model as a FastAPI service for: {description}. Include: inference endpoint, batch prediction, model versioning, monitoring.",
                    tools_needed=["file_write", "shell"]),
            ],
        )

        # ===== Dashboard =====
        self._pipelines["dashboard"] = PipelineSpec(
            pipeline_type=PipelineType.DASHBOARD,
            name="Analytics Dashboard",
            description="Build a data analytics dashboard",
            tech_stack={"frontend": "Next.js + Recharts/Tremor", "backend": "FastAPI", "database": "PostgreSQL"},
            steps=[
                PipelineStep(stage=PipelineStage.PLANNING, name="Dashboard Design",
                    description="Design dashboard layout and metrics",
                    prompt_template="Design a dashboard for: {description}. Include: KPIs, charts, filters, date ranges, export capabilities, real-time updates."),
                PipelineStep(stage=PipelineStage.FRONTEND, name="Dashboard UI",
                    description="Build dashboard frontend",
                    prompt_template="Build the dashboard UI for: {description}. Use Next.js, Tremor components, responsive grid, dark mode, chart interactivity.",
                    tools_needed=["file_write"]),
                PipelineStep(stage=PipelineStage.BACKEND, name="Dashboard API",
                    description="Build data aggregation API",
                    prompt_template="Build the dashboard API for: {description}. Include: aggregation queries, caching, real-time WebSocket updates, CSV export.",
                    tools_needed=["file_write"]),
            ],
        )

    def get_pipeline(self, pipeline_type: str) -> PipelineSpec | None:
        """Get a pipeline template."""
        return self._pipelines.get(pipeline_type)

    def list_pipelines(self) -> list[dict]:
        """List all available pipelines."""
        return [
            {
                "type": p.pipeline_type.value,
                "name": p.name,
                "description": p.description,
                "tech_stack": p.tech_stack,
                "steps": len(p.steps),
            }
            for p in self._pipelines.values()
        ]

    async def build(self, description: str, pipeline_type: str = "auto",
                    model_router=None, tool_registry=None) -> PipelineResult:
        """
        Build a complete project from a natural language description.

        Auto-detects the best pipeline if type is 'auto'.
        """
        result = PipelineResult()
        start_time = time.time()

        # Auto-detect pipeline type
        if pipeline_type == "auto":
            pipeline_type = self._detect_pipeline_type(description)

        pipeline = self._pipelines.get(pipeline_type)
        if not pipeline:
            result.success = False
            result.errors.append(f"Pipeline '{pipeline_type}' not found")
            return result

        result.steps_total = len(pipeline.steps)

        # Execute pipeline steps
        for step in pipeline.steps:
            step.status = "running"
            prompt = step.prompt_template.replace("{description}", description)

            try:
                if model_router:
                    response = await model_router.generate(
                        model=self.config.default_model if self.config else "claude-sonnet-4-6",
                        messages=[
                            {"role": "system", "content": f"You are building: {description}. Current stage: {step.stage.value}"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=8192,
                    )
                    step.result = response.get("content", "")
                else:
                    step.result = f"[Would execute: {step.name}]"

                step.status = "completed"
                result.steps_completed += 1

            except Exception as e:
                step.status = "failed"
                result.errors.append(f"{step.name}: {str(e)}")

        result.total_duration_ms = (time.time() - start_time) * 1000
        result.success = result.steps_completed == result.steps_total
        result.summary = f"Built {pipeline.name}: {result.steps_completed}/{result.steps_total} steps completed"
        return result

    def _detect_pipeline_type(self, description: str) -> str:
        """Auto-detect the best pipeline type from the description."""
        desc_lower = description.lower()

        type_keywords = {
            "saas": ["saas", "subscription", "billing", "multi-tenant", "pricing"],
            "api": ["api", "rest", "graphql", "endpoint", "backend only"],
            "ml": ["machine learning", "model", "predict", "train", "classify", "neural"],
            "dashboard": ["dashboard", "analytics", "metrics", "chart", "kpi", "monitor"],
            "website": ["website", "landing", "portfolio", "blog", "homepage"],
            "web_app": ["web app", "application", "webapp", "full-stack", "fullstack"],
        }

        for pipe_type, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return pipe_type

        return "website"  # Default

    @property
    def stats(self) -> dict:
        return {
            "total_pipelines": len(self._pipelines),
            "available": self.list_pipelines(),
        }
