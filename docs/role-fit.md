# Why this project exists

This is a public, synthetic example of a support assistant that must find the right business policy, explain its evidence, and execute a permitted action without crossing customer boundaries. It gives a reviewer code they can run and inspect when commercial product code is private.

The business hypothesis is that evidence-backed answers and reliable handoffs can reduce repetitive support work while leaving uncertain requests to a person. This repository does not claim a measured reduction in support costs, customer adoption, production traffic, or commercial deployment. It is an AI-assisted portfolio project created in October 2026.

## What relevant roles actually ask for

Research checked on **13 October 2026**. These are selected employer-authored postings relevant to applied AI and integration engineering, not a representative labor-market survey. The matrix identifies useful engineering evidence; it does not promise ATS ranking, interviews, eligibility, or compensation.

| Employer / role | Requirements evidenced by the posting | Implication for this demo |
|---|---|---|
| [Armeta — AI Engineer, Forward Deployed](https://kz.linkedin.com/jobs/view/ai-engineer-forward-deployed-at-armeta-4455875126) | Python, typing/Pydantic, FastAPI, agent frameworks including LangGraph, tool calling, structured output, retrieval/embeddings/vector databases, golden datasets and regression evaluation, cost/latency control; customer-facing delivery. | Show a small workflow with clear states, validated boundaries, source-backed answers, a reproducible evaluation suite and explicit operational limits. A demo cannot establish years of commercial experience or customer deployment. |
| [ITS Partner — AI Backend Engineer, Nexus](https://kz.linkedin.com/jobs/view/ai-backend-engineer-nexus-at-its-partner-4470316283) | Python/FastAPI, asynchronous APIs, model integrations, streaming and tool calls, RAG/MCP, external integrations, mocked services, LLM evaluations, guardrails, Docker and CI. Bedrock is specifically mentioned. | Offer an inspectable API and provider boundary, behavioral tests and CI. A local provider adapter is not evidence of Bedrock, AWS operations, real-time voice or production concurrency experience. |
| [SearchApi — AI Integrations Engineer](https://jobs.ashbyhq.com/SearchApi/f0bf74e7-12bc-4daf-94c2-6fb7b53a3132/) | Python/TypeScript/Ruby integrations, agent/RAG frameworks, MCP, useful examples and documentation, GitHub/CI. The employer emphasizes merged contributions to repositories maintained by others. | Make installation, inputs, outputs and failure behavior easy to review. This personal repository complements external contributions; it does not replace them or prove that others use an integration. |
| [AgentPress — Full-Stack AI Engineer](https://wellfound.com/jobs/4362207-full-stack-ai-engineer?autoOpenApplication=true) | A TypeScript-first full-stack product with RAG, tool execution, agent orchestration, streaming, PostgreSQL/PGVector, MCP, client integrations and monitoring. | Demonstrate transferable retrieval and integration decisions. Do not pretend a Python demo covers their TypeScript frontend, embedded widgets, production monitoring, or required working-hour overlap. |

## Scope and priorities

The [README](../README.md) is the source of truth for the features implemented and commands verified in this revision. The following is the rationale for choosing a compact support workflow, not a claim that every capability in the job postings has been built.

| Priority | Capability | Useful review evidence |
|---|---|---|
| Core | Tenant-scoped retrieval and source provenance | Search operates on the authorized corpus; responses identify source/chunk versions. Cross-tenant requests and revoked or superseded sources belong in tests. |
| Core | Grounding and abstention | A supported question returns relevant evidence. Missing or insufficient evidence produces an explicit handoff/abstention rather than a fabricated policy. |
| Core | Evaluation separate from unit tests | A checked-in synthetic query set evaluates evidence retrieval and abstention; behavioral tests verify permissions and action side effects. Report corpus size and cases instead of generalizing a small score. |
| Core | Safe tool execution | Identity is supplied by the application, arguments are validated, allowed operations are explicit, and retries do not duplicate a ticket. |
| High | LangGraph orchestration | Actual graph nodes and branches make retrieval, answering and refusal inspectable. Installing a framework alone is not the feature. |
| High | FastAPI and typed contracts | One understandable request/response contract, bounded input, dependency boundaries, predictable errors and API tests. Demo credentials must not be described as production authentication. |
| High | Traceability and reproducibility | Request/step timing and evidence references help explain a failure; CI runs repeatable checks. Avoid logging raw secrets or customer content. |
| Later, when measured | Learned embeddings, a vector service, reranking, provider tracing | Compare against the local retrieval baseline on a suitable corpus. Identify the embedding model, external dependencies, costs, permissions and changed failure modes. |

## Research principles behind the design

- **Authorization precedes retrieval and generation.** Tenant metadata is useful only when the application enforces it from trusted identity. Every retrieval branch must respect the same scope. Microsoft's [multitenant RAG architecture guidance](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag) describes the underlying separation of identity, orchestration and grounding data.
- **Retrieved documents remain untrusted input.** Citations and delimiters help inspection; they do not grant documents authority to change permissions or invoke tools. Source lifecycle, provenance and adversarial tests matter alongside prompts. See the [OWASP RAG Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html).
- **Choose orchestration for control flow.** LangGraph supports combining predictable steps with model-driven steps. A deterministic path is useful for reproducible tests; it must be labeled separately from live-model behavior. See the [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview).
- **Improve retrieval from evidence.** Lexical, semantic and contextual retrieval solve different failure modes. Anthropic's [Contextual Retrieval article](https://www.anthropic.com/engineering/contextual-retrieval) provides one evaluated approach; its reported results are not this project's results. A small synthetic corpus cannot demonstrate broad retrieval quality.

These principles are independently supported by public sources. No private research text, client architecture, private corpus or proprietary implementation is included here.

## Deliberate boundaries

GraphRAG, fine-tuning, autonomous multi-agent teams, Kubernetes, billing, a full helpdesk interface and a fleet of model providers are not necessary to explain this workflow. They require a concrete use case and evaluation before they justify their complexity. MCP can be a useful next integration when a real client consumes the tools, rather than a wrapper added solely for a keyword.

Likewise, deterministic vectors or lexical features must not be described as learned semantic embeddings; extractive answers must not be presented as validated generative-model performance. Passing synthetic regression checks establishes only the behavior covered by those cases. The useful interview discussion is what the system does, how a failure is detected, and what would need to change for a real deployment.
