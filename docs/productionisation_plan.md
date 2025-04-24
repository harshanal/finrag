# FinRAG – Production Deployment & Operations Plan

## 1. Project Overview

**Project Scope:** Transition the FinRAG prototype, currently featuring a ChromaDB vector store, Cohere Reranker, and a two-step agent (Specify Expression -> Extract Values), into a robust, scalable production system capable of supporting real-time financial question answering.

---

## 2. Objectives

- Productionise the FinRAG prototype for commercial use, building upon the successful two-step agent architecture.
- Ensure scalability, security, and reliability for the vector store (ChromaDB or alternative) and agent components.
- Incorporate MLOps best practices for continuous deployment, monitoring (including W&B insights), and model/prompt governance.
- Provide explainability and auditability suitable for financial services compliance, leveraging the intermediate outputs logged by the current prototype.

---

## 3. Agile Development Roadmap (Based on Prototype)

**Methodology:** Scrum (2‑week sprints)

### Phase Breakdown

| Phase                              | Sprint Count | Key Deliverables                                                                                                                                |
| ---------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | 
| Phase 1 – Discovery & Planning     | **½ sprint** | Scope & SLAs locked, production architecture design (confirming vector store, hosting), backlog groomed                                             |
| Phase 2 – Architecture & Infra      | **1 sprint** | Production Vector Store setup (e.g., Managed ChromaDB/alternative), GitHub Actions CI/CD, baseline IaC (Terraform/Pulumi)                             |
| Phase 3 – Feature Development       | **3 sprints** | • Robust data ingestion/embedding pipeline<br>• Agent refinement (prompt tuning, error handling)<br>• Python `eval` hardening<br>• API endpoints (FastAPI)<br>• Minimal React UI | 
| Phase 4 – MLOps & Deployment        | **1 sprint** | Monitoring setup (metrics, logging, W&B dashboards), cost & latency dashboards, blue-green deployment strategy                                  |
| Phase 5 – QA & Hardening           | **1 sprint** | Load/performance testing, security testing (pen-test, dependency scans), prompt injection safeguards, compliance checks                        |
| Phase 6 – Pilot & Handover         | **½ sprint** | Runbooks, demo day, stakeholder training, final documentation                                                                                   |

**Estimated Timeline:** **7 sprints ≈ 14 weeks (≈ 3½ months)**
**Dev Cycle:** Bi‑weekly demos, JIRA‑managed tickets, Sprint Planning / Grooming / Retro cadence

---

## 4. Team Composition & Resource Plan (Example)

*This remains largely the same as an initial estimate, adjust based on final scope.*
| Role                       | FTE | Duration      | Responsibility                                                                   |
| -------------------------- | --- | ------------- | -------------------------------------------------------------------------------- | 
| Tech Lead / Full‑stack Eng.| 1   | Full project  | End‑to‑end design, React UI, code reviews, infra decisions                        |
| ML / Backend Engineer      | 1   | Ph 2 – 5      | Retrieval layer optimisation, agent prompt tuning, evaluation framework hardening | 
| Platform / MLOps Engineer  | 0.5 | Ph 2 – 6      | Cloud setup, CI/CD, monitoring, cost optimisation                                 |
| QA / SecOps (Contract)     | 0.5 | Ph 5 – 6      | Automated load / pen / fuzz testing, security remediation                         |
| PM / Scrum Master          | 0.25| Full project  | Agile cadence, stakeholder alignment, risk tracking                               |

---

## 5. Technical Architecture (Proposed for Production)

**Backend:** Python (e.g., FastAPI) for API layer.
**Frontend:** React (or chosen framework).
**LLM:** OpenAI GPT-4o / GPT-4o-mini (or alternative, model selection informed by prototype W&B results) with versioned prompts.
**Retrieval:** 
    - **Vector Store:** ChromaDB (consider managed cloud options) or alternative (e.g., pgvector, managed vector DBs) indexed with `all-mpnet-base-v2` or similar Sentence Transformer.
    - **Reranker:** Cohere ReRank API.
    - *Future:* Consider adding BM25/sparse vector fusion before reranking.
**Evaluation:** Extend prototype's `eval.py` and **Weights & Biases (W&B)** integration for ongoing monitoring.

**Services:**

- Embedding computation service (potentially GPU-accelerated).
- Asynchronous task queue (e.g., Celery + Redis or AWS SQS) for indexing/embedding jobs.
- Secure execution environment for Python `eval()`. 

---

## 6. MLOps & Deployment Plan

**CI/CD:** GitHub Actions + Docker + Poetry (or chosen dependency manager).
**Model/Prompt Lifecycle:**

- Prompt versioning integrated with evaluation runs (tracked in W&B).
- Canary deployment or A/B testing for prompt/model changes.
- Metrics logging: Execution accuracy, failure rates (specify, extract, eval), latency, cost (tracked via W&B and infra monitoring).

**Monitoring Tools:**

- **Weights & Biases (W&B)** for experiment tracking, model/prompt performance, evaluation results.
- Infrastructure monitoring (e.g., Prometheus/Grafana, CloudWatch, Datadog) for system health, latency, resource usage.

**Testing:**

- Unit + integration tests for retrieval, agent steps, and execution.
- Model performance regression suite using curated evaluation sets.
- Eval results stored, versioned, and linked in W&B.

**Security & Compliance:**

- Secure API keys via cloud provider secret management (e.g., AWS Secrets Manager, GCP Secret Manager).
- Role-based access control (RBAC) for API and infrastructure.
- Input validation and sanitisation to mitigate prompt injection.
- Data redaction/encryption (if handling sensitive PII beyond source documents).
- Audit logs for requests, agent steps, and final answers.

---

## 7. Maintenance & Support Plan

| Activity                     | Frequency  | Resource           | Tool/Method                          |
| ---------------------------- | ---------- | ------------------ | ------------------------------------ |
| Performance Regression Eval  | Bi-weekly  | ML/Backend Eng     | `eval.py` / W&B Dashboard |
| Prompt Review & Tuning       | Monthly    | ML/Backend Eng     | W&B Analysis, Manual Review        |
| LLM/Embedding Model Updates  | Quarterly+ | ML/Backend Eng     | Re-evaluation, W&B Comparison      |
| Cost Auditing                | Monthly    | Platform Eng + PM  | Cloud Billing + Usage Logs         |
| Dependency/Security Updates  | Monthly    | Platform/Backend Eng | Dependabot, Snyk, etc.             |
| Bug Fixes                    | Ad-hoc     | Backend/Full-stack | GitHub Issues + SLA tracking       |
| New Feature Requests         | Quarterly  | PM                 | Stakeholder Engagement, Backlog    |

---

## 8. Deliverables Summary

- Production-ready FinRAG API service based on the two-step agent architecture.
- Scalable and secure backend infrastructure with monitoring and alerting.
- Production-grade data ingestion and embedding pipeline for ChromaDB (or chosen vector store).
- MLOps pipeline: CI/CD, prompt/model versioning, evaluation harness integrated with W&B.
- Comprehensive documentation:
  - README, Setup Guide, Deployment Runbook
  - API Documentation
  - W&B Dashboard Guide
  - DevOps SOP and support escalation paths



