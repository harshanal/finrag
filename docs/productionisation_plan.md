# FinRAG – Production Deployment & Operations Plan

## 1. Project Overview

**Project Scope:** Transition the FinRAG prototype into a robust, scalable production system capable of supporting real-time financial question answering from uploaded financial documents (reports, tables, etc.) using LLMs and symbolic reasoning.

---

## 2. Objectives

- Productionize the FinRAG prototype for commercial use.
- Ensure scalability, security, and reliability.
- Incorporate MLOps best practices for continuous deployment, monitoring, and model governance.
- Provide explainability and auditability suitable for financial services compliance.

---

## 3. Agile Development Roadmap (Revised)

**Methodology:** Scrum (2‑week sprints)

### Phase Breakdown

| Phase                              | Sprint Count | Key Deliverables                                                                                                  |
| ---------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------- |
| Phase 1 – Discovery & Planning     | **½ sprint** | Scope & SLAs locked, architecture sketch, backlog groomed                                                         |
| Phase 2 – Architecture & Infra      | **1 sprint** | Supabase + pgvector setup, GitHub Actions CI/CD, baseline Terraform                                                |
| Phase 3 – Feature Development       | **3 sprints** | • Hybrid retrieval SQL<br>• Planner prompt v2 with JSON retry<br>• DSL v2 executor with unit guard<br>• Minimal React UI |
| Phase 4 – MLOps & Deployment        | **1 sprint** | Prometheus / Grafana monitoring, cost & latency dashboards, blue‑green rollout                                     |
| Phase 5 – QA & Hardening           | **1 sprint** | Load‑ & fuzz‑tests @100 QPS, pen‑test fixes, prompt‑injection safeguards                                           |
| Phase 6 – Pilot & Handover         | **½ sprint** | Runbooks, demo day, stakeholder training, final documentation                                                     |

**Estimated Timeline:** **7 sprints ≈ 14 weeks (≈ 3½ months)**  
**Dev Cycle:** Bi‑weekly demos, JIRA‑managed tickets, Sprint Planning / Grooming / Retro cadence

---

## 4. Team Composition & Resource Plan 

| Role                       | FTE | Duration      | Responsibility                                                    |
| -------------------------- | --- | ------------- | ---------------------------------------------------------------- |
| Tech Lead / Full‑stack Eng.| 1   | Full project  | End‑to‑end design, React UI, code reviews, infra decisions       |
| ML / Backend Engineer      | 1   | Ph 2 – 5      | Retrieval layer, planner prompt, evaluation harness              |
| Platform / MLOps Engineer  | 0.5 | Ph 2 – 6      | Cloud setup, CI/CD, monitoring, cost optimisation                |
| QA / SecOps (Contract)     | 0.5 | Ph 5 – 6      | Automated load / pen / fuzz testing, security remediation        |
| PM / Scrum Master          | 0.25| Full project  | Agile cadence, stakeholder alignment, risk tracking              |

---

## 5. Technical Architecture

**Backend:** Python (FastAPI), PostgreSQL / DynamoDB\
**Frontend:** React (or Streamlit POC)\
**LLM:** OpenAI GPT-4o with prompt versioning + safety checks\
**Retrieval:** Hybrid BM25 + vector DB (Pinecone or FAISS)\
**Evaluation:** OpenAI Evals or custom W&B integration

**Services:**

- Embedding caching layer (e.g., Redis)
- Asynchronous task queue (Celery + Redis or AWS SQS)
- Modular Tool Executor (DSL + secure sandbox)

---

## 6. MLOps & Deployment Plan

**CI/CD:** GitHub Actions + Docker + Poetry\
**Model Lifecycle:**

- Prompt templating version control
- Canary deployment for prompt changes
- Metrics logging: program match rate, execution accuracy, latency, cost

**Monitoring Tools:**

- Weights & Biases or Prometheus/Grafana
- Custom dashboards for prompt versions and tool failures

**Testing:**

- Unit + integration tests for each tool
- Model performance regression suite
- Eval results stored and versioned in `outputs/`

**Security & Compliance:**

- Secure API keys via AWS Secrets Manager
- Role-based access control (RBAC)
- Data redaction and encryption
- Retention policies for audit logs

---

## 7. Maintenance & Support Plan

| Activity               | Frequency  | Resource     | Tool                         |
| ---------------------- | ---------- | ------------ | ---------------------------- |
| Model drift evaluation | Monthly    | ML Engineer  | Custom eval script / W&B     |
| Prompt updates         | Bi-monthly | Tech Lead    | Versioned prompt tests       |
| LLM cost auditing      | Monthly    | PM + Backend | Cloud billing + usage logs   |
| Bug fixes              | Ad-hoc     | Backend, QA  | GitHub Issues + SLA tracking |
| New feature requests   | Quarterly  | PM           | Client engagement backlog    |

---

## 8. Deliverables Summary

- Production-ready FinRAG pipeline (retriever, planner, executor)
- Secure and scalable backend with monitoring
- React-based demo and internal testing UI
- MLOps pipeline: CI/CD, model versioning, test harness
- Comprehensive documentation:
  - README, Setup Guide, Deployment Checklist
  - Evaluation metrics, dashboards
  - DevOps SOP and support escalation paths



