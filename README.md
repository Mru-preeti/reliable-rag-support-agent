# Aster & Row Customer Support AI Agent

An agentic customer support system designed for **Aster & Row**, an outdoor equipment and lifestyle brand. Built with deterministic privacy controls, hybrid BM25 + dense semantic retrieval, generalized conflict detection, and strict anti-injection defenses.

---

## 1. System Overview & Problem Solved

Customer support AI assistants frequently suffer from critical vulnerabilities:
1. **Hallucination & Policy Invention**: Inventing return windows, free shipping exceptions, or warranty coverage without authoritative backing.
2. **Privacy Leaks**: Exposing sensitive customer PII (emails, physical addresses) and internal warehouse/risk data (`risk_score`, `warehouse_note`).
3. **Action Dishonesty**: Falsely claiming to have completed actions (e.g. canceling orders, issuing credit card refunds) when the agent has no API authorization or mutation capability.
4. **Authority & Conflict Blindness**: Citing superseded or draft documents, or arbitrarily resolving active policy contradictions.

This system provides **deterministic, code-level safety boundaries** around an LLM generator, guaranteeing that customer privacy, document authority, and action honesty rules are enforced programmatically.

---

## 2. Architecture & Major Components

```text
                                Customer Query
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Session & Multi-Turn    │
                        │      Memory Manager       │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Security Pre-Filter     ├────────► [Immediate Refusal]
                        │  (PII/Secret Exfiltration)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Intent & Tool Router    │
                        └──────┬─────────────┬──────┘
             Order Intent      │             │     Policy / Knowledge Intent
            (with Order ID)    ▼             ▼     (or Mixed Inquiry)
    ┌──────────────────────────────┐     ┌──────────────────────────────┐
    │     Safe Order Tool          │     │ Hybrid Knowledge Retriever   │
    │  - Normalization (ORD-XXXX)  │     │  - Okapi BM25 + Stemming     │
    │  - PII & Risk Redaction      │     │  - Dense MiniLM Embeddings   │
    │  - Status Precedence & ETA   │     │  - Authority Precedence      │
    │  - 30-Min Window Calculation │     │  - Generalized Conflict Det. │
    └──────────────┬───────────────┘     └──────────────┬───────────────┘
                   │                                    │
                   └─────────────────┬──────────────────┘
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │    Context Encapsulator   │
                        │    (<untrusted_data>)     │
                        └─────────────┬─────────────┘
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │    LLM Provider Layer     │
                        │  (OpenAI / Mock Provider) │
                        └─────────────┬─────────────┘
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │  Post-Generation Guard    │
                        │  1. PII Regex Redaction   │
                        │  2. Citation Validation   │
                        │  3. Source Pruning        │
                        │  4. Action Honesty Check  │
                        └─────────────┬─────────────┘
                                     │
                                     ▼
                          Structured AgentResponse
```

### Component Classification

| Component | Nature | Function |
| :--- | :--- | :--- |
| `src/tools.py` (`OrderService`) | **Deterministic Code** | Normalizes order IDs, validates statuses, suppresses stale ETAs, calculates 30-min cancellation windows, redacts PII and internal fields. |
| `src/knowledge.py` (`KnowledgeBase`) | **Deterministic Code & Local ML** | Parses YAML frontmatter, filters superseded/draft docs, computes BM25 Okapi & 384d MiniLM embeddings, detects active policy conflicts. |
| `src/session.py` (`SessionManager`) | **Deterministic Code** | Manages isolated, bounded multi-turn conversation sessions. |
| `src/prompt.py` | **Prompt Engineering** | Enforces XML untrusted data boundaries, citation formatting `[filename: Section Heading]`, and anti-injection instructions. |
| `src/providers.py` | **LLM Abstraction** | Interfaces with OpenAI API (`OpenAIProvider`) or deterministic offline mock provider (`DeterministicMockLLMProvider`). |
| `src/agent.py` (`SupportAgent`) | **Orchestrator & Safety Guard** | Routes requests, encapsulates untrusted context, sanitizes PII leaks, prunes irrelevant sources, and enforces action honesty. |
| `src/evaluation.py` | **Deterministic Benchmark** | Evaluates 27 multi-turn and adversarial scenarios across 11 safety & quality dimensions. |

---

## 3. Core Safety & Engineering Mechanisms

### A. Order Privacy & Status Integrity (`src/tools.py`)
- **Strict Privacy Whitelist**: Customer records in `data/orders.json` contain sensitive fields (`customer_name`, `email`, `shipping_address`, `warehouse_note`, `risk_score`). The tool strips all non-whitelisted fields in Python before returning results.
- **Status Precedence**:
  - `cancelled` (`ORD-1004`) and `returned` (`ORD-1008`): `estimated_delivery` is strictly suppressed to `None` so stale ETAs are never reported.
  - `exception` (`ORD-1010`): Sets `requires_handoff = True`.
  - `shipped` without ETA (`ORD-1011`): Preserves `None` without hallucinating dates.
- **Cancellation Window**: Uses the snapshot timestamp `2026-08-15T12:00:00Z` to evaluate if a `pending` order was placed $\le 30$ minutes ago (`ORD-1001` eligible vs `ORD-1002` processing/ineligible).

### B. Metadata-Aware Hybrid Knowledge Retrieval (`src/knowledge.py`)
- **Authority Filtering**: Documents with `status: superseded` (e.g. `02-returns-policy-legacy.md`) or `policy_authority: none` / `draft` (e.g. `14-internal-content-migration-notes.md`) are strictly filtered out before retrieval.
- **Hybrid Retrieval**: Combines Okapi BM25 ($k_1=1.5, b=0.75$) with dense 384-dimensional semantic embeddings (`all-MiniLM-L6-v2`) to accurately retrieve paraphrased queries without brittle keyword matching.
- **Multi-Intent Decomposition**: Splits compound queries into sub-queries so secondary topics (e.g. return window + warranty) are not crowded out.

### C. Generalized Conflict Detection (`src/knowledge.py`)
- Automatically detects contradictions between active official documents on shared topics (e.g. `11-product-care.md` hand-washing vs `12-breeze-tumbler-product-card.md` dishwasher-safe).
- **Conflict Handling Policy**: When an active conflict is detected, the agent explains both perspectives factually, refuses to arbitrarily pick one side, includes both source citations, and routes to a human support specialist with `handoff = True`.

### D. Action Honesty & Eligibility (`src/agent.py`, `src/providers.py`)
- The agent explicitly evaluates transaction eligibility from deterministic tool data.
- It informs the user whether an action is eligible (`ORD-1001`) or ineligible (`ORD-1002`), but **never falsely claims to have executed cancellations, refunds, or address changes**, escalating directly to human staff.

### E. Source Relevance Pruning (`src/agent.py`)
- Retrieved candidate chunks are filtered against the actual substantive claims in the final answer (`overlap >= 2` content tokens), preventing unrelated warranty or care sections from being cited in simple return policy answers.

---

## 4. Benchmark Evaluation Suite (`src/evaluation.py`)

The evaluation suite tests **27 behavior-level cases** across 11 core quality & safety dimensions:

| Dimension | Visible Cases | Adversarial / Edge Cases | Pass Rate |
| :--- | :---: | :---: | :---: |
| **A. Knowledge Retrieval & Precedence** | 2 | 1 | **100%** |
| **B. Grounded Answers & Paraphrasing** | 2 | 2 | **100%** |
| **C. Citation Correctness & Pruning** | 2 | 2 | **100%** |
| **D. Order Tool Usage & Normalization** | 3 | 1 | **100%** |
| **E. Privacy & PII Protection** | 1 | 2 | **100%** |
| **F. Prompt Injection Resistance** | 1 | 1 | **100%** |
| **G. Multi-Turn Conversation Memory** | 1 | 1 | **100%** |
| **H. Session Isolation** | 1 | 1 | **100%** |
| **I. Authoritative Source Conflicts** | 1 | 1 | **100%** |
| **J. Insufficient-Information Abstention** | 1 | 1 | **100%** |
| **K. Action Honesty & Eligibility** | 1 | 2 | **100%** |
| **Total** | **15** | **12** | **100.0% (27/27)** |

---

## 5. Development Bug Diary (Failures, Root Causes, & Fixes)

1. **Heuristic Keyword Scoring on Semantic Paraphrases**
   - *Failure*: Initial keyword matching failed to retrieve relevant documents for paraphrased queries such as `"What is the penalty fee if I ship something back?"`.
   - *Root Cause*: Relying on exact string matches rather than statistical and semantic representation.
   - *Fix*: Implemented Okapi BM25 with Porter stemming and dense local embeddings using `SentenceTransformer("all-MiniLM-L6-v2")`.
   - *Regression Test*: `TestTrueParaphrases` in `tests/test_knowledge_retrieval.py`.

2. **False Conflict Detection Across Divergent Geographic Scopes**
   - *Failure*: Conflict detector flagged false contradictions between domestic shipping processing times (1–2 days) and Canadian delivery estimates (5–9 days).
   - *Root Cause*: Conflict detection extracted shared general tokens like `"shipping"` without checking section heading entity alignment or destination scope.
   - *Fix*: Added scope divergence filtering (ignoring numerical divergence between domestic and international sections) and required section-level entity alignment.
   - *Regression Test*: `TestGeneralizedConflictDetection.test_no_false_conflict_for_trailplus_exception`.

3. **Arbitrary Conflict Resolution & Hallucinated Safe Guidance**
   - *Failure*: When encountering the Breeze Tumbler contradiction between `11-product-care.md` and `12-breeze-tumbler-product-card.md`, the agent recommended hand-washing as definitive guidance.
   - *Root Cause*: Conflict response template selected one policy option rather than presenting the conflict transparently.
   - *Fix*: Refactored orchestrator to state both conflicting official sources factually without picking a side, attaching both citations and flagging `conflict_detected = True` and `handoff = True`.
   - *Regression Test*: `test_active_source_conflict_does_not_arbitrarily_choose` in `tests/test_agent_orchestration.py`.

4. **Action Requests Missing Action Honesty & Eligibility Determination**
   - *Failure*: For cancellation requests like `"Cancel ORD-1001"`, the agent merely repeated the pending status without evaluating eligibility or explaining execution limits.
   - *Root Cause*: Order lookup response was purely status-oriented and did not assess transaction eligibility against the 30-minute placement window.
   - *Fix*: Integrated deterministic eligibility evaluation into action responses, distinguishing eligible (`ORD-1001`) from ineligible (`ORD-1002`) orders, explaining that automated agents cannot perform database mutations, and routing to human support.
   - *Regression Test*: `TestActionHonestyAndEligibility` in `tests/test_agent_orchestration.py`.

5. **Extraneous Citations in Top-K Context**
   - *Failure*: Simple return policy questions cited unrelated warranty (`07-warranty.md`) and product care (`11-product-care.md`) sections merely because they shared the word `"backpack"`.
   - *Root Cause*: Attaching all retrieved top-k chunks regardless of whether the final answer depended on them.
   - *Fix*: Added substantive token overlap pruning (`overlap >= 2` substantive tokens against generated answer), keeping only citations that directly support the response.
   - *Regression Test*: `test_standard_return_policy_with_citations` in `tests/test_agent_orchestration.py`.

---

## 6. How to Run the Project

### Environment Setup
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install lightweight dependencies
pip install -r requirements.txt
```

### Configuration (`.env`)
Create a `.env` file (copied from `.env.example`):
```ini
# Optional: Required only for live OpenAI completions.
# When omitted, the system uses the 100% deterministic offline mock provider.
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

### Running Tests & Evaluation
```powershell
# 1. Run full test suite (54 unit, integration, and benchmark tests)
python -m pytest tests/ -v

# 2. Run standalone evaluation benchmark (27 visible + adversarial cases)
python run_eval.py

# 3. Run interactive end-to-end demonstration (all 11 scenarios)
python demo.py
```

---

## 7. Limitations & Honest Assessment

1. **Local Model Cold Start**: Loading `SentenceTransformer("all-MiniLM-L6-v2")` into memory on initial boot takes $\approx 1.5$ seconds; once loaded, sub-millisecond in-memory inference is achieved.
2. **Deterministic Mutation Safeguard**: The system is intentionally non-mutating. While it accurately assesses eligibility for cancellations, refunds, replacements, and address updates, transactional mutations are deliberately routed to human specialists.
