# Template Design Specification — TEL-C2-001

**Template ID:** TEL-C2-001
**Template Name:** TelecomPlanServiceQAAgent
**Category:** Cat 2 (multi-step domain workflow — retrieval pattern)
**Industry:** TEL

## Position in the AgentCore architecture

- **Agent class:** `TelecomPlanServiceQAAgent` (alias `Graph`)
- **L1 Base (framework base class):** `AgentBaseGraph` — direct framework inheritance
- **Inner graph base:** `BaseGraph` — `DomainWorkflowGraph`
- **Pattern:** two-layer nested architecture — the fixed five-node outer backbone plus a
  `GraphNode` in the `main` slot wrapping an inner domain workflow
- **Three-layer separation:**
  - State: flat `TypedDict` (never a Pydantic model — msgpack checkpoints corrupt
    silently on one); structured fields stored as JSON strings via `to_json()` /
    `from_json()`
  - Node: `FunctionNode` subclasses overriding `execute(self, state) -> dict` only —
    no extra parameters; config knobs flow through State, never a per-call argument
  - Graph: composition via `register_nodes()`; the outer `add_edges()` is not overridden

## Purpose

A telecom plan and service question-answering agent. Customer-service representatives,
sales staff and self-service-portal customers ask a natural-language question about
carrier plans, features, pricing, contract terms or service procedures; the agent
retrieves grounded passages from the plan and service catalog and assembles a
catalog-grounded answer with citations, behind an output boundary that enforces the
published invariant below.

Answer assembly is deterministic: keyword retrieval plus rule-based composition, with no
model call. See *Answer synthesis* below for the seam a model-backed builder would use.

## Architecture overview

### Outer backbone (`AgentBaseGraph`)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max_retry)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | framework default | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | trust gate; shape checks; instruction-injection screen; caller-data contract; identifier screen → `validated_input` | `VERIFIED_EXTERNAL` |
| main | `PlanServiceQAGraphNode` (`GraphNode`) | delegates to the inner `DomainWorkflowGraph`; maps the inner `formatted_answer` → outer `result` | — (delegation) |
| post_process | `PostProcessNode` | the output boundary — enforces the published invariant | `VERIFIED_EXTERNAL` |
| finalize | framework default | response_metadata, total_time_ms | — (framework) |

A node whose incoming state already carries an error status is skipped by the framework
before `execute()` runs, so a refusal at `pre_process` genuinely stops the pipeline: the
`main` slot never executes and routing sends the run straight to `finalize` with an empty
output.

### Inner graph (`DomainWorkflowGraph` — `BaseGraph`, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner nodes declare `required_trust_level = TrustLevel.ANONYMOUS`. The external
trust gate lives on the outer backbone; a stricter inner level would deny a real
authenticated invoke at runtime.

| Node | Responsibility | Input State | Output State |
|------|----------------|-------------|--------------|
| `InputValidateNode` | Re-validate the caller-data contract; parse the question (plain text or a JSON envelope); normalise and cap it; resolve filters | `validated_input`, `input_context` | `search_query`, `query_filters`, `caller_catalog`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval over the caller-supplied catalog slice, or the bundled sample corpus: tokenise, score title/tags/content overlap, apply the category filter | `search_query`, `query_filters`, `caller_catalog`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Category-match boost, relevance floor, cap at `top_k` | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded assembly from the ranked passages only, with numbered citation markers | `ranked_documents` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Render the answer document: body, Sources list, standing disclaimer | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data flow

```
user_input + input_context
  → PreProcessNode                              → validated_input (+ caller data validated)
  → PlanServiceQAGraphNode.extract_input        → inner DomainWorkflowGraph.invoke(...)
        → input_validate                        → search_query / query_filters / caller_catalog
        → retrieve                              → retrieved_documents
        → rerank_filter                         → ranked_documents
        → generate_answer                       → grounded_answer / citations
        → output_format                         → formatted_answer
     get_output() → {formatted_answer, citations, status, ...}
  → PlanServiceQAGraphNode.merge_output         → result, plan_service_answer
  → PostProcessNode                             → formatted_output (boundary-enforced)
```

## The caller-data contract

`POST /invoke` accepts an optional `input_context` object next to the question. The full
contract lives in `src/caller_contract.py`; the accepted fields are:

| Field | Type | Bound |
|---|---|---|
| `channel` | string | inert identifier `[a-z0-9_]{1,32}` — metadata only |
| `category` | string | inert identifier; restricts retrieval to one catalog category |
| `top_k` | number | whole number, 1–20 |
| `score_threshold` | number | 0.0–1.0 |
| `catalog_documents` | list | ≤ 25 entries; each `{id, title, category, source, content, tags}` with per-field caps and a 100,000-character total budget |

Three rules hold across every field:

1. **Fail closed.** An out-of-contract value stops the retrieval path. Nothing is
   clamped or silently ignored — a caller who asked for something the agent cannot
   honour gets told so, not a different answer.

   **Completion is not the same as answering.** The run then ends with
   `AgentStatus.SUCCESS`, which reports that the request was handled safely to a
   defined end — not that an answer was produced. A value the caller can correct
   (an out-of-contract parameter, an empty or over-long question) ends this way so
   the caller receives the reason and can send a corrected request on the same
   conversation; terminating instead would end the calling surface's turn and
   surface only an exception type, leaving the reason reachable solely from the
   audit trail. The reason travels as `error_code` in State (`EMPTY_INPUT`,
   `QUESTION_TOO_LONG`, `INVALID_REQUEST`), every later domain node passes through
   without doing work once it is set, and `PostProcessNode` renders it as a static
   caller-facing sentence.

   Two classes keep terminating, and must not be folded into the above: content
   refused outright (an instruction-override payload — re-sending a reworded variant
   is not a correction), and a breach of a contract the caller cannot influence.
2. **Name the field, never the value.** Rejected caller data does not round-trip into
   logs or the response.
3. **Every number is parsed as finite and in range.** `float()` accepts `"NaN"` and
   `"Infinity"`, and a JSON body may carry them unquoted; every comparison against NaN is
   `False`, so an unchecked NaN threshold silently disables the filter it configures.
   Booleans are rejected too — in Python they are integers, and would otherwise pass as
   0 or 1.

`catalog_documents` is what makes the public path do real work. Plan catalogs are
operator-specific and change constantly, so a deployment passes the slice it wants
answered from; the bundled sample corpus at `config/kb/telecom_kb.json` is the fallback
that keeps the template runnable and its tests deterministic.

The framework's `GraphNode` does not forward `input_context` into an inner graph, so this
template bridges it explicitly: `extract_input()` stashes the value and the inner graph's
`_extra_initial_state()` reads it back (`src/graph/context_bridge.py`). The bridge uses a
`ContextVar`, so concurrent invocations in one process cannot observe each other's data.

## Runtime configuration

`config/agent.yaml` is the flat discovery manifest — every key at root level, `class:` a
single dotted import path. Runtime parameters live in `config/config.yaml`, which the
registry loads separately and passes to the graph constructor as `config=`:

```
max_retry: 3
timeout_s: 30
retrieval: {top_k, score_threshold, kb_path, hybrid_search}
llm:       {temperature, max_tokens}
```

Every entry point reads this file and passes it as `Graph(config=...)`. The HTTP adapter
does so when constructing the agent, so a standalone process honours the declared values
rather than running on framework defaults; `PlanServiceQAGraphNode._parent_config()` receives it from the
outer graph and forwards the `retrieval` and `llm` blocks to the inner graph, which republishes
`retrieval` into inner state as the JSON field `retrieval_config`. `RetrieveNode` and
`RerankFilterNode` read `top_k` / `score_threshold` from there — bounds-checking them like
any other input — and fall back to module defaults mirroring the file only when the field
is unseeded, which happens when a node is exercised on its own.

## Output invariant

Declared once in `src/output_schema.py`, rendered by `OutputFormatNode`, enforced by
`PostProcessNode`. Every answer this agent emits:

- **(a)** carries the standing informational disclaimer;
- **(b)** carries a Sources section in which every `[n]` marker used in the answer body
  resolves to a listed source;
- **(c)** contains no direct subscriber identifier and no credential-shaped string.

The boundary applies four layers, and the order is part of the design:

| # | Layer | On violation |
|---|---|---|
| 1 | **Credential scan**, run on the document exactly as assembled | withhold the whole document, error status |
| 2 | **Verbatim caller text** — the answer is built from catalog passages and citations only, so an embedding of the caller's own question is caller-controlled text on the external surface | replace the fragment, audit event |
| 3 | **Subscriber identifiers** — mobile numbers, e-mail addresses, 13-plus-digit account/device numbers, national identifier shapes | replace the token, audit event |
| 4 | **Grounding, then disclaimer** | a dangling `[n]` withholds the document; a missing disclaimer is restored, audit event |

Layer 1 runs first on purpose: a scan that looks for a pattern must see the text before
anything rewrites it, or an earlier redaction can break a key or token into pieces the
scanner no longer recognises.

Layer 3 is deliberately narrower than the identifier screen on the input side. A catalog
publishes support hotlines and short numeric codes on purpose; deleting those would
silently remove the answer the caller asked for. The input screen, which sees text a
subscriber typed, is broad by contrast.

**On rounding.** Some agents in this family publish monetary aggregates and snap them onto
a coarse grid so individual figures cannot be recovered. That is deliberately *not* done
here, and the grid does not apply: this agent answers questions about published plan
pricing, where `¥6,980 per month` **is** the answer — rounding it to `¥7,000` would make
the agent wrong rather than discreet. Nothing rendered is an aggregate over private
records; every figure is quoted from a catalog passage that is cited beside it. The
invariant this boundary enforces instead is (a)–(c) above, applied to every answer.

## Security gates

- **Trust gate:** every node declares `required_trust_level`. `PreProcessNode` and
  `PostProcessNode` require `VERIFIED_EXTERNAL`; the standalone adapter raises an
  authenticated bearer caller to that level (`INVOKE_AUTH_TOKEN`).
- **Input screens (owned by this template):** shape checks, an instruction-injection
  screen, the caller-data contract, and the subscriber-identifier screen — all in
  `PreProcessNode`, and re-applied for the caller contract in `InputValidateNode`. The
  refusal is the template's own, proven by calling `execute()` directly with no framework
  wrapper in front of it, so the guarantee holds wherever the agent runs. The framework
  also masks scanned input fields at every node boundary; that is an additional layer, not
  the one this template relies on.
- **Injection-screen anchoring:** every pattern is anchored on whole words and a specific
  phrase shape. A bare substring screen fails in the direction that hurts most — it
  refuses ordinary work. `act as` matches inside *impact assessment*, `system prompt`
  matches the handset dialog a subscriber is actually asking about, and `ignore` matches
  *can I ignore the data-cap warning*. The suite probes both directions: each family
  refuses its attack forms, and real catalog sentences and support questions are accepted
  untouched.
- **Output boundary:** the four layers above, each with its own audit event, so an
  operator can tell which one fired. The gate is a module-level function called from
  `execute()`, not an instance method: the framework auto-wraps node gate hooks, and
  defining one on the node would collide with that machinery.
- **Audit:** every node emits exactly one domain event on its success path
  (`emit_trace_event`); the framework emits the node lifecycle events itself. Domain event
  names: `pre_process_complete`, `input_validate_complete`, `input_validate_refused`,
  `retrieve_complete`, `rerank_filter_complete`, `generate_answer_complete`,
  `output_format_complete`, `post_process_complete`, `input_validation_failed`,
  `output_withheld`, `output_caller_text_redacted`, `output_identifiers_redacted`,
  `output_disclaimer_restored`.

## State definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `NotRequired[str]` | normalised, identifier-stripped question | outer |
| `plan_service_answer` | `NotRequired[str]` | final answer, mapped from the inner `formatted_answer` | outer |
| `search_query` | `NotRequired[str]` | normalised search query | inner |
| `query_filters` | `NotRequired[Optional[str]]` (JSON) | validated `category` / `top_k` / `score_threshold` | inner |
| `caller_catalog` | `NotRequired[Optional[str]]` (JSON) | caller-supplied catalog slice | inner |
| `retrieval_config` | `NotRequired[Optional[str]]` (JSON) | forwarded runtime `retrieval` block | inner |
| `retrieved_documents` | `NotRequired[Optional[str]]` (JSON) | scored candidates | inner |
| `ranked_documents` | `NotRequired[Optional[str]]` (JSON) | passages that cleared the floor | inner |
| `grounded_answer` | `NotRequired[str]` | answer body with citation markers | inner |
| `citations` | `NotRequired[Optional[str]]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `NotRequired[str]` | rendered answer document | inner |
| `intake_notes` | `NotRequired[Optional[str]]` (JSON) | parse notes; carries no caller values | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State constraints (mandatory):**

- Flat `TypedDict` only — primitives and JSON-serialisable types.
- Structured fields stored as JSON strings via `to_json()` / `from_json()`, by every
  producer *and* every consumer.
- Domain fields are `NotRequired[...]`, so the TypedDict is valid before any node writes.
- `formatted_output` is not re-declared — that backbone field stays framework-owned.
- No tokens, keys, credentials or raw subscriber identifiers in State.
- The invocation context travels via `config["configurable"]`, never in State.
- No Pydantic models, dataclasses or arbitrary Python objects.

## Answer synthesis

Answer assembly is deterministic end to end: retrieval is keyword scoring, and
`GenerateAnswerNode` composes the body from the ranked passages (a lead sentence plus one
cited excerpt per passage). There is no model call and no model client dependency; the
`llm` block is forwarded for a model-backed builder but no node reads it, and no system
prompt is loaded at runtime.

`config/prompts/answer_synthesis_prompt.md` documents the upgrade seam: a model-backed
`GenerateAnswerNode` would synthesise over the same `ranked_documents` input and emit the
same `grounded_answer` / `citations` contract, so no other node changes.

## Composition pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation:** `propagate` — inner failures re-raise as `SubgraphError`.
- Inner domain nodes run at `ANONYMOUS`; the outer boundary slots at `VERIFIED_EXTERNAL`.

## Import isolation

- [x] No platform-internal SDK imports.
- [x] Import targets are `framework/` and `shared/` only.
- [x] No intermediate agent class names appear in any base position.

## Design decision record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Framework base | `AgentBaseGraph` | `AutonomousBaseGraph` | **`AgentBaseGraph`** | Fixed multi-step retrieval workflow, no autonomous loop |
| Composition | standalone slots | `GraphNode` → inner `BaseGraph` | **`GraphNode` → inner `BaseGraph`** | A five-step domain workflow exceeds a single `main` node; nesting keeps the outer backbone untouched |
| Answer synthesis | rule-based assembly | model call | **rule-based** | Deterministic and testable; a model-backed builder swaps in at the documented seam |
| Catalog corpus | external vector store | caller-supplied slice + bundled sample | **caller-supplied slice + bundled sample** | Catalogs are operator-specific and change constantly; the retrieval contract is store-agnostic, so a vector store swaps in behind `RetrieveNode` alone |
| Node config parameter | `execute(state, config=None)` | `execute(self, state)` + State-seeded config | **`execute(self, state)`** | The framework node contract is exactly `execute(self, state) -> dict`; config flows through State |
| Output invariant | monetary rounding grid | disclaimer + grounding + identifier screen | **disclaimer + grounding + identifier screen** | Published plan prices are the answer; rounding them would make the agent wrong. Nothing rendered is an aggregate over private records |

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.
