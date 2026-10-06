# Test Specification — TEL-C2-001

**Template ID:** TEL-C2-001
**Template Name:** TelecomPlanServiceQAAgent
**Category:** Cat 2 (nested retrieval workflow)

This document is the contract the delivered tests implement
(`tests/unit/` + `tests/proof_of_boundary/`). Every table row below names a
test file that ships in this repository.

## 1. Scope and conventions

- Per-node unit tests for the five inner domain nodes and the two outer
  boundary nodes.
- The caller-data contract as its own suite: bounded, finite, inert,
  fail-closed — probed in both directions.
- Manifest and runtime-config consistency, and bundled-catalog integrity.
- Retrieval quality over the bundled catalog (golden queries).
- Inner-graph and outer-graph composition.
- Boundary proofs: import isolation, State serialization safety, invoke order,
  human-in-the-loop propagation (conditional), server boot, and end-to-end
  behaviour through the real ASGI `/invoke`.

**Invocation convention.** Per-node tests invoke the node as `node(state)`, so
the framework's own gates run first, exactly as they do in production. The
state builder sets an authenticated caller level for the two boundary nodes
and an anonymous one for the five inner domain nodes.

**One deliberate exception.** The instruction-injection tests call
`execute()` directly, with no framework wrapper in front. The framework also
refuses high-confidence injection before `execute()` runs, so an end-to-end
call cannot distinguish "the template refused" from "something in front of it
refused". Calling `execute()` directly is what proves the refusal belongs to
this template — which is what makes it hold wherever the agent runs.
Assertions stay behavioural: an error status and nothing carried forward,
never a match on a gate's wording.

**Nodes take no config argument.** `execute(self, state)` is the only
signature. A test that needs to exercise a configured knob seeds the State
field `retrieval_config` — the same field the inner graph republishes at
runtime — and still invokes through `node(state)`.

**Input masking.** The framework's input gate masks `user_input`,
`validated_input` and `llm_response`, and only those field names: e-mail
addresses, grouped digit runs and Title-Case name bigrams all become
`[MASKED]` before `execute()` runs. Plan names in this domain ("Unlimited
Plus") are Title-Case bigrams, so positive-path payloads seeded into those
fields are written in lowercase to keep the fixture intact. `search_query`,
`retrieved_documents`, `ranked_documents`, `citations`, `grounded_answer`,
`formatted_answer` and `result` are not scanned, so tests seeding them
directly are free of this interaction. `PreProcessNode`'s own identifier
screen layers underneath the framework mask and catches shapes the generic
patterns miss, such as a solid ungrouped digit run.

**Audit muting.** `shared.*` is never stubbed through `sys.modules` — the
framework imports `shared.security` at load time. The domain audit emitter is
muted by an autouse fixture patching `src.nodes.<module>.emit_trace_event`;
an audit assertion re-patches the same attribute with a spy and asserts on the
event payload.

## 2. Unit tests

### 2.1 Caller-data contract — `test_caller_contract.py`

| Case | Expected |
|------|----------|
| Non-finite matrix per numeric field | `"NaN"`, `"Infinity"`, `"-Infinity"`, and the raw float forms are all refused, naming the field |
| Booleans | refused — they are integers in Python and would pass as 0 or 1 |
| Out-of-range and non-numeric values | refused; fractional values refused for whole-number fields |
| A correctable rejection does not end the conversation | via `invoke()`: empty, whitespace, over-long, out-of-range `top_k`, bad `category` → `status=success`, the response names what to change, no answer; refused instruction-override content still terminates |
| Error content | the message names the field and never repeats the value |
| Inert identifiers | `channel` / `category` / tags accept `[a-z0-9_]{1,32}` and nothing else |
| Catalog entry count, per-field caps, total text budget | each cap refuses beyond its bound, naming the field |
| Caller passages | flattened to one line, stray citation markers neutralised, identifiers stripped at ingest |
| Injection screen — attacks | each family refuses its attack forms |
| Injection screen — real work | 16 genuine support questions containing the same words are accepted |
| Injection screen — own corpus | every bundled catalog title and passage reads clean |

### 2.2 Input boundary — `test_pre_process_node.py`

| Case | Expected |
|------|----------|
| Valid question | success; `validated_input` set; `enriched_context` carries channel and source |
| Empty, whitespace, missing, non-string, over-long | declined with `error_code` (`EMPTY_INPUT` / `QUESTION_TOO_LONG`); no `validated_input`; the run completes so the caller can correct and retry |
| Injection attempts (direct `execute()`) | refused and terminal (`status=ERROR`); nothing carried forward; the payload is not echoed into the error |
| Ordinary questions with the same words (direct `execute()`) | accepted |
| Out-of-contract caller data | fails closed; the error names the field, not the value |
| Identifier screen | no raw account number, phone number or e-mail survives, whichever layer acted |
| Audit | `pre_process_complete` on the accepted path, `input_validation_failed` (reason only) on a refusal |

### 2.3 Request parsing — `test_input_validate_node.py`

| Case | Expected |
|------|----------|
| Plain text | the whole string becomes `search_query`; filters are all unset |
| Whitespace | collapsed to single spaces |
| JSON envelope | `query` / `question` alias, `category`, `top_k` parsed |
| Malformed JSON | read as plain text plus a parse note |
| Out-of-contract envelope values | fail closed, naming `query.top_k` / `query.category` |
| Caller data channel | supplies the filters, and wins over the envelope |
| Caller catalog | carried forward as a JSON string; absent key when none supplied |
| Oversize query | truncated with a note |
| Empty request | noted, not refused — it degrades to a no-coverage answer |

### 2.4 Retrieval — `test_retrieve_node.py`

| Case | Expected |
|------|----------|
| Happy path | the plan-pricing query returns `kb-001` first |
| Ordering | scores strictly descending, all above zero |
| Entry shape | keys `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| Category filter | only entries of that category; `kb-004` first for billing |
| Empty query | no candidates |
| Configured `kb_path` override | an unreadable file degrades with a note and an empty list |
| No configured block | the module default resolves; `kb-001` first |
| Notes | appended to prior notes, never clobbered |

### 2.5 Rerank and floor — `test_rerank_filter_node.py`

| Case | Expected |
|------|----------|
| Relevance floor | 0.5 dropped against the 0.70 default |
| Configured threshold / depth overrides | honoured |
| Caller override | wins over the deployment default, bounded by the contract |
| Out-of-contract configured value | falls back to the documented default rather than clamping into a different filter |
| Category boost | +0.1, re-ranked ahead, capped at 1.0 |
| Garbage entries | skipped, or scored 0.0 and dropped |
| Tie-break | deterministic, identifier ascending |

### 2.6 Answer assembly — `test_generate_answer_node.py`

| Case | Expected |
|------|----------|
| Citation markers | `[1]`, `[2]` with titles |
| Caller question | never echoed into the answer body |
| Citations list | refs mirror ranked order; id, title and source carried |
| Groundedness | the body traces to ranked passages only |
| No coverage | the no-coverage answer; empty citations |

### 2.7 Answer rendering — `test_output_format_node.py`

| Case | Expected |
|------|----------|
| Full document | heading, body, `## Sources` rows, disclaimer; success |
| Blank source | no empty parenthesis suffix |
| Disclaimer | present on every answer |
| No citations | the explicit "none" sources line |
| Missing body | fallback text; success |

### 2.8 Output boundary — `test_post_process_node.py`

| Case | Expected |
|------|----------|
| Clean answer | passes through byte-identical |
| Empty result | forwarded unchanged, not fatal |
| Credential shapes (API key, credential assignment, web token, bearer token) | the whole document is withheld; the secret appears in neither surfaced field |
| Layer order | the credential scan runs on the document as assembled, before any redaction |
| Nested strings | scanned recursively |
| Verbatim caller text | replaced; a short fragment is not treated as an echo; every blocked field is checked |
| Subscriber identifiers | mobile numbers (grouped, bare, international), e-mail, 13-plus-digit runs removed |
| Published catalog detail | hotline `0120-000-111`, price `6,980`, `30GB`, product code `5G-1000` and the year `2026` all survive |
| Dangling citation marker | the answer is withheld as ungrounded |
| Missing disclaimer | restored |
| Layer counters | zero on a clean document; one per layer that fired |

### 2.9 Manifest and runtime config — `test_config_manifest.py`

| Case | Expected |
|------|----------|
| Manifest shape | flat — no `agent:` block; id, namespace, enabled present |
| Class contract | `class:` resolves to the agent class in `src/graph/graph.py`; `name` matches |
| Classification | Cat 2, TEL, deterministic generation mode |
| Declared requirements | no secret and no extra — answer assembly makes no model call |
| Trust level | the declared level equals both boundary nodes' `required_trust_level` |
| Runtime file | `max_retry` / `timeout_s` live in `config/config.yaml`, not the manifest; within the framework ceiling; human-in-the-loop not enabled |
| `load_runtime_config()` | reads the file |
| Retrieval block | mirrors the node module defaults; `kb_path` exists |
| `_parent_config()` | forwards the runtime blocks; never empty, and falls back when the file is unreadable |
| Bundled catalog | JSON list of at least five entries; unique ids; required keys per entry |

### 2.10 Retrieval quality — `test_retrieval_quality.py`

| Case | Expected |
|------|----------|
| Ten golden queries, one per catalog category | the expected entry is ranked first |
| Relevance floor | every survivor at or above 0.70 |
| Citation integrity | every survivor id exists in the catalog |
| Precision | the plan-pricing query keeps only `kb-001` |
| Category filter | billing filter keeps only billing entries, `kb-004` first |
| No coverage | an out-of-domain query yields zero survivors and the no-coverage answer |

### 2.11 Trust gate — `test_trust_gate.py`

| Case | Expected |
|------|----------|
| Anonymous caller on an inner node | admitted; a normalised query is produced |
| Anonymous caller at either boundary | denied; the execute-only output key is absent |
| Authenticated caller at either boundary | admitted |
| Declared matrix | boundary nodes require an authenticated caller; inner nodes admit anonymous |

## 3. Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| Case | Expected |
|------|----------|
| Composition | inherits `BaseGraph`; registers exactly the five domain nodes; no initialize or finalize |
| Seeded initial state | `_extra_initial_state()` republishes the retrieval block as a JSON string and seeds the bridged caller data |
| Output shape | `get_output()` emits the merge contract; `route()` returns END on error |
| End to end | a full inner invoke succeeds; the rendered answer carries the disclaimer and a `kb-001` citation; node history is the five nodes in order |

### 3.2 Outer graph — `test_graph_composition.py`

| Case | Expected |
|------|----------|
| Composition | inherits `AgentBaseGraph`; the `Graph` alias exists; `add_edges()` is not overridden |
| Backbone slots | compile fills all five with the expected classes |
| `get_subgraph()` | returns the inner graph carrying the forwarded retrieval config |
| `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| `merge_output()` | the inner answer reaches both `plan_service_answer` and `result`; changed keys only |
| Config fallback | `_parent_config()` is never empty, even with an unreadable runtime file |
| End to end | an authenticated invoke succeeds and traverses the output boundary |
| Denial | an anonymous invoke errors with an empty output; the output boundary is not traversed |
| Serialization helpers | `to_json` / `from_json` round-trip; None and malformed input handled |

## 4. Boundary proofs

| File | Expected |
|------|----------|
| `test_import_isolation.py` | no platform-internal import anywhere under `src/` |
| `test_state_safety.py` | `State` has no credential-named fields and no model or invocation-context annotations. **PB-5 auto-waived — checkpointing disabled** (`config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`); the conditional gate and traversal helper ship with the stub |
| `test_pb_invoke_order.py` | a full invoke over the payload byte-equal to `deploy/invoke_payload.json` succeeds with the exact expected node history |
| `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — this template declares no human-in-the-loop step**; the conditional skip stub is retained |
| `test_server_boot.py` | importing the server does not raise; the module-level agent is this template's class, compiled; a fresh instance fills all five slots; `/health` reports the agent |
| `test_invoke_e2e.py` | end-to-end behaviour through the real ASGI `/invoke` — see below |

### 4.1 End-to-end through `/invoke` — `test_invoke_e2e.py`

Nothing is stubbed: every request crosses the entry-point auth, the outer trust
and input boundary, the caller-data bridge into the inner graph, all five
domain nodes, and the output boundary.

| Case | Expected |
|------|----------|
| Grounded question | a real cited answer; the catalog price is reproduced exactly, not rounded |
| No coverage | a success outcome with the explicit no-coverage answer |
| Caller catalog | replaces the bundled corpus — the answer could only have come from the caller's slice, which proves the bridge carried it |
| Forged structure in a caller passage | cannot create a second Sources section or a source the answer then cites |
| Identifiers in a caller passage | never reach the answer |
| Non-finite matrix per numeric field | refused, with no output |
| Every other out-of-contract field | refused, with no output |
| Valid overrides | accepted, and the depth override visibly reaches retrieval |
| Empty question | refused |
| Oversize caller object | refused at the adapter with 413 |
| Published invariant | the disclaimer is present on every answer, every citation marker resolves, and the question is not echoed |
| Unauthenticated caller | 401 |

> **Mandatory before review:** import isolation, State safety, invoke order,
> server boot and the end-to-end suite. The human-in-the-loop proof applies
> only to templates that declare such a step; this one does not, so its skip
> must not block the gate.

## 5. Execution summary

- Runner: the real framework wheel, `python -m pytest tests/`
- Total: 346 — 345 passed, 1 skipped (the human-in-the-loop conditional stub)
- Determinism: no model call, no network; retrieval and answer assembly are
  rule-based, so every result is reproducible

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `TelecomPlanServiceQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.
