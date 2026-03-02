# New Chat Handoff (Smart Stack)

Use this file when starting a fresh chat. It is prefilled with current project context so you do not need to rebuild context manually.

## 1) Current Snapshot (Autofilled)

- Workspace: `/Users/pranjal/garage/smart_stack`
- Branch: `master`
- Context source of truth: `/Users/pranjal/garage/smart_stack/CUMULATIVE_CONTEXT.md`
- Latest architecture milestone: **Phase 1 Policy Foundation implemented**
  - query types: `generic|constrained|attribute|identity`
  - adaptive policy shared by Search + Chat
  - policy confidence fallback to generic
  - bounded gate/top-k adaptivity
- Latest test status:
  - `python -m unittest tests.test_query_planner tests.test_query_policy tests.test_search_rerank tests.test_intent_ranking`
  - `Ran 17 tests ... OK`

## 2) Important Runtime Flags

- `SMART_STACK_ADAPTIVE_POLICY_ENABLED` (default intended: `1`)
- `SMART_STACK_LEGACY_PATCHES_ENABLED` (default intended: `1` for safe rollout)
- `SMART_STACK_POLICY_BASE_SIMILARITY_GATE`
- `SMART_STACK_POLICY_GATE_ADJUSTMENT_MAX`
- `SMART_STACK_POLICY_MAX_TOPK_MULTIPLIER`
- `SMART_STACK_POLICY_CONFIDENCE_FALLBACK_THRESHOLD`

## 3) What To Include In New Chat (Minimum Required)

1. **Single objective**
   - Example: `Fix ranking for query: "white check shirt having meal"`.
2. **Scope files** (absolute paths)
   - Example: `/Users/pranjal/garage/smart_stack/mm_stack/search_engine.py`
3. **Exact repro commands + JSON output**
   - Include at least 2 failing command outputs.
4. **Flag values used in test run**
   - Especially adaptive/legacy and gate-related vars.
5. **Acceptance criteria**
   - Behavioral, not vague.

## 4) Copy-Paste Prompt Block For New Chat

```md
Task: [one exact goal]

Workspace: /Users/pranjal/garage/smart_stack
Read first: /Users/pranjal/garage/smart_stack/CUMULATIVE_CONTEXT.md

Current state:
- Phase 1 adaptive policy foundation is implemented.
- Keep local-only behavior and RAM-safe operation.

Scope (edit only):
- [absolute file path 1]
- [absolute file path 2]

Do not touch:
- [optional absolute paths]

Repro:
1. [command]
2. [command]

Observed output (paste JSON snippets):
- [snippet 1]
- [snippet 2]

Runtime flags used:
- SMART_STACK_ADAPTIVE_POLICY_ENABLED=[value]
- SMART_STACK_LEGACY_PATCHES_ENABLED=[value]
- SMART_STACK_POLICY_BASE_SIMILARITY_GATE=[value]
- SMART_STACK_POLICY_GATE_ADJUSTMENT_MAX=[value]
- SMART_STACK_POLICY_MAX_TOPK_MULTIPLIER=[value]
- SMART_STACK_POLICY_CONFIDENCE_FALLBACK_THRESHOLD=[value]

Acceptance criteria:
1. [expected ranking/answer behavior]
2. [expected confidence/fallback behavior]
3. [no regression in query class X]
4. [tests/commands to pass]
```

## 5) Quick Command Pack (Optional)

Run these before new chat and paste outputs:

```bash
cd /Users/pranjal/garage/smart_stack

git branch --show-current
git status --short

./.venv/bin/python -m unittest \
  tests.test_query_planner \
  tests.test_query_policy \
  tests.test_search_rerank \
  tests.test_intent_ranking

# Example repros (replace query)
./.venv/bin/python /Users/pranjal/garage/smart_stack/mm_cli.py search "code" -n 8 --json
./.venv/bin/python /Users/pranjal/garage/smart_stack/mm_cli.py chat "code" --json -n 4
```

## 6) Notes

- Keep each new chat focused on one phase/bug to avoid context bloat.
- If task is complete, open a fresh chat for the next objective.
- Update `/Users/pranjal/garage/smart_stack/CUMULATIVE_CONTEXT.md` after major milestones.
