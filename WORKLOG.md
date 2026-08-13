# v3 notebook worklog index

This is the coordination index for notebook-author sessions. The
[ALCHEMI tutorial guide](TUTORIAL_GUIDE.md) is the single
source for curriculum, teaching, visual, helper, and review decisions. Use the
`$alchemi-tutorial-authoring` skill to apply it.

## Logs

| ID | Owned notebook directory | Owned log |
|---|---|---|
| 01 | `notebooks/01-atomicdata-batch/` | [AtomicData and Batch](worklog/01-atomicdata-batch.md) |
| 02 | `notebooks/02-zarr-data-loading/` | [Zarr data loading](worklog/02-zarr-data-loading.md) |
| 03 | `notebooks/03-model-interfaces-composition/` | [Model interfaces and composition](worklog/03-model-interfaces-composition.md) |
| 04 | `notebooks/04-hooks/` | [Hooks](worklog/04-hooks.md) |
| 05 | `notebooks/05-base-dynamics/` | [BaseDynamics](worklog/05-base-dynamics.md) |

Future advanced work, currently unassigned:

- [06 — GPU pipelines and profiling](worklog/06-gpu-pipelines-profiling.md)
- [07 — Training and fine-tuning](worklog/07-training-finetuning.md)
- [08 — Domain decomposition](worklog/08-domain-decomposition.md)
- [Integration](worklog/integration.md)

## Rules

1. Read the canonical authoring guide, `environment/README.md`,
   `shared/README.md`, this index, the owned log, and relevant shared requests.
2. Inspect every owned file and the live notebook state before editing.
3. Write only the assigned notebook directory and paired log during notebook
   authoring. Root and shared changes belong to a bounded integration pass.
4. Edit `.ipynb` files through the notebook MCP/live VS Code bridge.
5. Use `./scripts/v3-run` for every Python, test, and Jupyter command. Treat the
   lock, Python version, and runtime pins as fixed inputs.
6. Append dated log entries and preserve earlier entries. Record decisions,
   changed files, checks, blockers, shared requests, answers, and the next
   action.
7. Use request IDs such as `N03-REQ-001`. Answer a request from the responding
   notebook's log and cite the same ID.
8. Record root and shared integration work in `worklog/integration.md`.

Within an assigned notebook directory, the author may edit the notebook, local
helpers, local tests, small generated outputs, and the local README. Root,
shared, environment, build, notice, and Git state remain integration-owned.

## Shared requests and integration

Start support code inside the notebook that teaches it. When two notebooks need
the same small helper, record a shared request and let an integration pass
decide whether to promote it.

An integration pass may reconcile shared terminology and links, promote proven
helpers, update root navigation or notices, run combined checks, and report
remaining target-hardware or rendered-review work. Keep each pass bounded and
record it in the integration log.

## Entry template

```markdown
## YYYY-MM-DD HH:MM TZ — short checkpoint

Owner: N01
Status: planned | in progress | blocked | ready for integration

Observed:
- ...

Changed:
- `path`

Validation:
- command or notebook action: result
- text review: short, direct, no motivational preamble or patronizing explanation
- plot review: shared dark style loaded; NVIDIA green is the primary series
- progress review: shared Rich black-and-green pattern used only where a cell visibly waits

Shared request:
- ID: N01-REQ-001
- For: N02 | integration | all
- Need: ...
- Why: ...
- Status: open | answered | resolved

Next:
- ...
```

Omit empty sections. Keep entries concise enough that another author can scan all logs before working.
