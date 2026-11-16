## Summary / Motivation

When `n_action_steps > 1`, `select_action` consumes that many distinct actions before planning again, but CEM always shifts the previous mean by one. For a five-step plan with three executed actions, the next initialization retains `[m1, m2, m3, m4, 0]` instead of the unexecuted tail `[m3, m4, 0, 0, 0]`. With finite CEM iterations this can change the next selected action.

## What changed

- Shift the previous mean by `n_action_steps`, leaving the newly exposed tail at zero.
- Keep the default one-step behavior, repeated-single-action behavior, and reset semantics.
- Add 16 CPU cases through real `select_action -> predict_action_chunk -> plan` calls, including batch sizes 1/2, chunk lengths 1/2/3/5, a fully consumed horizon, repeated actions, and reset.

## How was this tested

The tests control only CEM's Gaussian proposals; the action queue, planner, latent dynamics and value networks run normally. On unmodified main, the six multi-action cases fail and ten controls pass. With this change, all 16 pass.

Python 3.12.13, PyTorch 2.15.0, CPU. Exact commit: `69747d8f44403ac64c6f15bdb1429df8150de4f0`.

| Fork Fast Tests tier | Passed | Skipped |
| --- | ---: | ---: |
| Base | 2,585 | 540 |
| Dataset | 1,757 | 132 |
| Hardware | 7 | 72 |
| Visualization | 9 | 151 |

[Baseline](https://github.com/Afloat16/lerobot/actions/runs/43961818735) · [Final tests](https://github.com/Afloat16/lerobot/actions/runs/43963629106) · [Quality](https://github.com/Afloat16/lerobot/actions/runs/43963629165)

The Quality run passed after retrying a Go dependency download failure. To reproduce the focused check:

```bash
uv sync --locked --extra test
uv run pytest tests/policies/tdmpc/test_warm_start.py -q
```

## Checklist

- [x] Linting/formatting: all pre-commit hooks passed in fork CI.
- [ ] All tests pass locally: verification was executed in fork CI.
- [x] Documentation updated: warm-start comment.
- [ ] CI is green: upstream PR checks are pending.
- [ ] Community Review: review of #4692 is prepared; posting is pending.

## Reviewer notes

The repeat path uses one planned action multiple times and retains its existing one-step shift. This change adds no parameters or checkpoint fields. The regression establishes correct warm-start alignment; it does not measure trained-policy success rates.

AI assistance: ChatGPT assisted with implementation and regression tests.
