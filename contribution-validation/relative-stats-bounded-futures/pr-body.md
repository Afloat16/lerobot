## Summary / Motivation

With `num_workers > 1`, `compute_relative_action_stats` submits every expanded chunk batch and keeps all of the completed futures. Each future retains its full NumPy result even after the running statistics have consumed it. Intermediate memory therefore grows with the total number of valid chunks.

## What changed

Use a lazy submission iterator and refill a window of at most `num_workers` pending batches after consuming and releasing completed results. All valid chunks still contribute; the statistics formulas, batch size, public arguments, and serial path stay the same.

Seven regression cases cover completed-array lifetimes with two/three workers, every statistics field against serial processing, episode boundaries, excluded joints, unchanged input arrays, and worker-error propagation.

## Verification

The unmodified baseline fails the two retention cases (ten expanded arrays remain live), while five controls pass. All seven new cases and 53 existing statistics tests pass with the correction.

A fresh-process measurement using NumPy 2.2.6, 800,000 valid starts, 32-step chunks, four action dimensions and two workers produced:

| Measurement | Before | After |
| --- | ---: | ---: |
| Peak live expanded arrays | 16 | 2 |
| Expanded-array storage | 409,600,000 bytes | 51,200,000 bytes |
| Peak process RSS | 940,760 KiB | 754,052 KiB |
| Samples included in statistics | 25,600,000 | 25,600,000 |

All ten output fields were identical for that measurement. Input-column storage still scales with dataset size. These are memory measurements; no training or throughput improvement is claimed.

The broader local selection also hit 13 existing external-dataset fixture errors because `socksio` was missing for the configured SOCKS proxy. They are not counted as passes. The complete existing fork dependency-tier workflow is running on exact head `7dd35f28dd6d6e0fd94da353597eecb51c6ca14d`; [Fast Tests](https://github.com/Afloat16/lerobot/actions/runs/43972209997). [Quality](https://github.com/Afloat16/lerobot/actions/runs/43972209869) has passed both pre-commit and doctests.

## Checklist

- [x] Linting/formatting: fork pre-commit and doctests passed.
- [ ] All tests pass locally: 60 passed; 13 existing dataset fixtures were blocked as described above.
- [x] Documentation updated: inline explanation of the retained-result bound.
- [ ] CI is green: full fork Fast Tests are pending; upstream checks do not exist yet.
- [ ] Community Review: the review of #4692 is prepared, but the API submission was blocked and it has not been posted.

AI assistance: ChatGPT assisted with implementation and regression tests.
