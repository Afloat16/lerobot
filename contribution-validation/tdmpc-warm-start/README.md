# TDMPC action-chunk warm-start investigation

The production correction is `69747d8f44403ac64c6f15bdb1429df8150de4f0` on `fix/tdmpc-action-chunk-warm-start-20261116`. It changes only the TDMPC planner and 16 CPU regression cases relative to upstream `7910c20a75834ad3fa6a9792df4d44a614d1bba6`.

`ready.json` and `pr-body.md` contain the complete verified result; `prefilled-pr-url.txt` contains the actual comparison URL with title/body parameters. No upstream PR has been created. The required peer review was prepared but its one API submission returned 403. Its exact text and ordinary GitHub UI steps are preserved in `community-review-4692.md` and `community-review-steps.md`; the peer-review checkbox remains incomplete.

The baseline failed six multi-action cases while ten controls passed. All 16 pass after the change. The exact production head passed all four existing Fast Tests tiers and Quality. Test counts distinguish passed, skipped and deselected cases. These are fork CI results, not local tests or upstream CI results.

Full public upstream PR and fork history, including intermediate/reverted source and previous filenames, was examined. `history-baseline.json` is the original planner audit; the later consumption/training audits and all raw CI logs are split losslessly according to `evidence-parts.json`.

The state-only training investigation is excluded because issue #1090 already proposes that fix; the failed baseline and duplicate evidence remain preserved. Fork PRs are validation artifacts and do not count as upstream contributions.
