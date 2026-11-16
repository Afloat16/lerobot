# Bounded relative-action statistics batches

Production head: `7dd35f28dd6d6e0fd94da353597eecb51c6ca14d` on `fix/relative-stats-bounded-futures-20261116`. The production diff has two files only. This directory contains evidence and is excluded from that diff.

The real CPU baseline has two failing array-lifetime regressions and five passing controls. With the fix, those seven and 53 existing statistics tests pass. The broader local selection also hit 13 external-dataset fixture errors for a missing SOCKS transport dependency; they are preserved and not counted as passes. Existing fork Quality passed. Full Fast Tests were still running when this checkpoint was saved.

In the measured 800,000-start example, expanded results dropped from 16 live arrays (409,600,000 bytes) to two (51,200,000 bytes). Peak process RSS was 940,760 versus 754,052 KiB. Every output statistic was identical, including 25,600,000 samples. There is no throughput or trained-policy claim.

The history audit covers all 3,800 public upstream PR heads, 265 fork branches and other advertised refs, all intermediate source versions, merge-parent diffs and previous filenames. Four malformed historical training/factory files contain no target definitions or calls; their raw text and manual disposition are preserved. Eleven prior parallel variants all retain the complete eager Future list. The remaining same-named implementations use other APIs and accumulate all their chunks.

No upstream PR has been created. The peer review of #4692 remains unposted after one API 403. Its exact draft is ../tdmpc-warm-start/community-review-4692.md. Open the ordinary https://github.com/huggingface/lerobot/pull/4692/files page, recheck the reviewed head, paste the draft into Review changes and submit as Comment. Only after the actual review appears should its real URL be inserted into this PR body and the community-review checkbox marked complete. No prefilled review-form parameters are claimed. The full PR comparison title/body are in prefilled-pr-url.txt.
