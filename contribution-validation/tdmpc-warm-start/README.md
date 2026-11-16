# TDMPC warm-start validation

This branch is an internal validation branch based on upstream 7910c20a75834ad3fa6a9792df4d44a614d1bba6. No production change is included yet.

The regression exercises the real CPU TDMPC select_action -> predict_action_chunk -> plan call chain. Only CEM Gaussian proposals are controlled, so that the selected next action exposes the initialization after consuming a chunk. Cases cover chunk lengths 1, 2, 3, and 5, horizon 1 and 5, batch sizes 1 and 2, repeated single actions, and reset. Expected baseline outcome has not yet been executed.

The separate validation-only audit fetches public upstream branch, tag, and all PR-head histories plus all Afloat16 fork branches and tags. It records every relevant intermediate source blob, including reverted versions and former filenames discovered through rename history. Its output requires manual review; it does not automatically declare novelty.

Only the final source correction and regression test will be considered for an upstream PR after real validation and a complete audit. The history audit, this snapshot, and this README are not part of that proposed upstream diff.
