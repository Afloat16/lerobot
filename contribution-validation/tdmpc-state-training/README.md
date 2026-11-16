# State-only TDMPC training: baseline investigation

This fork-only branch contains CPU forward/backward/optimizer regressions for state-only, state+environment and state+image input, plus a complete public-history audit extended to training methods. No production change has been made and no result has yet been observed.

Current TDMPCConfig accepts state-only features, its encoder supports the state branch, and real state-only inference already ran in the warm-start investigation. The training forward currently chooses its horizon from an image or environment-state key even when neither is configured. This candidate must be tested and checked against every historical source variant before any production patch is proposed.

Raw historical PR text remains in the separate task-progress branch, preserving its original spelling without adding it to the validation source tree.
