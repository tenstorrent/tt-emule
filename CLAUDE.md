Always try to reduce the api surface included in jit_hw and try to use the real headers instead.
tt-emule backed software emulation always runs in slow dispatch mode.
We will always run on wormhole n150 architecture unless specified otherwise.
Always run the regression tests after code changes.
Compile using clang-20.
jit_hw and using real host APIs is the main use case and should be prioritized over standalone mode.
