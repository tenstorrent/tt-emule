Always try to reduce the api surface included in jit_hw and try to use the real headers instead.
tt-emule backed software emulation always runs in slow dispatch mode.
We will always run on wormhole n150 architecture unless specified otherwise.
Always run the regression tests after code changes.
Compile using clang-20.
Always log the full output of regression tests
When browsing the repo, consult STRUCTURE.md first — it indexes every file under src/ and include/ and the top-level symbols each contains.
Keep STRUCTURE.md up to date: when you add, remove, rename, or move a source file, or add/remove a top-level symbol (class/struct/enum/namespace/free function/macro), update its entry in STRUCTURE.md in the same change.
When fixing a failure in a mock API, the goal is always being faithful to the canonical silicon implementation. Avoid creating parallel or different code paths to avoid bugs. If it works in silicon, we must make it functional and correct in the mock API.
