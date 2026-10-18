# Developing HLSmith

Add implementation code under `hlsmith/`. Root Python modules retain historical imports and commands; extend their adapters only when an existing interface needs it. Keep generated experiments outside the tracked tree and register new studies before interpreting their outcomes.

Run `python scripts/check_offline.py -q` and `python reproducibility/paper/replay.py` for ordinary changes. For controller, provider or prompt changes, also compare captured responses with the frozen `v6` snapshot using `scripts/compare_v6_replay.py`. Hardware integration and paid model experiments have separate, explicit budgets.

Treat benchmark inputs, reference evidence and prompt profiles as versioned experimental inputs. Changed prompts or compiler contracts need new identities. Preserve failures, timeouts, selected source hashes and historical manifests rather than editing them to agree with a new result.

The package's source adapters deliberately share implementation module objects so older callers and tests can patch the same helpers. Keep that behavior when extracting further controller logic.

Document retired commands and preserve their evidence before removing them. Check live process dependencies before moving historical worktrees. The private archive and the unrelated C-to-CUDA checkout are outside the public source release.
