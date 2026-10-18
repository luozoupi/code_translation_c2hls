# Research commands

The active cleanup workflow uses `rerun_cleanup.py`, `compare_v6_replay.py`, and `check_offline.py`. Run each command with `--help` for its inputs. The paper's portable numerical replay is `../reproducibility/paper/replay.py`.

Training and corpus tools include `build_role_attributed_sft_dataset.py`, `export_model_aware_sft_dataset.py`, `train_agentic_sft_qlora.py`, and `serve_agentic_peft_openai.py`. Training is optional and is not part of the requested frozen-checkpoint rerun.

Other dated campaign commands are retained for the experiments they document. They may expect historical checkpoints, absolute paths, or frozen source files. Consult their docstrings and the matching history notes before invoking them. Their presence does not imply that a new run is authorized or that its original hosted model weights remain available.
