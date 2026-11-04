# Frozen v24 idle residual deployment

Use `scripts/export_cts_idle_checkpoint.py --checkpoint PATH --training TRAINING_ROOT --output NEW_BUNDLE` for a complete idle-head model bundle. Use the existing `scripts/prepare_cts_release.py` with explicit `--policy-kind idle-finetune` to prepare a self-contained local release including the updated contract validator. Its default remains ordinary CTS.

The canonical training loader verifies the pinned v24 baseline and frozen tensors, then the shared CTS exporter serializes the **complete** actor into TorchScript and ONNX. The actor's interface, perception, command feedback and execution semantics remain CTS nominal16 with independent raw command3. The new actor scope and `idle_finetune` provenance bind the head, baseline, checkpoint, configuration, update counter, gate and bounds. Existing ordinary CTS contracts follow the original validation branch.

The seven-file model bundle and whole release remain candidates. Copying only the model into an old release does not upgrade its Python contract validator. Install the matching release through the existing staged deployment workflow before selecting an idle candidate. No automatic board connection, runtime switch, SDK start, calibration or commissioning inheritance occurs during preparation. C++ inference and actuator control are unchanged. ARM parity, board timing and physical behavior remain separate acceptance steps.

The training checkout provides the local user entry `bash scripts/tools/m20_idle_finetune.sh CHECKPOINT --prepare-only` and the detailed guide `docs/m20_cts_idle_finetune_deploy.md`. It validates using the newly packaged runtime, not the training environment's policy implementation.
