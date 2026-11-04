# CTS stair finetune v2 deployment

This adapter supports v2 only. It exports the full trained encoder/decoder for raw forward vx > 0.0001 and the frozen v24 actor otherwise. The nominal 16-action ABI and existing execution control are unchanged. Do not use the ordinary CTS exporter for this checkpoint.

Prepare on the workstation with `scripts/prepare_cts_release.py --policy-kind stair-finetune --training TRAINING_ROOT --checkpoint CHECKPOINT --output NEW_CANDIDATE`. No hardware commands are issued.

The exported candidate is source plus models, not a built board release. The board operator must install the small Python compatibility adaptation into a **new isolated copy** of a built CTS release on **both** boards before selecting this bundle. Transfer this candidate through the normal safepush workflow. Keep `STAGED_NOT_ACCEPTED.txt` in that copy until offline acceptance. The adapter rejects current and links escaping the isolated copy. Run its `scripts/adapt_cts_stair_release.py --release BUILT_RELEASE` to inspect, then add `--apply` while the runtime is stopped. The adapter checks previous content, updates both source and installed Python packages, and backs up replaced files; unknown/newer versions are rejected. It neither switches models nor starts/stops processes. Re-run inspection to check that no updates remain. Do not use a candidate without the compiled SDK binaries and site configuration as BUILT_RELEASE.

From the canonical m20_student_deploy checkout, launch later with:

```bash
./scripts/start_cts_stair_finetune.sh start --aos user@10.21.41.1 --gos user@10.21.31.104 --release AOS_ISOLATED_RELEASE --gos-release GOS_ISOLATED_RELEASE
```

The default bundle is the exact requested v2/500 in the adjacent rl_training checkout. Physical serials default to the measured 1515 boards. Addresses are explicit connection arguments. For isolated built releases add `--release AOS_RELEASE --gos-release GOS_RELEASE`. `--dry-run` validates arguments and local bundle without SSH. Existing DEPLOY_PRONE / START_STUDENT and handset gates remain. Passwords are interactive and are never stored.

`stop --aos TARGET` does not require a valid model or GOS. Actual adaptation, model switching and starting were not performed in this task: another process owns robot operation. CPU ONNX parity does not certify physical locomotion or native ARM runtime acceptance; the normal operator must complete board offline acceptance before start.

## CTS nominal without policy HOLD

The production runner constructs StudentControlCore with forced HOLD disablement for CTS nominal_pd. An old site `hold_enabled=true` cannot re-enable a latch. Coverage, terrain age/session and scheduling discontinuities do not latch policy HOLD. Scheduling expiry discards an unpublished candidate and resumes on a fresh tick; previously issued support is retained during the gap. Heading feedback is unchanged. Invalid outputs, stale body state, hardware faults, handset loss and emergency stops retain the existing fault/stop paths.

This requires rebuilding `student_control` from the new SDK source; changing JSON or exporting a model alone is insufficient. After installing the new executable and matching SDK headers, rerun board offline acceptance. Confirm `[CTS_SCHEDULING] phase_guard=v4 holds_enabled=false` on RL entry. Never replace a running controller executable.

### 真机手柄与 MuJoCo 速度范围对齐（2026-11-04）

实际 GamepadInterface 入口对配置范围与 MuJoCo 手柄包络取交集：前后 ±0.7 m/s、横移 ±0.5 m/s、转向 ±0.6 rad/s。旧现场 robot.json 的更宽范围不能重新扩大入口；更窄范围仍保留。启动日志 `[COMMAND_RANGES]` 输出有效范围。报文已经是速度指令，因此采用夹紧，不对低速指令再次乘系数，松杆仍为零；不把训练采样下界 0.15 当作最低行走速度。

此变更通过标准一键启动入口加载的新 SDK 生效，无额外启动参数。保留 nominal_pd 无 HOLD；不改变模型、几何标定或关节 PD。MuJoCo 键盘具有 1 秒升速/0.3 秒松键斜坡，模拟摇杆没有自动插入该斜坡。名义 50 Hz 策略/200 Hz 发布不等于实时调度已达标；真机仍需验丢帧与运动。

本轮事实与回放数据保存在训练仓库 logs/evaluation/cts_stair_v2_1515_deploy_20261104/。原运行中 31.32 秒、1548 策略帧、52 帧前进指令超过 0.7；78 帧实测观测 TorchScript/ARM ONNX 回放最大动作误差 4.77e-6。速度范围差异确定，整体不稳定的唯一根因尚未确定。
