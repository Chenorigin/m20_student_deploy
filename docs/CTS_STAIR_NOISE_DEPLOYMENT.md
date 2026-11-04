# CTS 上楼降音策略：导出与一键启动

组合降音 checkpoint 必须用 `scripts/export_cts_stair_noise_checkpoint.py` 导出；普通 CTS exporter 不包含降音残差。完整 release 使用 `scripts/prepare_cts_release.py --policy-kind stair-noise --checkpoint CHECKPOINT --training TRAINING_ROOT --output NEW_RELEASE`。导出的模型仍接收 obs789、独立 raw_command3，返回 nominal action16。冻结 v24 与降音 head 均包含在 ONNX 内；原始前进命令超过已保存阈值才启用残差。

## 一键启动

在本仓库目录执行，地址填写当前连接的机器人 SSH 目标；GOS 经 AOS 连接：

```bash
./scripts/start_cts_stair_noise.sh --aos user@10.21.41.1 --gos user@10.21.31.104
```

默认包固定为相邻 `rl_training/logs/evaluation/cts_noise_deploy_20261104/candidate_8300/models/v1`，并校验其 checkpoint SHA；不会自动选最新模型。该包是部署适配验证候选，不代表训练效果最优。仓库不是相邻目录时增加 `--training /path/to/rl_training`；选其它已导出的降音包时增加 `--bundle /path/to/models/v1`。可增加 `--expect-aos-serial`、`--expect-gos-serial` 锁定已核验的硬件身份。

脚本调用训练仓库现有 `scripts/tools/m20_robot.py start --bundle ...`：双板身份及模型检查 → 交互 `DEPLOY_PRONE` → 双板离线校验及切换 → 原有感知和 SDK 启动流程。SSH/sudo 密码只在终端输入，不写配置；原 `START_STUDENT`、commissioning、手柄和急停门禁保留。双板部署异常即停止，不自动继续启动。每次 start 都明确选择该包。

```bash
# 仅验证参数及本地模型，不连接机器人
./scripts/start_cts_stair_noise.sh --aos user@10.21.41.1 --gos user@10.21.31.104 --dry-run
# 停止仅需 AOS；不检查模型包，不要求 GOS 在线
./scripts/start_cts_stair_noise.sh stop --aos user@10.21.41.1
# 双板状态
./scripts/start_cts_stair_noise.sh status --aos user@10.21.41.1 --gos user@10.21.31.104
```

## 本次板端适配与边界

1515 两板现用 release 为 `20261103_cts_v29_model2500`。升级 source/installed Python CTS 加载器及 bundle validator，保留现用 v29 兼容；未改 C++ 运控、标定、PD 或模型链接。两板的数据盘 `/var/opt/robot/data/cts_stair_noise_20261104` 保存候选包、安装哈希收据和 `adapter_backup` 原文件；这是备份，勿当运行源码。

实测：CPU 导出/ONNX 对拍、AOS 原生 C++ 128 帧对拍、GOS 元数据和两板 source/installed 导入验证。启动脚本仅做离线测试，本轮未切换模型、未启动电机。沿用现有 `nominal_pd`，没有加入缺少硬件接线的 landing controller；真实上楼能力、声学降噪和启动后的时延仍需现场验证。详细证据在训练仓库 `logs/evaluation/cts_noise_deploy_20261104/`。


## CTS 公共运行基线对齐（2026-11-04）

默认启动目标改为两板数据盘 `/var/opt/robot/data/cts_noise_aligned_20261104`，不再借用可被其它策略改写的 `current`。这份独立 release 从 CTS 公共源代码重新构建；保存现场的 site/LIO/外参，配置按 `deploy/cts_runtime_baseline.json` 与所选模型 perception 对齐。配置与二进制不是只复制模型时顺带继承的旧文件。

`check_cts_runtime_alignment.py` 检查速度范围、policy HOLD、覆盖/时效/会话 HOLD、nominal_pd、PD/动作映射、本体观测和关节约束，以及雷达前视、量程、采集扩边；实际构造 mapper 验证采集网格与固定策略网格分离。数值唯一来源为 baseline JSON 和模型 sidecar；不在另一份启动参数中重复硬编码。

`build_board.sh` 成功后记录源码、已安装 Python、native lidar reader 和 AOS 控制/对拍二进制摘要。新启动入口在模型切换前检查，AOS/GOS 原启动脚本在起进程前再次检查；缺少配置、旧安装包、改过源码但未构建或摘要变化均拒绝。停止入口不执行对齐检查。

本 release 不继承旧 commissioning，配置内容变化后须由现场重新确认；软件对齐/离线对拍不等于已完成真机运动、声音或时延验收。本轮没有启动、切换 current 或停止另一 release 的进程。
