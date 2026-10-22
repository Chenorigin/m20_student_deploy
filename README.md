# M20 Student 模型部署工作区

仓库与本地目录统一命名为 `m20_student_deploy`，用于持续维护 Student 模型部署。当前随附的是 student-v3.5 的适配实现；下面的模型版本、`deployment_family`、校验脚本名及板端 release 路径仍表示具体模型的兼容性契约，不能仅通过重命名替换。部署其它版本时，应使用匹配的模型契约并完成对应验证。

GitHub 仓库包含源码、配置、当前模型文件和 `evidence/offline_models/` 固定测试输入；构建产物、运行状态和其它历史实测数据仅保留在本地。下文其它 `evidence/` 链接需在本地查看。`SOURCE_SHA256.json` 是原始部署副本的历史快照；打包时由 `scripts/package_release.py` 重新生成 payload 校验清单。

本目录从 `m20_heightmap_ws` 与 2026-10-19 实测 AOS/GOS 当前部署复制，原目录、两板原 release 和原 `current` 不改。

实际学生仍为 789→512→256→128→16 ELU MLP，教师的 residual 分支不进入学生推理。新适配位于 AOS `student_policy_runner.hpp`：只从实机本体和已送达地形计算 `legacy_v1_v3_5` 航向反馈；反馈后命令写入当前观测及本体历史。地形过期重算使用同一个原始命令和反馈前状态，历史只提交一次。前视 mask 与训练一致，GOS 地图/标定和消息 ABI 沿用现场版本。

正式 ONNX 仅把线性层计算改为 FP64：原 FP32 权重精确扩宽，每层线性输出回到 FP32 后执行原 ELU，外部输入/输出仍为 FP32。此举修正 ARM 与训练端不同 FP32 累加顺序造成的固定样本超差；数值门槛仍为 `atol=rtol=2e-5`。ORT 实际入口同时关闭预打包。模型 SHA256 为 `58694c118ae027f1b462dfb295c8ce6454f5c3c736710beca16564b9542bb64b`。

`v1` 是原有 789 维消息/profile 名称，不代表旧模型；模型 bundle 的 `deployment_family` 必须为 `m20_student_v3_5`。模型、反馈参数、默认关节位置、动作缩放、PD、观测缩放/顺序均校验，缺项或不兼容时拒绝。

## 当前模型及两板路径

checkpoint：`rl_training/logs/rsl_rl/deeprobotics_m20_student_v3_5/2026-10-19_12-12-39_599339_student_v3_5/model_2000.pt`。

模型、板端构建与验收的最终状态见 `evidence/final_summary.json`。两板独立目录均为：

```text
/home/user/m20_student_v3_5/releases/20261019_model2000_7370f3b3
```

本次只进行程序适配、构建和无关节命令的离线验收；没有开始 SDK 初始化、站立、行走，也没有继承旧 commissioning 凭据。新候选的 `STAGED_NOT_ACCEPTED.txt` 由 AOS 全部固定数值测试通过后自动移除，验收收据为 `runtime/offline_acceptance.json`。现场 commissioning 使用原有交互流程。

## 后续迭代

在有 CPU PyTorch、ONNX、ONNX Runtime、PyYAML、NumPy、pytest 的训练 Python 中运行：

```bash
python scripts/prepare_v35_release.py \
  --checkpoint /absolute/path/to/model_NEXT.pt \
  --output /absolute/path/to/new_release
```

默认读取该 checkpoint 同目录保存的 `env.yaml`；可显式传 `--env-yaml`。每次输出必须是新目录。程序从 checkpoint 读取 hidden_dims 和反馈参数，严格加载学生 actor，不依赖原教师 checkpoint；生成 TorchScript、默认 FP64 线性层 ONNX、训练参数契约及固定 ARM 对拍输入。只支持当前 MLP/ELU、无归一化、789→16、20ms 的部署 ABI；其它结构/语义改变会报错，需再次适配。

```bash
cd /absolute/path/to/new_release
python scripts/package_release.py --output /absolute/path/to/new_payload.tar.gz
```

使用 `safepush` 向两板新路径传输，再用 `scripts/stage_board_release.py` 核验 payload SHA256 和实测板卡序列号，解包到不存在的新目录；不覆盖原目录或切换原 current。板端先 `python3 scripts/prepare_board.py`，再读取余量并单线程构建：

```bash
python3 scripts/run_with_budget.py --limit-mb 3072 \
  --receipt runtime/build_resource.json -- bash scripts/build_board.sh
```

AOS 无电机离线验收：

```bash
source scripts/board_env.sh
python3 scripts/offline_board_models.py
# 固定板端数值与反馈测试全通过后，记录软件验收并解除该副本的临时软件阻断：
python3 scripts/accept_offline_release.py
```

GOS 只负责原感知链，使用该副本的配置/消息包；不加载神经网络、不发关节命令。两板均须核验源文件和 manifest，不把磁盘到货当作运行验收。

## 人工启动入口

在两板各自的上述独立目录中，完成已有 commissioning 流程后使用以下入口。AOS 在板内 `tmux` 会话运行，保留交互确认和手柄操作：

```bash
# GOS
sudo -E bash scripts/operator_start_gos.sh v1
# AOS（交互终端）
sudo -E bash scripts/operator_start_aos.sh v1
```

如该副本尚无本次开机的 commissioning 收据，先在两板分别 `source scripts/board_env.sh` 并交互运行 `python3 scripts/operator_commission.py --profile v1`。本次适配没有自动运行这些启动入口。

## 验证边界

CPU 对拍、负例和独立复审见 `evidence/`。数值门槛保留原值；ORT 所有实际推理入口均关闭预打包。历史实机观测重放只证明后端数值关系，不能代表新策略已上机。扩大随机/极值输入仍有数值超差，见 `evidence/numerics/report.md`。

保持现场原动作保护配置：轮目标额外 clip 关闭，与 Isaac 训练在极端动作上的 processed clip 不完全一致；本次没有顺手改变它。新模型实机站立、行走、楼梯能力均未验证。
