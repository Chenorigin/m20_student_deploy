# CTS 模型适配

新的模型族为 `m20_cts_student`，接收独立的 `obs` 和 `raw_command`，输出 nominal actions。
`obs` 延续现有 v1 地形消息和历史布局；模型包括训练后的 student encoder 和共享 decoder。
原始命令在航向反馈之前采样，不能从 `obs[6:9]` 恢复；即使原始零命令，反馈后的 yaw 也可能非零。
不需要给 GOS 地形消息加命令字段，命令属于 AOS 本地输入。

## 准备新候选

在训练 Python 环境中，从本工作区执行，显式选择一个已完整保存的 CTS checkpoint：

```bash
python scripts/prepare_cts_release.py \
  --checkpoint /absolute/path/to/cts/model_N.pt \
  --training /absolute/path/to/rl_training \
  --output /absolute/path/outside/source/cts_candidate
```

输出必须是不存在的新目录。使用训练工程正式 `load_policy` 严格验证 CTS 元数据、完整模型状态、
decoder bounds 和执行控制契约，然后只导出完整学生与共享 decoder。
教师、critic、优化器、训练用 retention 数据不进入部署文件。
候选不继承旧 runtime、commissioning、DDS 地址或启动许可，原有 release 与 current 不变。

只导出模型也可以：

```bash
python scripts/export_cts_checkpoint.py --checkpoint /absolute/model_N.pt \
  --training /absolute/rl_training --output /absolute/new_model_dir
```

现有 `export_checkpoint.py --profile v1` 在提供 `--training` 时也会识别 CTS；
`export_onnx.py` 与 `check_model.py` 已接入 CTS 分流。
CTS 使用独立的分支图导出，明确保存 FP32 精度，不调用只接受单输入 MLP 的旧 FP64 转换。
原有数值容差不变，ARM 必须实际复跑，不能用 x86 通过代替。

## 离线检查

```bash
python scripts/check_model.py --profile v1 \
  --model /absolute/new_model_dir/policy.onnx \
  --contract /absolute/new_model_dir/observation_contract.json --raw-command 0 0 0
python scripts/validate_cts_bundle.py --model-dir /absolute/new_model_dir
python scripts/offline_cts.py --model-dir /absolute/new_model_dir
```

CPU 验证输入包含未知地形、零/移动命令、原始与有效 yaw 不一致以及门控阈值。
`input.f32` 是固定离线样本：每行先保存 `obs`，随后保存独立 `raw_command`；
它不是新的地形消息格式。历史文件如果没有记录原始命令，不得自动补造 CTS 输入。

板端单线程构建完成后，直接执行无 ROS/SDK 的 C++ 模型探针：

```bash
python3 scripts/offline_cts.py --model-dir models/v1 \
  --probe install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe
```

此路径只需要 NumPy，不需要板端 PyTorch 或 Python ORT。`offline_board_models.py` 也会分流到它。
GOS 使用 `validate_cts_bundle.py --metadata-only` 检查摘要，不加载模型。
`load_config` 核对模型与现场前雷达几何、置信度公式；`Pipeline` 在重投影后继续执行模型前视 mask，
视野外/盲区保持 h=m=c=0。现场标定不同会拒绝启动，不会用模型参数改写物理外参。

新增 `cts_sensor_probe` 可从标准输入读取 JSON 帧数组，每帧提供 `proprio`、`terrain`、
`raw_command` 和 `yaw`；可用 `reset:true` 清空历史。输出观测和原始命令供完整 ONNX 使用。
它调用 `CTSObservation` 的实际 C++ mask、航向反馈和历史实现；不创建电机发布者。

## 实机边界

CTS 训练使用 Safe/Quiet 落脚控制。完整训练执行链需要经过验证的因果接触输入，
当前 SDK 尚无完成实测标定的硬件 binding。模型输出仍经原有动作缩放、关节映射和 PD 形成 SDK 指令。

用户选择的基础运动模式在原目录 `runtime/heightmap_v1.local.json` 中显式设置
`aos.cts_execution_mode="nominal_pd"`。它运行原 CTS10000 student+decoder，保留原始 nominal
动作与常规 PD，跳过触地反馈、地形落脚记忆和落脚修正，不构造落脚控制器。
`M20_CTS_FEEDBACK_CONTRACT` 在此模式不被读取，未知触地值不会伪造为有效零值。
该模式是明确选择的执行消融，不修改模型的训练契约或 ONNX 元数据，也不代表已经完成实机行走验收。

实时运行器现已接入 `obs789 + raw_command3`，原始命令在 heading feedback 前保存；
地形过期后的二次推理也使用同一原始命令。共享落脚公式与地形记忆已移植为 CPU C++，
按模型保存的 settings 加载，不复制另一组默认参数；源码哈希不匹配时拒绝。
SDK 工作者每个真实 5ms 周期执行一次控制，每四周期更新一次 nominal action/history。
漏拍、重复时钟、过期推理、退出后迟到结果均不得发布，不能补跑虚拟子步。
修正只作用于腿参考和增益；轮速与 previous nominal action 不受污染。
HOLD 从 SDK 实际发布缓存保持最后支撑目标/增益，轮速与前馈归零。

缺省模式及显式 `landing` 模式仍要求完整触地验证：控制公式和调用链已实现，但硬件触地验证仍缺。
`standalone_robot_controller=false` 与完整 `quiet_execution_control` 同时保存在 sidecar、TorchScript 和 ONNX。
Python 启动入口和 C++ `main` 在 ROS/FSM/SDK 初始化前核对执行模式和 CTS 模型身份；
只有显式 `nominal_pd` 可以在不依赖落脚反馈的情况下进入原 SDK 流程。
身体反馈、关节位置边界、时效、漏拍、迟到输出、唯一发布者、FSM 和急停逻辑继续生效。
`accept_offline_release.py` 支持 CTS ARM 探针与初始化前模式检查，收据明确为离线软件验收。

### 实测反馈接口

SDK 可通过 `M20_CTS_FEEDBACK_CONTRACT` 指定的 JSON 描述订阅只读 `std_msgs/String` 反馈。
描述包括 `topic/source_id/hardware_serial/world_frame/binding_sha256/publisher_node`；
订阅者要求唯一发布者，校验来源、形状、坐标系、采集时间、session/sequence。
配置订阅不构成硬件许可，也不解除 `main` 的执行能力门禁。
没有该接口时 `GetLandingFeedback()` 明确 unavailable；原 `GetContactForce()` 的零占位不被使用。

消息 schema 为 `m20_cts_landing_feedback.v1`，必须包含同一采样快照的 SDK16 关节位置/速度、
四轮世界支持点、对应三腿关节的完整点 Jacobian，以及**包含基座运动**的世界法向速度。
接触力必须为因果支持力并带逐轮 valid 和采集时刻，不能把未知填成有效零值。
顺序固定为 `joint_order=sdk16`、`leg_order=FL_FR_HL_HR`、`jacobian_columns=policy_leg_joints`。
`capture_monotonic_ns/contact_capture_monotonic_ns` 必须已转换到 **AOS CLOCK_MONOTONIC**；
GOS 原始 monotonic 时钟不可直接相减。ROS query 时间只用于关联原始地形帧。

地图反馈须提供精确匹配 `session/sequence/stamp_ros_s` 的 `terrain_query` 原始 query pose；
不能用当前姿态变换旧图。每格捕获年龄保持不变，重发不续命。无匹配 pose 时不产生新世界地图证据。
当前位姿跳变和地图 session 改变都会清理记忆，并拒绝跳变前捕获的迟到数据；旧 session 不可复活。
外部反馈源的接触估计、运动学、时钟映射和 actuator identification 必须由实测验证。
后续触地专项沿用用户确认已经验证的关节和机身反馈，不以触地专项要求重做整机验证。
未经标定的力矩只用于下面的诊断，不自动生成声称已验证的反馈源。

### 触地专项的只读采集

`scripts/capture_cts_contact.py` 直接订阅原有 `/JOINTS_DATA`、`/IMU` 与厂商运动状态，
不构造 SDK，不创建关节命令发布者。当前厂商状态消息没有逐轮触地或足端力字段。
工具保存到达 callback 的原始帧、原始 source header、AOS monotonic 接收时刻、端点身份与错误；
Foxy Python 回调没有逐帧 writer GID，因此不把图端点快照冒充逐帧来源证明。
非有限原始浮点数使用 `nonfinite_float64_be_hex` 保留 IEEE 位表示，不填零，后续帧继续采集。
best-effort depth 1 可能丢 DDS 帧；不能把保存成功称为源流零丢帧。

```bash
source scripts/board_env.sh
python3 scripts/capture_cts_contact.py --seconds 10 --output runtime/contact_capture_NEW
```

输出目录必须不存在。总帧数、体积与时长有上限；仅在 AOS 运行。接收原始帧与估计频率分别管理，
重计算按尝试限频，失败也不会退化成每个回调重算。重新生成几何使用原始 URDF，而非手填腿长：

```bash
python3 scripts/prepare_contact_geometry.py --urdf /absolute/original/M20.urdf \
  --output /absolute/new_contact_geometry.json
```

`contact_estimator.py` 按 URDF 质量、惯量、关节轴和轮子圆柱几何做完整递归逆动力学，
包含关节与基座运动。关节方向和 offset 只在原始 DDS 入口应用一次；没有发 offset reset 命令。
按 `M*qdd+C+g=tau_motor+J_support.T*F_external` 求三轴外力，保留有符号结果和 Jacobian 条件数，
不能把力矩绝对值、侧碰力或正负裁剪后的数字当成承载力。当前圆柱诊断几何也不冒充训练 Isaac 的胶囊轮。

该工具是诊断，`contact_valid` 始终为 false。速度差分噪声、力矩零偏/摩擦、IMU 杆臂与时钟对齐
仍影响估计；趴地时腹部、膝部、小腿接触也无法仅凭三个关节力矩排除。
下一步需逐轮已知离地与已知承载的对照数据及独立载荷/接触方向依据，验证轮上承载估计、
误检漏检和延迟后才能接入 CTS。无需因此重新运行已确认通过的关节/机身反馈验证。

现场确认可靠支撑、电机卸力后，每次只记录一条明确轮子标签，不能替未知的其它轮子补标签。
空载时必须由合适支架或吊具承托机身，让轮子自然悬空且无人托拉腿或轮子；手托轮子会引入真实外力，
不能标为无支撑样本。硬急停后还要确认编码器反馈继续更新；DDS mode/status 不是物理断电回读：

```bash
python3 scripts/capture_cts_contact.py --condition wheel_unloaded --leg FL --output runtime/FL_AIR_NEW
python3 scripts/capture_cts_contact.py --condition wheel_loaded --leg FL --output runtime/FL_LOAD_NEW
python3 scripts/compare_cts_contact.py --unloaded runtime/FL_AIR_NEW --loaded runtime/FL_LOAD_NEW \
  --output runtime/FL_COMPARISON_NEW.json
```

比较器验证原始记录摘要、同机同 boot、几何/robot/端点和逐轮标签，拒绝趴地无标签数据。
事先固定判据为 loaded Fz 的 p05 大于 max(0, unloaded p95)；重叠则不称为可分。
标签只是操作员报告，比较通过也不自动标为力传感器标定通过或授予电机控制权限。

### 原实时运行器的无电机回放

`cts_execution_probe` 直接调用与 SDK 共用的落脚公式/记忆/输入校验，使用固定序列与训练 Python 对拍。
`cts_live_replay` 进一步调用实际 `M20PolicyRunner`，只构造地形订阅者，不构造 SDK/FSM/关节发布者。
只允许隔离的 DDS domain 199、loopback；拒绝机器人 DDS profile。示例（在板端原目录内）：

```bash
python3 scripts/replay_cts_live.py --prepare runtime/cts_replay_fixture.json
source scripts/board_env.sh
export M20_ROBOT_CONFIG="$PWD/deploy/robot.json"
export M20_HEIGHTMAP_CONFIG="$PWD/runtime/heightmap_v1.local.json"
export M20_EXPECTED_GEOMETRY_HASH=$(python3 -c 'import os;from m20_heightmap.commission import geometry_hash;print(geometry_hash(os.getcwd(),os.environ["M20_HEIGHTMAP_CONFIG"]))')
export M20_VERIFIED_MODEL_CONTRACT_SHA256=$(sha256sum models/v1/observation_contract.json | cut -d ' ' -f1)
env -u FASTRTPS_DEFAULT_PROFILES_FILE -u FASTDDS_DEFAULT_PROFILES_FILE \
  ROS_DOMAIN_ID=199 ROS_LOCALHOST_ONLY=1 \
  install/m20_student_sdk/lib/m20_student_sdk/cts_live_replay models/v1/policy.onnx \
  < runtime/cts_replay_fixture.json > runtime/cts_replay_result.json
```

在有 Python ORT 的验证环境中用 `scripts/replay_cts_live.py --verify <结果> --model-dir <模型目录>`
核对真实运行器的动作、原始命令、proprio history 和每策略步的四次控制输出。
这些传感器为显式合成回放，结果不证明真实触地有效、执行器已辨识或真机已接管。

后续实机接线仍需提供经过验证的实测接触、运动学与执行器时序，并完成现场验证。
有限仿真继续使用训练工程 `deploy/deploy_mujoco/deploy_cts_student.py`，它已有完整共享控制器。

本轮证据写入 `evidence/cts_20261027/`。它只证明代码和模型接口可运行，不能证明完整楼梯能力。


本次禁止后退测试可在原 AOS 启动入口的环境中设置 `M20_FORWARD_ONLY=1`。它只在 gamepad 输入入口把负向 X 速度归零，前进、侧移、转向仍使用原命令范围；`0` 或不设置保留原行为，其他值拒绝启动。启动日志必须出现 `[COMMAND_SCOPE] forward_only=true`，并核对实际控制进程环境。该开关限制指令方向，不保证机身实际不向后移动，也不构成其他运动工况通过。
