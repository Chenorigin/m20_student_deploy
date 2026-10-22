# M20Pro 实机部署与操作步骤

> **2026-10-17：student-v3 10500与MuJoCo限制已部署，本次楼梯试验结束，SDK及本轮GOS感知已停止。** 用户报告15cm高/30cm深木楼梯五六次成功一次，失败为撞阶后才抬腿。完整trace已封存；前进约78%输入地形全未知、位姿反复清图，干净图单次仿真中学生的预抬时机也有疑点。见[楼梯复盘](../evidence/student_v3_20261017/stairs_diagnosis/diagnosis.md)及[本轮记录](../evidence/student_v3_20261017/summary.json)。排查未改生产参数；原ONNX数值失败、原门槛及用户启动例外均保留。真实手柄由现场操作，DDS返回不证明电机执行；全局日志有101条显式丢弃，正常策略/HOLD周期关联完整。下方10月16日为历史。

> **2026-10-16轮速调整已上机试验，SDK已停止：** 新狗两板入口为`20261016_trial_d136c514135c`。关闭两项地形HOLD后的实机运行没有HOLD，推摇杆后仍倒地；日志显示姿态超限后进入阻尼。用户批准的轮目标速度上限调整已写本地及两板当前配置并重绑凭据，已完成短时零速运行，控制周期HOLD提前结束本次试验，轮速行走效果未验证。状态凭证更新已在两板构建通过，最终状态见[迁移与启动记录](../evidence/migration_20261016/summary.json)。

本轮显式关闭覆盖率及地形时效HOLD，超龄或接收超时地形降为未知，单格继续老化清零；未来/非有限时间、契约、本体状态和所有权保护保留，配置见[唯一台账](DEPLOYMENT_PARAMETERS.md)。继续使用原v1模型，v0与14800离线配置未改。旧狗[历史联调验证](../evidence/admission_online_20261016/sdk_admission_trial_actual.json)不验证本次开关或新狗控制。

统一入口为`/home/user/m20_heightmap/current`。两项地形HOLD关闭后已作实机运行，SDK随后停止；本次轮速调整已作短时零速试验，控制周期HOLD提前结束，行走效果未验证；两板Python包已构建通过，SDK已停止。以下是后续启动的依赖与入口说明，最终状态以[本轮证据](../evidence/migration_20261016/summary.json)为准，不另起重复实例，也不借用旧狗凭据。

用户最新确认“现在SDK已开启，趴稳／支撑和急停均就绪”；此前关于SDK模式的确认已被用户更正，不能沿用为旧运行的控制权证据。原厂安装沿用共同几何候选，未独立重新测量外参或验证固件硬限位；运动按键和摇杆由现场用户操作。手柄重连后须重新使能。R2后的阻尼状态结束前，L1不会排队，先从日志确认Idle再操作。R1正常趴下；R2不能当作站立保持。

实现与历史依据：[故障复查](FALL_REAUDIT_20261015.md)、[原生DDS](NATIVE_DDS_20261015.md)、[修复与模型记录](TERRAIN_OFFLINE_FIX_20261016.md)。旧`STEP6_STATUS.md`及原狗回执不代表当前新狗状态。

下面使用 `v1`。v0/v1 的模型、契约、profile 与历史长度必须一起切换；本次实机验证为 v1。原厂服务保留，程序不自动停止它们。

## 1. 登录、版本与时钟

通过本机网卡直接连接新狗：AOS为`10.21.31.103`，GOS为`10.21.31.104`。当前不经过旧狗的`nvidia-jump`或`user-via-nvidia`入口；地址、板卡角色与运行身份以[本轮记录](../evidence/migration_20261016/summary.json)为准。

两板分别检查：

```bash
cd /home/user/m20_heightmap/current
readlink -f /home/user/m20_heightmap/current
cat evidence/board_build_receipt.json
cat evidence/deployment_receipt.json
```

构建阶段会准备本版本 DDS XML 和两版预览配置。DDS 仅广告已验证的板间网卡当前地址及回环，默认 AOS eth2、GOS eth0；手柄与传感器子网不会作为 GOS 的替代 locator。原始大点云由 GOS 原生 DDS 从本机原厂驱动读取；GOS 本机 ROS 2 大点云按实测回环 MTU 设片段上限，并配置短周期可靠重传，避免将本机消息切成过多小片而延迟 LIO。跨板 DDS 只保留 IMU、位姿、地形与状态，小型地形消息仍可能发生 IP 分片，须按真实接收与时效核验。网卡接线改变时先核对链路，可用 --interface 显式指定，缺失地址/MTU会拒绝。若需根据当前网卡地址重新生成 DDS XML：

```bash
python3 scripts/prepare_board.py
```

该命令只准备配置，不启动应用。使用本版本 `scripts/board_env.sh`，不要 source 会改写原厂 DDS XML 的脚本。AOS 可纯离线重查模型：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/offline_board_models.py
```

GOS 只读检查授时：

```bash
bash scripts/operator_clock_gos.sh inspect
```

本轮新狗IMU、位姿和地形的当前消息时间检查已通过，不等于已经获得长期授时稳定性证据。旧狗10月15日的FAULTY修复和临时授时override仅作历史，不能自动继承到新狗。`operator_clock_gos.sh apply`只针对已证实的TX timestamp timeout / FAULTY，不用于盲目处理LISTENING或改民用日期。

## 2. 预览配置与现场支持

机器人静止、稳定支撑，有人看守且能操作实体急停。你已确认雷达和 IMU 为原厂安装，配置据此提供候选外参与自体过滤；这不能替代静态几何核验。

两板的 `runtime/heightmap_v0.local.json` / `heightmap_v1.local.json` 若已准备，可直接用于感知预览。`preview_preparation.json` 明确标为 `preview_only`，**没有创建 `commissioned.json`，不会放行运控**。若预览配置缺失，可执行：

```bash
python3 scripts/prepare_board.py --prepare-preview
```

它不覆盖已有现场配置。不要手工伪造现场确认凭据。

## 3. 在 GOS 启动纯感知

```bash
cd /home/user/m20_heightmap/current
sudo -E bash scripts/operator_start_gos.sh v1
```

GOS已按本次轮速配置重启感知，初始化状态凭证更新已构建通过且实际预检通过；先核对本轮回执、current与实际进程，不另起重复实例。AOS无原始点云中继；GOS使用下方原生输入入口。

原厂雷达组播由现有 `rslidar_network` 在 GOS 融合、完成静态变换并发布 `/LIDAR/POINTS`。原生读取端使用独立进程和 `/usr/local/lib` 的原厂 ABI，通过 UDP 回环订阅该话题；核实唯一原厂 root 进程、PID/启动时间/可执行 inode 和完整 DDS writer GUID，按本机实测 participant=1 的 ABI 约定校验。它不创建 DDS 发布者或电机接口，也不修改原厂雷达配置或组播服务。这里的跨板传感器数据使用厂商现成组播链路，不能称作新增的 AOS→GOS DDS 大点云直发。

原始 metadata、header、所有点和每点时间经私有本机 Unix socket 传给 GOS 输入节点，保持字节不变。本机 socket 目录 root/0700、socket 0600，双方检查内核 peer credentials；Python 还核对实际 peer 的可执行文件哈希、PID 与启动时间。输入节点在 GOS 本机发布 `/m20/gos/points`、转发已对齐的 `/IMU` 到 `/m20/gos/imu`，并发布每帧源身份/新鲜度验证回执 `/m20/gos/cloud_receipt`；AOS 只取小回执，避免跨板拉取整云。

监督器同时启动本版本独立 Super-LIO、IMU 到训练 base 的位姿 adapter 和 CPU mapper。日志为 `runtime/native_cloud.log`、`native_inputs.log`、`lio.log`、`pose.log`、`heightmap.log`；Ctrl-C 只清理本监督器创建的子进程组及私有 socket。新输入/C++构建不启动 SDK；SDK 初始化仍由第 6 步操作者确认后执行。

## 4. 只读验收感知与静态几何

AOS：

```bash
cd /home/user/m20_heightmap/current
source scripts/board_env.sh
python3 scripts/live_preflight.py --profile v1
python3 scripts/terrain_snapshot.py --profile v1
```

检查连续真实点云、IMU、训练 base 位姿和地形消息；时间、来源、有效覆盖都需通过。AOS 预检以 GOS 实际完成解码、校验并发布后的逐帧收件凭据检查点云，避免再将大点云拉回 AOS；GOS 本机预检直接读取完整 PointCloud2。实际验收结果以本轮报告及对应证据为准。在线延迟、CPU 共载、LIO 位姿质量与覆盖必须按真实帧验证，不能只看 ROS 图或服务状态。

现场核对：11×17 顺序、前后/左右方向、未知格 `??`、离地高度与台阶高度。模型高度是 `base_z−ground_z−0.5`，升高的地面使高度通道下降；不要强行把平地填零。遮挡或过期区域应为未知，不能显示成可信平地。实际噪声和位姿不确定性尚未完成独立真值标定。

未通过时保留日志、只读快照、`/m20/gos/native_status`和高度图状态。当前参数及候选性质见[台账](DEPLOYMENT_PARAMETERS.md)。本轮初始化入口明确记录点云历史间隔与覆盖告警；这不能当作策略质量验收。

## 5. 确认真实手柄与 SDK 交接

输入使用原厂手柄 / App，经被动 AF_PACKET 接收；原厂 `basic_server` 继续独占 UDP/30000。没有改为 SSH 键盘，也不需要启动历史转发或 ECAN 桥。手机 App 只有在真实协议/按键验证后才能视为可用入口，不能因手柄没电就默认 App 已通过。

按厂商流程由你切 SDK 模式，确认原厂运动指令持续静默；保留 `rl_deploy` 的原厂低层数据和 `basic_server`。本版只允许一个可验证的原厂数据参与者及其最多一个静默命令端点，不以“端点暂时没消息”自动授予控制权。任何其它端点或实际外来关节命令都会阻断。

确认安装几何、软件保护范围、支撑与急停后，在两板分别执行：

```bash
python3 scripts/operator_commission.py --profile v1
```

只有实际核验后才输入 `CONFIRM_PHYSICAL_CHECKS`。凭据绑定当前 boot、profile、配置、机器人参数、安装参数与 LIO 配置。修改配置须重启整个感知并重新检查；不能重新签收后让旧进程冒充加载了新几何。

## 6. SDK 启动入口与当前操作

AOS在板端tmux内运行，避免SSH断开结束控制进程。状态凭证已在两板构建通过并通过实际预检；本次SDK短时运行因控制周期HOLD结束，现场确认趴下后已停止SDK；下面保留默认strict入口，不表示现在启动：

```bash
# AOS 上创建或进入一个本机 tmux 终端
 tmux new -s m20_student
cd /home/user/m20_heightmap/current
sudo -E bash scripts/operator_start_aos.sh v1
```

本轮经用户授权使用显式初始化模式：

```bash
sudo -E bash scripts/operator_start_aos.sh v1 supported-init
```

默认strict启动入口保留，supported-init本身仍只处理限定的初始化前告警，不限制L2。本轮另以两个独立配置开关关闭覆盖率与地形时效HOLD；Python接收及SDK使用全未知表示旧地形，SDK没有新鲜地形时也按未知继续推理。格点老化仍清零，未来/非有限时间、契约、几何/commissioning、本体和所有权保护保留；其他原因仍能触发HOLD。输入全未知不是策略质量通过，配置见[台账](DEPLOYMENT_PARAMETERS.md)。

显式supported-init且关闭地形时效HOLD时，预检须取得新鲜GOS状态凭证，核对配置、schema、commissioning、会话和worker状态，并确认状态与地形各有唯一发布者、来自同一GOS participant，才可将地形样本、尾龄或间隔不足列为未知输入告警。没有地形样本时，端点类型及schema声明不能证明数组内容正确；已收到消息的ABI、未来时间、几何等错误仍硬拒。点云、IMU、位姿、所有权和发布者检查不因此放宽。

入口先做真实手柄心跳被动观察、原厂进程/端口身份检查和只读感知/关节命令检查。手柄未连接、没有真实心跳时应拒绝，不能合成心跳绕过。本轮被动手柄、唯一原厂静默端点及IMU/pose/terrain检查已通过，点云历史间隔告警保留在[本轮证据](../evidence/migration_20261016/summary.json)。

全部通过后输入 `START_STUDENT`，表示你确认当前 SDK 模式、稳定支撑和急停。**这一步会启动 SDK 并发零偏初始化命令，不是仅加载权重等待。** 程序内部仍会在初始化前检查状态和命令来源。

此前未开启原厂SDK模式的推理及HOLD日志只证明软件执行。重新确认SDK模式后曾有覆盖HOLD与前栽反馈；随后关闭两项地形HOLD的实机运行持续推理且没有HOLD，推摇杆后仍倒地，日志明确记录姿态超限后切入阻尼再返回Idle。SDK已停止，现场确认稳住且无损伤，最初失稳原因尚未确定。用户批准仅放宽轮目标速度上限，已写配置并重绑凭据；腿部等其他限制保留。本次短时零速试验因控制周期HOLD提前结束，现场报告未能持续接管、已趴下；SDK已停止，轮速行走效果未验证。参数与复算边界见[台账](DEPLOYMENT_PARAMETERS.md)，最终状态见[本轮记录](../evidence/migration_20261016/summary.json)。

后续须结合日志、目标和反馈确认站立，不能把初始化通过当作接管通过。过期键不会重放；摇杆必须持续更新，心跳不会延续旧速度。R2是关节阻尼，不能当作站立保持或实体急停。

## 7. 停止与异常

在稳定支撑下按厂商流程退出 SDK 并恢复原厂控制。正常 SDK Stop 会发阻尼；感知终端 Ctrl-C 只停止本版进程组。

若收到外来关节命令或启动时的完整 DDS 端点集合变化，学生输出会锁定关闭，并把本体状态报告为不可用；它不会自动停厂商、重抢控制或自行恢复。**在所有权丢失时，为避免竞争，自身阻尼写入也被抑制；不要期待该 SDK 的 R2/Stop 替代实体急停或原厂接管。** 实际低层最后目标保持/失联保护尚未现场验收，需现场操作人员处理。

保留旧版本只用于审计与软件排障；旧包的指针回退不等于已经恢复可用运控。机器重启后重新核验授时、来源和现场凭据。
