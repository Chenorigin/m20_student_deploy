# 模型导出与部署参数

## 2026-10-17 实机手柄命令范围更新

按用户指定配置，`deploy/robot.json::command_operator_ranges` 更新为前后 ±1.2 m/s、横向 ±0.7 m/s、转向 ±1.0 rad/s。新范围在原训练输入契约内；模型、训练范围、关节与轮速限制、增益和 HOLD 开关均未变。手柄报文 X/Y/Yaw 按物理单位独立裁剪，不额外乘比例；输入范围内的数值保持原值。MuJoCo 文件未修改。

真实生产 `GamepadInterface` 构造器/解析器在隔离本机探针中通过 17 项检查，旧配置对新上限预期的正负两项失败，独立复跑得到同样结果；两次均无网络调用。证据见 [本轮记录](../evidence/gamepad_speed_20261017_174804/)。本机检查不代表提高速度后的实机运动稳定性。

两板旧配置和 GOS 日志备份位于 `/home/user/m20_heightmap/gamepad_speed_20261017_174804_before`。新 robot SHA 为 `733017b328358c9c70fda8748907e00077f5944e5ced825bb35e6946a531ec30`，内容绑定 geometry 为 `0f2d987e88b1659a2c159543e4a8c32e63387f702f4bfd2b4c3b70a647454a23`。手柄范围仅在 SDK 构造时读取；两板均须更新配置并重新 commission、重启 GOS 和 SDK，实际完成状态见本轮 `handoff.json`。

## 2026-10-17 HOLD 复核：仅恢复来源会话保护

当前 `deploy/robot.json` 的 `policy.terrain_source_session_hold_enabled=true` 已写入新狗 AOS/GOS；覆盖率、地形时效开关继续为 false，其他参数和模型未变。自研 SDK 保持停止，本项由下一次正常启动的 runner 读取；此次没有电机初始化或 L1/L2 动作。GOS 已按新内容身份重新启动，两板 commission 匹配，AOS 只读控制预检通过。

判断依据为 [本轮证据](../evidence/hold_reenable_20261017_165827/recorded_gate_analysis.json)：R5 移动完整可见片段 5787 周期仍有 44 个覆盖条件越线、8 个时效条件越线；最近站姿 2765 周期另有 24 个覆盖条件越线。两窗共 8552 个已记录周期均未出现会话切换。越线数是各开关独立判断的条件次数，开启后首次会锁存，不能称多次独立 HOLD；日志丢失和切片边界限制保留，不能推导未来必不触发。

当前真实 runner 与 v3 ONNX 的隔离 x86 夹具 20 项及独立复跑均通过：同源递增消息继续推理；开关 true 时 A→B、新会话过期导致清帧仍锁存，false 时继续；好帧不自动解锁，有效重新交接后恢复。该夹具不创建电机发布者或 SDK 硬件接口，不替代 ARM 闭环实测。运行中 GOS 重启或位姿 epoch 重置导致换源时，本项仍会按设计 HOLD。

robot SHA 为 `0ef323e1e9915b53c50256f8c745d5998daae00b6ef794f1f2d1be500f182901`，新内容绑定 geometry 为 `90543ff954c523ea721181870b58089a107638f37f586f585e1118e85427f813`；runtime 高程配置字节未变。两板改前配置及 GOS 日志保存在 `/home/user/m20_heightmap/hold_reenable_20261017_165827_before`。这次恢复不修复过温、摇杆零输入或位姿跳变的根因；不得因此直接恢复行走。

## 2026-10-17 R5 前向扇形输入（已部署，已完成一轮移动核验）

权威运行配置为两板当前 release 的 `runtime/heightmap_v1.local.json`，快照见 `evidence/terrain_dynamic_fix_20261017/candidate_r5/`。前向扇形使用已有 `site.json` 的唯一前向雷达位置、yaw 与 135° 水平视场，经完整 LiDAR→body 外参计算；区域内每条原始点记录、所有字段及时间不变，没有 stride、ring 抽稀或点平均。保留全扫描真实 max-time 原始点作为时间端点；它可在扇形外，不代表该点会被拟合为前方地面。LIO 的完整扫描发布在该裁剪前，原始 DDS 源仍提供完整融合消息，不能将此描述为厂商雷达接口按区域订阅。

| 项目 | 当前 R5 | 改前 / 边界 |
|---|---|---|
| 云输入独立时效 `runtime.cloud_input_timeout_s` | 0.30s | 原共用 0.20s；单独提高云处理预算 |
| 本体/位姿 `runtime.input_timeout_s` | 0.20s | 不变 |
| 地形格龄 `estimator.max_map_age_s` | 0.50s | 不变，不延长旧地形寿命 |
| 发布前 query 位姿年龄 | 0.08s | 不变 |
| SDK 对齐年龄 | 0.12s | 不变 |
| 自研进程 CPU 亲和 | native C++ 2、pose 3、mapper 4、LIO 5、fit worker 6、native Python 7 | 原厂进程亲和未改；实际逐线程收据在本轮证据中 |
| query 来源诊断 | 本次启动环境 `M20_TERRAIN_QUERY_TRACE=1` | 默认关闭；实际日志记录每个最终有效格所选 scan/cell/最老及最新支持时间 |

原来的几何支持、点数、XY 宽度、方差、未来时间与来源检查不变。模型、MuJoCo 对齐限制、控制器与 LIO V6 二进制不变。候选包 SHA 为 `e963b7dea6e23ad3315203d16c2509b7eb6f37eeea04587e12bd147d43c035d2`。19 对实机保真核对与 325 帧 ARM 回放分开报告；随后已完成新实机移动采集，实际前进输入未出现整图未知。阶高提示短缺、日志边界及未验收提前抬腿的范围见本轮 report，不把任何一次 valid 当作全部地形正确。

## 2026-10-17 student-v3 与日志候选

锁定checkpoint为昨晚v3运行的`model_10500.pt`，任务来源为`Rough-Deeprobotics-M20-Student-v3`，保存格式与部署profile仍为v1，不能将其误认为原v1权重。哈希与完整证据见[本轮记录](../evidence/student_v3_20261017/summary.json)。TorchScript对拍通过；ONNX在本机和AOS独立CPU探针均有3/512元素超过原`atol=rtol=2e-5`，最大绝对差`0.00017547607421875`。用户最初要求保留门槛、仅准备日志；本轮随后明确撤销数值失败的启动阻拦。原图及诚实的`passed=false`元数据已安装两板`models/v1`，旧模型为`models/v1_before_student_v3_10500`；原失败收据、归档marker、用户例外与逐项安装回执保留。阈值及失败结论没有更改。

用户明确要求对齐限制，并裁定“以 MuJoCo 部署为准，列出训练差异”。权威为`deploy_mujoco.py::M20Contract.apply_pd`及审计中锁定的`M20.xml`。全16关节表、轴/局部旋转、逐项源哈希见[四方审计](../evidence/student_v3_20261017/limits_training_mujoco_audit.json)，实机改前/改后见[对照表](../evidence/student_v3_20261017/limits_before_after.json)。这是控制目标口径对齐；本轮已做真实策略运行与下发采集，总PD力矩/机械坐标仍未实证等价。

| 当前选用项目 | 本轮配置 / 结果 | v3训练差异 |
|---|---|---|
| 归一化动作 | 保留±100 | 相同 |
| 腿位置目标 | `q_default + scale*clip(raw)`；`joint_target_position_clip_enabled=false`，同时旁路processed与额外物理盒裁剪 | 训练仍有processed±100；对目前腿动作尺度不触发 |
| 轮速度目标 | `5*clip(raw)`；`wheel_target_velocity_clip_enabled=false`，同时旁路processed与额外±50裁剪；数学范围±500rad/s | 训练processed目标±100；USD solver约79.3是另一层物理速度属性 |
| 腿目标slew | false，无额外变化率限制 | 相同 |
| 手动命令 | 配置`command_operator_ranges`，后续用户指定更新见本页手柄命令范围更新；真实gamepad handler读取 | 训练采样范围±2/±1/±2仍作为策略输入契约保留 |
| 腿standing/反馈范围 | 按MuJoCo XML逐关节名写入；反馈超界原拒绝逻辑保留 | USD的hipx、hipy、knee部分范围/方向不同，见审计；不将URDF当作训练USD |
| PD增益 | 腿80/2，轮0/0.6 | 两边相同 |
| 总PD力矩 | MuJoCo为腿±76.4、轮±21.6Nm；实机SDK发kp/q/kd/dq，尚无总限矩等价证据 | 不把tau_ff=0或元数据当作限矩执行 |
| 其他运行保护 | 姿态、时效、所有权、急停路径与原数值门槛保留 | 仿真终止定义、延迟与viewer过渡不等同这些实机保护 |

新开关缺省true，老配置继续原目标裁剪；非布尔值拒绝。旧`software_velocity_limits_sdk_rad_s`及processed盒仍留作旧模式回退参数，开关false时不参与当前目标生成。HOLD继续保持最后一次已发腿目标，该目标现在可能超出XML物理范围；HOLD并不将其重置到合法关节位置。本轮根据用户当前现场确认重新运行交互commissioning，没有续用旧boot凭据。

日志候选参数由`student_trace.hpp`及启动wrapper提供，默认如下：

| 项 | 值 / 语义 |
|---|---|
| 异步队列 | 最多512事件；锁竞争、队列满、构造失败、写失败均显式计入dropped |
| 每次文件上限 | 512MiB；`M20_TRACE_MAX_BYTES`可覆盖，封顶标为记录失败，不自动改变运控 |
| 磁盘保留 | 1GiB；启动要求保留量加文件预算均可用，运行中由写盘线程检查 |
| 完整策略输入 | wrapper显式设置`M20_TRACE_FULL_OBSERVATION=1`；每次推理另存完整输入与原始输出 |
| 理论时间 | 首个策略调用为episode起点，按模型策略周期和调用序号递推；标记`nominal_from_first_policy_tick`，不是已测得的FSM硬件deadline |
| 实际时间 | Unix UTC纳秒与本机boot内monotonic纳秒；分别记录接收、推理起止、DDS发布前后 |
| 异步输入 | steering/terrain没有已知调度deadline，理论调度字段为null并注明原因；地形query源时间独立保留，不伪造为发布时刻 |
| 完整性验收 | `.health.json`必须final=true、dropped=0、attempted=written且无error；失败尾行另标partial_tail |
| 理论关联 | episode、policy_tick连接同周期观测、每次推理及最终writer；DDS返回不代表电机执行确认 |

本轮真实运行将文件上限覆盖为1GiB，磁盘保留2GiB；第一轮final health为attempted38931/written38867/dropped64，不能称完整无缺失。第二轮楼梯试验已结束，SDK及本轮感知已停止；final health为attempted133995/written133894/dropped101，final=true、writer_stopped=true、无写盘错误。7484正常策略周期与541个HOLD均唯一关联输入/推理/最终发送，但不能称整场无损。楼梯预瞄缺失与学生预抬能力的证据及边界见[楼梯复盘](../evidence/student_v3_20261017/stairs_diagnosis/diagnosis.md)，此次排查未改生产参数。理论时间仍是episode内名义周期，异步输入及非策略命令为null；工作站wall clock与两板约有264.1秒偏差（SSH中点测量不确定度约0.53秒），不得直接混排跨机wall时间。原始ns及host/boot标识全部保留。GOS一次PTP读数为SLAVE、offsetFromMaster460ns，但不等于已建立UTC可追溯性。见`live_deployment_attempt/clock_offset_probe.json`及`gos_ptp_inspect.log`。合成数据性能仅用于预算；策略实际控制与前进行为须依据真实命令、反馈及现场观察验收。

以下为10月16日历史状态及参数。

> **本轮状态：** 新狗两板`20261016_trial_d136c514135c`关闭地形HOLD后已作实机试验，没有HOLD但仍倒地；姿态超限触发阻尼。用户批准仅将轮目标速度上限调至50rad/s，已写本地及两板当前`deploy/robot.json`并重绑commissioning，新一轮已执行130次零速推理，随后控制周期HOLD；轮速行走效果未验证。状态凭证相关Python包两板构建和实际预检均通过，SDK已停止。继续使用原v1的789维ONNX，v0与14800离线配置未改。证据见[迁移与启动记录](../evidence/migration_20261016/summary.json)。

新狗通过本机网卡直连AOS `10.21.31.103`、GOS `10.21.31.104`，此前已完成ARM/Foxy迁移构建。旧运行中的覆盖与`no fresh terrain`HOLD因当时未开原厂SDK模式，只作软件证据。确认实际SDK模式后曾在约0.22s、11次零命令推理后触发覆盖HOLD并收到前栽反馈，二者因果未证实。随后两个开关均false的真实试验持续391个策略周期、约7.8s，全部实际推理、0次HOLD，267帧有效格为0而整帧均新鲜；用户推摇杆后仍倒地。日志先记录姿态`30.4/-14.1/-18.5°`，横滚超过SDK的30°保护线，随后才由`rl_control`进入阻尼，不能说地形HOLD导致本次倒地。最初失稳原因仍待确定；SDK已正常停止，现场确认稳住且未损伤。原始记录见[SDK日志](../evidence/migration_20261016/aos_terrain_hold_off_sdk.log)与[控制trace](../evidence/migration_20261016/aos_terrain_hold_off_control_trace.jsonl)。

用户先要求禁用覆盖HOLD，随后要求不因地形有效期或整帧时效进入HOLD。两个独立布尔开关已在两板安装配置中读回false：覆盖不足不再拒绝，整帧超龄或SDK接收超时降为全未知，单格仍按真实年龄清零，不伪造有效地面。未来/非有限时间、契约、数组范围、本体及所有权保护保留；缺字段默认true，非布尔值拒绝。上一轮AOS构建exit0、耗时约82s，GOS Python构建exit0、耗时约1.64s；本次状态凭证Python包两板构建和实际预检已通过，SDK执行短时零速推理后因控制周期HOLD结束，现场确认趴下后已停止SDK。

用户最新确认原厂SDK已开启、稳定支撑和实体急停就绪；旧运行不能沿用此前已被更正的SDK模式确认。原厂安装沿用共同几何候选，未独立重新测量外参、标定噪声或验证固件硬限位。commissioning和初始化通过不等于这些测量或真实控制交接已经完成。

初始读取用户指定的 `/home/ubuntu/桌面/student_v0.pt` 与 `student_v1.pt`，使用 `torch.load(..., map_location="cpu", weights_only=True)`。只将 `student_state_dict` 导出为确定性 actor，教师、优化器和回放数据不进入部署模型。

## 生成的文件

| 文件 | 内容 |
|---|---|
| `models/v0/policy.pt`、`models/v1/policy.pt` | 学生 TorchScript，仅 actor |
| `models/v0/policy.onnx`、`models/v1/policy.onnx` | 静态 batch=1、输入 `obs`、输出 `actions` 的 CPU ONNX |
| 各模型目录的 `observation_contract.json` | 从对应 checkpoint 原样提取的观测契约 |
| 各模型目录的 `deployment.json` | checkpoint/导出模型 SHA256、训练来源、TorchScript 对拍记录 |
| 各模型目录的 `policy.onnx.json` | ONNX 形状、SHA256、CPU provider 与对拍误差 |
| `deploy/robot.json` | 关节、动作、SDK 参数数值台账与逐文件来源 SHA256 |
| `deploy/heightmap_v0.json`、`deploy/heightmap_v1.json` | 已配置相对模型路径的感知模板；现场标定仍为 false |

实际 v0 为 618→512→256→128→16，实际 v1 为 789→512→256→128→16，都是 ELU 且无 running observation normalization。v1 history_steps=4，历史排列与前述契约一致。两份学生指向同一教师，已核对教师 checkpoint SHA256。

保存时 v0 perception.level=0.25，v1=1.0；具体训练噪声保存在各模型的 provenance 中。实机使用真实测量，不重复注入这些噪声或人为添加一步动作延迟。

## v1 准入与发布预算（10 月 16 日在线修订）

| 配置项 | 两板安装配置已读回 | 含义 |
|---|---:|---|
| `aos.coverage_hold_enabled` | false | 仅禁用覆盖率HOLD；缺字段默认为true，必须是布尔值 |
| `aos.terrain_timing_hold_enabled` | false | 地形接收超时/整帧超龄降为全未知，不再因这些门槛HOLD；缺字段默认true |
| `aos.min_valid_fraction` | 保留0.05（5%） | 当前开关关闭时不参与拒绝；重新开启后至少10/187格 |
| `aos.max_alignment_age_s` | 保留0.12s（120ms） | 当前决定旧地形何时降为未知，不因超过该值HOLD；推理后地形超龄HOLD也随开关关闭 |
| `runtime.max_publish_query_age_s` | 0.03s（30ms） | GOS调用发布前的查询位姿年龄预算 |
| `estimator.max_map_age_s` | 原v1为0.5s | 单格真实观测寿命；与查询位姿时效分开检查 |
| `deploy/robot.json`轮目标速度上限 | 50rad/s（此前10rad/s） | 用户明确批准的wheel-only实机联调值；本地及两板配置已生效；短时零速运行，行走效果未验证 |
| 同文件腿关节目标变化速度限制 | 保留2.5rad/s | 腿部限制、位置范围及其他参数不随轮速调整 |
| supported-init GOS状态凭证 | 至少1条，发布年龄不超过2s | 仅时效HOLD关闭时使用；未来容差沿用`runtime.future_tolerance_s`，状态与地形发布者须各唯一且同participant |

失稳前355帧包含1420个轮目标请求，实测请求绝对值峰值43.2285547rad/s；按`ceil(43.2285547*1.1/5)*5=50rad/s`取上限。对同一批固定请求复算，旧10rad/s上限截断129个，新50rad/s上限截断0个。这仅证明该窗口请求不再被原轮速上限截断，不是闭环重放，不能证明放宽后动作、反馈或稳定性会保持，也不是硬件或固件限位验证。腿关节限制及其他参数保留，commissioning已按新内容重新绑定；不能宣称轮速调整已解决倒地。

迁移前的旧狗百帧及边界验证是在地形保护开启时执行，见[历史候选回放](../evidence/admission_online_20261016/sdk_admission_trial_replay.json)和[实际SDK runner](../evidence/admission_online_20261016/sdk_admission_trial_actual.json)，不验证本次开关后的控制行为。当前两个false时，全未知、单格过期或整帧超龄不会因覆盖/时效触发HOLD，但契约、非法时间、本体或所有权故障仍会保护。若仅关闭时效而重新开启覆盖，全未知仍可能触发覆盖HOLD；两个开关相互独立。

本轮先检查覆盖开关关闭时零有效格可接收、开启时拒绝，以及当时时效开关默认开启时过期仍拒绝，并核对缺字段和错误类型；后续仅补双关闭时过期变未知、非有限时间仍拒绝。另维护两处已有配置断言并限定运行三个参数例通过，没有全套复跑或新动作。这些检查不替代SDK、ARM闭环或栽倒问题验收，证据见[本轮记录](../evidence/migration_20261016/summary.json)。

本轮修改正式原v1的`deploy/heightmap_v1.json`、Python接收/配置读取及学生SDK地形判断，v0及14800离线配置未调整。新增字段改变内容绑定哈希，不能沿用前轮凭据；板端参数、构建和commissioning状态须以本次实际回执为准。未声明开关的旧配置默认保留对应HOLD，未声明发布预算则沿用消费时限。

默认strict启动入口保留，supported-init只影响初始化前限定告警。显式supported-init且`terrain_timing_hold_enabled=false`时，新增可靠GOS状态凭证：核对node、profile、schema版本、内容绑定geometry、commissioned、odometry、无worker故障、有限新鲜时间及uint64会话；状态与地形端点须各唯一、节点均为`m20_heightmap_gos`且GID前缀属于同一participant，已收到地形时还核对会话。凭证通过才可将地形样本数、尾龄或间隔质量不足转为明确的未知输入告警，记录`strict_passed`和`unknown_input_mode`；无地形样本只能核对端点类型和schema声明，不能验数组。已收到消息的ABI、未来时间、几何等错误仍硬拒，点云、IMU、位姿、所有权及发布者检查不豁免；未获凭证或时效HOLD开启时保留原检查。两个地形开关不是“禁用所有HOLD”，不宣称策略质量通过。

原离线预算取保存窗口中两板接收年龄差的最大值上取整、SDK 允许的最长周期及实测推理上取整：`60 − 11 − 25 − 1 = 23ms`，再按 5ms 向下取为 20ms，剩余 3ms 给 DDS 调用、其他处理和波动。该推导仅适用于原离线候选；在线修订后不能继续宣称保有同样余量。两板观察差包含时钟与观察器调度，**不是已证明的网络尾延迟上界**；剩余余量也未经过板端验证。

训练的`max_delay_s`指雷达包采集到到达的延迟，随后每个策略步仍用当前位姿查询历史地图，不能为同样长的查询位姿延迟提供训练背书。现有日志只有训练覆盖率均值，没有单环境单帧的可靠覆盖下限。原离线复算保留旧消费门槛，后续按用户要求放宽并已应用到新狗；这些参数仍是工程联调值，不能称为模型最优值或安全极限。历史推导与范围见[预算记录](../evidence/terrain_fix_20261016/admission_budget.json)。

本次轮速候选实机记录：130次零速度推理，跨度约2.58s；第2.62s一次控制间隔39.834ms触发`policy tick discontinuity`，原代码要求20ms±5ms。49帧使用新鲜地形、5帧整帧过期降未知、76帧接收超时降未知，这些地形情况未触发HOLD。轮目标峰值6.833rad/s，尚未超过原上限，不能据此评价轮速放宽效果；策略运行期间最大绝对roll/pitch约0.86°/0.83°。现场报告“未切入策略，已趴下”，SDK已停止，原厂服务和GOS感知保留。[实机回执](../evidence/migration_20261016/aos_wheel50_trial_result.json)保存了完整策略段及HOLD首尾节选来源；未改控制周期保护，也未继续新一轮动作。

## 参数依据与冲突处理

远端 `https://arcusgit.vip.cpolar.cn/Dawning/m20_pro_perceptive_lidar` 当前要求认证，现有凭据未能读取。已找到同一 origin 的本机检出 `/home/ubuntu/桌面/m20_pro_perceptive_lidar`，HEAD 为 `76cf139ab67cc0bb9ba40990d5a4c0e48ca42de5`。该检出的训练目录有其他未提交修改，本轮只读使用部署目录的指定 SDK 文件，SHA256 已记录；**未核对远端最新版本**。

训练参数取 checkpoint 固定教师对应的 `params/env.yaml`；参考仓库用于 SDK 硬件顺序、方向、初始零偏、消息接口，并交叉验证动作比例和 PD。

**参考 runner 的默认髋/膝角与这两份学生的训练姿态不同。** 已采用学生训练默认姿态；旧 SDK 姿态保存在 `reference_legacy_default_position_policy_rad` 作为审计依据，`reference_default_is_used=false`。准确数值只维护在 `deploy/robot.json`。

关节处理分为策略顺序（12 腿关节后接 4 轮）、SDK 顺序（每腿 hipx/hipy/knee/wheel）与原始电机坐标三层。`robot_config.py` 提供转换，测试用独立按名字构造的映射验证。

**SDK 的 `GetJointPosition/Velocity` 和 `SetJointCommand` 已做方向/零偏转换，使用这些 API 时不能再套 raw 转换。** 直接读写原始 drdds 消息时才使用 raw 公式。参考 `ResetPositionOffset` 可按实际状态将髋/膝零偏加减整圈；配置保存初始值，运行时必须使用实际有效值，helper 接受 `effective_offset_deg` 覆盖。

动作按训练顺序执行 `clip(raw_action) → scale/offset → clip(processed_target)`，再重排为 SDK 的 `[kp,position,kd,velocity,torque_ff]`。轮动作进入速度列、轮子 kp 为零；腿动作进入位置列。**训练的宽 action clip 不是电机硬限位**，真实控制器仍需官方限位和控制权检查。

初次导出阶段只生成参数与纯CPU变换函数，没有启动SDK。当前新狗已经执行真实初始化；SDK的`Start()`会发初始化命令，`Stop()`会发阻尼，不能当作无动作的只读探针。

## 路径迁移和使用

导出模型不依赖原 checkpoint 路径；`deployment.json` 中桌面路径仅作来源记录。部署 JSON 的模型和契约路径相对于**配置文件所在目录**解析；整体拷贝工作区到板端，无需保留桌面目录。

当前实机部署模板及SDK启动入口使用`models/v1/policy.onnx`，配套契约为同目录`observation_contract.json`；通用开发模板中的TorchScript路径不代表现场入口。当前学生SDK runner直接读取地形与SDK本体快照，在进程内拼接789维并推理；Python `aos_adapter`是可选外接接口。**改变模型路径不会让旧63D runner自动接受新契约**，必须使用对应学生runner和契约。

```bash
# 在有 Torch 的训练 Python 环境重新导出。路径都可指定，无 GPU/Isaac。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/export_checkpoint.py \
  --checkpoint /新的路径/model.pt --profile v1 --output /新的版本目录/v1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/export_onnx.py \
  --model /新的版本目录/v1/policy.pt \
  --contract /新的版本目录/v1/observation_contract.json \
  --profile v1 --output /新的版本目录/v1/policy.onnx
```

输出要求新目录/新文件，保留旧版。ONNX 导出额外需要 `onnx` 和 CPU `onnxruntime`，不是 GOS 点云核心的依赖。

## 尚需确认的现场项

旧狗三板核查、故障复验与楼梯采样仅作历史。新狗两板已经构建并启动，话题、当前消息时间、原厂端点及SDK初始化有[本轮运行证据](../evidence/migration_20261016/summary.json)。下面保留实际验收边界，不把初始化或有限窗口预检当作长期实时性、独立标定或闭环成功。

| 未确定项 | 缺口 |
|---|---|
| 雷达→训练 base、IMU→base 外参与自体区域 | 用户确认原厂安装，沿用共同候选；未对新狗独立重新测量外参 |
| 位姿来源和 frame | 已使用独立LIO及显式base位姿adapter，有限窗口消息检查通过；动态位姿精度未独立验收 |
| 实际话题、ROS域、时钟 | 新狗已按当前网卡生成DDS并收到原生输入小回执、IMU/pose/terrain；长期授时稳定性尚未证明 |
| 逐点时间与 header | 原厂融合云经GOS本机原生链保留逐点时间；已有输入检查不代替运动中的去畸变验收 |
| 测距与位姿不确定性 | 当前工程估计，不能以训练噪声代替真机标定 |
| SDK零偏、限位、接管/退出 | 最新双关闭试验无HOLD但仍倒地，姿态超限后进入阻尼；轮速调整已应用，零速短试验因控制周期HOLD结束，行走效果未验证；SDK已停止 |
| 目标板资源和依赖 | 两板ARM/Foxy构建及实际启动已有证据；有限窗口含质量告警，持续共载实时性未通过完整验收 |

部署模板继续保持`calibration_confirmed=false`；新狗runtime依据本轮现场确认另行准备并绑定凭据，不能将其解释为已完成独立外参或硬限位测量。

## 初次导出验证（历史）

- TorchScript 保存后重新加载，各 128 组 CPU 输入对拍原学生 actor，最大绝对差 0。
- ONNX 静态输入/输出形状正确，各 32 组 CPU 输入与 TorchScript 在 `atol=rtol=2e−5` 下对拍通过；最大绝对误差见 `policy.onnx.json`，不是位级一致。
- 32 项 CPU 回归通过，新增 SDK/策略重排、raw 坐标、零动作姿态、动作缩放、处理后截断与相对模型路径检查。
- 初次导出时未执行跨板通信、SDK接管与行走；新狗现已有确认SDK模式后的实际推理与失稳记录，不能沿用更早未开SDK模式的日志作为物理交接证据，连续行走尚未通过。

独立同 family 复核已实际重新推导参数、检查学生权重来源并复跑两份 ONNX 的 CPU 对拍，报告见 [deployment_independent_review.json](../evidence/deployment_independent_review.json)。异构 family 复核仍未完成。
