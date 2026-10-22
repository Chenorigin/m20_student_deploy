# 原生 DDS 输入与验证（2026-10-15）

本页说明仍采用的原生 DDS 输入链路及 10 月 15 日历史输入验证。TCP/LZ4 中继和差分/Zstd 候选已废弃。当前控制修复、两次接管结果和未上板的调度修复见 [故障复验](FALL_REAUDIT_20261015.md) 与 [离线修复](TERRAIN_OFFLINE_FIX_20261016.md)。

## 官方依据与实际链路

[官方 DDS 接口总览](https://alidocs.dingtalk.com/i/p/OlnXRl7ed542DGLp/docs/o14dA3GK8g5y4QGySnyegPppV9ekBD76)说明内部中间件是 DrDDS/FastDDS。[激光雷达章节](https://alidocs.dingtalk.com/i/p/OlnXRl7ed542DGLp/docs/20eMKjyp81RbA0YbUe21nQMZWxAZB1Gv)说明，GOS 已有原厂组播接收、解析及融合输出；`/LIDAR/POINTS` 虽标为 `lidar_link`，驱动内部已经转换到 `base_link`。因此本程序保持该点云外参为单位变换，不重复应用前后雷达安装位置。[IMU 章节](https://alidocs.dingtalk.com/i/p/OlnXRl7ed542DGLp/docs/YndMj49yWjPnDl2nIRj1Qyo4J3pmz5aA)说明 `/IMU` 的轴方向也已经由驱动对齐；保留真实物理位置的杆臂补偿，不再旋转一次。

实际采用：原厂雷达组播 → GOS 原厂 `rslidar_network` → **原生 FastDDS UDP 回环订阅** → 私有本机 Unix socket → GOS 本机 ROS 2 PointCloud2 → 独立 LIO/位姿适配/CPU 高程估计 → DDS 小型地形观测回 AOS。AOS 上不再额外读取并转发一份大点云。必须区分：厂商跨板传感器链路是既有组播，本程序原生 DDS 从 GOS 本机原厂驱动取云；这不是新增的“AOS→GOS 原生 DDS 大点云直发”。

公开文档来源哈希记录在 `evidence/native_dds_20261015/official_document_sources.json`。没有执行文档中启用服务、安装软件或修改原厂配置的命令；现有原厂数据链路已经有实际帧。

## 原生接收与身份

新增 `m20_native_inputs` 是独立原生 ABI 的 C++ 包，不加载 ROS Foxy 的另一版 FastDDS。只创建 PointCloud2 DataReader，不创建 DataWriter 或控制通道；仅使用 UDP 回环，禁用本进程内置 SHM。核实唯一 root 原厂 `rslidar_network` 可执行路径、PID/启动时间及 exe inode，周期复查并锁定完整 writer GUID。

实测 GOS 原厂数据 GUID 的 participant 编号为 1；原 AOS 本机接收实现要求 0，两者不能混用。新 GOS 专用判定固定已验证的本机 host/PID/counter 约定；布局变化就拒绝并重新核查，不猜、不自动回退。完整 writer 变化、来源歧义或进程重启会停止原有输入。该身份约定用于本机器人可信板内链路的误接线防护，不是密码学认证。

Unix socket 在 root/0700 私有目录内、权限 0600；服务端检查内核 peer UID，Python 端检查 root peer、实际运行 exe 哈希、PID 和启动时间。native_session/native_sequence 跨本机收件锁定；乱序、重放、原生进程重启或源 GUID 改变后锁停。应用仅保留最新帧，Unix 重连不重放旧候选；无新云时有界等待，不永久占住断开的客户端。

所有原始字段、header、每点绝对时间、点字节及 padding 保持不变。本链路无压缩、无降采样、无传输前裁剪、无伪造平地。原始时间、地图老化、覆盖率与未来时间门槛不变，v0/v1 ABI 与噪声/置信度公式也不变。

## 已执行证据

- 初始原生读取在 10 秒窗口收到约 100 帧，但被旧 AOS 身份约定拒收。随后诊断观察到唯一 GOS 原厂 GUID，host/PID 匹配、counter=1；没有因为收到帧就关闭身份校验。
- 按 GOS 实测身份约定的纯读取探针，约 10 秒收到 **96 帧**，零来源拒收；末点到回调收件点龄平均 **22.26 ms**，范围 **16.60–34.38 ms**。原始整帧跨度约 0.1 秒。见 `evidence/native_dds_20261015/gos_native_probe_freshness_v3.json`。这是原生输入阶段的数据，不是整条高程链路通过的证据；该诊断二进制也不能冒充后续生产构建。
- 当前本机 Python 回归 **69 项通过**，pyflakes 和 shell 语法检查通过，包括源身份、GOS counter、原始 IPC 字节、分片读取、长度读取前拒绝、源 session/序号锁定与恢复不自动解锁。后续独立审查及板端全链路结果另存本目录 evidence；同 family 审查不冒充异构。

## 构建和启动关系

两板同版本构建后，`prepare_board.py` 准备本版本 DDS XML、原生输入说明和预览配置，不创建现场准入凭据。GOS 构建会包含 `m20_native_inputs`，不需要启动 SDK。旧 TCP 桥和 AOS 中继入口已移除。

新 GOS 启动入口是 `sudo -E bash scripts/operator_start_gos.sh v1`。它启动自己的原生读取端、`gos_native_inputs`、独立 LIO、pose adapter 和 mapper；最终布局为CPU4 mapper回调、CPU5 LIO/pose、CPU6原生读取和独立计算worker、CPU7 Python输入。启动前必须读取实时余量及这些核的实际占用，不停止其他人的项目来腾资源。新源发布 `/m20/gos/points`、`/m20/gos/imu`、每帧 `/m20/gos/cloud_receipt` 与 `/m20/gos/native_status`；AOS 只用小回执核实原始云输入，不反向跨板订阅整个点云。

10 月 15 日原生输入历史窗口中 AOS 观测校验为 477/482，随后控制修复版本已实际接管，但两次均因地形超龄锁存 HOLD。不能用早期采集比例验收持续控制。板端最后核实版本、现场结果及最新离线修复以 [故障复验](FALL_REAUDIT_20261015.md)、[操作步骤](REAL_ROBOT_STEPS.md) 和 [离线修复](TERRAIN_OFFLINE_FIX_20261016.md) 为准。
