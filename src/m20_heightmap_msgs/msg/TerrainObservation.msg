# header.stamp: query pose time in synchronized ROS clock, NOT publication time.
# frame_id: gravity aligned, origin at base, X/Y following robot yaw.
std_msgs/Header header
uint32 schema_version
string profile
uint64 session_id
uint64 sequence
bool motion_compensated
string geometry_hash
bool operator_commissioned
# Channel-major order: 11 ascending Y rows x 17 ascending X columns.
float32[187] height
float32[187] validity
float32[187] confidence
# Diagnostics for aging after transport; these do not enter the model.
float32[187] variance
float32[187] age
