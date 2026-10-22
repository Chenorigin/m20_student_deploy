std_msgs/Header header
# New session_id on controller restart, policy entry, or episode reset.
uint64 session_id
uint64 step
# Already scaled current 57D in training order. At 50 Hz, never LiDAR rate.
float32[57] values
