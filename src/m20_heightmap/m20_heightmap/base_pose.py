"""Explicit IMU-pose to training-base adapter; retains source capture timestamps."""
import json
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from .geometry import rotation
from .ros_nodes import q_multiply, sensor_qos

class BasePoseAdapter(Node):
    def __init__(self):
        super().__init__('m20_base_pose_adapter')
        path=self.declare_parameter('site_config','').value
        if not path:raise ValueError('site_config path is required')
        self.cfg=json.load(open(path))['pose_adapter']
        if not self.cfg['factory_installation_confirmed']:raise ValueError('sensor installation not confirmed')
        self.Rbi=rotation(self.cfg['imu_to_base_xyzw'])
        self.tib=-self.Rbi.T@np.asarray(self.cfg['imu_position_base_m'])
        self.pose_variance_floor=np.square(self.cfg['pose_std'])
        self.pose_variance_floor_max=float(np.max(self.pose_variance_floor))
        self.lever_variance_scale=(1+float(np.linalg.norm(self.tib)))**2
        self.last_covariance_bytes=None
        self.converted_covariance=None
        self.last_stamp=None
        # Transport the latest pose. The mapper retains capture-time history;
        # processing an old DDS backlog cannot improve current query freshness.
        self.output=self.create_publisher(Odometry,self.cfg['output_topic'],sensor_qos(1))
        self.create_subscription(Odometry,self.cfg['input_topic'],self.receive,sensor_qos(1))

    def receive(self,msg):
        try:
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            elapsed=self.get_clock().now().nanoseconds*1e-9-stamp
            if not np.isfinite(elapsed) or elapsed<-.005 or elapsed>.2:raise ValueError('LIO pose timestamp stale/future')
            if msg.header.frame_id!=self.cfg['input_world_frame'] or msg.child_frame_id!=self.cfg['input_child_frame']:raise ValueError('LIO IMU pose frame differs from declared source contract')
            if self.last_stamp is not None and stamp<=self.last_stamp:raise ValueError('reordered LIO pose')
            q=msg.pose.pose.orientation;p=msg.pose.pose.position
            qi=np.array([q.x,q.y,q.z,q.w]);Rwi=rotation(qi)
            pb=np.array([p.x,p.y,p.z])+Rwi@self.tib
            qb=q_multiply(qi,np.asarray(self.cfg['imu_to_base_xyzw'])*[-1,-1,-1,1])
            if not np.isfinite(pb).all():raise ValueError('nonfinite LIO position')
            out=Odometry()
            out.header.stamp.sec=msg.header.stamp.sec;out.header.stamp.nanosec=msg.header.stamp.nanosec
            out.header.frame_id=self.cfg['output_world_frame'];out.child_frame_id=self.cfg['output_base_frame']
            out.pose.pose.position.x,out.pose.pose.position.y,out.pose.pose.position.z=map(float,pb)
            out.pose.pose.orientation.x,out.pose.pose.orientation.y,out.pose.pose.orientation.z,out.pose.pose.orientation.w=map(float,qb)
            out.twist=msg.twist  # Passed through read-only; never mutate the input.
            covariance=np.asarray(msg.pose.covariance).reshape(6,6)
            covariance_bytes=covariance.tobytes()
            if covariance_bytes!=self.last_covariance_bytes:
                if not np.isfinite(covariance).all() or np.any(np.diag(covariance)<0) or not np.allclose(covariance,covariance.T,atol=1e-6) or np.linalg.eigvalsh(covariance)[0]<-1e-6:raise ValueError('invalid LIO covariance')
                # Preserve norm2 for accepted nearly symmetric matrices too.
                bound=max(float(np.linalg.norm(covariance,ord=2)),self.pose_variance_floor_max)
                # ||J|| <= 1+lever; retain the common spectral covariance bound.
                converted=np.diag(self.pose_variance_floor+bound*self.lever_variance_scale).ravel().tolist()
                self.last_covariance_bytes=covariance_bytes
                self.converted_covariance=converted
            out.pose.covariance=self.converted_covariance
            self.last_stamp=stamp;self.output.publish(out)
        except ValueError as exc:self.get_logger().warning(str(exc))

def main():
    rclpy.init();node=None
    try:node=BasePoseAdapter();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
