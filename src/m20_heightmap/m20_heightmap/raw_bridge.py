"""AOS native selected raw stream to one lossless TCP stream; no motor interface."""
import json
from pathlib import Path
import socket
import struct
import threading
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from .ros_nodes import sensor_qos
from .runtime import load_config
from .raw_packet import cloud_message
from .raw_transport import Codec, LatestServer, exact


class RawBridge(Node):
    def __init__(self):
        super().__init__('m20_aos_raw_bridge')
        if Path('/opt/robot/os_type').read_text().strip() != 'CA9B_AOS':
            raise ValueError('AOS native input only')
        config_path = Path(self.declare_parameter('config','').value)
        self.cfg = load_config(config_path)
        settings = json.loads((config_path.parent/'raw_transport.json').read_text())
        self.codec = Codec()
        self.server = LatestServer(settings['address'])
        self.session = uuid.uuid4().hex
        self.path = self.declare_parameter('socket_path','').value
        # Small IMU and endpoint/status messages still use ROS2. PointCloud2 is
        # published only on GOS, so it cannot fan out across the USB link.
        imu_qos = sensor_qos(4)
        imu_qos.reliability = ReliabilityPolicy.RELIABLE
        self.imus = self.create_publisher(Imu,'/m20/aos/imu',imu_qos)
        self.status_pub = self.create_publisher(String,'/m20/aos/raw_status',1)
        self.create_subscription(Imu,'/IMU',self.imu,sensor_qos(8))
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.count = 0
        self.imu_count = 0
        self.compress_ms = None
        self.wire_bytes = None
        self.stop = threading.Event()
        self.sock = None
        self.thread = threading.Thread(target=self.reader,daemon=True)
        self.thread.start()
        self.create_timer(.005,self.publish_latest)
        self.create_timer(1.,self.report)

    def reader(self):
        while not self.stop.is_set():
            try:
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
                    self.sock = sock
                    sock.settimeout(1.)
                    sock.connect(self.path)
                    while not self.stop.is_set():
                        size = struct.unpack('<I',exact(sock,4))[0]
                        if not 1 <= size <= 4096:
                            raise ValueError('native header budget')
                        meta = json.loads(exact(sock,size))
                        amount = meta['bytes']
                        if type(amount) is not int or not 0 < amount <= 16000000:
                            raise ValueError('native payload budget')
                        blob = exact(sock,amount)
                        with self.lock:
                            self.latest = (meta,blob)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                self.error = str(exc)
                self.stop.wait(.1)
            finally:
                self.sock = None

    def publish_latest(self):
        with self.lock:
            packet,self.latest = self.latest,None
        if packet is None:
            return
        meta,blob = packet
        try:
            cloud_message(meta,blob,self.cfg,self.get_clock().now().nanoseconds*1e-9)
            meta = dict(meta,session=self.session,sequence=self.count+1)
            start = time.perf_counter()
            frame = self.codec.encode(meta,blob)
            self.compress_ms = (time.perf_counter()-start)*1000
            self.wire_bytes = len(frame)
            self.server.submit(frame)
            self.count += 1
            self.error = None
        except (ValueError,TypeError,KeyError) as exc:
            self.error = str(exc)

    def imu(self,msg):
        if msg.header.frame_id not in ('','imu_link'):
            self.error = 'unknown IMU frame'
            return
        msg.header.frame_id = 'imu_link'
        self.imus.publish(msg)
        self.imu_count += 1

    def report(self):
        msg = String()
        msg.data = json.dumps({'board':'CA9B_AOS','source':'selected_local_rslidar_udp',
            'transport':'tcp_lz4_v1','address':self.server.address,'port':self.server.port,
            'session':self.session,'clouds':self.count,'imus':self.imu_count,
            'compress_ms':self.compress_ms,'wire_bytes':self.wire_bytes,
            'sent':self.server.sent,'last_error':self.error,'network_error':self.server.error,'commands':0})
        self.status_pub.publish(msg)

    def destroy_node(self):
        self.stop.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=2)
        self.server.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = RawBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
