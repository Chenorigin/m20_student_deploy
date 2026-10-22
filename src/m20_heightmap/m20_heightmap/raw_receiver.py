"""GOS restores unchanged raw scans locally; no command/policy publishers."""
import json
from pathlib import Path
import socket
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from .ros_nodes import sensor_qos
from .runtime import load_config
from .raw_packet import cloud_message
from .raw_transport import Codec, endpoint_from_status, receive_frame


class RawReceiver(Node):
    def __init__(self):
        super().__init__('m20_gos_raw_receiver')
        if Path('/opt/robot/os_type').read_text().strip() != 'CA9B_GOS':
            raise ValueError('raw receiver is GOS only')
        self.cfg = load_config(self.declare_parameter('config','').value)
        self.codec = Codec()
        q = sensor_qos()
        q.reliability = ReliabilityPolicy.RELIABLE
        self.clouds = self.create_publisher(PointCloud2,self.cfg['topics']['cloud'],q)
        self.receipts = self.create_publisher(String,'/m20/gos/cloud_receipt',1)
        self.status = self.create_publisher(String,'/m20/gos/transport_status',1)
        self.create_subscription(String,'/m20/aos/raw_status',self.endpoint,1)
        self.condition = threading.Condition()
        self.address = None
        self.endpoint_seen = 0.
        self.session = None
        self.writer = None
        self.sequence = 0
        self.latest = None
        self.error = None
        self.count = 0
        self.stop = threading.Event()
        self.source_fault = threading.Event()
        self.sock = None
        self.worker = threading.Thread(target=self.reader,daemon=True)
        self.worker.start()
        self.create_timer(.005,self.publish_latest)
        self.create_timer(1.,self.report)

    def endpoint(self,msg):
        try:
            address = endpoint_from_status(json.loads(msg.data))
            publishers = len(self.get_publishers_info_by_topic('/m20/aos/raw_status'))
            if publishers == 0:
                self.error = 'raw endpoint discovery not ready'
                return
            if publishers != 1:
                raise ValueError('ambiguous raw endpoint publishers')
            with self.condition:
                if self.session is not None and address[2] != self.session:
                    raise ValueError('raw source restarted; restart perception after inspection')
                if self.address is not None and self.address != address:
                    raise ValueError('raw endpoint changed within session')
                self.session = address[2]
                self.address = address
                self.endpoint_seen = time.monotonic()
                self.condition.notify_all()
        except (ValueError,KeyError,TypeError) as exc:
            self.latch_source(str(exc))

    def latch_source(self,error):
        self.error = error
        self.source_fault.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def reader(self):
        while not self.stop.is_set():
            if self.source_fault.is_set():
                self.stop.wait(.2)
                continue
            with self.condition:
                self.condition.wait_for(lambda: self.stop.is_set() or (self.address and time.monotonic()-self.endpoint_seen<3.),timeout=.2)
                if self.stop.is_set():
                    return
                if not self.address or time.monotonic()-self.endpoint_seen>=3.:
                    continue
                address = self.address
            try:
                with socket.create_connection(address[:2],timeout=1.) as sock:
                    self.sock = sock
                    sock.settimeout(.5)
                    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                    while not self.stop.is_set() and not self.source_fault.is_set():
                        start = time.perf_counter()
                        (meta,blob),wire_bytes = receive_frame(sock,self.codec)
                        if (meta.get('session') != address[2] or type(meta.get('sequence')) is not int or
                                meta['sequence'] <= self.sequence):
                            self.latch_source('raw session/sequence rejected')
                            raise ValueError('raw session/sequence rejected')
                        writer = (meta['writer_guid'],meta['driver_pid'])
                        if self.writer is not None and self.writer != writer:
                            self.latch_source('raw native writer changed')
                            raise ValueError('raw native writer changed')
                        self.writer = writer
                        self.sequence = meta['sequence']
                        with self.condition:
                            self.latest = (meta,blob,wire_bytes,(time.perf_counter()-start)*1000)
            except (OSError,ValueError,TypeError,KeyError) as exc:
                self.error = str(exc)
                self.stop.wait(.1)
            finally:
                self.sock = None

    def publish_latest(self):
        with self.condition:
            packet,self.latest = self.latest,None
        if packet is None or self.source_fault.is_set():
            return
        if time.monotonic()-self.endpoint_seen >= 3.:
            self.error = 'raw endpoint heartbeat expired'
            return
        publishers = len(self.get_publishers_info_by_topic('/m20/aos/raw_status'))
        if publishers != 1:
            self.error = 'raw endpoint discovery absent/ambiguous'
            if publishers > 1:
                self.latch_source('ambiguous raw endpoint publishers')
            return
        meta,blob,wire_bytes,read_ms = packet
        try:
            now = self.get_clock().now().nanoseconds*1e-9
            msg,end = cloud_message(meta,blob,self.cfg,now)
            self.clouds.publish(msg)
            self.count += 1
            receipt = String()
            receipt.data = json.dumps({'protocol':'tcp_lz4_v1','board':'CA9B_GOS','session':self.session,
                'sequence':meta['sequence'],'stamp':meta['sec']+meta['nanosec']*1e-9,
                'end_stamp':float(end),'frame':meta['frame'],'received_at':now,
                'points':meta['width']*meta['height'],'raw_bytes':len(blob),'wire_bytes':wire_bytes,
                'stream_read_ms':read_ms,'writer_guid':meta['writer_guid'],'driver_pid':meta['driver_pid'],
                'crc_verified':True,'published_local':True,'clouds':self.count,'commands':0})
            self.receipts.publish(receipt)
            self.error = None
        except (ValueError,KeyError,TypeError) as exc:
            self.error = str(exc)

    def report(self):
        msg = String()
        msg.data = json.dumps({'board':'CA9B_GOS','clouds':self.count,'sequence':self.sequence,
            'session':self.session,'writer':self.writer,'last_error':self.error,'commands':0})
        self.status.publish(msg)

    def destroy_node(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.worker.join(timeout=2)
        return super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = RawReceiver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
