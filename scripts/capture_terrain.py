#!/usr/bin/env python3
"""Capture actual terrain for an offline viewer; subscribers only, no actuation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy
from std_msgs.msg import String
from m20_heightmap_msgs.msg import TerrainObservation
from m20_heightmap.runtime import load_config
from m20_heightmap.receiver import receive_terrain


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=('v0','v1'),default='v1');ap.add_argument('--seconds',type=float,default=8);ap.add_argument('--output',required=True);args=ap.parse_args()
    if not 3<=args.seconds<=20:raise ValueError('capture bounded to 3..20 seconds')
    dst=Path(args.output)
    if dst.exists():raise ValueError('preserve existing capture')
    root=Path(__file__).resolve().parents[1];cfg=load_config(root/'runtime'/('heightmap_'+args.profile+'.local.json'))
    rclpy.init();node=rclpy.create_node('m20_terrain_visualization_capture');frames=[];receipts=[];status={};counts={'terrain':0,'accepted':0,'rejected':0};first=time.monotonic();last_saved=[-1.]
    def terrain(msg):
        counts['terrain']+=1;now=node.get_clock().now().nanoseconds*1e-9;error=None;corrected=None
        try:corrected=receive_terrain(msg,now,cfg).tolist();counts['accepted']+=1
        except ValueError as exc:error=str(exc);counts['rejected']+=1
        elapsed=time.monotonic()-first
        if elapsed-last_saved[0]<.09:return
        last_saved[0]=elapsed
        arrays={k:[float(v) for v in getattr(msg,k)] for k in ('height','validity','confidence','variance','age')}
        if any(len(v)!=187 or not np.isfinite(v).all() for v in arrays.values()):return
        frames.append(dict(arrays,elapsed_s=elapsed,stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            receive_stamp=now,frame=msg.header.frame_id,session=int(msg.session_id),sequence=int(msg.sequence),
            profile=msg.profile,motion_compensated=msg.motion_compensated,operator_commissioned=msg.operator_commissioned,
            receive_gate_error=error,receive_adjusted_hmc=corrected))
    def receipt(msg):
        try:
            value=json.loads(msg.data)
            if isinstance(value,dict):receipts.append(dict(value,elapsed_s=time.monotonic()-first))
        except ValueError:pass
    def health(msg,name):
        try:status[name]=json.loads(msg.data)
        except ValueError:status[name]={'parse_error':True}
    q=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    subs=[node.create_subscription(TerrainObservation,cfg['topics']['terrain'],terrain,q),
          node.create_subscription(String,'/m20/gos/cloud_receipt',receipt,q)]
    for name in ('/m20/gos/native_status',cfg['topics']['status']+'/m20_heightmap_gos'):
        subs.append(node.create_subscription(String,name,lambda m,n=name:health(m,n),q))
    while time.monotonic()-first<args.seconds:rclpy.spin_once(node,timeout_sec=.02)
    result={'schema':'m20_actual_terrain_capture_v1','profile':args.profile,'board':Path('/opt/robot/os_type').read_text().strip(),
        'duration_s':time.monotonic()-first,'counts':counts,'frames':frames,'raw_cloud_receipts':receipts,'latest_status':status,
        'source':'real ROS2 TerrainObservation, no synthetic values','model_height_formula':'base_z-ground_z-0.5',
        'grid':{'nx':17,'ny':11,'x_min_m':-.8,'y_min_m':-.5,'step_m':.1},'motor_publishers_created':0}
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    node.destroy_node();rclpy.shutdown()
    print(json.dumps({'output':str(dst),'frames':len(frames),'counts':counts,'motor_publishers_created':0}))
    if not frames:raise SystemExit(1)


if __name__=='__main__':main()
