#!/usr/bin/env python3
"""Read-only terrain preview for the operator's static geometry check."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy
from m20_heightmap_msgs.msg import TerrainObservation
from m20_heightmap.runtime import load_config
from m20_heightmap.receiver import receive_terrain

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['v0','v1'],default='v1');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];cfg=load_config(root/'runtime'/('heightmap_'+args.profile+'.local.json'))
    rclpy.init();node=rclpy.create_node('m20_terrain_static_snapshot');records=[]
    def receive(msg):
        try:records.append(receive_terrain(msg,node.get_clock().now().nanoseconds*1e-9,cfg))
        except ValueError as error:node.get_logger().warning(str(error))
    subscription=node.create_subscription(TerrainObservation,cfg['topics']['terrain'],receive,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
    if not records:raise RuntimeError('No usable terrain; do not start SDK control')
    h,m,c=records[-1];known=m==1
    print(json.dumps({'frames':len(records),'valid_fraction':float(known.mean()),'height_known_min_m':float(h[known].min()),'height_known_max_m':float(h[known].max()),'height_known_median_m':float(np.median(h[known])),'confidence_known_median':float(np.median(c[known])),'formula':'base_z - ground_z - 0.5; positive means ground farther below base'},indent=2))
    print('11 Y rows, 17 X columns; Y and X both ascending. ?? means unobserved.')
    for row in range(11):print(' '.join('{:+.2f}'.format(float(h[row*17+i])) if m[row*17+i] else '  ?? ' for i in range(17)))
    node.destroy_subscription(subscription);node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
