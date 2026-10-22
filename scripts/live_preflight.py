#!/usr/bin/env python3
"""Bounded read-only live readiness gate; never constructs a motor publisher."""
import argparse
import os
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy
from sensor_msgs.msg import PointCloud2,Imu
from nav_msgs.msg import Odometry
from drdds.msg import JointsDataCmd
from std_msgs.msg import String
from m20_heightmap_msgs.msg import TerrainObservation
from m20_heightmap.runtime import load_config
from m20_heightmap.receiver import receive_terrain
from m20_heightmap.commission import geometry_hash,commission_matches
from m20_heightmap.cloud import read_cloud,validate_cloud_time
from m20_heightmap.ownership import check_command_endpoints
from m20_heightmap.native_packet import validate_receipt

ROOT=Path(__file__).resolve().parents[1]

def validate_gos_status(value, cfg, expected_geometry, now):
    """Attest the live producer identity without asserting terrain quality."""
    if (not isinstance(value,dict) or value.get('node')!='m20_heightmap_gos' or
            value.get('profile')!=cfg['profile'] or
            type(value.get('terrain_schema_version')) is not int or value['terrain_schema_version']!=2 or
            value.get('geometry_hash')!=expected_geometry or value.get('operator_commissioned') is not True or
            value.get('pose_mode')!='odometry' or 'worker_fault' not in value or value['worker_fault'] is not None or
            value.get('mapper_cloud_topic',cfg['topics']['cloud'])!=cfg['runtime']['mapper_cloud_topic']):
        raise ValueError('GOS status identity/configuration rejected')
    stamp=value.get('published_at_s');session=value.get('source_session')
    if type(stamp) not in (int,float) or not np.isfinite([stamp,now]).all():
        raise ValueError('GOS status nonfinite time')
    if not -cfg['runtime']['future_tolerance_s']<=now-stamp<=2.:
        raise ValueError('GOS status stale/future')
    if type(session) is not int or not 0<=session<2**64:
        raise ValueError('GOS status source session rejected')
    return float(stamp),session

def supported_init_quality_warnings(checks, errors, cfg):
    """Only soften quality history for supported SDK initialization, not policy entry."""
    warnings=[]
    for name in ('cloud','mapper_subset_receipt'):
        if name not in checks:continue
        cloud=checks[name]
        if (not cloud['passed'] and cloud['unique_stamps']>=3 and
                cloud['last_receive_age_s'] is not None and
                cloud['last_receive_age_s']<cfg['runtime']['input_timeout_s'] and
                cloud['max_receive_gap_s'] is not None and
                cfg['runtime']['input_timeout_s']<=cloud['max_receive_gap_s']<=cfg['estimator']['max_map_age_s']):
            cloud['strict_passed']=False
            cloud['passed']=True
            warnings.append(name+': recent cloud gap exceeds input timeout; latest verified sample is fresh')
    attested=(cfg['aos'].get('terrain_timing_hold_enabled',True) is False and
              checks.get('gos_status_attestation',{}).get('passed') is True)
    if attested:
        terrain=checks['terrain']
        terrain['strict_passed']=terrain['passed']
        terrain['unknown_input_mode']=not terrain['passed']
        if not terrain['passed']:
            terrain['passed']=True
            warnings.append('Terrain sample/tail/gap quality not ready; verified GOS identity permits SDK unknown terrain input')
    remaining=[]
    for error in errors:
        if error=='Insufficient valid terrain coverage':
            warnings.append(error)
        else:remaining.append(error)
    return sorted(set(warnings)),remaining

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['v0','v1'],default='v1');ap.add_argument('--control',action='store_true');ap.add_argument('--seconds',type=float,default=8)
    ap.add_argument('--supported-init',action='store_true',help='Allow transient quality warnings for supported SDK initialization; policy admission remains enforced by the runner')
    args=ap.parse_args()
    if args.supported_init and not args.control:raise ValueError('--supported-init requires --control')
    if not 3<=args.seconds<=20:raise ValueError('read-only probe is bounded to 3..20 seconds')
    config_path=ROOT/'runtime'/('heightmap_'+args.profile+'.local.json')
    cfg=load_config(config_path)
    attest_unknown=args.supported_init and cfg['aos'].get('terrain_timing_hold_enabled',True) is False
    if args.control:
        commission=json.loads((ROOT/'runtime/commissioned.json').read_text())
        for key in ('factory_geometry_checked','software_limits_checked','operator_controls_start','physical_estop_available'):
            if commission.get(key) is not True:raise ValueError('commissioning check missing: '+key)
        if commission.get('profile')!=args.profile:raise ValueError('commissioning profile mismatch')
        if not commission_matches(ROOT,config_path,args.profile):raise ValueError('commissioning configuration or boot changed; repeat physical checks')
    expected_geometry=geometry_hash(ROOT,config_path)
    rclpy.init();node=rclpy.create_node('m20_student_readonly_preflight')
    records={'cloud':[],'imu':[],'pose':[],'terrain':[],'motor_command':[],'mapper_subset_receipt':[]};errors=[]
    q=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    cloud_q=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE)
    aos=Path('/opt/robot/os_type').read_text().strip()=='CA9B_AOS'
    mapper_topic=cfg['runtime']['mapper_cloud_topic']
    separate_mapper=mapper_topic!=cfg['topics']['cloud']
    cloud_identity={'session':None,'sequence':0}
    status_records=[];terrain_sessions=set()
    status_topic=cfg['topics']['status']+'/m20_heightmap_gos'
    def gos_status(msg):
        try:
            value=json.loads(msg.data)
            stamp,session=validate_gos_status(value,cfg,expected_geometry,node.get_clock().now().nanoseconds*1e-9)
            if status_records and session!=status_records[-1][2]:raise ValueError('GOS status source session changed')
            status_records.append((time.monotonic(),stamp,session))
        except (ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
    def cloud_receipt(msg):
        try:
            value=json.loads(msg.data);when=time.monotonic()
            stamp,_=validate_receipt(value,cfg,node.get_clock().now().nanoseconds*1e-9)
            if cloud_identity['session'] is not None and value['session']!=cloud_identity['session']:raise ValueError('GOS cloud session changed')
            if value['sequence']<=cloud_identity['sequence']:raise ValueError('GOS cloud sequence repeated/regressed')
            cloud_identity.update(session=value['session'],sequence=value['sequence'])
            if separate_mapper:records['mapper_subset_receipt'].append((when,stamp))
            if aos:records['cloud'].append((when,stamp))
        except (ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
    def callback(kind,msg):
        when=time.monotonic();stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if kind=='motor_command':records[kind].append((when,stamp));return
        age=node.get_clock().now().nanoseconds*1e-9-stamp
        try:
            if kind=='cloud':
                r=cfg['runtime']
                _,offsets=read_cloud(msg,r['max_cloud_points'],r['max_cloud_bytes'],r['point_time_field'],r['point_time_scale_s'],r['point_time_mode'])
                _,end=validate_cloud_time(stamp,node.get_clock().now().nanoseconds*1e-9,r,offsets)
                age=node.get_clock().now().nanoseconds*1e-9-end
            age_limit=cfg['runtime']['cloud_input_timeout_s'] if kind=='cloud' else cfg['runtime']['input_timeout_s']
            if not np.isfinite(age) or age< -cfg['runtime']['future_tolerance_s'] or age>age_limit:raise ValueError(kind+' stamp stale/future')
            if kind=='cloud' and msg.header.frame_id not in cfg['lidar_extrinsics']:raise ValueError('cloud frame')
            if kind=='pose' and (msg.header.frame_id!=cfg['frames']['world'] or msg.child_frame_id!=cfg['frames']['base']):raise ValueError('base pose frame')
            if kind=='terrain':
                if not msg.motion_compensated:raise ValueError('terrain is not odometry-compensated')
                if args.control and (not msg.operator_commissioned or msg.geometry_hash!=expected_geometry):raise ValueError('GOS geometry or commissioning differs from AOS')
                receive_terrain(msg,node.get_clock().now().nanoseconds*1e-9,cfg)
                if attest_unknown:terrain_sessions.add(msg.session_id)
            records[kind].append((when,stamp))
        except ValueError as exc:errors.append(str(exc))
    subs=[]
    for kind,typ,topic in [('cloud',PointCloud2,mapper_topic),('imu',Imu,cfg['topics']['imu']),('pose',Odometry,cfg['topics']['odom']),('terrain',TerrainObservation,cfg['topics']['terrain'])]:
        if kind=='cloud' and aos:
            subs.append(node.create_subscription(String,'/m20/gos/cloud_receipt',cloud_receipt,cloud_q))
            continue
        subs.append(node.create_subscription(typ,topic,lambda m,k=kind:callback(k,m),cloud_q if kind=='cloud' else q))
    if args.control:subs.append(node.create_subscription(JointsDataCmd,'/JOINTS_CMD',lambda m:callback('motor_command',m),q))
    if separate_mapper and not aos:
        subs.append(node.create_subscription(String,'/m20/gos/cloud_receipt',cloud_receipt,cloud_q))
    if attest_unknown:subs.append(node.create_subscription(String,status_topic,gos_status,cloud_q))
    start=time.monotonic()
    while time.monotonic()-start<args.seconds:rclpy.spin_once(node,timeout_sec=.02)
    now=time.monotonic();checks={};passed=True
    for kind in ('cloud','imu','pose','terrain')+(('mapper_subset_receipt',) if separate_mapper else ()):
        values=records[kind];unique=len(set(v[1] for v in values));tail=now-values[-1][0] if values else None
        required=3 if kind in ('cloud','mapper_subset_receipt') else 10
        max_gap=max(np.diff([v[0] for v in values])) if len(values)>1 else None
        ok=bool(unique>=required and tail is not None and tail<cfg['runtime']['input_timeout_s'] and max_gap is not None and max_gap<cfg['runtime']['input_timeout_s'])
        checks[kind]={'samples':len(values),'unique_stamps':unique,'last_receive_age_s':tail,'max_receive_gap_s':max_gap,'passed':ok};passed &= ok
    for topic in (cfg['topics']['cloud'],cfg['topics']['terrain']):
        count=len(node.get_publishers_info_by_topic(topic));checks[topic]={'publishers':count,'passed':count==1};passed &= count==1
    if aos:
        count=len(node.get_publishers_info_by_topic('/m20/gos/cloud_receipt'))
        checks['/m20/gos/cloud_receipt']={'publishers':count,'passed':count==1};passed &= count==1
    if separate_mapper:
        endpoints=[node.get_publishers_info_by_topic(topic) for topic in
                   (cfg['topics']['cloud'],mapper_topic,'/m20/gos/cloud_receipt')]
        matched=False
        if all(len(group)==1 for group in endpoints):
            full,front,receipt=(group[0] for group in endpoints)
            prefixes=[bytes(endpoint.endpoint_gid)[:12] for endpoint in (full,front,receipt)]
            matched=(all(endpoint.node_name=='m20_gos_native_inputs' for endpoint in (full,front,receipt)) and
                     full.node_namespace==front.node_namespace==receipt.node_namespace and
                     full.topic_type==front.topic_type=='sensor_msgs/msg/PointCloud2' and
                     receipt.topic_type=='std_msgs/msg/String' and
                     all(len(prefix)==12 and any(prefix) for prefix in prefixes) and
                     prefixes[0]==prefixes[1]==prefixes[2])
        ok=bool(matched and cloud_identity['session'] is not None)
        checks['mapper_cloud_source']={'topic':mapper_topic,'publishers':len(endpoints[1]),
            'same_native_participant':matched,'validated_subset_receipt':cloud_identity['session'] is not None,'passed':ok}
        passed &= ok
    if attest_unknown:
        status_publishers=node.get_publishers_info_by_topic(status_topic)
        terrain_publishers=node.get_publishers_info_by_topic(cfg['topics']['terrain'])
        endpoint_match=False
        if len(status_publishers)==len(terrain_publishers)==1:
            status_endpoint,terrain_endpoint=status_publishers[0],terrain_publishers[0]
            status_prefix,terrain_prefix=(bytes(endpoint.endpoint_gid)[:12] for endpoint in (status_endpoint,terrain_endpoint))
            endpoint_match=(status_endpoint.node_name==terrain_endpoint.node_name=='m20_heightmap_gos' and
                            status_endpoint.node_namespace==terrain_endpoint.node_namespace and
                            status_endpoint.topic_type=='std_msgs/msg/String' and
                            terrain_endpoint.topic_type=='m20_heightmap_msgs/msg/TerrainObservation' and
                            len(status_prefix)==len(terrain_prefix)==12 and any(status_prefix) and status_prefix==terrain_prefix)
        latest=status_records[-1] if status_records else None
        age=None if latest is None else node.get_clock().now().nanoseconds*1e-9-latest[1]
        fresh=bool(age is not None and np.isfinite(age) and -cfg['runtime']['future_tolerance_s']<=age<=2.)
        session_match=bool(latest is not None and (not terrain_sessions or terrain_sessions=={latest[2]}))
        checks['gos_status_attestation']={'samples':len(status_records),'last_publish_age_s':age,
            'status_publishers':len(status_publishers),'terrain_publishers':len(terrain_publishers),
            'same_gos_participant':endpoint_match,'source_session':None if latest is None else latest[2],
            'terrain_session_matches':session_match,'passed':bool(fresh and endpoint_match and session_match)}
        passed &= checks['gos_status_attestation']['passed']
    if args.control:
        commands=node.get_publishers_info_by_topic('/JOINTS_CMD')
        try:
            vendor_pid=int(os.environ['M20_VERIFIED_VENDOR_PID'])
            identity=check_command_endpoints(node.get_publishers_info_by_topic('/JOINTS_DATA'),commands,vendor_pid=vendor_pid)
            quiet=not records['motor_command']
        except ValueError as exc:
            identity={'error':str(exc)};quiet=False
        checks['existing_motor_stream']={'messages':len(records['motor_command']),'publishers':len(commands),'identity':identity,'passed':quiet};passed &= quiet
    warnings=[]
    if args.supported_init:
        warnings,errors=supported_init_quality_warnings(checks,errors,cfg)
        passed=all(check['passed'] for check in checks.values())
    passed &= not errors
    result={'passed':bool(passed),'profile':args.profile,'checks':checks,'errors':sorted(set(errors)),
            'supported_initialization_only':args.supported_init,'quality_warnings':warnings,
            'policy_readiness_not_asserted':args.supported_init,
            'unknown_input_mode':bool(args.supported_init and checks['terrain'].get('unknown_input_mode',False)),
            'cloud_evidence':('GOS per-frame verified native DDS source/subset receipt' if aos else 'local PointCloud2 '+mapper_topic),'motor_publishers_created':0}
    print(json.dumps(result,indent=2));node.destroy_node();rclpy.shutdown()
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
