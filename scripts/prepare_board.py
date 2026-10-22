#!/usr/bin/env python3
"""Prepare release-local DDS and site configuration. Never starts applications."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as E

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/m20_heightmap'))
from m20_heightmap.native_packet import PROTOCOL, SOURCE

def dds_addresses(interfaces, board, requested=None):
    # These are the factory board-to-board links observed on this M20Pro.
    # Other AOS networks serve the handset/sensors and must not be advertised
    # as alternative GOS locators. Resolve current IPs, never store device IPs.
    selected=requested or {'CA9B_AOS':'eth2','CA9B_GOS':'eth0'}[board]
    addresses=['127.0.0.1']
    for interface in interfaces:
        if interface['ifname']==selected:
            addresses.extend(a['local'] for a in interface.get('addr_info',[]) if a['family']=='inet' and a['local']!='127.0.0.1')
    if len(addresses)<2:raise ValueError('no IPv4 address on selected board link: '+selected)
    return selected,addresses

def dds_payload_limit(interfaces, selected):
    mtu=next((i.get('mtu') for i in interfaces if i['ifname']==selected),None)
    if type(mtu) is not int or mtu<600:raise ValueError('missing/invalid MTU for '+selected)
    # Physical-link reference; the raw TCP stack uses the actual link MTU.
    # DDS large samples now stay local, so its datagram limit is separate below.
    return min(1400,mtu-20-8)

def dds_local_limit(interfaces):
    mtu=next((i.get('mtu') for i in interfaces if i['ifname']=='lo'),None)
    if type(mtu) is not int or mtu<1500:raise ValueError('missing/invalid loopback MTU')
    # Large raw DDS samples now stay on GOS. A small physical-link ceiling
    # fragmented local samples into thousands of packets and delayed LIO init.
    return min(64000,mtu-28)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-preview',action='store_true');ap.add_argument('--interface',help='board-to-board interface; factory defaults: AOS eth2, GOS eth0');args=ap.parse_args()
    board=Path('/opt/robot/os_type').read_text().strip()
    if board not in ('CA9B_AOS','CA9B_GOS'):raise ValueError('unsupported board')
    data=json.loads(subprocess.check_output(['ip','-j','-4','addr','show'],text=True))
    selected,addresses=dds_addresses(data,board,args.interface)
    physical_payload_limit=dds_payload_limit(data,selected)
    payload_limit=dds_local_limit(data)
    ns='http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles';E.register_namespace('',ns)
    tag=lambda s:'{'+ns+'}'+s
    root=E.Element(tag('profiles'));descriptors=E.SubElement(root,tag('transport_descriptors'));transport=E.SubElement(descriptors,tag('transport_descriptor'))
    E.SubElement(transport,tag('transport_id')).text='m20_release_udp';E.SubElement(transport,tag('type')).text='UDPv4'
    whitelist=E.SubElement(transport,tag('interfaceWhiteList'))
    for address in addresses:E.SubElement(whitelist,tag('address')).text=address
    for name,value in [('sendBufferSize','8388608'),('receiveBufferSize','8388608'),('maxMessageSize',str(payload_limit))]:E.SubElement(transport,tag(name)).text=value
    participant=E.SubElement(root,tag('participant'),{'profile_name':'m20_release','is_default_profile':'true'});rtps=E.SubElement(participant,tag('rtps'))
    user=E.SubElement(rtps,tag('userTransports'));E.SubElement(user,tag('transport_id')).text='m20_release_udp';E.SubElement(rtps,tag('useBuiltinTransports')).text='false'
    publisher=E.SubElement(root,tag('publisher'),{'profile_name':'m20_local_delivery','is_default_profile':'true'})
    times=E.SubElement(publisher,tag('times'))
    for name,nsvalue in [('initialHeartbeatDelay',1000000),('heartbeatPeriod',20000000),('nackResponseDelay',1000000),('nackSupressionDuration',0)]:
        item=E.SubElement(times,tag(name));E.SubElement(item,tag('sec')).text='0';E.SubElement(item,tag('nanosec')).text=str(nsvalue)
    E.SubElement(publisher,tag('historyMemoryPolicy')).text='PREALLOCATED_WITH_REALLOC'
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    E.ElementTree(root).write(str(runtime/'dds_udp.xml'),encoding='UTF-8',xml_declaration=True)
    # Resolve at installation time. No fixed device address or global port.
    (runtime/'native_input.json').write_text(json.dumps({'protocol':PROTOCOL,'source':SOURCE,
        'cloud_path':'factory multicast -> GOS rslidar_network -> native DDS loopback -> local PointCloud2',
        'board_link_interface':selected,'raw_pointcloud_relay_on_AOS':False},indent=2)+'\n')
    previews=[]
    if args.prepare_preview:
        for profile in ('v0','v1'):
            path=runtime/('heightmap_'+profile+'.local.json')
            if path.is_symlink():raise ValueError('preserve unexpected preview symlink')
            if not path.exists():
                cfg=json.loads((ROOT/'deploy'/('heightmap_'+profile+'.json')).read_text())
                cfg['calibration_confirmed']=True
                path.write_text(json.dumps(cfg,indent=2)+'\n')
            previews.append(str(path))
        (runtime/'preview_preparation.json').write_text(json.dumps({'preview_only':True,'physical_checks_performed':False,'commissioning_receipt_created':False,'application_started':False},indent=2))
    print(json.dumps({'board':board,'dds_interface':selected,'dds_interfaces':addresses,'max_udp_payload_bytes':payload_limit,'physical_mtu_payload_bytes':physical_payload_limit,'large_cloud_transport':PROTOCOL,'runtime':str(runtime),'preview_configs':previews,'application_started':False,'commissioning_receipt_created':False}))

if __name__=='__main__':main()
