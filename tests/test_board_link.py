import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('m20_prepare_board',Path(__file__).resolve().parents[1]/'scripts/prepare_board.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)

def test_board_link_does_not_advertise_handset_or_sensor_networks():
    def iface(name,address):return {'ifname':name,'addr_info':[{'family':'inet','local':address}]}
    interfaces=[iface('eth0','192.0.2.10'),iface('eth1','198.51.100.10'),iface('eth2','203.0.113.10')]
    assert prepare.dds_addresses(interfaces,'CA9B_AOS')==('eth2',['127.0.0.1','203.0.113.10'])
    assert prepare.dds_addresses(interfaces,'CA9B_GOS')==('eth0',['127.0.0.1','192.0.2.10'])
    assert prepare.dds_addresses(interfaces,'CA9B_AOS','eth1')==('eth1',['127.0.0.1','198.51.100.10'])
    with pytest.raises(ValueError):prepare.dds_addresses(interfaces[:2],'CA9B_AOS')

def test_rtps_datagrams_fit_actual_link_mtu():
    assert prepare.dds_payload_limit([{'ifname':'eth2','mtu':1500}],'eth2')==1400
    assert prepare.dds_payload_limit([{'ifname':'eth2','mtu':1280}],'eth2')==1252
    with pytest.raises(ValueError):prepare.dds_payload_limit([{'ifname':'eth2'}],'eth2')
    with pytest.raises(ValueError):prepare.dds_payload_limit([{'ifname':'eth2','mtu':500}],'eth2')

def test_large_local_dds_limit_uses_actual_loopback_mtu():
    assert prepare.dds_local_limit([{'ifname':'lo','mtu':65536}])==64000
    assert prepare.dds_local_limit([{'ifname':'lo','mtu':32000}])==31972
    for interfaces in ([],[{'ifname':'lo'}],[{'ifname':'lo','mtu':True}],[{'ifname':'lo','mtu':1000}]):
        with pytest.raises(ValueError):prepare.dds_local_limit(interfaces)
