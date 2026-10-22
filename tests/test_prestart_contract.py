from types import SimpleNamespace as NS
import numpy as np
import pytest
from m20_heightmap.cloud import validate_cloud_time
from m20_heightmap.ownership import check_command_endpoints


def test_fused_scan_uses_last_acquisition_for_transport_freshness():
    r={'scan_duration_s':.21,'input_timeout_s':.2,'future_tolerance_s':.005}
    oldest,newest=validate_cloud_time(1.,1.224,r,np.array([0.,.1,.2]))
    assert oldest==1. and newest==1.2
    # Transport freshness does not rewrite the oldest measurement time.
    assert 1.224-oldest>.2 and 1.224-newest<.03
    for offsets,now in [(np.array([0.,.22]),1.224),(np.array([-.2,.2]),1.224),
                        (np.array([0.,.2]),1.1),(np.array([0.,.2]),1.401),
                        (np.array([0.,np.nan]),1.224)]:
        with pytest.raises(ValueError):validate_cloud_time(1.,now,r,offsets)
    with pytest.raises(ValueError):validate_cloud_time(1.,1.224,r,None)


def endpoint(pid=2936,name='_CREATED_BY_BARE_DDS_APP_',entity=1):
    gid=b'\x01\x0f\x1a\xeb'+pid.to_bytes(2,'little')+b'\x11\x22'+bytes(4)+entity.to_bytes(4,'big')+bytes(8)
    return NS(endpoint_gid=gid,node_name=name)


def test_vendor_endpoint_exception_is_narrow_and_excludes_only_own_gid():
    data=endpoint(entity=2);command=endpoint();own=endpoint(10101,name='m20_student',entity=3)
    check_command_endpoints([data],[],vendor_pid=2936)
    check_command_endpoints([data],[command],vendor_pid=2936)
    check_command_endpoints([data],[own,command],own_gid=own.endpoint_gid,vendor_pid=2936)
    for ds,cs,kw in [([],[],{}),([data,data],[],{}),([data],[command,command],{}),
                     ([data],[endpoint(2940)],{}),([data],[endpoint(name='named_ros_node')],{}),
                     ([endpoint(name='named_data')],[],{}),([data],[],{'vendor_pid':2940}),
                     ([data],[own,command],{}),([data],[],{'vendor_pid':65536})]:
        with pytest.raises(ValueError):check_command_endpoints(ds,cs,**kw)
