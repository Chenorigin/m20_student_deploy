"""Reconstruct the unchanged native PointCloud2 and check acquisition freshness."""
import array
import numpy as np
from .cloud import read_cloud, validate_cloud_time
from .raw_transport import validate_meta


def cloud_message(meta, blob, cfg, now, expected_source='AOS_local_native_UDP', return_points=False):
    from sensor_msgs.msg import PointCloud2, PointField
    validate_meta(meta, len(blob), expected_source)
    if meta['frame'] not in cfg['lidar_extrinsics']:
        raise ValueError('native cloud frame rejected')
    msg = PointCloud2()
    msg.header.stamp.sec = meta['sec']
    msg.header.stamp.nanosec = meta['nanosec']
    msg.header.frame_id = meta['frame']
    for field in ('width','height','point_step','row_step'):
        setattr(msg, field, meta[field])
    msg.is_bigendian = meta['bigendian']
    msg.is_dense = meta['dense']
    msg.fields = [PointField(name=f['name'],offset=f['offset'],datatype=f['datatype'],count=f['count']) for f in meta['fields']]
    msg.data = array.array('B', blob)
    r = cfg['runtime']
    points, offsets = read_cloud(msg,r['max_cloud_points'],r['max_cloud_bytes'],r['point_time_field'],r['point_time_scale_s'],r['point_time_mode'])
    _, end = validate_cloud_time(meta['sec']+meta['nanosec']*1e-9, now, r, offsets)
    if return_points:
        return msg, end, points, offsets
    return msg, end


def subset_cloud(msg, keep):
    """Copy selected complete point records, preserving acquisition metadata.

    Row padding is skipped; padding inside each point and every field's bytes
    are retained unchanged. The derived cloud is a single unorganized row.
    """
    from sensor_msgs.msg import PointCloud2

    keep = np.asarray(keep)
    count = msg.width*msg.height
    if keep.dtype != np.bool_ or keep.shape != (count,):
        raise ValueError('Point subset mask must match the original cloud')
    if (msg.width < 0 or msg.height <= 0 or msg.point_step <= 0 or
            msg.row_step < msg.width*msg.point_step or len(msg.data) != msg.height*msg.row_step):
        raise ValueError('Malformed cloud strides/buffer for point subset')
    indices = np.flatnonzero(keep)
    payload = b''
    if len(indices):
        records = np.ndarray((msg.height, msg.width), dtype=np.dtype('V'+str(msg.point_step)),
                             buffer=msg.data, strides=(msg.row_step, msg.point_step))
        payload = records[indices//msg.width, indices%msg.width].tobytes()
    result = PointCloud2()
    result.header = msg.header
    result.fields = msg.fields
    result.height = 1
    result.width = len(indices)
    result.point_step = msg.point_step
    result.row_step = result.width*result.point_step
    result.is_bigendian = msg.is_bigendian
    result.is_dense = msg.is_dense
    result.data = array.array('B', payload)
    return result
