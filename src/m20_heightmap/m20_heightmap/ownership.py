"""Narrow endpoint check for the vendor's silent SDK-mode participant."""
def check_command_endpoints(data, commands, own_gid=None, vendor_pid=None):
    if len(data)!=1:
        raise ValueError('Exactly one JOINTS_DATA publisher is required')
    prefix=bytes(data[0].endpoint_gid)[:12]
    if data[0].node_name!='_CREATED_BY_BARE_DDS_APP_':
        raise ValueError('JOINTS_DATA is not a native vendor endpoint')
    if len(prefix)!=12 or not any(prefix):
        raise ValueError('Missing JOINTS_DATA participant identity')
    if vendor_pid is not None and (not 0<vendor_pid<65536 or prefix[4:6]!=vendor_pid.to_bytes(2,'little')):
        raise ValueError('JOINTS_DATA participant does not match verified vendor process')
    foreign=[p for p in commands if own_gid is None or bytes(p.endpoint_gid)!=bytes(own_gid)]
    if len(foreign)>1:
        raise ValueError('Multiple external JOINTS_CMD endpoints')
    if foreign:
        p=foreign[0]
        if p.node_name!='_CREATED_BY_BARE_DDS_APP_' or bytes(p.endpoint_gid)[:12]!=prefix:
            raise ValueError('External command endpoint is not the official data participant')
    return {'external_endpoints':len(foreign),'official_data_prefix':prefix.hex()}
