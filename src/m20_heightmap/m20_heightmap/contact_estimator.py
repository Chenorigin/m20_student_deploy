"""Read-only inverse dynamics diagnostics, never a contact validity authority.

For each leg, tau_model = tau_motor + J_support.T @ force_world.
Torque bias, transmission/friction, geometry and impact bandwidth need measured
validation before these estimates can become CTS contact observations.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

LEGS = ('fl', 'fr', 'hl', 'hr')
SUFFIXES = ('hipx', 'hipy', 'knee', 'wheel')
ORDER = tuple(f'{leg}_{joint}_joint' for leg in LEGS for joint in SUFFIXES)


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError('Invalid finite vector: ' + name)
    return result


def rotation(axis, angle):
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)


def rpy_rotation(angles):
    r, p, y = angles
    return rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)


def urdf_geometry(path):
    """Extract original inertials/joints/cylinders without copying visual meshes."""
    raw = Path(path).read_bytes()
    if len(raw) > 2*1024**2 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Oversized or entity-bearing robot description')
    root = ET.fromstring(raw)
    joints = {j.attrib['name']: j for j in root.findall('joint')}
    links = {link.attrib['name']: link for link in root.findall('link')}
    if set(joints) != set(ORDER) or len(joints) != 16 or len(links) != 17:
        raise ValueError('Expected the original 16-joint M20 tree')
    def origin(element):
        item = element.find('origin')
        return {key: [float(x) for x in (item.get(key, '0 0 0') if item is not None else '0 0 0').split()]
                for key in ('xyz', 'rpy')}
    document = {'schema': 'm20_contact_geometry.v1', 'urdf_sha256': hashlib.sha256(raw).hexdigest(),
                'joint_order': list(ORDER), 'legs': [], 'hardware_geometry_verified': False}
    for leg in LEGS:
        chain = []
        for suffix in SUFFIXES:
            name = f'{leg}_{suffix}_joint'
            joint = joints[name]
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent != ('base_link' if not chain else chain[-1]['child']):
                raise ValueError('Unexpected joint chain: ' + name)
            if joint.get('type') not in ('continuous', 'revolute'):
                raise ValueError('Expected revolute joint: ' + name)
            inertial = links[child].find('inertial')
            i = inertial.find('inertia').attrib
            inertia = [[float(i['ixx']), float(i['ixy']), float(i['ixz'])],
                       [float(i['ixy']), float(i['iyy']), float(i['iyz'])],
                       [float(i['ixz']), float(i['iyz']), float(i['izz'])]]
            chain.append(dict(name=name, parent=parent, child=child, origin=origin(joint),
                              axis=[float(x) for x in joint.find('axis').get('xyz').split()],
                              mass=float(inertial.find('mass').get('value')),
                              com=origin(inertial), inertia=inertia))
        collisions = links[f'{leg}_wheel'].findall('collision')
        if len(collisions) != 1 or collisions[0].find('geometry/cylinder') is None:
            raise ValueError('Expected exactly one cylinder per wheel')
        cylinder = collisions[0].find('geometry/cylinder')
        document['legs'].append(dict(joints=chain, cylinder=dict(origin=origin(collisions[0]),
                                       radius=float(cylinder.get('radius')),
                                       half_width=.5*float(cylinder.get('length')))))
    ContactModel(document)  # Validate exported geometry before writing it.
    return document


class ContactModel:
    def __init__(self, geometry):
        if geometry.get('schema') != 'm20_contact_geometry.v1' or geometry.get('joint_order') != list(ORDER):
            raise ValueError('Invalid contact geometry schema/order')
        if len(geometry.get('legs', [])) != 4:
            raise ValueError('Expected four leg chains')
        self.chains = []
        for leg, item in zip(LEGS, geometry['legs']):
            chain = []
            if len(item['joints']) != 4:
                raise ValueError('Expected four joints per leg')
            for suffix, j in zip(SUFFIXES, item['joints']):
                if j['name'] != f'{leg}_{suffix}_joint' or j['mass'] <= 0 or not np.isfinite(j['mass']):
                    raise ValueError('Invalid joint identity/mass')
                axis = vector(j['axis'], 3, 'axis')
                if abs(np.linalg.norm(axis)-1) > 1e-8:
                    raise ValueError('Joint axis must be normalized')
                inertia = np.asarray(j['inertia'], dtype=float)
                if (inertia.shape != (3, 3) or not np.isfinite(inertia).all()
                        or not np.allclose(inertia, inertia.T, atol=1e-12)
                        or np.min(np.linalg.eigvalsh(inertia)) <= 0):
                    raise ValueError('Invalid link inertia')
                def transform(o):
                    return vector(o['xyz'], 3, 'origin'), rpy_rotation(vector(o['rpy'], 3, 'rpy'))
                p, r = transform(j['origin']); com, ir = transform(j['com'])
                chain.append((p, r, axis, j['mass'], com, ir@inertia@ir.T))
            c = item['cylinder']; cp, cr = transform(c['origin'])
            if not np.isfinite([c['radius'], c['half_width']]).all() or min(c['radius'], c['half_width']) <= 0:
                raise ValueError('Invalid wheel geometry')
            self.chains.append((chain, cp, cr, c['radius'], c['half_width']))

    def dynamics(self, q, dq, ddq, base_rotation, base_specific_force, omega_body, alpha_body):
        """World support/J and required actuator torque with no external contact.

        Specific force is a_base - gravity, expressed in the base frame. Thus
        stationary upright input is [0,0,+g], never zero. IMU lever arm and frame
        must already be compensated by the caller. No finite differences here.
        """
        q, dq, ddq = (vector(x, 16, n) for x, n in ((q, 'q'), (dq, 'dq'), (ddq, 'ddq')))
        rbase = np.asarray(base_rotation, dtype=float)
        if (rbase.shape != (3, 3) or not np.isfinite(rbase).all()
                or not np.allclose(rbase.T@rbase, np.eye(3), atol=1e-8)
                or abs(np.linalg.det(rbase)-1) > 1e-8):
            raise ValueError('Invalid base rotation')
        accel = rbase@vector(base_specific_force, 3, 'base specific force')
        omega = rbase@vector(omega_body, 3, 'omega')
        alpha = rbase@vector(alpha_body, 3, 'alpha')
        torque, supports, jacobians = [], [], []
        for leg, (chain, cp, cr, radius, half_width) in enumerate(self.chains):
            p, r, a, w, aw = np.zeros(3), rbase.copy(), accel.copy(), omega.copy(), alpha.copy()
            states = []
            for k, (origin, rot, axis, mass, com, inertia) in enumerate(chain):
                j = 4*leg+k
                offset = r@origin
                p = p+offset
                a = a+np.cross(aw, offset)+np.cross(w, np.cross(w, offset))
                axis_world = r@rot@axis
                aw = aw+axis_world*ddq[j]+np.cross(w, axis_world)*dq[j]
                w = w+axis_world*dq[j]
                r = r@rot@rotation(axis, q[j])
                com_world = r@com
                force = mass*(a+np.cross(aw, com_world)+np.cross(w, np.cross(w, com_world)))
                iw = r@inertia@r.T
                moment = iw@aw+np.cross(w, iw@w)+np.cross(com_world, force)
                states.append((p.copy(), axis_world, force, moment))
            centre, axis = p+r@cp, r@cr[:, 2]
            radial = np.array([0., 0., 1.])-axis[2]*axis
            if np.linalg.norm(radial) < 1e-6:
                raise ValueError('Wheel axis vertical: support direction undefined')
            support = centre-radius*radial/np.linalg.norm(radial)-half_width*np.sign(axis[2])*axis
            supports.append(support)
            # Jacobian is for the material point now at support, as mj_jac.
            jacobians.append(np.column_stack([np.cross(s[1], support-s[0]) for s in states[:3]]))
            total_f, total_n, previous_p = np.zeros(3), np.zeros(3), states[-1][0]
            values = []
            for point, axis_world, force, moment in reversed(states):
                total_n = moment+total_n+np.cross(previous_p-point, total_f)
                total_f = total_f+force
                values.append(float(axis_world@total_n)); previous_p = point
            torque.extend(reversed(values))
        return dict(model_torque_sdk_nm=np.asarray(torque), support_relative_world_m=np.asarray(supports),
                    point_jacobian_world_m_per_rad=np.asarray(jacobians))

    def diagnose(self, measured_torque, **state):
        result = self.dynamics(**state)
        measured = vector(measured_torque, 16, 'measured torque')
        forces, conditions, reasons = [], [], []
        for leg, jac in enumerate(result['point_jacobian_world_m_per_rad']):
            condition = float(np.linalg.cond(jac.T)); conditions.append(condition if np.isfinite(condition) else None)
            if not np.isfinite(condition) or condition > 100:
                forces.append([None]*3); reasons.append('ill_conditioned_support_jacobian'); continue
            residual = result['model_torque_sdk_nm'][4*leg:4*leg+3]-measured[4*leg:4*leg+3]
            forces.append(np.linalg.solve(jac.T, residual).tolist())
            reasons.append('unvalidated_torque_bias_friction_and_contact_direction')
        return {**{key: value.tolist() for key, value in result.items()},
                'estimated_force_world_N': forces, 'jacobian_condition': conditions,
                'contact_valid': [False]*4, 'invalid_reason': reasons,
                'hardware_validated': False,
                'scope': 'inverse_dynamics_diagnostic; not a measured support-force sensor; not CTS admission'}


def save_geometry(urdf, output):
    value = urdf_geometry(urdf)
    with Path(output).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
