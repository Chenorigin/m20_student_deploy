"""Read robot-side inference parameters from this run's saved env.yaml."""
import re
import yaml


def derive(path, clip):
    env = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    names = [f'{leg}_{joint}_joint' for leg in ('fl','fr','hl','hr')
             for joint in ('hipx','hipy','knee')] + [f'{leg}_wheel_joint' for leg in ('fl','fr','hl','hr')]
    def rules(mapping, selected):
        result = []
        for name in selected:
            values = [float(v) for k,v in mapping.items() if re.fullmatch(k,name)]
            if len(values) != 1:
                raise ValueError('ambiguous training joint rule: '+name)
            result.append(values[0])
        return result
    terms = env['observations']['teacher']
    expected_terms = ['base_ang_vel','projected_gravity','velocity_commands','joint_pos','joint_vel','actions','height_scan']
    active_terms = [key for key,value in terms.items() if isinstance(value,dict) and 'func' in value]
    if active_terms != expected_terms:
        raise ValueError('teacher proprio term order changed; new deployment adaptation required')
    functions = ['isaaclab.envs.mdp.observations:base_ang_vel',
        'isaaclab.envs.mdp.observations:projected_gravity',
        'm20_student_v1.v3_5_environment:effective_velocity_commands',
        'rl_training.tasks.manager_based.locomotion.velocity.mdp.observations:joint_pos_rel_without_wheel',
        'isaaclab.envs.mdp.observations:joint_vel_rel',
        'm20_student_v1.environment:last_issued_action',
        'isaaclab.envs.mdp.observations:height_scan']
    for key,scale,function in zip(expected_terms,[.25,1.,1.,1.,.05,1.,1.],functions):
        term=terms[key]
        expected_clip=[-1.,1.] if key=='height_scan' else [-100.,100.]
        if (term['func']!=function or float(term['scale'])!=scale or
                list(map(float,term['clip']))!=expected_clip or term['modifiers']!='null' or
                int(term['history_length'])!=0):
            raise ValueError('hardcoded proprio semantics changed: '+key)
    for key in ('joint_pos','joint_vel'):
        asset=terms[key]['params']['asset_cfg']
        if asset['joint_names']!=names or asset['preserve_order']!='true' or asset['name']!='robot':
            raise ValueError('proprio joint order changed: '+key)
    if terms['joint_pos']['params']['wheel_asset_cfg']['joint_names']!=names[12:]:
        raise ValueError('wheel position masking changed')
    for key,selected in [('joint_pos',names[:12]),('joint_vel',names[12:])]:
        action=env['actions'][key]
        if (action['joint_names']!=selected or action['preserve_order']!='true' or
                action['use_default_offset']!='true' or float(action['offset'])!=0 or
                action['asset_name']!='robot'):
            raise ValueError('action semantics changed: '+key)
    if any(float(value)!=0 for value in env['scene']['robot']['init_state']['joint_vel'].values()):
        raise ValueError('nonzero default joint velocity is not supported')
    if int(env['decimation'])!=4 or float(env['sim']['dt'])!=.005:
        raise ValueError('SDK servo/policy timing changed')
    sdk_names = [f'{leg}_{joint}_joint' for leg in ('fl','fr','hl','hr')
                 for joint in ('hipx','hipy','knee','wheel')]
    joints = env['scene']['robot']
    actuator = joints['actuators']
    dt = float(env['sim']['dt'])*int(env['decimation'])
    if dt != .02:
        raise ValueError('SDK supports the trained 20ms/decimation4 contract only')
    return {'policy': {'dt_s': dt, 'normalized_action_clip': [-float(clip),float(clip)],
                       'processed_target_clip_policy': [list(map(float,env['actions']['joint_pos' if i<12 else 'joint_vel']['clip']['.*'])) for i in range(16)]},
            'joints': {'policy_order': names, 'sdk_order': sdk_names,
                       'default_position_policy_rad': rules(joints['init_state']['joint_pos'],names),
                       'action_scale_policy': rules(env['actions']['joint_pos']['scale'],names[:12])+[float(env['actions']['joint_vel']['scale'])]*4,
                       'kp_sdk': [float(actuator['wheel' if 'wheel' in n else 'joint']['stiffness']) for n in sdk_names],
                       'kd_sdk': [float(actuator['wheel' if 'wheel' in n else 'joint']['damping']) for n in sdk_names]},
            'command_training_ranges': {k: list(map(float,env['commands']['base_velocity']['ranges'][k])) for k in ('lin_vel_x','lin_vel_y','ang_vel_z')}}
