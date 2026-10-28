#!/usr/bin/env python3
"""Summarize physical-contact traces without rerunning or changing the observer."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('contact_benchmark', Path(__file__).with_name('verify_cts_contact_sim.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def summarize(folder):
    report = json.loads((folder / 'results.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    if not report.get('completed'):
        raise ValueError('Incomplete experiment')
    groups = {}
    traces = {}
    for case in report['cases']:
        name = '{variant}_{scene}_{seed}'.format(**case)
        path = folder / (name + '.jsonl')
        if benchmark.sha(path) != case['trace_sha256']:
            raise ValueError('Trace hash mismatch: ' + name)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        traces[name] = rows
        groups.setdefault((case['variant'], case['scene']), []).append((case, rows))
    aggregated = []
    for (variant, scene), cases in groups.items():
        item = dict(variant=variant, scene=scene, seeds=len(cases), comparisons={},
                    side_contact_frames=sum(case['side_contact_frames'] for case, _ in cases),
                    nonwheel_contact_frames=sum(case['nonwheel_contact_frames'] for case, _ in cases))
        # Metrics are evaluated per rollout (FSM resets between seeds), then
        # errors are pooled. Never join separate runs into a fake transition.
        for observer in ('ideal_world_N', 'causal_world_N', 'aligned_delayed_world_N'):
            errors, disagreements, total = [], 0, 0
            false_support = missed_support = invalid = 0
            for _, rows in cases:
                truth = np.maximum(np.asarray([r['truth_world_N'] for r in rows])[:, :, 2], 0.)
                estimate = np.asarray([r[observer] for r in rows], dtype=float)[:, :, 2]
                if observer == 'aligned_delayed_world_N':
                    truth, estimate = truth[:-1], estimate[1:]
                measurement = benchmark.metrics(truth, estimate, protocol)
                finite = np.isfinite(estimate)
                errors.extend(np.abs(estimate - truth)[finite].tolist())
                total += measurement['samples']
                invalid += int((~finite).sum())
                false_support += measurement['fsm_false_support_samples']
                missed_support += measurement['fsm_missed_support_samples']
                disagreements += measurement['fsm_false_support_samples'] + measurement['fsm_missed_support_samples']
            item['comparisons'][observer] = dict(force_abs_error_p95_N=float(np.quantile(errors, .95)),
                force_abs_error_max_N=float(max(errors)), force_mae_N=float(np.mean(errors)),
                contact_fsm_agreement=1-disagreements/total, samples=total,
                invalid_samples=invalid, fsm_false_support_samples=false_support,
                fsm_missed_support_samples=missed_support)
        aggregated.append(item)
    result = dict(cases=aggregated, completed_runs=len(report['cases']), steps=report['simulation_steps'],
                  current_implementation_pass=report['screening_all_cases_pass'], hardware_admission_granted=False)
    (folder / 'aggregate.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    panels = [('matched_cylinder', 'drop', None), ('source_mjcf', 'drop', (.17, .28)),
              ('source_mjcf', 'wall', None), ('source_mjcf', 'flat', (.5, 1.5))]
    for ax, (variant, scene, window) in zip(axes.flat, panels):
        rows = traces[f'{variant}_{scene}_12703']
        t = np.asarray([r['time_s'] for r in rows])
        for key, label, color, linestyle in [
            ('truth_world_N', 'MuJoCo wheel net +Z', 'black', '-'),
            ('causal_world_N', 'Delivered current-frame difference', '#D55E00', '-'),
            ('aligned_delayed_world_N', 'Experimental aligned (5ms older)', '#0072B2', '--')]:
            values = np.asarray([r[key] for r in rows], dtype=float)[:, 0, 2]
            if key == 'truth_world_N':
                values = np.maximum(values, 0.)
            ax.plot(t, values, label=label, color=color, linestyle=linestyle, linewidth=1.2)
        if scene == 'wall':
            ax.plot(t, [r['upward_normal_N'][0] for r in rows], color='#009E73',
                    label='Upward-normal adapter oracle', linewidth=1.)
        for threshold in ('contact_on_force_N', 'contact_off_force_N'):
            ax.axhline(protocol['settings'][threshold], color='grey', linewidth=.6, linestyle=':')
        if window:
            ax.set_xlim(*window)
        if scene == 'flat':
            ax.set_ylim(60, 105)
        ax.set_title(f'{variant} / {scene}, FL, seed 12703')
        ax.set_xlabel('Availability time (s)')
        ax.set_ylabel('Signed estimated / true force (N)')
        ax.grid(alpha=.2)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=2, fontsize=9)
    fig.savefig(folder / 'contact_force_comparison.png', dpi=160)
    plt.close(fig)

    lines = ['# CTS 触地估计物理仿真验证（2026-10-27）', '',
        '**结论：现有采集诊断实现未通过动态触地力对齐验证，不能据此解除 CTS 真机触地门禁。**', '',
        f'本轮完成 {len(report["cases"])} 个独立 rollout、{report["simulation_steps"]} 个真实物理步进；MuJoCo {report["mujoco_version"]}，CPU 单线程。',
        '仿真以固定关节 PD 保持名义姿态，包含离地、平地承载、低高度落地、轮子撞低矮立面。没有运行 CTS10000 策略闭环、训练或真机指令。', '',
        '## 对照结果', '',
        '下表为四轮、三个种子的绝对力误差 95% 分位（N）；时序对齐实验输出对应上一物理帧，获取时已延迟 5 ms。', '',
        '| 物理模型 | 场景 | 理想加速度 | 现有差分诊断 | 5 ms 对齐实验 | 轮侧碰帧数 |',
        '|---|---|---:|---:|---:|---:|']
    for item in aggregated:
        values = [item['comparisons'][key]['force_abs_error_p95_N'] for key in
                  ('ideal_world_N', 'causal_world_N', 'aligned_delayed_world_N')]
        lines.append(f'| {item["variant"]} | {item["scene"]} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {item["side_contact_frames"]} |')
    lines += ['', '## 已定位的问题', '',
        '- 现有诊断把当前帧的力矩、姿态、比力，与前一速度区间的差分加速度混合使用。Euler 物理轨迹上的上一帧对齐消融大幅降低落地误差，支持时间错配为主要原因；这不是硬件传感器时间同步的证明。',
        '- 对齐实验的低误差是对上一源时刻而言。独立审查把同一延迟输出故意对当前真值比较，source_mjcf/drop 的 p95 仍约 26.7 N；不能把延迟输出直接当作当前力。下图按实际可获取时刻画线，保留可见的 5 ms 延迟。',
        '- 对齐后，轮子同时撞立面和地面仍有明显误差。当前估计器把所有外力等效为轮底单点力；多个接触点的合力矩不能一般地等效为该点的三维力。改变差分时序不能消除这一误差。',
        '- 平地/落地中触地二值 FSM 一致，并不表示力值一致；承载远高于阈值时，大幅力值误差也可能不改变触地状态。所有峰值、漏检和误报统计保留在 results.json / aggregate.json。',
        '- FSM 指标仅在有限力估计被假设有效的前提下比较力滞回/确认递推。生产诊断的 contact_valid 仍全为 false；本轮没有执行完整 LandingController 的 air_time、其它支撑、phase 和增益控制链。',
        '- Isaac 训练的轮子净接触力 world-Z 与 MuJoCo 适配器仅累计向上法向支撑力的定义，在侧碰含摩擦时不等价。两种 oracle 分别统计，不能混称训练真值。', '',
        '## 验证方法与反例', '',
        '- 每步在同一 pre-integration 状态执行 mj_forward 后读取 q/dq/qacc、实际 qfrc_actuator 和 mj_contactForce，再由 mj_step 推进真实接触轨迹。没有通过人为设置接触力再反算力矩构造正例。',
        '- 理想加速度路径使用仿真 qacc，属于可观测性上限；现有路径直接调用交付的 Diagnostics.consume，经过原始 SDK 顺序、方向和偏置转换，只用后向速度差分。对齐消融只在本仿真脚本中实现，没有修改部署实现。',
        '- oracle 通过接触点 Jacobian 累加力和力矩，其广义力与 qfrc_constraint 交叉校验；零力和反向力在承载场景均触发失败。未把异常输入替换为有效零。',
        '- 冻结筛查界限：绝对力误差 p95 ≤ 4 N、FSM 一致率 ≥ 99%、有限估计比例 ≥ 99%。这是本轮工程筛查界限，不能作为厂家精度指标。当前路径保留第一帧差分预热，因此短离地回合因可用率不足而筛查失败。延迟消融按有对应源时刻的帧评估，不含首帧；其可用率分母不同，不能把通过项数量直接与当前路径比较。',
        '- 第一轮 wall 未形成轮子侧碰，原始结果保留在 ../cts_contact_sim_20261027；v2 只修正立面位置和高度并增加时序对齐消融，保持原判据。', '',
        '## 来源、范围与未验证项', '',
        f'- 部署参照模型：`{protocol["mjcf"]}`。source_mjcf 保留物理参数，仅剔除可视 mesh，使用部署加载器相同的轮 armature 0.00243216。各运行 XML 与脚本共同定义仿真；armature 是编译后赋值。',
        '- matched_cylinder 将腿部质量、惯量和轮柱体替换为当前估计器参数，以隔离模型误差；matched_capsule 仅把轮柱体换为 capsule 做形状扰动，不是原 Isaac USD 的严格复刻。',
        '- 仿真输入采用基座原点的理想同步 IMU；另留 uncompensated_imu_world_N 记录原模型 IMU 杠杆臂未补偿的敏感性。没有覆盖实际扭矩偏置、摩擦标定、编码器噪声、DDS 异步时钟。',
        '- 现有诊断离线每 5 ms 计算一次；真机采集入口限 20 Hz，CTS 策略为 50 Hz。本轮未验实际传输、保持与过期行为，不能把源时刻对齐指标当作部署实时性能。',
        '- 未运行实际 Isaac/PhysX、CTS10000 行走/楼梯性能或实机验证。所有 contact_valid 与 hardware_admission 保持 false。',
        '- 协议、源文件 SHA、模型变体、种子、全部轨迹与指标可在本目录 protocol.json / results.json 中核对。', '',
        '- 同 Codex family reviewer 独立复跑 480 个物理步、从 800 帧白名单输入重算，并核对所有 v2 轨迹/XML SHA；缺异构 family。独立结果在 independent_review/。既有 contact_estimator 回归测试本轮 21 项通过，不能替代以上物理力值验证。', '',
        '## 复现', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ubuntu/miniconda3/envs/m20_wzh/bin/python \\',
        f'  {benchmark.ROOT}/scripts/verify_cts_contact_sim.py \\',
        f'  --protocol {folder}/protocol.json --output /tmp/m20_contact_sim_new_run',
        '```', '',
        'MuJoCo 官方语义：[mj_forward / mj_step / mj_contactForce](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)、',
        '[同一时刻的动力学字段与 contact frame](https://mujoco.readthedocs.io/en/latest/programming/simulation.html)。', '']
    (folder / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(folder / 'REPORT.md'), plot=str(folder / 'contact_force_comparison.png'), groups=len(aggregated))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder.resolve())
