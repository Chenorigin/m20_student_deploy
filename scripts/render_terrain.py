#!/usr/bin/env python3
"""Offline standalone HTML + PNG from a real capture_terrain.py receipt."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture');ap.add_argument('--output-prefix',required=True);args=ap.parse_args()
    src=Path(args.capture);data=json.loads(src.read_text());frames=data.get('frames',[])
    if data.get('schema')!='m20_actual_terrain_capture_v1' or not frames:raise ValueError('real terrain capture with frames required')
    for frame in frames:
        for name in ('height','validity','confidence'):
            a=np.asarray(frame[name]);
            if a.shape!=(187,) or not np.isfinite(a).all():raise ValueError('malformed captured grid')
    prefix=Path(args.output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.html','.png'):
        if prefix.with_suffix(suffix).exists():raise ValueError('preserve existing visualization')
    payload=base64.b64encode(json.dumps(data,allow_nan=False).encode()).decode()
    html='''<!doctype html><meta charset="utf-8"><title>M20 真机高程观测</title>
<style>body{background:#101722;color:#e8edf5;font:16px system-ui;margin:28px}h1{font-size:25px}.cards{display:flex;gap:18px;flex-wrap:wrap}.card{background:#182333;padding:16px;border-radius:10px}canvas{width:425px;height:300px;max-width:100%;background:#253040}.muted{color:#aab8cb}#state{padding:12px;border-left:4px solid #e6a84c;background:#293044;margin:16px 0}input{width:min(800px,90vw)}#tip{white-space:pre-wrap}button{padding:7px 14px}code{color:#9bd8fa}</style>
<h1>M20Pro · 真机 187 点高程观测</h1><p class="muted">原始 ROS 2 数据记录 · CPU 感知 · SDK 未启动 · 每格 0.1 m · 无数据格显示灰色，不填平地</p>
<div id="state"></div><button id="play">播放</button> <input id="slider" type="range" min="0" step="1"><span id="frame"></span>
<div class="cards"><div class="card"><h3>模型高度 h（m）</h3><canvas id="height" width="510" height="360"></canvas></div><div class="card"><h3>有效性（0 / 1）</h3><canvas id="validity" width="510" height="360"></canvas></div><div class="card"><h3>置信度（0–1，质量分数）</h3><canvas id="confidence" width="510" height="360"></canvas></div></div>
<p>X 水平向右为机器人前方，Y 向上为机器人左侧；右侧墙在 Y 负方向。中心十字为训练 base_link。</p><p>h = clip(base_z − ground_z − 0.5, −1, 1)。地面升高，h 下降。置信度尚非经独立真值标定的概率。</p><pre id="tip">移到栅格上查看实际数值</pre><pre id="stats"></pre>
<script>const D=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('PAYLOAD'),c=>c.charCodeAt(0))));const S=document.getElementById('slider');S.max=D.frames.length-1;S.value=S.max;let timer=null;
function cells(f){return f.receive_adjusted_hmc||[f.height,f.validity,f.confidence]}
function draw(){let k=+S.value,f=D.frames[k],[h,m,c]=cells(f),known=h.filter((_,i)=>m[i]===1),lo=known.length?Math.min(...known):0,hi=known.length?Math.max(...known):1;if(hi-lo<.02){lo-=.01;hi+=.01}const valid=m.filter(x=>x===1).length;
document.getElementById('frame').textContent=` ${k+1}/${D.frames.length} · ${f.elapsed_s.toFixed(2)}s`;
document.getElementById('state').textContent=`${f.receive_gate_error?'接收门禁未通过：'+f.receive_gate_error:'本帧地形接收检查通过；不表示 SDK/现场准入通过'} | 有效 ${valid}/187 | 帧龄 ${(1000*(f.receive_stamp-f.stamp)).toFixed(1)}ms | sequence ${f.sequence}`;
for(const [name,a] of [['height',h],['validity',m],['confidence',c]]){let el=document.getElementById(name),ctx=el.getContext('2d');ctx.clearRect(0,0,510,360);for(let iy=0;iy<11;iy++)for(let ix=0;ix<17;ix++){let i=iy*17+ix,x=35+ix*26,y=24+(10-iy)*26,v=a[i];ctx.fillStyle=m[i]!==1?'#5b6270':name==='validity'?'#65cba6':name==='height'?`hsl(${230-230*Math.max(0,Math.min(1,(v-lo)/(hi-lo)))},70%,55%)`:`hsl(160,60%,${20+55*v}%)`;ctx.fillRect(x,y,25,25);if(m[i]!==1){ctx.fillStyle='#d7dbe2';ctx.font='9px sans-serif';ctx.fillText('?',x+10,y+16)}}ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(250,167);ctx.lineTo(262,167);ctx.moveTo(256,161);ctx.lineTo(256,173);ctx.stroke();ctx.fillStyle='white';ctx.font='12px sans-serif';ctx.fillText('X -0.8 m  ← base →  +0.8 m (front)',35,332);ctx.fillText(name==='height'?`height range ${lo.toFixed(3)} … ${hi.toFixed(3)} m`:'0 … 1; unknown = grey',35,351);el.onmousemove=e=>{let r=el.getBoundingClientRect(),ix=Math.floor(((e.clientX-r.left)*510/r.width-35)/26),iy=10-Math.floor(((e.clientY-r.top)*360/r.height-24)/26);if(ix<0||ix>16||iy<0||iy>10)return;let i=iy*17+ix;document.getElementById('tip').textContent=`index ${i} · x=${(-.8+ix*.1).toFixed(1)}m · y=${(-.5+iy*.1).toFixed(1)}m\nh=${h[i].toFixed(4)}m  valid=${m[i]}  confidence=${c[i].toFixed(4)}\nvariance=${f.variance[i].toExponential(3)}  sender_age=${f.age[i].toFixed(3)}s`}}
document.getElementById('stats').textContent=JSON.stringify({source:D.source,board:D.board,profile:D.profile,duration_s:D.duration_s,counts:D.counts,latest_status:D.latest_status},null,2)}
S.oninput=draw;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}if(+S.value===+S.max)S.value=0;timer=setInterval(()=>{if(+S.value>=+S.max){clearInterval(timer);timer=null;return}S.value=+S.value+1;draw()},100)};draw();</script>'''.replace('PAYLOAD',payload)
    prefix.with_suffix('.html').write_text(html)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    f=frames[-1];h,m,c=np.asarray(f['receive_adjusted_hmc'] if f['receive_adjusted_hmc'] is not None else [f['height'],f['validity'],f['confidence']]);known=m==1
    fig,axes=plt.subplots(2,3,figsize=(15,8),gridspec_kw={'height_ratios':[2,1]},layout='constrained')
    for ax,a,title,palette,limits in zip(axes[0],[h,m,c],['Model height h (m)','Validity','Confidence (quality score)'],['RdBu_r','Greens','viridis'],[(None,None),(0,1),(0,1)]):
        image=np.ma.array(a.reshape(11,17),mask=(m.reshape(11,17)!=1));cmap=cm.get_cmap(palette).copy();cmap.set_bad('#85858a')
        im=ax.imshow(image,origin='lower',extent=[-.85,.85,-.55,.55],cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest',aspect='equal')
        ax.plot(0,0,'k+',ms=12,mew=2);ax.set(title=title,xlabel='X (m), positive = front',ylabel='Y (m), positive = left');fig.colorbar(im,ax=ax,shrink=.65)
    times=[x['elapsed_s'] for x in frames];fractions=[sum(x['validity'])/187 for x in frames]
    axes[1,0].plot(times,fractions);axes[1,0].set(xlabel='Capture elapsed (s)',ylabel='Sender valid fraction',ylim=(0,1));axes[1,0].grid(alpha=.25)
    rr=data.get('raw_cloud_receipts',[])
    if rr:axes[1,1].plot([x['elapsed_s'] for x in rr],[(x['received_at']-x['end_stamp'])*1000 for x in rr])
    axes[1,1].set(xlabel='Capture elapsed (s)',ylabel='GOS newest-point age (ms)');axes[1,1].grid(alpha=.25)
    axes[1,2].axis('off');axes[1,2].text(0,1,'REAL ROS2 CAPTURE\n'+data['board']+' / '+data['profile']+'\nValid cells: '+str(int(known.sum()))+'/187\nGate: '+str(f['receive_gate_error'] or 'terrain receive passed')+'\nSDK NOT STARTED\nUnknown = grey; no fabricated floor',va='top',wrap=True)
    fig.suptitle('M20Pro real terrain: h = base_z - ground_z - 0.5 | no control admission',fontsize=14)
    fig.savefig(prefix.with_suffix('.png'),dpi=150);plt.close(fig)
    print(json.dumps({'html':str(prefix.with_suffix('.html')),'png':str(prefix.with_suffix('.png')),'source_capture':str(src),'frames':len(frames),'valid_cells_last':int(known.sum())}))


if __name__=='__main__':main()
