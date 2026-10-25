"""Only promote cold/hot and partial-occupancy qualified winners, preserving all evidence."""
import argparse,hashlib,json,statistics,importlib.util
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dispatch',type=Path,required=True,help='completed dispatch stage directory')
parser.add_argument('--decode',type=Path,required=True,help='completed decode stage directory')
parser.add_argument('--prefill',type=Path,required=True,help='completed prefill stage directory')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--selection',type=Path,required=True,help='accept/reject evidence JSON')
a=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
stages={'a1-layout':a.dispatch,'a2-dot-layout':a.decode,'a3-prefill-v3':a.prefill}
entries={};family_entries={};rejected=[];sources={}
for stage,family in [('a1-layout','dispatch'),('a2-dot-layout','decode'),('a3-prefill-v3','prefill')]:
 family_entries[family]={}
 p=stages[stage]/'tuning.json'
 assert json.loads((p.parent/'done.json').read_text())['returncode']==0
 sources[stage]=hashlib.sha256(p.read_bytes()).hexdigest()
 for r in json.loads(p.read_text())['rows']:
  if family=='dispatch' and r['M'] not in [1,2,4,8,16,32]:continue
  baseline='reference' if family=='prefill' else 'gemv' if r['M']<=8 else 'wmma'
  win=r['winner'];key=':'.join(str(r[x]) for x in ['kind','M','N','K','epi','stride0'])
  if win==baseline:continue
  med=r['median_ms'];hot=r['hot_median_ms'];partial=r.get('partial_median_ms',{})
  cold_gain=med[baseline]/med[win];hot_gain=hot[baseline]/hot[win]
  partial_gain=partial[baseline]/partial[win] if partial else None
  if min(cold_gain,hot_gain)<1.10 or partial_gain is not None and partial_gain<1.05:
   rejected.append(dict(key=key,stage=stage,winner=win,cold=cold_gain,hot=hot_gain,partial=partial_gain));continue
  if win.startswith('scalar-'):continue
  impl,cfg=(win.split('-')[0],list(map(int,win.split('-')[1:]))) if '-' in win else (win,None)
  if family=='decode' and impl not in ['dot','wmma']:continue
  row=dict(implementation=impl,family=family,cfg=cfg,
           evidence=dict(stage=stage,cold_gain=cold_gain,hot_gain=hot_gain,partial_gain=partial_gain))
  # Decode replacements supersede A1 for the same bucket; prefill includes M16/32.
  entries[key]=row;family_entries[family][key]=row
runtime=json.loads((a.prefill/'environment.json').read_text())
spec=importlib.util.spec_from_file_location('qualified_dispatch',repo/'startlux_prism/kernels/amd_w4a8_dispatch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=dict(schema=1,kernel_identity=module.kernel_identity(),runtime={k:runtime[k] for k in ['architecture','torch','triton','hip']},
          sources=sources,entries=entries,family_entries=family_entries)
a.output.write_text(json.dumps(data,indent=2)+'\n')
a.selection.write_text(json.dumps(dict(accepted=entries,rejected=rejected),indent=2)+'\n')
print('qualified entries',len(entries),'threshold rejected',len(rejected))
