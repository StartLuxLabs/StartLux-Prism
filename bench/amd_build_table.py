"""Only promote cold/hot and partial-occupancy qualified winners, preserving all evidence."""
import argparse,hashlib,json,statistics,importlib.util
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dispatch',type=Path,help='completed dispatch stage directory')
parser.add_argument('--decode',type=Path,help='completed decode stage directory')
parser.add_argument('--prefill',type=Path,help='completed prefill stage directory')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--selection',type=Path,required=True,help='accept/reject evidence JSON')
parser.add_argument('--tuning', type=Path, help='unified real-weight qualified experiment JSON')
parser.add_argument('--base-table', type=Path, help='existing same-identity table to preserve')
a=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
if a.tuning:
 from amd_table_selection import merge_qualified
 if not a.base_table:parser.error('--tuning requires --base-table')
 data=json.loads(a.tuning.read_text());base=json.loads(a.base_table.read_text())
 spec=importlib.util.spec_from_file_location('qualified_dispatch',repo/'startlux_prism/kernels/amd_w4a8_dispatch.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result,decisions=merge_qualified(data,base,module.kernel_identity(),data['runtime'])
 result.setdefault('sources',{})[data['experiment']]=hashlib.sha256(a.tuning.read_bytes()).hexdigest()
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 a.selection.write_text(json.dumps(decisions,indent=2)+'\n')
 print('qualified entries',len(result['entries']))
 raise SystemExit(0)
if not all((a.dispatch,a.decode,a.prefill)):parser.error('provide --tuning/--base-table or all three legacy stages')
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
  # Historical stages may have different sources or methods: do not silently rank them.
  if key in entries and (entries[key]['implementation'],entries[key]['cfg']) != (row['implementation'],row['cfg']):
   raise ValueError(f'Conflicting historical candidates for {key}; remeasure together with --tuning')
  entries[key]=row;family_entries[family][key]=row
runtime=json.loads((a.prefill/'environment.json').read_text())
spec=importlib.util.spec_from_file_location('qualified_dispatch',repo/'startlux_prism/kernels/amd_w4a8_dispatch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=dict(schema=1,kernel_identity=module.kernel_identity(),runtime={k:runtime[k] for k in ['architecture','torch','triton','hip']},
          sources=sources,entries=entries,family_entries=family_entries)
a.output.write_text(json.dumps(data,indent=2)+'\n')
a.selection.write_text(json.dumps(dict(accepted=entries,rejected=rejected),indent=2)+'\n')
print('qualified entries',len(entries),'threshold rejected',len(rejected))
