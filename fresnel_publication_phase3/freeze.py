import csv, hashlib, itertools, json, pathlib
d=pathlib.Path(__file__).resolve().parent
old=d.parent/'fresnel_publication_phase2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
previous=json.loads((old/'experiment_manifest.json').read_text())['geometries']
cases=list(itertools.product(('4','10'),('1','3','10'),('1','4'),('0.1','0.3')))
if any(list(c) in previous for c in cases): raise ValueError('overlap')
files=[d/'frozen_selection_rule.py',old/'adaptive_certification.py',old/'adaptive_results.csv',old/'experiment_manifest.json',*(old/'frozen').glob('repaired*.py'),d/'experiment.py']
manifest=dict(order=['p','nf','ar','theta'],cases=cases,excluded=previous,models=['Pade','Fresnel'],radius='5',degrees=[16,24,32,48,64],bits=[256,512],relative='0.01',absolute='0.00001',seconds_per_method=30,process_mib=512,fixed=[64,512],method_order='ABC cyclic by case/model index, serial cold workers, one repeat',selection='complete crossed design, two new p levels, NF1/3/10 and AR1/4 including joint stress corners; matched original pilot budget',sha256={str(p.relative_to(d.parent)):sha(p) for p in files})
with (d/'holdout_manifest.json').open('x') as f: json.dump(manifest,f,indent=2)
