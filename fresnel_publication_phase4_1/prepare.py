"""Freeze the holdout before opening any new certificate."""
import csv, hashlib, itertools, json, sys
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
P4=ROOT/'fresnel_publication_phase4'
KEYS=('nf','ar','p','theta')
MAN=HERE/'holdout_manifest.json'
PIN=HERE/'holdout_manifest.sha256'
SOURCES=HERE/'original_sha256.json'

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require(test,msg):
    if not test: raise ValueError(msg)

def key(r):
    return tuple(Fraction(r[k]) for k in KEYS)

def original_check():
    old=load(SOURCES)
    roots={k.split('/')[0] for k in old}
    actual={p.relative_to(ROOT).as_posix():digest(p) for name in roots
            for p in (ROOT/name).rglob('*') if p.is_file()}
    require(actual==old,'Original file inventory or digest changed')
    pins=load(P4/'audits/policy_source_hashes.json')
    require(all(actual[k]==v for k,v in pins.items()),'Phase 4 freeze mismatch')
    return len(actual)

def prepare():
    count=original_check()
    if MAN.exists() or PIN.exists():
        frozen()
        print('Already frozen:',digest(MAN), flush=True)
        return
    old=set()
    for phase,filename,group in [('2','experiment_manifest.json','geometries'),('3','holdout_manifest.json','cases')]:
        obj=load(ROOT/f'fresnel_publication_phase{phase}'/filename)
        for row in obj[group]:old.add(key(dict(zip(('p','nf','ar','theta'),row))))
    old.update(key(r) for r in load(P4/'experiments/holdout_manifest.json')['cases'])
    for path in [ROOT/'fresnel_publication_phase2/adaptive_results.csv',ROOT/'fresnel_publication_phase2/benchmark_results.csv',
                 P4/'experiments/development_results.csv',P4/'experiments/development_policy_comparison.csv']:
        with path.open(encoding='utf-8-sig',newline='') as f:
            old.update(key(r) for r in csv.DictReader(f) if all(r.get(k) for k in KEYS))
    cases=[];changes=[]
    for vals in itertools.product(('12','14','16'),('3','4'),('6','10'),('0.2','0.3')):
        r=dict(zip(KEYS,vals))
        if key(r) in old:
            orig=r.copy()
            for replacement in ('0.21','0.31','0.22','0.32'):
                r={**orig,'theta':replacement}
                if key(r) not in old and key(r) not in {key(c) for c in cases}:break
            changes.append(dict(original=orig,replacement=r))
        require(key(r) not in old and key(r) not in {key(c) for c in cases},'overlap')
        cases.append(r)
    inputs=[p for p in HERE.glob('*.py')]+[SOURCES]
    record=dict(cases=cases,replacements=changes,reserve='same NF, AR and p; first unused theta in 0.21,0.31,0.22,0.32',prior_physical_count=len(old),models=['Pade','Fresnel'],methods=['B','C'],degree=128,bits=1024,scan_bits=512,radius='5',seconds=90,memory_mib=768,absolute='1e-5',relative='0.01',original_files=count,source_pins={p.name:digest(p) for p in inputs})
    with MAN.open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
    with PIN.open('x') as f:f.write(digest(MAN)+'\n')
    print('Frozen',len(cases),'physical cases;',len(changes),'replacements;',digest(MAN))

def frozen():
    require(digest(MAN)==PIN.read_text().strip(),'Manifest hash mismatch')
    m=load(MAN)
    require(all(digest(HERE/k)==v for k,v in m['source_pins'].items()),'Runner/source pin mismatch')
    return m

if __name__=='__main__':prepare()

