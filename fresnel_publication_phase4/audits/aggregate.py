"""Reconcile expected run paths, held-out inputs, prior proofs and exact endpoints."""
from pathlib import Path
import sys,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import *
from collections import defaultdict
def endpoints(rec):
    e=rec['obligations']['endpoints']['evidence']
    return dy(e['lower']),dy(e['upper'])
def bounds(records):
    return max(x[0] for x in records)<=min(x[1] for x in records)
def demand(v,txt):
    if not v:raise ValueError(txt)
def main():
    manifest=read(E/'holdout_manifest.json')
    dev=loadcsv(E/'development_results.csv')
    hold=loadcsv(E/'holdout_results.csv')
    repeated=loadcsv(E/'repeated_runtime.csv')
    dp=loadcsv(E/'development_policy_comparison.csv')
    demand(len(hold)==144 and len(repeated)==15,'missing evaluation rows')
    versions=read(D/'audits/policy_source_hashes.json')
    demand(all(sha(ROOT/k)==v for k,v in versions.items()),'frozen hash mismatch')
    old=read(ROOT/'fresnel_publication_phase3/holdout_manifest.json')
    panels=defaultdict(list);prior=[];count=0
    def record(r,params,n,b,rho):
        nonlocal count
        obj=read(D/r['file']);rec=obj['certificate']
        expected=dict(**params,nothing=None);expected.pop('nothing')
        expected.update(degree=n,bits=b,rho=rho,Q=None)
        demand(obj['expected']==rec['inputs']==expected,'input mismatch')
        proof=obj['verification']
        demand(proof['verified'] is True and proof['inputs']==expected and proof['verifier_sha256']==sha(D/'src/rho_independent_verifier.py') and rec['generator_sha256']==sha(D/'src/rho_certifier.py'),'verifier provenance')
        l,u=endpoints(rec)
        met=u-l<=F('1e-5') and l>0 and 2*(u-l)<=F('.01')*(u+l)
        demand(met is rec['target_met'] and met is proof['target_met'],'incorrect target')
        panels[tuple(params[k] for k in ('p','nf','ar','theta','model'))].append((l,u))
        count+=1
        return met
    for r in dev:
        if r['verified']!='True':continue
        i=int(r['case'])
        params=dict(zip(old['order'],old['cases'][i-1]),radius='5',model=r['model'])
        record(r,params,int(r['n']),int(r['bits']),r['rho'])
        if r['rho']=='2' and r['n']=='64' and r['bits']=='512':
            pair=endpoints(read(ROOT/f"fresnel_publication_phase3/runs/{i:02}_{r['model']}_C/n64_b512.json")['certificate'])
            demand(bounds([pair,endpoints(read(D/r['file'])['certificate'])]),'rho=2 regression');prior.append(r['file'])
    keys=set()
    for r in hold+repeated+dp:
        i=int(r['case']);group=r['group']
        p=manifest['cases'][i-1] if group!='devpolicy' else dict(zip(old['order'],old['cases'][i-1]))
        params=dict(**p,model=r['model'],radius='5')
        if group=='holdout':
            key=i,r['model'],r['method']
            demand(key not in keys and all(r[k]==v for k,v in params.items()),'holdout manifest');keys.add(key)
        folder=D/'records'/f"{group}_{i}_{r['model']}_{r['method']}_{r['rep']}"
        attempts=read(folder/'attempts.json') if (folder/'attempts.json').exists() else []
        actual=[]
        for a in attempts:
            actual.append(record(a,params,a['n'],a['bits'],a['rho']))
        demand((r['met']=='True')==(any(actual) and r['status']=='complete'),'aggregation')
    demand(len(keys)==144 and len(prior)==16,'cohort or regression')
    demand(all(bounds(v) for v in panels.values()),'disjoint enclosures')
    demand(read(D/'audits/wsl_regeneration.json')['overlap'],'Ubuntu regeneration')
    demand(len(loadcsv(D/'audits/contour_replay.csv'))==128,'contour replay')
    tests=loadcsv(D/'audits/negative_integrity_tests.csv')
    demand(all(r['passed']=='True' for r in tests) and len(tests)==10,'negative tests')
    demand(read(D/'audits/source_hash_audit.json')['unchanged'],'protected originals')
    demand(read(D/'audits/wsl_probe.json')['verified'],'Ubuntu replay')
    dump(D/'audits/verification_results.json',dict(status='PASS',checked_at_issuance_by='rho_independent_verifier.check(expected)',archived_attempts_reconciled=count,original_rho2_overlaps=len(prior),holdout_method_runs=len(keys),overlap_groups=len(panels),negative_tests=len(tests)-1,positive_controls=1,wsl_representatives=2,independent_development_contour_replays=128,trust='Theorem, reviewed Python code, and shared FLINT/Arb. Independent mathematical replay at issuance; final aggregation checks provenance, exact endpoints, inputs and intersections.'))
if __name__=='__main__':main()


