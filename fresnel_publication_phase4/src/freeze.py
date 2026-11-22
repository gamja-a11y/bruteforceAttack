from experiment import *
import itertools
def main():
    cases=[dict(p=p,nf=nf,ar=ar,theta=t) for p,nf,ar,t in itertools.product(('6','14'),('3','10','14'),('1','3'),('0.1','0.25'))]
    seen=set()
    for name,key in [('fresnel_publication_phase2/experiment_manifest.json','geometries'),('fresnel_publication_phase3/holdout_manifest.json','cases')]:
        seen.update(tuple(map(F,v)) for v in read(ROOT/name)[key])
    if len(seen)!=30 or any(tuple(map(F,c.values())) in seen for c in cases):raise ValueError('overlap')
    if (E/'holdout_manifest.json').exists():raise FileExistsError('already frozen')
    dev=loadcsv(E/'development_results.csv')
    policies=loadcsv(E/'development_policy_comparison.csv')
    if len(policies)!=64 or len({r['case']+r['model'] for r in dev if r['n']=='128'})!=16:raise ValueError('incomplete development')
    manifest=dict(cases=cases,models=['Pade','Fresnel'],design='two new p levels crossed with three NF levels, two AR and theta levels; 16/24 geometries have NF>=10; controlled coordinate pairs',excluded_count=len(seen),radius='5',seconds=90,process_mib=768,concurrent=4,order='Pade then Fresnel; rotate A,B,C by (2*case+model_index)%3; four concurrent cold workers; repeats alone',goals=['absolute<=1e-5','L>0 and symmetric relative<=.01'],A='original Phase 3 policy, <=64/512, same new math backend, 90s; original 30s historical data separate',B='same higher degree scheduler as C, fixed rho 2',C='finite rho scan, then joint degree/precision scheduler',rhos=RHOS,degrees=NS,bits=BITS,stop='first verified certificate meeting both goals, end of policy, deadline or error; retain every attempt',analysis='all 48 denominators per method; pair times on common successes; label >100x both tolerances; repeats excluded from denominators',repeated='smallest numbered Fresnel case with all methods successful; five interleaved repetitions per method',cpu=os.environ.get('PROCESSOR_IDENTIFIER'),platform=platform.platform(),python=sys.version)
    dump(E/'holdout_manifest.json',manifest)
    dump(D/'policy_manifest.json',dict(methods={k:manifest[k] for k in 'ABC'},development_mapping='Cfixedbits is development Method C (always 1024); C is development Method D and final holdout C',development_runs=len(dev),selection='minimum certified delta at each degree, threshold and precision ladder in higher_degree_baseline.py',budgets={k:manifest[k] for k in ('seconds','process_mib','rhos','degrees','bits')},degree64_historical='Phase 3 A/B/C each 30/48 at 30 seconds; NF10 0/16'))
    files=list((D/'src').glob('*.py'))+[D/'policy_manifest.json',E/'holdout_manifest.json',E/'development_results.csv',E/'development_policy_comparison.csv',E/'rho_sweep_results.csv',ROOT/'fresnel_publication_phase3/frozen_selection_rule.py',ROOT/'fresnel_publication_phase3/experiment.py',ROOT/'fresnel_publication_phase3_audit_repaired/repaired_analyze.py',ROOT/'fresnel_publication_phase2/adaptive_certification.py']
    dump(D/'audits/policy_source_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
if __name__=='__main__':main()

