"""Replay all manifest runs. Failed runs are reported and excluded from successes."""
import csv,json,platform,sys,time
from collections import Counter
import flint
from repaired_analyze import *

def prior(name):
    with (PH/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    start=time.monotonic()
    m=manifest()
    old=prior('holdout_results.csv')
    baseline={(int(r['case']),r['model'],r['method']):r for r in old}
    previous_diag=prior('interval_width_diagnostics.csv')
    diag_by_key={(int(r['case']),r['model'],r['method'],int(r['n']),int(r['bits'])):r for r in previous_diag}
    need(len(old)==len(baseline)==144 and len(previous_diag)==len(diag_by_key)==312,'legacy_inventory')
    runs=PH/'runs'
    checks=[]
    details=[]
    matched=[]
    failures=[]
    expected={f'{i:02}_{model}_{method}' for i in range(1,len(m['cases'])+1) for model in m['models'] for method in 'ABC'}
    inventory(expected,{p.name for p in runs.iterdir()})
    for i,c in enumerate(m['cases'],1):
        for model in m['models']:
            intervals=[]
            paired=dict(case=i,model=model,**dict(zip(m['order'],c)))
            for method in 'ABC':
                folder=runs/f'{i:02}_{model}_{method}'
                try:
                    rows,best,s=inspect_run(folder,m,i,model,method)
                    for row in rows:
                        key=(i,model,method,row['degree'],row['bits'])
                        row['old_gram_norm_width']=diag_by_key[key]['gram_norm_width']
                        row['recovered_positive']=F(row['old_gram_norm_width'])==0 and F(row['gram_norm_width_lower'])>0
                    selected=rows[best]
                    met=selected['absolute_met'] and selected['relative_met']
                    reference=baseline[i,model,method]
                    need(reference['status']==s['status'] and all(reference[k]==s[k] for k in m['order']) and reference['degree']==str(selected['degree']) and reference['bits']==str(selected['bits']) and reference['absolute_width']==dec(selected['final_width'],True) and reference['lower']==dec(selected['L'],False) and reference['upper']==dec(selected['U'],True),'legacy_result_mismatch')
                    checks.append(dict(**paired_base(i,model,c,m),method=method,target_met=met,independently_verified=len(rows),best_degree=selected['degree'],best_bits=selected['bits'],L=selected['L'],U=selected['U'],absolute_width=selected['final_width'],relative_width=selected['relative_width'],field_delta=selected['field_delta'],historical_seconds=s['seconds'],historical_peak_worker_mib=s['peak_worker_mib']))
                    details.extend(rows)
                    intervals.extend((r['L'],r['U']) for r in rows)
                    paired.update({method+'_target_met':met,method+'_historical_seconds':s['seconds'],method+'_historical_peak_worker_mib':s['peak_worker_mib']})
                except Exception as exc:
                    failures.append(dict(run=folder.name,error=str(exc)))
            try: overlap(intervals)
            except AuditError as exc: failures.append(dict(case=i,model=model,error=str(exc)))
            matched.append(paired)
        print(f'{i}/24',flush=True)
    if checks: table(D/'repaired_holdout_results.csv',checks)
    if details: table(D/'repaired_interval_diagnostics.csv',details)
    if matched and not failures: table(D/'repaired_matched_benchmark.csv',matched)
    summary=dict(replayed=len(details),successful={k:sum(r['target_met'] for r in checks if r['method']==k) for k in 'ABC'},recovered=sum(r['recovered_positive'] for r in details),failures=failures,seconds=time.monotonic()-start,platform=platform.platform(),python=sys.version,flint=flint.__version__,native=flint.__FLINT_VERSION__)
    (D/'replay_summary.json').write_text(json.dumps(summary,indent=2))
    print(summary)
    if failures: raise SystemExit(1)

def paired_base(i,model,c,m): return dict(case=i,model=model,**dict(zip(m['order'],c)))

if __name__=='__main__': main()

