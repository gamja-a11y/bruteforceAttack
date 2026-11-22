"""Parent enforces a common deadline per cold policy process, including startup."""
from compute import *
from frozen_rho_policy import candidates
import subprocess,platform,statistics
from concurrent.futures import ThreadPoolExecutor
E=D/'experiments'
def memory_limit(mib):
    if os.name=='nt':
        sys.path.append(str(ROOT/'fresnel_publication_phase3'))
        # Load explicitly; this file has the same basename as the old driver.
        import importlib.util
        spec=importlib.util.spec_from_file_location('phase3_job_limit',ROOT/'fresnel_publication_phase3/experiment.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod.limit(mib)
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(mib*2**20,mib*2**20))
def worker(tag,params,method,m):
    guard=memory_limit(m['process_mib'])
    out=D/'records'/tag
    series=candidates(params,method)
    rows=[]
    last=None
    while True:
        try:n,b,rho,scans=series.send(last)
        except StopIteration:break
        if scans:dump(out/'contours.json',scans)
        path=out/f'n{n}_b{b}_r{rho}.json'
        dump(out/'pending.json',dict(n=n,bits=b,rho=rho))
        last=evaluate(params,n,b,rho,path)
        rows.append(last)
        dump(out/'attempts.json',rows)
        (out/'pending.json').unlink()
    return True
def trial(task):
    group,i,params,method,rep,m=task
    tag=f"{group}_{i}_{params['model']}_{method}_{rep}"
    out=D/'records'/tag
    out.mkdir()
    tic=time.perf_counter()
    try:
        res=subprocess.run([sys.executable,'-B',__file__,'worker'],input=json.dumps([tag,params,method,m]),capture_output=True,text=True,timeout=m['seconds'])
        status='complete' if res.returncode==0 else 'worker_failure'
        reason=res.stderr.strip()[-1000:]
    except subprocess.TimeoutExpired:
        status,reason='timeout','common method deadline'
    seconds=time.perf_counter()-tic
    if seconds>m['seconds'] and status=='complete':status,reason='timeout','parent wall time exceeded deadline'
    rows=read(out/'attempts.json') if (out/'attempts.json').is_file() else []
    selected=min(rows,key=lambda r:F(r['width'])) if rows else {}
    ok=bool(selected.get('met')) and status=='complete'
    if status=='complete' and not ok:reason='degree or precision limit'
    scans=read(out/'contours.json') if (out/'contours.json').is_file() else []
    record=dict(group=group,case=i,**params,method=method,rep=rep,met=ok,status=status,reason=reason,wall_s=seconds,peak_mib=max((r['peak_mib'] for r in rows),default=None),issued_verified=len(rows),attempts=len(rows)+int((out/'pending.json').exists()),selection_contours=len({r['rho'] for r in scans}),selection_contour_s=sum(max(x['contour_s'] for x in scans if x['rho']==rho) for rho in {r['rho'] for r in scans}),over_target_100x=bool(ok and F(selected['width'])<=F('1e-7') and F(selected['relative'])<=F('0.0001')))
    for k in ('n','bits','rho','L','U','width','relative','delta','file'):record[k]=selected.get(k)
    dump(out/'result.json',record)
    return record
def run_group(group):
    m=read(E/'holdout_manifest.json')
    for name,h in read(D/'audits/policy_source_hashes.json').items():
        if sha(ROOT/name)!=h:raise ValueError(name)
    jobs=[]
    if group=='holdout':
        for i,case in enumerate(m['cases'],1):
            for j,model in enumerate(m['models']):
                p=dict(**case,model=model,radius='5')
                start=(2*i+j)%3
                for method in ('ABC'*2)[start:start+3]:jobs.append((group,i,p,method,1,m))
    elif group=='repeated':
        successes={r['case'] for r in loadcsv(E/'holdout_results.csv') if r['model']=='Fresnel' and r['met']=='True'}
        r=loadcsv(E/'holdout_results.csv')
        eligible=[int(i) for i in successes if sum(v['met']=='True' for v in r if v['case']==i and v['model']=='Fresnel')==3]
        if not eligible:raise ValueError('no common success')
        i=min(eligible)
        for k in range(5):
            for method in ('ABC'*2)[k%3:k%3+3]:
                jobs.append((group,i,dict(**m['cases'][i-1],radius='5',model='Fresnel'),method,k+1,m))
    else:raise ValueError(group)
    name='holdout_results.csv' if group=='holdout' else 'repeated_runtime.csv'
    rows=[]
    with ThreadPoolExecutor(max_workers=m['concurrent'] if group=='holdout' else 1) as pool:
        for row in pool.map(trial,jobs):
            rows.append(row);writecsv(E/name,rows)
            if len(rows)%12==0:print(group,len(rows),'/',len(jobs),flush=True)
def loadcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(*json.load(sys.stdin))
    else:run_group(sys.argv[1])

