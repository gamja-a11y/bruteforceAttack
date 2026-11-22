"""Development grid. Each fixed-rho candidate uses a cold job, concurrency four."""
from compute import *
from concurrent.futures import ProcessPoolExecutor
import itertools,multiprocessing
def case(i,model):
    vals=read(ROOT/'fresnel_publication_phase3/holdout_manifest.json')['cases'][i-1]
    return dict(zip(('p','nf','ar','theta'),vals),radius='5',model=model)
def work(job):
    i,model,n,b,rho=job
    params=case(i,model)
    path=D/f'records/dev_{i}_{model}_{n}_{b}_{rho}.json'
    tag=dict(case=i,**params,n=n,bits=b,rho=rho)
    start=time.perf_counter()
    try: row=evaluate(params,n,b,rho,path)
    except Exception as exc: row=dict(verified=False,met=False,reason=repr(exc),elapsed_s=time.perf_counter()-start)
    return dict(tag,**row) if False else {**tag,**row}
def main():
    cases=[(i,m) for i in range(1,25) for m in ('Pade','Fresnel') if case(i,m)['nf']=='10']
    sweeps=[]
    jobs=[]
    for i,model in cases:
        p=case(i,model)
        entries=scan(p,512,RHOS)
        sweeps.extend(dict(case=i,**p,**e) for e in entries)
        # Full degree ladder and precision ladder, including a controlled
        # three-precision comparison at degree 96 for each geometry/model.
        for n in NS:
            for b in (BITS if n==96 else [512 if n<=80 else 1024]):
                jobs.append((i,model,n,b,'2'))
            best=min((e for e in entries if e.get('n')==n),key=lambda e:F(e['delta']))
            if best['rho']!='2': jobs.append((i,model,n,512 if n<=80 else 768 if n==96 else 1024,best['rho']))
    writecsv(D/'experiments/rho_sweep_results.csv',sweeps)
    rows=[]
    with ProcessPoolExecutor(max_workers=4,max_tasks_per_child=1) as pool:
        for row in pool.map(work,jobs):
            rows.append(row)
            writecsv(D/'experiments/development_results.csv',rows)
            if len(rows)%16==0: print(len(rows),len(jobs),flush=True)
if __name__=='__main__': main()

