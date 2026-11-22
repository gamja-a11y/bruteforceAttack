import ctypes, hashlib, json, sys, time
from ctypes import wintypes as w
from fractions import Fraction
from pathlib import Path
import frozen_selection_rule as rule
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'fresnel_publication_phase2'))
import adaptive_certification as pilot

def limit(mib):
    class Basic(ctypes.Structure):
        _fields_=[('times',ctypes.c_int64*2),('flags',w.DWORD),('minws',ctypes.c_size_t),('maxws',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('schedule',w.DWORD)]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',ctypes.c_uint64*6),('process',ctypes.c_size_t),('job',ctypes.c_size_t),('peakprocess',ctypes.c_size_t),('peakjob',ctypes.c_size_t)]
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.restype=w.HANDLE
    handle=k.CreateJobObjectW(None,None)
    data=Extended()
    data.basic.flags=0x100
    data.process=mib*2**20
    if not handle or not k.SetInformationJobObject(w.HANDLE(handle),9,ctypes.byref(data),ctypes.sizeof(data)) or not k.AssignProcessToJobObject(w.HANDLE(handle),w.HANDLE(-1)):
        raise ctypes.WinError(ctypes.get_last_error())
    return handle

def stats(rec):
    ev={k:v['evidence'] for k,v in rec['obligations'].items()}
    r=pilot.rational
    lo,hi=map(r,ev['gram']['G'])
    return hi-lo,r(ev['remainder']['delta']),r(ev['contour']['M']),r(ev['interpolation']['epsilon'])

def new_search(params,dest,m):
    dest.mkdir()
    rt,at=map(Fraction,(m['relative'],m['absolute']))
    n,b=rule.start(params)
    start=time.perf_counter()
    rows=[]
    best=None
    common=None
    reason='candidate limits exhausted'
    status='no_verified_interval'
    while True:
        left=m['seconds_per_method']-(time.perf_counter()-start)
        if left<=0:
            reason='deadline'
            break
        res=pilot.candidate(params,n,b,left)
        rec=res.pop('certificate',None)
        ver=res.pop('verification',None)
        row=dict(degree=n,bits=b,**res)
        rows.append(row)
        if not rec or ver['verified'] is not True:
            reason=res['reason'] or 'verification failed'
            break
        name=f'n{n}_b{b}.json'
        (dest/name).write_text(json.dumps(dict(certificate=rec,verification=ver)))
        l,u,width,rel,rok,aok=pilot.metric(rec,rt,at)
        common=(l,u) if common is None else (max(l,common[0]),min(u,common[1]))
        if common[0]>common[1]:
            best=None
            status='integrity_failure'
            reason='disjoint verified intervals'
            break
        row.update(lower=pilot.decimal_out(l,False),upper=pilot.decimal_out(u,True),absolute_width=pilot.decimal_out(width,True),relative_width=pilot.decimal_out(rel,True) if rel is not None else None,certificate=name,relative_met=rok,absolute_met=aok)
        if best is None or width<best_width:
            best,best_width=row,width
        status='verified_target_unmet'
        if rok and aok:
            best=row
            status,reason='target_met','requested targets met'
            break
        g,d,M,_=stats(rec)
        choice=rule.advance(n,b,l,g,d,M,Fraction(params['nf']),at,rt)
        if choice is None: break
        n,b=choice
    final=dict(**params,status=status,reason=reason,attempts=rows,best=best,seconds=time.perf_counter()-start,peak_worker_mib=max((r.get('peak_worker_mib',0) for r in rows),default=0))
    (dest/'search.json').write_text(json.dumps(final))

def main():
    m=json.loads((D/'holdout_manifest.json').read_text())
    for name,h in m['sha256'].items():
        if hashlib.sha256((D.parent/name).read_bytes()).hexdigest()!=h: raise ValueError(name)
    limit(m['process_mib'])
    runs=D/'runs'
    runs.mkdir(exist_ok=True)
    for i,c in enumerate(m['cases']):
        for j,model in enumerate(m['models']):
            params=dict(zip(m['order'],c),radius='5',model=model)
            k=(2*i+j)%3
            for method in ('ABC'*2)[k:k+3]:
                dest=runs/f'{i+1:02}_{model}_{method}'
                if (dest/'search.json').exists(): continue
                if method=='B': new_search(params,dest,m)
                else: pilot.search(params,dest,target_relative_width=m['relative'],target_absolute_width=m['absolute'],maximum_degree=64,maximum_bits=512,timeout_seconds=30,fixed=(64,512) if method=='C' else None)
            print(i+1,model,flush=True)

if __name__=='__main__': main()
