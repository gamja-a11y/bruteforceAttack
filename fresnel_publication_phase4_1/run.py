"""Serial Ubuntu driver. No arguments runs the registered holdout."""
import fcntl, os, resource, signal, subprocess, sys, time
from prepare import HERE,ROOT,P4,MAN,load,digest,require,frozen,original_check
import json
sys.dont_write_bytecode=True
sys.path.insert(0,str(P4/'src'))
from common import dy,dec,F

SMOKE=dict(p='4',nf='10',ar='1',theta='0.1',model='Pade',radius='5')
def save(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf8')
    tmp.replace(path)

def choose(params,method):
    if method=='B':return '2'
    from frozen_rho_policy import RHOS,scan,choose as policy
    scans=scan(params,512,RHOS)
    # Exclude the three prohibited degrees via the frozen policy's existing API.
    choice=policy(scans,params['model'],params['theta'],{64,80,96})
    if not choice:
        reasons=[r.get('reason','') for r in scans]
        raise ValueError('no certified contour: '+' | '.join(reasons))
    require(choice[:2]==(128,1024),'Unexpected frozen degree/precision')
    return choice[2]

def expected(params,rho):
    return dict(**params,rho=rho,degree=128,bits=1024,Q=None)

def measures(cert):
    ev={k:v['evidence'] for k,v in cert['obligations'].items()}
    l,u=[dy(ev['endpoints'][k]) for k in ('lower','upper')]
    rel=2*(u-l)/(u+l) if u+l else None
    success=l>0 and u-l<=F('1e-5') and rel<=F('.01')
    return dict(M=dec(dy(ev['contour']['M'])),epsilon=dec(dy(ev['interpolation']['epsilon'])),delta=dec(dy(ev['remainder']['delta'])),lower=dec(l,False),upper=dec(u),width=dec(u-l),relative=dec(rel) if rel is not None else None,target_met=success)

def worker(folder,params,method,cap):
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    stage='contour_bound_failure'
    result={}
    try:
        from rho_certifier import certify
        from rho_independent_verifier import check
        rho=choose(params,method)
        result['rho']=rho
        stage='certification_failure'
        req=expected(params,rho)
        cert=certify(**req)
        values=measures(cert)
        cert['target_met']=values['target_met']
        stage='verification_failure'
        proof=check(cert,req)
        require(proof['verified'] is True,'Independent verifier rejected')
        save(folder/'certificate.json',cert)
        result.update(**values,verified=True,reason='' if proof['target_met'] else 'target_width_unmet',certificate_sha256=digest(folder/'certificate.json'))
    except Exception as exc:
        message=str(exc)
        if isinstance(exc,MemoryError):stage='memory_limit'
        elif any(x in message for x in ('complex domain','square-root cut')):stage='analyticity_failure'
        elif 'contour' in message.lower() and stage=='certification_failure':stage='contour_bound_failure'
        result.update(reason=stage,detail=message)
    save(folder/'worker.json',result)

def jobs(m,smoke):
    if smoke:
        for method in 'BC':yield 'smoke',SMOKE,method
        return
    for i,c in enumerate(m['cases'],1):
        for j,model in enumerate(m['models']):
            for method in ('BC' if (i+j)%2 else 'CB'):
                yield f'{i:02}_{model}',dict(**c,radius=m['radius'],model=model),method

def replay(folder,params,method,result):
    require(result['params']==params and result['method']==method and result['manifest']==digest(MAN),'Run binding mismatch')
    if result.get('verified'):
        require(digest(folder/'certificate.json')==result['certificate_sha256'],'Certificate hash mismatch')
        from rho_independent_verifier import check
        req=expected(params,choose(params,method))
        cert=load(folder/'certificate.json')
        proof=check(cert,req)
        require(proof['verified'] is True and result['rho']==req['rho'] and all(result[k]==v for k,v in measures(cert).items()),'Certificate/result mismatch')
        require(result['reason']==('' if proof['target_met'] else 'target_width_unmet'),'Outcome mismatch')
    else:
        require(not result.get('target_met') and result.get('reason'),'Unverified success')
    return result

def saved(folder,params,method):
    require((folder/'receipt').read_text().strip()==digest(folder/'result.json'),'Result hash mismatch')
    return replay(folder,params,method,load(folder/'result.json'))

def launch(folder,params,method,m):
    start=time.monotonic()
    with (folder/'stderr.log').open('xb') as err:
        proc=subprocess.Popen([sys.executable,'-B',__file__,'worker',str(folder),json.dumps(params),method,str(m['memory_mib']*2**20)],stderr=err)
        try:
            while True:
                pid,status,usage=os.wait4(proc.pid,os.WNOHANG)
                elapsed=time.monotonic()-start
                if pid:break
                if elapsed>=m['seconds']:
                    proc.kill()
                    _,status,usage=os.wait4(proc.pid,0)
                    break
                time.sleep(.02)
        except BaseException:
            if proc.poll() is None:proc.kill();proc.wait()
            raise
        proc.returncode=os.waitstatus_to_exitcode(status)
    elapsed=time.monotonic()-start
    result=dict(params=params,method=method,n=128,bits=1024,manifest=digest(MAN),wall_s=elapsed,peak_mib=usage.ru_maxrss/1024,verified=False,target_met=False,rho='2' if method=='B' else None)
    if elapsed>m['seconds']:result['reason']='timeout'
    elif proc.returncode or not (folder/'worker.json').exists():
        detail=(folder/'stderr.log').read_text(errors='replace')[-2000:]
        result.update(reason='memory_limit' if any(x in detail.lower() for x in ('out of memory','cannot allocate','memoryerror','unable to allocate')) else 'worker_exit',detail=detail,exit_code=proc.returncode)
    else:result.update(load(folder/'worker.json'))
    save(folder/'result.json',result)
    (folder/'receipt').write_text(digest(folder/'result.json'))
    return result

def compare_smoke(folder,result):
    require(result['verified'] and result['target_met'],'Smoke did not meet target')
    rho=result['rho']
    ref=load(P4/f'records/dev_9_Pade_128_1024_{rho}.json')['certificate']
    actual=load(folder/'certificate.json')
    req=expected(SMOKE,rho)
    require(actual['inputs']==req==ref['inputs'],'Smoke reference inputs')
    def ends(c):
        e=c['obligations']['endpoints']['evidence']
        return [dy(e[k]) for k in ('lower','upper')]
    a,b=ends(actual),ends(ref)
    require(max(a[0],b[0])<=min(a[1],b[1]) and ref['target_met'] is actual['target_met'],'Smoke interval mismatch')
    return dict(rho=rho,reference_overlap=True,identical=(a==b))

def main(smoke):
    m=frozen();original_check()
    base=HERE/('smoke' if smoke else 'runs')
    base.mkdir(exist_ok=True)
    with (HERE/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not smoke:
            for label,p,method in jobs(m,True):
                d=HERE/'smoke'/f'{label}_{method}'
                compare_smoke(d,saved(d,p,method))
        comparisons={}
        for label,p,method in jobs(m,smoke):
            folder=base/f'{label}_{method}'
            if (folder/'receipt').is_file():
                result=saved(folder,p,method);action='reused'
            else:
                if folder.exists():
                    folder.rename(base/f'{folder.name}.interrupted.{time.time_ns()}')
                folder.mkdir()
                result=launch(folder,p,method,m);action='finished'
            msg=f"{label} {method} {action} {result['reason'] or 'PASS'} {result['wall_s']:.2f}s"
            print(msg,flush=True)
            with (base/'progress.log').open('a') as f:print(msg,file=f,flush=True)
            if smoke:comparisons[method]=compare_smoke(folder,result)
        if smoke:save(base/'comparison.json',comparisons)

if __name__=='__main__':
    if sys.argv[1:2]==['worker']:
        from pathlib import Path
        worker(Path(sys.argv[2]),json.loads(sys.argv[3]),sys.argv[4],int(sys.argv[5]))
    elif sys.argv[1:] in ([],['--smoke']):main(bool(sys.argv[1:]))
    else:raise SystemExit('usage: python -B run.py [--smoke]')

