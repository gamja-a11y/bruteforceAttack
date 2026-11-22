"""Tolerance-driven search over the frozen continuous-disk residual certifier."""
import argparse
import ctypes
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"
DEGREES = (16, 24, 32, 48, 64, 80, 96, 128, 160)
BITS = (256, 512, 768, 1024, 1536)
HASHES = {"repaired_certification.py":"954f078c856ef6ec6c15a0a1a383714ae4ad025250c5ecbf64904e6f6a75e8db",
          "repaired_independent_verification.py":"219f26d04908572d87c9524264e9ae6aa51dee5d51a342618f040552e8d9227a"}


def check_sources():
    for name, ref in HASHES.items():
        if hashlib.sha256((FROZEN/name).read_bytes()).hexdigest()!=ref:
            raise ValueError("Frozen source hash mismatch: "+name)


def peak_mib():
    if sys.platform!="win32":
        import resource
        scale=1 if sys.platform=="darwin" else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale/2**20
    from ctypes import wintypes as w
    class Counters(ctypes.Structure):
        _fields_=[("cb",w.DWORD),("faults",w.DWORD)]+[(k,ctypes.c_size_t) for k in
             ("peak","ws","pp","p","np","n","page","peak_page")]
    proc=ctypes.windll.kernel32.GetCurrentProcess
    proc.restype=w.HANDLE
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=(w.HANDLE,ctypes.POINTER(Counters),w.DWORD)
    c=Counters()
    c.cb=ctypes.sizeof(c)
    if not get(proc(),ctypes.byref(c),c.cb):
        raise OSError("peak working set unavailable")
    return c.peak/2**20


def rational(pair):
    return Fraction(int(pair[0]))*Fraction(2)**pair[1]


def decimal_out(x,upper):
    with localcontext() as ctx:
        ctx.prec=22
        ctx.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        return str(Decimal(x.numerator)/Decimal(x.denominator))


def worker(request):
    check_sources()
    sys.path.insert(0,str(FROZEN))
    import repaired_certification as issuer
    import repaired_independent_verification as verifier
    start=time.perf_counter()
    rec=issuer.certify(**request)
    issue=time.perf_counter()-start
    expected={**request,"Q":None}
    for k in ("p","nf","theta","ar","radius"):
        expected[k]=issuer.scalar(request[k],k)
    verified=verifier.verify(rec,expected)
    return dict(certificate=rec,verification=verified,issue_s=issue,
                verify_s=time.perf_counter()-start-issue,peak_worker_mib=peak_mib())


def candidate(params,n,b,limit):
    start=time.perf_counter()
    req=dict(**params,degree=n,bits=b)
    try:
        run=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker"],
                           input=json.dumps(req),capture_output=True,text=True,timeout=limit)
    except subprocess.TimeoutExpired:
        return dict(status="timeout",reason="wall-clock limit during candidate",seconds=time.perf_counter()-start)
    if run.returncode:
        return dict(status="rejected",reason=run.stderr.strip().splitlines()[-1],seconds=time.perf_counter()-start)
    result=json.loads(run.stdout)
    result.update(status="verified",reason="",seconds=time.perf_counter()-start)
    return result


def metric(rec,rt,at):
    ev=rec["obligations"]
    ends=ev["endpoints"]["evidence"]
    l,u=(rational(ends[k]) for k in ("lower","upper"))
    w=u-l
    rel=2*w/(u+l) if u+l else None
    rel_ok=(l>0 and rel is not None and rel<=rt) if rt is not None else None
    abs_ok=(w<=at) if at is not None else None
    return l,u,w,rel,rel_ok,abs_ok


def target(value):
    if value is None:
        return None
    f=Fraction(str(value))
    if f<=0:
        raise ValueError("Width targets must be finite and positive")
    return f


def search(params,dest,*,target_relative_width=None,target_absolute_width=None,
           maximum_degree=160,maximum_bits=1536,timeout_seconds=120,fixed=None):
    rt,at=target(target_relative_width),target(target_absolute_width)
    if rt is None and at is None:
        raise ValueError("Supply at least one width target")
    if not math.isfinite(timeout_seconds) or timeout_seconds<=0:
        raise ValueError("timeout must be finite and positive")
    if type(maximum_degree)!=int or type(maximum_bits)!=int or maximum_degree<1 or maximum_bits<1:
        raise ValueError("candidate limits must be positive integers")
    ns=[x for x in DEGREES if x<=maximum_degree]
    bs=[x for x in BITS if x<=maximum_bits]
    if fixed:
        if fixed[0] not in ns or fixed[1] not in bs:
            raise ValueError("fixed candidate exceeds limits")
        ns,bs=[fixed[0]],[fixed[1]]
    check_sources()
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    best=None
    score=None
    common=None
    attempts=[]
    reason="candidate limits exhausted"
    for n in ns:
        for b in bs:
            remaining=timeout_seconds-(time.perf_counter()-start)
            if remaining<=0:
                reason="timeout before next candidate"
                break
            result=candidate(params,n,b,remaining)
            rec=result.pop("certificate",None)
            verified=result.pop("verification",None)
            row=dict(degree=n,bits=b,**result)
            attempts.append(row)
            if result["status"]=="timeout":
                reason=result["reason"]
                break
            if rec is None:
                continue
            l,u,w,rel,rel_ok,abs_ok=metric(rec,rt,at)
            payload=dict(certificate=rec,verification=verified)
            file=dest/f"n{n}_b{b}.json"
            file.write_text(json.dumps(payload,separators=(",",":")),encoding="utf8")
            row.update(lower=decimal_out(l,False),upper=decimal_out(u,True),absolute_width=decimal_out(w,True),
                       relative_width=None if rel is None else decimal_out(rel,True),relative_met=rel_ok,
                       absolute_met=abs_ok,certificate=file.name)
            common=(l,u) if common is None else (max(l,common[0]),min(u,common[1]))
            if common[0]>common[1]:
                reason="integrity failure: verified intervals disjoint"
                best=None
                break
            penalties=[]
            if at is not None:
                penalties.append(w/at)
            if rt is not None:
                penalties.append((rel if rel is not None and l>0 else Fraction(2))/rt)
            rank=(max(penalties),w)
            met=(rel_ok is not False and abs_ok is not False)
            if best is None or rank<score or met:
                best,score=row,rank
            if met:
                reason="requested targets met"
                break
            # The analytic remainder is insensitive to working precision.
            # Spend additional bits at this degree only if rounding contributes
            # substantially to the observed width.
            delta=rational(rec["obligations"]["remainder"]["evidence"]["delta"])
            if w<=4*delta:
                break
        if reason!="candidate limits exhausted":
            break
    elapsed=time.perf_counter()-start
    status=("integrity_failure" if reason.startswith("integrity") else "target_met" if reason=="requested targets met"
            else "verified_target_unmet" if best else "no_verified_interval")
    if not best and attempts and all(r["status"]=="rejected" for r in attempts):
        reason=attempts[-1]["reason"]
    final=dict(**params,target_relative_width=target_relative_width,target_absolute_width=target_absolute_width,
               maximum_degree=maximum_degree,maximum_bits=maximum_bits,timeout_seconds=timeout_seconds,
               status=status,reason=reason,seconds=elapsed,peak_worker_mib=max((r["peak_worker_mib"] for r in attempts
                   if "peak_worker_mib" in r),default=None),best=best,attempts=attempts)
    (dest/"search.json").write_text(json.dumps(final,indent=2),encoding="utf8")
    return final


def main():
    if sys.argv[1:]==["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin)),allow_nan=False))
        return
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",choices=("Pade","Fresnel"),required=True)
    for k,default in dict(p="12",nf="10",theta="0.3",ar="1",radius="5").items():
        ap.add_argument("--"+k,"--"+k.upper(),default=default)
    for k in ("target_relative_width","target_absolute_width"):
        ap.add_argument("--"+k,"--"+k.replace("_","-"))
    for k,default in dict(maximum_degree=160,maximum_bits=1536,timeout_seconds=120).items():
        ap.add_argument("--"+k,"--"+k.replace("_","-"),type=float if k=="timeout_seconds" else int,default=default)
    ap.add_argument("--out",type=Path,required=True)
    args=vars(ap.parse_args())
    physical={k:args.pop(k) for k in ("p","nf","theta","ar","radius","model")}
    report=search(physical,args.pop("out"),**args)
    print(json.dumps({k:v for k,v in report.items() if k!="attempts"},indent=2))


if __name__=="__main__":
    main()

