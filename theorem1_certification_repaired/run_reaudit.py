"""Run baseline, adversarial, historical and optimized-mode checks from this folder."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from flint import arb as A, ctx
import repaired_certification as r
import repaired_independent_verification as verifier

HERE=Path(__file__).resolve().parent
OLD=HERE/"previous_audit"
CASES=[dict(model=m) for m in ("Pade","Fresnel")]
CASES.extend(dict(p=p,ar=ar,nf="1",theta="0.1",degree=8,bits=256,model="Fresnel")
             for p,ar in [(1,1),(2,1),(2,4),(2,"0.25"),(12,1),(12,2),(12,4),(16,4)])
CASES.extend([dict(p=2,ar="0.25",theta="0.1",Q="0.0625000000000000001",degree=8,bits=256),
              dict(p="1.5",ar="0.5",radius="3",nf="0.3",degree=8,bits=256),
              dict(model="Fresnel",degree=4,bits=256),dict(model="Pade",degree=32,bits=80),
              dict(model="Pade",degree=96,bits=1024)])


def evidence(cert,key):
    return cert["obligations"][key]["evidence"]


def need(ok,reason):
    if not ok:
        raise RuntimeError(reason)


def negative_tests():
    cases=[]
    def reject(label,operation):
        try:
            operation()
        except ValueError as exc:
            cases.append(dict(test=label,result="PASS",detail=str(exc),optimized=sys.flags.optimize))
        else:
            cases.append(dict(test=label,result="FAIL",detail="accepted invalid input or evidence",optimized=sys.flags.optimize))
    for k,vals in dict(nf=[0,-10,"NaN","Infinity"],theta=[0,"NaN",-1],ar=[0,-1,"NaN"],
                       p=[0,-1,"Infinity","NaN",True,[]],radius=[0,-1,"NaN"],
                       degree=[0,-1,1.5,True,"8"],bits=[0,1,32,True,"256"],
                       model=["Fresnle","padee","",None],Q=[0,-1,"NaN","Infinity"]).items():
        for val in vals:
            reject(f"{k}={val!r}",lambda k=k,val=val:r.certify(**({"degree":4,"bits":128}|{k:val})))
    reject("undersized prescribed Q",lambda:r.certify(p=2,ar="0.25",theta="0.1",Q="0.0624999999",degree=4))
    reject("ellipse includes cut",lambda:r.certify(Q="8",degree=4))
    for args in [(A(-1),A(0),A(0)),(A(1),A(-1),A(0)),(A(1),A(0),A(-1)),(A("nan"),A(0),A(0))]:
        reject("invalid Gram/remainders "+str(args),lambda args=args:r.endpoints(*args))
    with patch.object(r,"disk_gram",return_value=A(-1)):
        reject("injected negative Gram upper through certifier",lambda:r.certify(degree=4))
    with patch.object(r,"endpoints",return_value=(A(2),A(1))):
        reject("injected reversed endpoints through certifier",lambda:r.certify(degree=4))
    valid=r.certify(p=2,nf=1,theta="0.1",degree=6,bits=128,model="Fresnel")
    verifier.verify(valid)
    mods=[
        (("generator_sha256",),"0"*64),
        (("inputs","model"),"Pade"),(("inputs","theta"),"0.2"),
        (("obligations","identity","passed"),False),
        (("obligations","geometry","evidence","Q"),["1",-40]),
        (("obligations","analyticity","evidence","branch"),[["1",5],["1",6]]),
        (("obligations","contour","evidence","arc_uppers"),[]),
        (("obligations","contour","evidence","M"),["0",0]),
        (("obligations","interpolation","evidence","epsilon"),["0",0]),
        (("obligations","source","evidence","mu00"),[["0",0],["0",0]]),
        (("obligations","gram","evidence","G"),[["0",0],["0",0]]),
        (("obligations","gram","evidence","G"),[["1",1],["1",0]]),
        (("obligations","remainder","evidence","delta"),["0",0]),
        (("obligations","endpoints","evidence","upper"),["0",0])]
    for path,val in mods:
        damaged=deepcopy(valid)
        pos=damaged
        for k in path[:-1]:
            pos=pos[k]
        pos[path[-1]]=val
        reject("altered certificate "+"/".join(path),lambda:verifier.verify(damaged))
    return cases


def historical(records,baseline):
    """Run the historical test bodies unchanged, rebinding their public interfaces."""
    sys.path.insert(0,str(OLD))
    import independent_counterexample_tests as audit
    def setup(A,case):
        p,n,t,ar=map(A,case)
        ax,ay,z,w,K,m,Q=r.geometry(p,n,t,ar,A(5))
        return p,ax,ay,z,w,K,m,Q
    def old_worker(case,n,kind):
        try:
            cert=r.certify(p=case[0],nf=case[1],theta=case[2],degree=n,bits=max(256,8*n),model=kind)
            return dict(status=cert["status"],kind=kind)
        except ValueError as exc:
            return dict(status="unverified",kind=kind,reason=str(exc))
    def old_v6(case,n,bits):
        try:
            cert=r.certify(p=case[0],nf=case[1],theta=case[2],degree=n,bits=bits)
            ends=evidence(cert,"endpoints")
            return dict(status=cert["status"],lower=ends["lower"],upper=ends["upper"])
        except ValueError as exc:
            return dict(status="unverified",lower=None,upper=None,reason=str(exc))
    def old_interp(A,C,P,K,Q,n,kind):
        try:
            return r.spectrum(K,Q,n,kind)
        except ValueError as exc:
            raise ArithmeticError(str(exc)) from exc
    # Historical tests expect ArithmeticError from the kernel, and a failure
    # row from a worker. The actual public API raises ValueError.
    native=r.residual
    def old_res(q,k,kind,C):
        try:
            return native(q,k,kind,C)
        except ValueError as exc:
            raise ArithmeticError(str(exc)) from exc
    proxy=SimpleNamespace(worker=old_worker,residual=old_res,interpolate=old_interp,disk_gram=r.disk_gram)
    def worker_with_gram(case,n,kind):
        with patch.object(r,"disk_gram",proxy.disk_gram):
            return old_worker(case,n,kind)
    proxy.worker=worker_with_gram
    v=SimpleNamespace(certify=old_v6,kernel=lambda q,k:native(q,k,"Pade"),
                      aperture_transform=lambda A,C,c,p,a,z,Q:r.source_transform(c,p,a,a,z,Q,A(5))[0],
                      disk_gram=lambda A,R,terms:r.disk_gram(terms,max(i+j for i,j in terms)))
    with ExitStack() as stack:
        stack.enter_context(patch.object(audit,"std",proxy))
        stack.enter_context(patch.object(audit,"v",v))
        stack.enter_context(patch.object(audit.ind,"setup",setup))
        stack.enter_context(patch.object(audit.ind,"certify",lambda c,n,b:r.certify(
            p=c[0],nf=c[1],theta=c[2],ar=c[3],degree=n,bits=b)))
        q,k=audit.kernel_tests()
        audit.polynomial_tests()
        audit.adversarial(q,k)
        comparisons=[]
        for cert in records:
            d=cert["inputs"]
            ends=evidence(cert,"endpoints")
            reference=next((row for row in baseline if d["p"]=="12" and d["ar"]=="1"
                      and row["kind"]==d["model"] and d["theta"]=="0.3" and d["nf"]=="10"),None)
            overlaps=None
            if reference:
                a,b=map(verifier.point,reference["original_dyadics"])
                overlaps=bool(verifier.point(ends["lower"])<=b and verifier.point(ends["upper"])>=a)
            comparisons.append(dict(p=d["p"],NF=d["nf"],theta=d["theta"],AR=d["ar"],
                          engine="repaired",kind=d["model"],degree=d["degree"],bits=d["bits"],
                          **ends,archive_overlap=overlaps,endpoints_display=str(ends)))
        with patch.object(audit,"RUNS",comparisons):
            audit.comparison_tests()
        precise=[row for row in comparisons if row["kind"]=="Pade" and row["degree"]==96]
        need(len(precise)==2,"missing precision comparison")
        a,b=precise
        audit.check("repaired fixed degree 768->1024",
              verifier.point(a["lower"])<=verifier.point(b["upper"]) and
              verifier.point(b["lower"])<=verifier.point(a["upper"]),"common point in fixed-degree intervals","G")
    return [dict(test="previous audit / "+a["test"],result="PASS" if a["result"] in ("PASS","DETECTED") else "FAIL",
                 detail=a.get("observation",a.get("evidence","")),optimized=0,
                 category="negative" if a in audit.MUTATIONS and not a["test"].startswith("undersized Q") else "regression")
            for a in audit.CHECKS+audit.MUTATIONS]


def dump_json(path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def dump_csv(path,rows):
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--negative-only",action="store_true")
    ap.add_argument("--out",type=Path,default=HERE)
    args=ap.parse_args()
    if args.negative_only:
        print(json.dumps(negative_tests()))
        return
    out=args.out.resolve()
    if out!=HERE:
        out.mkdir(parents=True,exist_ok=False)
    names=["corrected_certificate.json","certificate_verification.json","regression_results.csv",
           "negative_tests.csv","final_integrity_report.md"]
    need(not any((out/f).exists() for f in names),"choose a new output directory")
    certs,checks,reg=[],[],[]
    baseline=json.loads((OLD/"final_certificate.json").read_text(encoding="utf-8"))["baseline"]
    for index,params in enumerate(CASES):
        print("Certification and independent replay",index+1,params,flush=True)
        rec=r.certify(**params)
        certs.append(rec)
        expected=dict(p="12",nf="10",theta="0.3",ar="1",radius="5",degree=96,bits=768,model="Pade",Q=None)
        expected.update({k:str(v) if k in ("p","nf","theta","ar","radius","Q") else v for k,v in params.items()})
        verified=verifier.verify(rec,expected)
        checks.append(verified)
        with ctx.workprec(2048):
            geo=evidence(rec,"geometry")
            Q=verifier.point(geo["Q"])
            ends=evidence(rec,"endpoints")
            lo,hi=[verifier.point(ends[k]) for k in ("lower","upper")]
            detail="10 stages independently replayed"
            if index<2:
                matches=[item for item in baseline if item["kind"]==rec["inputs"]["model"]]
                need(bool(matches),"missing archival interval")
                for item in matches:
                    oldlo,oldhi=map(verifier.point,item["original_dyadics"])
                    if lo>oldhi or hi<oldlo:
                        dump_json(out/"baseline_failure.json",dict(new=rec,old=item))
                        raise ValueError("disjoint baseline intervals; comparison aborted")
                detail+=f"; overlaps {len(matches)} archived interval(s)"
            if params.get("ar")=="0.25":
                need(Q>=A(1)/16,"p=2 AR=.25 continuous maximum")
            reg.append(dict(test="physical certificate "+str(index+1),result="PASS",detail=detail,
                       **rec["inputs"],qmax_upper=float(verifier.point(geo["qmax"][1])),
                       actual_Q=float(Q),lower_display=float(lo),upper_display=float(hi)))
    oldrows=historical(certs,baseline)
    reg.extend(row for row in oldrows if row["category"]=="regression")
    neg=negative_tests()
    proc=subprocess.run([sys.executable,"-B","-O",str(Path(__file__).resolve()),"--negative-only"],
                        capture_output=True,text=True,check=True)
    optimized=json.loads(proc.stdout)
    need(bool(optimized) and all(x["optimized"]==1 for x in optimized),"optimized interpreter not exercised")
    neg+=optimized
    neg.extend(row for row in oldrows if row["category"]=="negative")
    dump_csv(out/names[2],reg)
    dump_csv(out/names[3],neg)
    dump_json(out/names[0],certs)
    dump_json(out/names[1],dict(records=checks,certificate_sha256=sha256((out/names[0]).read_bytes()).hexdigest(),
                               independent_verifier_sha256=sha256((HERE/"repaired_independent_verification.py").read_bytes()).hexdigest()))
    failed=[row["test"] for row in reg+neg if row["result"]!="PASS"]
    grade="FAIL" if failed else "PASS"
    lines=["# Theorem 1 재감사","",f"Mathematical theorem: PASS. Implementation integrity: {grade}.",
           "",f"정상 인증 {len(certs)}건, 이전 감사 및 구간 비교 {len(oldrows)}건, 추가 거부·변조 검사 일반 {len(optimized)}건 및 -O {len(optimized)}건.",
           "기준 조건은 p=12, NF=10, θ=0.3, AR=1, R=5, n=96, 768비트이다. 다음 숫자는 표시용이다.",
           "", "|잔차|하한|상한|기존 구간과의 교집합|","|---|---:|---:|---|"]
    for row in reg[:2]:
        lines.append(f"|{row['model']}|{row['lower_display']:.16g}|{row['upper_display']:.16g}|확인|")
    lines += ["","|반례|수정 전|수정 후|","|---|---|---|",
              f"|p=1, AR=1|Q=0.01953125로 과소평가하고 인증|q_max≈{reg[2]['qmax_upper']:.11g}, Q={reg[2]['actual_Q']:.12g}, 양측 인증|",
              f"|p=2, AR=0.25|Q=0.0107421875로 과소평가하고 인증|q_max=0.0625, Q={reg[5]['actual_Q']:.12g}, 양측 인증|",
              "|Fresnle|Padé를 계산하고 요청한 이름으로 인증|모델 검사에서 ValueError|",
              "|NF=-10|음수 상한과 역전된 구간을 인증|물리 매개변수 검사에서 ValueError|",
              "",f"실패한 검사: {', '.join(failed) or '없음'}.",
              "판정은 새 인증 진입점과 명시한 스칼라 회절 문제에 적용된다. 저차수의 넓은 구간도 유효할 수 있다.",
              "독립 검증기는 512개 호, 보간 상한, 별도 DCT-I/모멘트/Gram 계산 및 이진 끝점을 확인했다.",
              "신뢰 기반과 미검증 범위는 remaining_proof_obligations.md에 기록했다."]
    with (out/names[4]).open("x",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print("Implementation integrity:",grade,flush=True)
    need(not failed,"failed tests: "+str(failed))


if __name__=="__main__":
    main()

