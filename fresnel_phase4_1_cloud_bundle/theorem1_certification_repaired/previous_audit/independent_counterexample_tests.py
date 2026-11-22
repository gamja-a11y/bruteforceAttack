"""Replay the frozen implementations and audit their mathematical contracts."""
import argparse
import ast
import csv
import hashlib
import importlib.metadata as md
import json
from math import comb
from pathlib import Path
import platform
import random
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
INPUT = HERE / "inputs"
sys.path.insert(0, str(INPUT))
import flint
from flint import acb as C, arb as A, acb_poly, arb_poly, ctx
import mpmath as mp
import sympy as sp
import fresnel_certified_bound_v6 as v
import fresnel_independent_audit as ind
import fresnel_standard_certification as std

BASE = ("12", "10", "0.3")
MODELS = ("Pade", "Fresnel")
CHECKS, MUTATIONS, RUNS = [], [], []
FILES = ["mathematical_to_code_matrix.csv", "proof_obligations.json",
         "mutation_test_results.csv", "interval_enclosure_tests.csv",
         "certification_reproduction.json", "final_certificate.json", "overall_audit_summary.md"]
MODULES = {"v6": v, "ind": ind, "std": std}
ARCHIVES = {
    "v6": ("certified_bound_v6_verified/certificate.json", "records"),
    "ind": ("fresnel_independent_audit_verified/audit_report.json", "certificates"),
    "std": ("standard_fresnel_verified/standard_fresnel_certificate.json", "records")}


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def table(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def check(name, result, detail, stage, status="interval-checked"):
    CHECKS.append(dict(test=name, result="PASS" if result else "FAIL", stage=stage,
                       evidence=str(detail), method=status))
    return bool(result)


def mutation(name, mechanism, observed, caught, legacy=False):
    MUTATIONS.append(dict(test=name, mechanism=mechanism, observation=str(observed),
                          result="DETECTED" if caught else "FALSE ACCEPT",
                          scope="original callable" if legacy else "independent oracle"))


def binary(x):
    m, e = x.man_exp()
    return [str(m), int(e)]


def ball(x):
    require(x.is_finite(), "nonfinite ball")
    return dict(lower=binary(x.lower()), upper=binary(x.upper()), display=str(x))


def exact(pair):
    return A((int(pair[0]), pair[1]))


def outer(gram, error):
    require(gram.is_finite() and gram.upper() >= 0, "invalid Gram upper endpoint")
    require(error.is_finite() and error >= 0, "invalid remainder")
    root = gram.lower().sqrt() if gram.lower() > 0 else A(0)
    lower = (root-error.upper()).lower()
    upper = (gram.upper().sqrt()+error.upper()).upper()
    return max(A(0), lower), upper


def q_geometry(p, ax, ay, z):
    if p <= 2:
        radius = ax if ax > ay else ay if ay > ax else ax.union(ay)
    else:
        t = 2*p/(p-2)
        radius = (ax**t + ay**t)**(1/t)
    return ((radius+5)/z)**2


def archive(engine, case, degree, kind):
    path, key = ARCHIVES[engine]
    items = json.loads((INPUT/path).read_text(encoding="utf-8"))[key]
    return next((r for r in items if
                 (r["p"],r["nf"],r["theta"],str(r.get("ar",1)),r["degree"],r.get("kind","Pade"))
                 == (*case,degree,kind)), None)


def replay(engine, kind, degree, bits, case=(*BASE,"1")):
    """Read locals only on return; execute the actual archived functions."""
    module = MODULES[engine]
    allowed = {f.__code__ for m in MODULES.values() for f in vars(m).values()
               if hasattr(f,"__code__") and f.__module__ in {x.__name__ for x in MODULES.values()}}
    seen, state = {}, {}
    def trace(frame, event, arg):
        if event == "call" and frame.f_code in allowed:
            frame.f_trace_lines = False
            name = frame.f_globals["__name__"]+"."+frame.f_code.co_name
            seen[name] = seen.get(name,0)+1
            return capture
    def capture(frame, event, arg):
        if event == "return" and arg is not None:
            loc, name = frame.f_locals, frame.f_code.co_name
            if name in ("interpolate","interpolation","polynomial"):
                state.update(M=loc["M"], eps=arg[1])
            if name == "certify":
                state.update(G=loc["gram"] if "gram" in loc else loc["squared"],
                             slack=loc["slack"] if "slack" in loc else loc["e"],
                             lo=arg["lower"], hi=arg["upper"])
                if engine=="v6":
                    state["ends"]=loc["final_low"],loc["final"]
                else:
                    state["ends"]=(loc["lo"].lower() if loc["lo"]>0 else A(0),loc["hi"].upper())
        return capture
    print("Replay",engine,kind,case,degree,bits,flush=True)
    sys.settrace(trace)
    try:
        if engine == "std":
            require(bits == max(256,8*degree) and case[3]=="1","standard worker contract")
            row = std.certify(case[:3],degree,kind)
        elif engine == "v6":
            row = v.certify(case[:3],degree,bits)
        else:
            row = ind.certify(case,degree,bits)
    finally:
        sys.settrace(None)
    p,ax,ay,z,lam,k,_,Q = ind.setup(A,case)
    q = q_geometry(p,ax,ay,z)
    h = Q/2
    rect = C(h+A(0,1)*5*h/4,A(0,1)*3*h/4)
    guards = {"Re(1+q)":ball((1+rect).real)}
    if kind == "Pade":
        for key,value in [("phase denominator",1+3*rect/4+rect**2/16),
                          ("amplitude denominator",1+7*rect/4+35*rect**2/48)]:
            guards[key]=ball(value.real)
    area = 4*ax*ay*(1+1/p).gamma()**2/(1+2/p).gamma()
    lo,hi = outer(state["G"],state["slack"])
    # Exact dyadic endpoints; printed Arb values are themselves enclosures.
    rawlo,rawhi = A(row["lower"]), A(row["upper"])
    same = rawlo.overlaps(lo) and rawhi.overlaps(hi)
    # The independent implementation may form the same rigorous bound in a
    # different rounding order. Compare its reported interval with ours.
    require(all(exact(g["lower"])>0 for g in guards.values())
            and q<Q and area.overlaps(100*A.pi()),"physical/analytic guard failed")
    require(Q==A(row.get("q_upper",row.get("q_interval_max"))) and hi>=lo,"domain or interval")
    orig_lo,orig_hi=state["ends"]
    check(f"outward serialization {engine} {kind} {degree} {bits}",
          rawlo.contains(orig_lo) and rawhi.contains(orig_hi) and orig_lo>=0 and orig_hi>=orig_lo,
          f"original endpoints encoded in Arb strings; exact dyadics retained","G")
    last=None
    max_arc=A(0)
    for j in range(512):
        ang=(2*A.pi()*j/512).union(2*A.pi()*(j+1)/512)
        require(last is None or last.overlaps(ang),"gap between adjacent arcs")
        arc=C(h+5*h*ang.cos()/4,3*h*ang.sin()/4)
        val=(std.residual(arc,k,kind,C) if engine=="std" else
             v.kernel(arc,k) if engine=="v6" else ind.arb_kernel(arc,k,C)).abs_upper()
        max_arc=max(max_arc,val)
        last=ang
    require(state["M"]>=max_arc and state["eps"]>=4*state["M"]/2**degree,
            "boundary maximum or interpolation remainder not outward")
    target = archive(engine,case,degree,kind)
    overlap = None
    if target and "lower" in target:
        overlap = bool(lo < A(target["upper"]).lower() and hi > A(target["lower"]).upper())
    r=dict(engine=engine,kind=kind,p=case[0],NF=case[1],theta=case[2],AR=case[3],
           roi_radius=5,illumination=1,degree=degree,bits=bits,ax=ball(ax),ay=ball(ay),
           z=ball(z),wavelength=ball(lam),K=ball(k),area=ball(area),q_max=ball(q),Q=ball(Q),
           rho=2,arc_count=512,rectangle=dict(real=ball(rect.real),imag=ball(rect.imag)),
           guard_intervals=guards,M=ball(state["M"]),largest_replayed_arc_upper=ball(max_arc),epsilon=ball(state["eps"]),
           delta=ball(state["slack"]),G=ball(state["G"]),lower=binary(lo),upper=binary(hi),
           endpoints_display=[str(lo),str(hi)],original_endpoints=[row["lower"],row["upper"]],
           original_dyadics=[binary(orig_lo),binary(orig_hi)],
           endpoints_overlap_recomputed=same,archive_overlap=overlap,call_counts=seen,
           status="interval-certified",seconds=row.get("seconds"))
    require(rawlo.lower()<=hi and rawhi.upper()>=lo,"disjoint original and audit")
    RUNS.append(r)
    return r


def kernel_tests():
    ctx.prec=768
    p,ax,ay,z,lam,K,qmax,Q=ind.setup(A,(*BASE,"1"))
    t,k=sp.symbols("t k", positive=True)
    q=t*t-1
    psi=(q/2+q**2/4)/(1+3*q/4+q*q/16)
    da=t**-3-(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48)
    eqs=[t-1-q/(1+t),t-1-q/2+q*q/(2*(1+t)**2),
         1/(1+q)-1+q/(1+q),t-1-psi-(t-1)**5/(16+12*q+q*q),
         da-(t-1)**5*(t*t+5*t+1)/(t**3*(48+84*q+35*q*q))]
    a,b,X,Y=sp.symbols("a b X Y")
    exp_identity=sp.expand(Y*((a-b)+a*(X/Y-1))-(a*X-b*Y))
    check("symbolic residual identities",all(sp.cancel(e)==0 for e in eqs) and exp_identity==0,
          "Five rational identities; X=exp(iKphi), Y=exp(iKpsi) nonzero, expm1=exp-1","B","symbolic")
    for label,qr in [("zero",0),("small","1e-60"),("physical max",qmax),("near Q",Q-A(2)**-200),("Q",Q)]:
        q=C(qr); t=(1+q).sqrt(); i=C(0,1)
        amp=1/(1+q)+i/(K*t**3)
        rs=amp*(i*K*(t-1)).exp()
        for kind in MODELS:
            other=(i*K*q/2).exp() if kind=="Fresnel" else (
                1/(1+q)+i*(1+q/4-q*q/48)/(K*(1+7*q/4+35*q*q/48))
                )*(i*K*(q/2+q*q/4)/(1+3*q/4+q*q/16)).exp()
            direct=rs-other
            got=std.residual(q,K,kind,C)
            vals=[got] if kind=="Fresnel" else [got,v.kernel(q,K),ind.arb_kernel(q,K,C)]
            check(kind+" "+label,all(x.overlaps(direct) for x in vals),str(direct),"B")
    return Q,K


def polynomial_tests():
    ctx.prec=384
    for d in (1,2,3):
        c=[C(0)]*d+[C(1)]
        terms=v.aperture_transform(A,C,c,A(2),A(1),A(1),A(1))
        ref=ind.source_shift(A,C,c,A(2),A(1),A(1),A(1),A(1))
        for (i,j),value in terms.items():
            k=d-i-j
            closed=A.pi()*comb(d,k)**2*comb(d-k,i)*25**(i+j)/(k+1)
            require(value.overlaps(C(closed)) and ref[i,j].overlaps(C(closed)),"circle monomial")
        check("all circle coefficients q^"+str(d),True,f"{len(terms)} coefficients, exact gamma/translation/closed form","E")
    rng=random.Random(718409)
    with mp.workdps(75):
        nodes,weights=mp.gauss_quadrature(10,"legendre")
        for d in (2,3,5):
            coeff=[(rng.randint(-5,5),rng.randint(-5,5)) for j in range(d+1)]
            cs=[C(*c) for c in coeff]
            terms=v.aperture_transform(A,C,cs,A(2),A(1),A(2),A(1)/4)
            alt=ind.source_shift(A,C,cs,A(2),A(1),A(1),A(2),A(1)/4)
            X,Y=mp.mpf(3)/8,-mp.mpf(1)/4
            direct=mp.mpc(0)
            for u,w in zip(nodes,weights):
                r=(1+u)/2
                for j in range(32):
                    phi=2*mp.pi*j/32
                    q=(X-r*mp.cos(phi))**2+(Y-r*mp.sin(phi))**2
                    direct+=mp.polyval([mp.mpc(*c) for c in reversed(coeff)],q)*r*w*mp.pi/32
            field=sum(c*(A(3)/40)**(2*i)*(A(-1)/20)**(2*j) for (i,j),c in terms.items())
            empirical=C(A(str(direct.real)),A(str(direct.imag)))
            diff=(field-empirical).abs_upper()
            check(f"seeded direct polar polynomial degree {d}",diff<A("1e-65") and
                  all(c.overlaps(alt[key]) for key,c in terms.items()),f"absolute discrepancy <= {diff}; 75 digits, 10 x 32","E","empirical + interval comparison")
            def moment(i,j):
                return (A(i)+A(1)/2).gamma()*(A(j)+A(1)/2).gamma()/(A.pi()*A(i+j+2).gamma())
            pair=sum(c*b.conjugate()*moment(i+k,j+l) for (i,j),c in terms.items() for (k,l),b in terms.items())
            got=v.disk_gram(A,arb_poly,terms)
            check(f"unpacked Hermitian Gram {d}",pair.real.overlaps(got) and pair.imag.contains(0)
                  and got.overlaps(ind.gram(A,arb_poly,terms)) and got>0,str(got),"F")


def adversarial(Q,K):
    # Exercise historical callables, including inputs outside their published grid.
    for case,engine in [(("1","1","0.1","1"),"std"),
                        (("2","1","0.1","0.25"),"ind")]:
        p,ax,ay,z,lam,k,q,Qi=ind.setup(A,case)
        actual=q_geometry(p,ax,ay,z)
        record=(std.worker(case[:3],8,"Fresnel") if engine=="std" else ind.certify(case,8,256))
        mutation("undersized Q "+engine,"valid aperture; opposite rim points attain q_max",
                 f"actual {actual}; Q {Qi}; {record['status']}",
                 actual<=Qi or record["status"]!="interval-certified",True)
    r=std.worker(BASE,8,"Fresnle")
    mutation("misspelled residual","unknown model label",r["status"]+" "+r["kind"],
             r["status"]!="interval-certified",True)
    r=v.certify(("12","-10","0.3"),4,256)
    mutation("negative NF","nonpositive wavelength",f"lo {r['lower']} hi {r['upper']} {r['status']}",
             r["status"]!="interval-certified",True)
    # A mutation of an internal operation tests detection, not the correctness
    # of the unmodified operation. Originals on disk are never edited.
    with patch.object(std,"disk_gram",return_value=A(-1)):
        r=std.worker(BASE,4,"Fresnel")
    mutation("negative Gram upper","in-memory injected Gram=-1",r["status"],r["status"]!="interval-certified",True)
    lo,hi=outer(A(0,1),A(1)/4)
    check("Gram straddles zero",lo==0 and hi>=1,"sqrt(max(0,Glo)); G in [-1,1]","G")
    for Qbad in ("7.99","8","8.01"):
        try:
            std.interpolate(A,C,acb_poly,A(1),A(Qbad),4,"Fresnel")
            ok=True
        except ArithmeticError:
            ok=False
        check("branch threshold "+Qbad,ok==(A(Qbad)<8),str(ok),"C")
    for kind,den in [("Fresnel",-1),("Pade","-0.99")]:
        # -0.99 is past the rightmost amplitude pole. A positive-real
        # certificate can reject a regular point on this side.
        try:
            std.residual(C(A(den)),K,kind,C)
            rejected=False
        except ArithmeticError:
            rejected=True
        if den==-1:
            mutation("sqrt cut","q=-1",rejected,rejected,True)
        else:
            check("sufficient pole test",rejected,"regular point q=-0.99 rejected by sufficient Re>0 check","C")
    # An analytic point with negative real denominator, above -1.
    # Disjoint angle unions leave a hole even if each remaining box is valid.
    ids=list(range(512)); ids.remove(173)
    mutation("arc coverage","remove closed arc 173",len(ids),ids!=list(range(512)))
    h=Q/2
    bad_rect=C(h+A(0,1)*h,A(0,1)*3*h/4)
    mutation("ellipse rectangle","reduce real semiaxis to h",str(bad_rect.real),not bad_rect.real.contains(h+5*h/4))
    # D=T_(n+1); interpolation aliases to T_(n-1).
    point=mp.cos(mp.pi/4)
    alias_error=abs(mp.chebyt(4,point)-mp.chebyt(2,point))
    mutation("Lobatto alias","claim discarded tail enters once at every x; T4 aliases to T2 (n=3)",
             "at x=0 true error 2, tail coefficient magnitude 1",abs(sp.chebyshevt(4,0)-sp.chebyshevt(2,0))==2)
    nodes=[A(1),A(0),A(0)]
    mutation("duplicate node","replace distinct Lobatto nodes",str(nodes),any((u-v).contains(0) for i,u in enumerate(nodes) for v in nodes[:i]))
    grid=[(1+(A.pi()*j/96).cos())/2 for j in range(97)]
    check("strict denominator differences",all(not (x-y).contains(0) for i,x in enumerate(grid) for y in grid[:i]),"97 Lobatto nodes","D")
    # Independent simple witnesses for the algebraic proof obligations.
    for name,bad,good,why in [
        ("omit odd parity rule",A(1),A(0),"unit disk integral s_x vanishes"),
        ("source scaling",A.pi()/2,A.pi()/8,"unit source at origin, z=2, Q=1, q^1"),
        ("gamma denominator",A(4)*A.pi(),A.pi(),"unit disk constant moment, omitted p^2"),
        ("omit conjugation",A(-1),A(1),"constant field i"),
        ("delete signed cross terms",A(1)+A(1)/8,A(1)-A(1)/2+A(1)/8,"field 1-X on unit disk"),
        ("stride alias",A(1)/4,A(1)/8,"n=1, stride=2 maps Y^2 to X; <Y^2>=1/8"),
        ("ROI normalization",25*A.pi(),A(1),"field 1 on radius-5 disk, squared norm"),
        ("nonunit illumination",A(1),A(2),"illumination 2 doubles the uniform field remainder"),
        ("carrier modulus",A(1),A(2),"multiplying both fields by 2 doubles the norm"),
        ("inward lower rounding",A(1)+A(2)**-50,A(1),"degenerate exact norm 1"),
        ("omit interpolation slack",A(0),A(1),"norm En=0, true constant residual has norm 1")]:
        mutation(name,why,f"mutated {bad}, required {good}",not bad.overlaps(good))
    # Scalar geometry countercheck and outward endpoint round trip.
    check("dyadic serialization",all(exact(binary(x))==x for x in [Q.lower(),K.upper(),A(0),A(-3)/8]),
          "signed integer mantissa times 2^exponent","G")
    check("valid sign carrier",C(0,K).exp().abs_lower()<=1<=C(0,K).exp().abs_upper(),"real K","B")


def comparison_tests():
    baseline=[r for r in RUNS if (r["p"],r["NF"],r["theta"],r["AR"])==(*BASE,"1")]
    def endpoints(r):
        return exact(r["lower"]),exact(r["upper"])
    for kind in MODELS:
        rows=[r for r in baseline if r["kind"]==kind]
        los,his=zip(*(endpoints(r) for r in rows))
        check(kind+" all degree/precision intersection",max(los)<=min(his),
              f"{len(rows)} intervals; common intersection [{max(los)}, {min(his)}]","G")
    for r in RUNS:
        if r["archive_overlap"] is not None:
            check(f"archive {r['engine']} {r['kind']} {r['degree']} {r['bits']} {r['p']} {r['AR']}",
                  r["archive_overlap"],str(r["endpoints_display"]),"G")
    # Exact nesting is not required by the theorem. Record it explicitly.
    for engine in ("v6","ind"):
        rows=sorted([r for r in baseline if r["engine"]==engine and r["degree"]==96],key=lambda r:r["bits"])
        for prev,new in zip(rows,rows[1:]):
            l,h=endpoints(prev); a,b=endpoints(new)
            check(f"fixed degree precision {engine} {prev['bits']}->{new['bits']}",a<=h and b>=l,
                  f"overlap; higher-precision interval nested={bool(l<=a and b<=h)}","G")
    for r in baseline:
        if r["degree"]<=4 or r["bits"]<=80:
            check(f"resolution limitation {r['degree']} {r['bits']}",exact(r["lower"])==0,
                  str(r["endpoints_display"]),"D","observed width; finite wide enclosure allowed")


# Ranges refer to immutable input snapshots. Each listed function also appears
# in the recorded runtime trace of the corresponding replay.
MAPPING = [
("A","source area, AR and a_eff","p>0; ax,ay>0","Cor.1","v6 physical 27-36;ind setup 48-58","all circle coefficients","area and physical scales change"),
("A","continuous q upper bound","p>=2 in v6, AR=1; independent p=2 requires ax>=ay","Cor.1","v6 physical 30-36;ind setup 49-58","undersized Q","interpolation tail outside domain is uncontrolled"),
("B","exact Padé residual","real K nonzero; principal t","Defs.","v6 kernel 14-24;ind arb_kernel 16-29;std residual 17-39","Pade ","wrong field difference"),
("B","exact conventional residual","unit amplitude, quadratic phase","Defs.","std residual 17-39","Fresnel ","wrong physical model"),
("B","identity of residual label","declared model must be one of two names","Defs.","std residual 24-39;std worker 197-202","misspelled residual","wrong model certified under supplied name"),
("B","common carrier and norm prefactor","unit illumination; real K; lambda,z>0","Thm.1","std certify 86-97;v6 certify 115-130;ind certify 148-156","negative NF","negative slack or erroneous normalization"),
("C","analytic square root and no denominator zeros","strict positive real enclosures over enclosing rectangle","Lem.1","v6 kernel 17-24;ind arb_kernel 18-29;std residual 19-39","branch threshold","a hidden pole or cut voids contour proof"),
("C","ellipse containment","h=Q/2; rho=2","Lem.1","v6 interpolation 40-50;ind polynomial 62-69;std interpolate 43-49","ellipse rectangle","verification might miss a singularity"),
("D","upper M on entire closed ellipse","512 contiguous angular intervals; maximum modulus","Lem.1","v6 interpolation 42-51;ind polynomial 63-70;std interpolate 44-50","arc coverage","tail underestimated"),
("D","Lobatto polynomial and error constant","n>=1; rho>1; no repeated nodes","Lem.1","v6 interpolation 51-62;ind polynomial 70-79;std interpolate 50-58","Lobatto","wrong polynomial or aliased tail bound"),
("E","scaled source moments and coordinate parity","axis-aligned centered uniform superellipse","Lem.2, Cor.1","v6 aperture_transform 65-84;ind source_shift 82-102","seeded direct polar","wrong source field"),
("F","normalized Gram of real and imaginary parts","normalized centered disk; signed Hermitian terms","Lem.2","v6 disk_gram 87-109;ind gram 105-125","unpacked Hermitian","incorrect squared norm"),
("F","Kronecker index injective under squaring","0<=j+l<=2n<stride","Lem.2","v6 disk_gram 88-108","stride alias","cross terms assigned to wrong moment"),
("G","uniform remainder at field level","normalized measure, finite positive area","Thm.1","v6 certify 124-130;std certify 92-97;ind certify 148-156","omit interpolation slack","true field can escape final interval"),
("G","round lower down, upper up; Gram nonnegative","finite G, G+>=0, delta>=0","Prop.1","v6 certify 124-136;std certify 92-106;ind certify 148-156","Gram","invalid/reversed certificate"),
("G","replay identical norm and compare endpoints","same raw fields, ROI and geometric parameters","Prop.1","v6 certify 112-136;std certify 84-106;ind certify 128-156","archive","results cannot be linked to the same quantity"),
("G","endpoint persistence","ball strings parsed outward or exact dyadics; display floats informational","Prop.1","v6 certify 131-136;std certify 98-106;ind certify 153-156","dyadic serialization","decimal rounding can shrink interval"),
]


def write_reports(out):
    tests=CHECKS+MUTATIONS
    matrix=[]
    for stage,statement,assum,ref,loc,pattern,harm in MAPPING:
        evidence=[r for r in tests if pattern.lower() in r["test"].lower()]
        require(evidence,statement)
        for entry in loc.split(";"):
            mod,fn,lines=entry.split()
            file=Path(MODULES[mod].__file__)
            left,right=map(int,lines.split("-"))
            defn=next(n for n in ast.parse(file.read_text(encoding="utf-8")).body if getattr(n,"name",None)==fn)
            require(defn.lineno<=left<=right<=defn.end_lineno,f"bad line range {entry}")
            result=", ".join(sorted(set(r["result"] for r in evidence)))
            matrix.append(dict(stage=stage,mathematical_statement=statement,mathematical_assumption=assum,
                equation_or_theorem=ref,source_filename="inputs/"+file.name,function_name=fn,exact_source_lines=lines,
                verification_test=" | ".join(r["test"] for r in evidence),verification_result=result,
                certification_status="scope restriction / original path FAIL" if "FALSE ACCEPT" in result else
                "FAIL" if "FAIL" in result else "audited within stated assumptions",failure_consequence=harm))
    paths=[p for p in INPUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [HERE/name for name in ("independent_counterexample_tests.py","audit_request.txt",
               "run_audit.ps1","theorem1_final.tex","theorem1_proof_audit.md")]
    hashes={p.relative_to(HERE).as_posix():digest(p) for p in paths}
    table(out/FILES[0],matrix)
    save(out/FILES[1],dict(obligations=matrix,observations=CHECKS,mutations=MUTATIONS))
    table(out/FILES[2],MUTATIONS)
    table(out/FILES[3],CHECKS)
    save(out/FILES[4],dict(runs=RUNS,interval_convention="signed mantissa times 2^exponent; inclusive",
                            tested_intersections=[r for r in CHECKS if r["stage"]=="G"]))
    failures=[r["test"] for r in MUTATIONS if r["result"]=="FALSE ACCEPT"]
    unexpected=[r["test"] for r in CHECKS if r["result"]=="FAIL"]
    decision="FAIL" if failures or unexpected else "CONDITIONAL PASS"
    evidence_files={name:digest(out/name) for name in FILES[:5]}
    baseline=[r for r in RUNS if r["degree"]==96 and r["bits"]==768 and r["AR"]=="1"]
    summary=dict(mathematical_theorem="PASS with stated positive scales, illumination and nondegenerate interpolation assumptions",
                 implementation_integrity=decision,scope="historical callable paths; four baseline replays evaluated separately",
                 counterexamples=failures,unexpected_test_failures=unexpected,
                 runtime=dict(python=sys.version,platform=platform.platform(),flint=flint.__FLINT_VERSION__,
                              packages={k:md.version(k) for k in ("python-flint","sympy","mpmath","numpy","scipy")}),
                 sources_sha256=hashes,evidence_sha256=evidence_files,baseline=baseline,
                 replay="powershell -ExecutionPolicy Bypass -File ./run_audit.ps1 -OutputDirectory replay_01",
                 proof_obligations=matrix,
                 trusted_base="Python, FLINT/Arb ball semantics, operating system and archived source bytes; no formal verifier",
                 remediation=["validate finite positive physical inputs before any loop","compute axis maximum for p<=2 regardless of axis order",
                              "whitelist model names","explicitly reject nonfinite G, delta, final endpoints, inverted intervals; preserve checks with -O",
                              "serialize signed dyadic endpoints and hash inputs","replay negative tests before publication"])
    save(out/FILES[5],summary)
    (out/FILES[6]).write_text(
        "# Theorem 1 감사 결과\n\n수정된 수학 정리 **PASS**, 원본 구현의 인증 무결성 **"+decision+"**.\n\n"
        +f"실제 인증 경로 {len(RUNS)}회 실행. 기준 조건의 768비트 네 경로는 아래 표와 같다. "
        +"표의 숫자는 표시용이며 정확한 양 끝점은 final_certificate.json의 이진 유리수이다.\n\n"
        +"|구현|잔차|하한 (표시)|상한 (표시)|보관 구간과 교차|\n|---|---|---:|---:|---|\n"
        +"".join(f"|{r['engine']}|{r['kind']}|{float(exact(r['lower'])):.16g}|{float(exact(r['upper'])):.16g}|{r['archive_overlap']}|\n" for r in baseline)
        +"\n거짓 인증 반례: "+", ".join(failures)+". 추가 검사 실패: "+(", ".join(unexpected) or "없음")+".\n\n"
        +"논문에 사용할 수 있는 주장은 명시한 가정 아래의 양측 상계·하계와 해당 물리 조건에서 재현된 연속 원판 노름이다. "
        +"일반 입력의 인증, 기존 연구 대비 독창성, 형식 검증은 별도의 증거가 필요하다. "
        +"제출 전에는 입력 검증과 연속 q 범위, 유한성과 끝점 순서 검사를 수정하고 반례를 다시 실행해야 한다.\n\n"
        +"원고와 증명 감사 문서는 폴더 루트의 theorem1_final.tex 및 theorem1_proof_audit.md에 있다. "
        +"모든 계산 파일은 새 폴더 이름을 지정해 재생성할 수 있다.\n\n"
        +"~~~powershell\n.\\run_audit.ps1 -OutputDirectory replay_01\n~~~\n",encoding="utf-8")
    print(decision,":",failures,"unexpected test failures:",unexpected,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE)
    args=parser.parse_args()
    out=args.out.resolve()
    if out!=HERE:
        out.mkdir(parents=True,exist_ok=False)
    require(not any((out/f).exists() for f in FILES),"Choose a fresh result directory")
    require(__debug__,"Archived code uses assertions; run with optimizations disabled")
    Q,K=kernel_tests()
    polynomial_tests()
    adversarial(Q,K)
    for engine,kind,n,b,c in [
            ("v6","Pade",96,768,(*BASE,"1")),("ind","Pade",96,768,(*BASE,"1")),
            ("std","Pade",96,768,(*BASE,"1")),("std","Fresnel",96,768,(*BASE,"1")),
            ("v6","Pade",96,1024,(*BASE,"1")),("ind","Pade",96,1024,(*BASE,"1")),
            ("std","Pade",64,512,(*BASE,"1")),("std","Fresnel",64,512,(*BASE,"1")),
            ("std","Fresnel",4,256,(*BASE,"1")),("v6","Pade",64,80,(*BASE,"1")),
            ("ind","Pade",64,640,("16","10","0.3","4")),
            ("ind","Pade",12,256,("2","0.1","0.02","1"))]:
        replay(engine,kind,n,b,c)
    ctx.prec=1200
    comparison_tests()
    write_reports(out)


if __name__=="__main__":
    main()



