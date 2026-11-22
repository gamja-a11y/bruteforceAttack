"""Check proof evidence without importing the certificate generator."""
import argparse
from decimal import Decimal
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import re

import flint
from flint import arb as A, acb as C, acb_poly, arb_poly, ctx

ROOT=Path(__file__).resolve().parent
STAGES="physical geometry identity analyticity contour interpolation source gram remainder endpoints".split()


def need(condition,reason):
    if not condition:
        raise ValueError(reason)


def point(pair):
    need(type(pair) is list and len(pair)==2 and type(pair[1]) is int and
         type(pair[0]) is str and re.fullmatch(r"-?[0-9]+",pair[0]),"invalid signed dyadic")
    x=A((int(pair[0]),pair[1]))
    need(x.is_finite() and x.is_exact(),"nonfinite or inexact endpoint")
    return x


def interval(pair):
    need(type(pair) is list and len(pair)==2,"malformed enclosure")
    l,h=map(point,pair)
    need(l<=h,"inverted interval")
    return l,h


def encloses(pair,value,label):
    l,h=interval(pair)
    need(value.is_finite() and l<=value.lower() and value.upper()<=h,label)


def parse_inputs(d):
    need(set(d)==set("p nf theta ar radius degree bits model Q rho".split()),"unrecognized input schema")
    need(type(d["model"]) is str and d["model"] in ("Fresnel","Pade"),"model")
    need(type(d["degree"]) is int and d["degree"]>=1,"degree")
    need(type(d["bits"]) is int and d["bits"]>=64,"precision")
    need(type(d["rho"]) is str and Decimal(d["rho"]).is_finite() and Decimal(d["rho"])>1,"rho")
    for k in ("p","nf","theta","ar","radius","Q"):
        if k=="Q" and d[k] is None:
            continue
        need(type(d[k]) is str and Decimal(d[k]).is_finite() and Decimal(d[k])>0,"invalid "+k)


def contour_check(ev,Q,K,model,rho):
    r=A(rho)
    major=Q*(r+1/r)/4
    minor=Q*(r-1/r)/4
    h=Q/2
    rect=C(h+A(0,1)*major,A(0,1)*minor)
    margins={"branch":(1+rect).real}
    if model=="Pade":
        margins.update(phase=(1+3*rect/4+rect*rect/16).real,
                       amplitude=(1+7*rect/4+35*rect*rect/48).real)
    need(set(margins)==set(ev["analyticity"]),"missing analytic guard")
    for k,v in margins.items():
        # At the original precision the rectangular enclosure is reproducible.
        encloses(ev["analyticity"][k],v,k)
        need(interval(ev["analyticity"][k])[0]>0,k)
    cont=ev["contour"]
    need(cont["rho"]==rho and cont["panels"]==len(cont["arc_uppers"])==512,"incomplete contour")
    maximum=point(cont["M"])
    i=C(0,1)
    for j,entry in enumerate(cont["arc_uppers"]):
        angle=(2*A.pi()*j/512).union(2*A.pi()*(j+1)/512)
        q=C(h+major*angle.cos(),minor*angle.sin())
        t=(1+q).sqrt()
        invcube=(1/t)**3
        amp=1/(1+q)+i/K*invcube
        if model=="Fresnel":
            psi=q/2
            diff=-q*q/(2*(1+t)**2)
            da=-q/(1+q)+i/K*invcube
        else:
            psi=(q/2+q*q/4)/(1+3*q/4+q*q/16)
            diff=q/(1+t)-psi
            da=i/K*(invcube-(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48))
        upper=((i*K*psi).exp()*(da+amp*(i*K*diff).expm1())).abs_upper()
        need(upper.is_finite() and upper<=point(entry)<=maximum,f"contour panel {j}")
    return maximum


def interpolate(n,Q,K,model):
    # Independent DCT-I of direct differences; no Newton divided differences.
    cos=[(A.pi()*j/n).cos() for j in range(2*n)]
    nodes=[(1+cos[j])/2 for j in range(n+1)]
    vals=[]
    i=C(0,1)
    for s in nodes:
        q=Q*s
        t=(1+q).sqrt()
        rs=(1/(1+q)+i/(K*t**3))*(i*K*(t-1)).exp()
        if model=="Fresnel":
            approx=(i*K*q/2).exp()
        else:
            amp=1/(1+q)+i*(48+12*q-q*q)/(K*(48+84*q+35*q*q))
            approx=amp*(i*K*4*q*(q+2)/(16+12*q+q*q)).exp()
        vals.append(rs-approx)
    prev,cur=acb_poly([1]),acb_poly([-1,2])
    total=acb_poly()
    for k in range(n+1):
        a=sum(v*cos[(j*k)%(2*n)]/(2 if j in (0,n) else 1) for j,v in enumerate(vals))*2/n
        total+=prev*(a/2 if k in (0,n) else a)
        prev,cur=cur,2*acb_poly([-1,2])*cur-prev
    return [total[i] for i in range(n+1)],min((a-b).lower() for a,b in zip(nodes,nodes[1:]))


def independent_gram(coeff,p,ax,ay,z,Q,radius):
    # Sum by source moments. Multiply rows as ordinary y-polynomials to avoid
    # the generator's Kronecker packing and check signed cross terms.
    n=len(coeff)-1
    scale=z*z*Q
    g=[(A(2*j+1)/p).gamma() for j in range(n+1)]
    den=[(1+A(2*j+2)/p).gamma() for j in range(n+1)]
    xp=[(ax*ax/scale)**i for i in range(n+1)]
    yp=[(ay*ay/scale)**i for i in range(n+1)]
    rp=[(radius**2/scale)**i for i in range(n+1)]
    bc=[[comb(2*k,2*i) for i in range(k+1)] for k in range(n+1)]
    terms=[[C(0) for j in range(n-i+1)] for i in range(n+1)]
    samples={}
    from math import factorial
    fac2=[factorial(2*k) for k in range(n+1)]
    kernel=[]
    for b in range(n+1):
        col=[]
        for a in range(n-b+1):
            v=4*ax*ay*g[a]*g[b]*xp[a]*yp[b]/(p*p*den[a+b])
            if a+b<=1: samples[a,b]=v
            col.append(v/fac2[a])
        kernel.append(arb_poly(col))
    for l in range(n+1):
        top=n-l
        row=acb_poly([coeff[k+l]*comb(k+l,k)*fac2[k] for k in range(top,-1,-1)])
        for b in range(l+1):
            product=row*kernel[b]
            for i in range(top+1):
                terms[i][l-b]+=product[top-i]*bc[l][b]/fac2[i]
    re,im=[],[]
    for i,row in enumerate(terms):
        vals=[c*rp[i+j] for j,c in enumerate(row)]
        re.append(arb_poly([c.real for c in vals]))
        im.append(arb_poly([c.imag for c in vals]))
    power,fac=[A(1)],[1]
    for k in range(1,2*n+2):
        power.append(power[-1]*(A(k)-A(1)/2))
        fac.append(fac[-1]*k)
    result=A(0)
    for i in range(n+1):
        for j in range(i+1):
            poly=re[i]*re[j]+im[i]*im[j]
            result+=sum(v*power[i+j]*power[k]/fac[i+j+k+1] for k,v in enumerate(poly.coeffs()))*(1 if i==j else 2)
    return samples,result


def verify(record,expected):
    need(isinstance(expected,dict),"external specification required")
    need(record.get("status")=="interval-certified","no issued certificate")
    need(record["schema"]==1 and record["generator_sha256"]==sha256(
        (ROOT/"rho_certifier.py").read_bytes()).hexdigest(),"generator hash mismatch")
    need((record["flint"],record["native_flint"])==(flint.__version__,flint.__FLINT_VERSION__),"backend version")
    d=record["inputs"]
    parse_inputs(d)
    if expected is not None:
        need(d==expected,"requested inputs differ")
    need(set(record["obligations"])==set(STAGES),"incomplete obligations")
    ev={}
    for s in STAGES:
        item=record["obligations"][s]
        need(item["passed"] is True and isinstance(item["evidence"],dict),"missing evidence "+s)
        ev[s]=item["evidence"]
    n,bits,model=d["degree"],d["bits"],d["model"]
    need(ev["identity"]==dict(model=model,illumination=1,carrier="removed",measure="disk area average"),"model identity")
    with ctx.workprec(bits):
        Q=point(ev["geometry"]["Q"])
        need(Q>0 and Q.is_exact(),"physical Q")
        qhi=interval(ev["geometry"]["qmax"])[1]
        need(qhi<=Q,"continuous q exceeds domain")
        requested=d["Q"]
        chosen=((qhi*1024).ceil()/1024 if requested is None else A(requested).lower())
        need(chosen==Q,"chosen Q disagrees with inputs")
        th=A(d["theta"])
        K=2*A.pi()*(10/th)/(10*th/A(d["nf"]))
        M=contour_check(ev,Q,K,model,d["rho"])
    with ctx.workprec(max(2*bits+128,12*n+256)):
        p,nf,th,ar,R=[A(d[k]) for k in ("p","nf","theta","ar","radius")]
        z,lam=10/th,10*th/nf
        K=2*A.pi()*z/lam
        # Dirichlet area inversion using Gamma(1+x)=x Gamma(x).
        ay=(100*A.pi()*p*p*(1+2/p).gamma()/(4*ar*(1/p).gamma()**2)).sqrt()
        ax=ar*ay
        if p<=2:
            reach=ay if ar<=1 else ax
        else:
            t=2*p/(p-2)
            reach=ay*(1+ar**t)**(1/t)
        for k,v in [("ax",ax),("ay",ay),("qmax",((R+reach)/z)**2)]:
            encloses(ev["geometry"][k],v,k)
        for k,v in [("z",z),("wavelength",lam),("K",K)]:
            need(v>0,k)
            encloses(ev["physical"][k],v,k)
        eps=point(ev["interpolation"]["epsilon"])
        delta=point(ev["remainder"]["delta"])
        need(eps>=(4*M/(A(d["rho"])**n*(A(d["rho"])-1))).upper() and delta>=0 and delta>=(100*A.pi()*eps/(lam*z)).upper(),"remainder understated")
        coeff,gap=interpolate(n,Q,K,model)
        need(ev["interpolation"]["nodes"]==n+1 and 0<point(ev["interpolation"]["minimum_gap"])<=gap,
             "Lobatto node separation")
        samples,G=independent_gram(coeff,p,ax,ay,z,Q,R)
        source=ev["source"]
        need(source["count"]==(n+1)*(n+2)//2 and ev["gram"]["stride"]==2*n+1,"polynomial dimensions")
        for k,v in [("mu00",samples[0,0]),("mu01_scaled",samples[0,1]),("mu10_scaled",samples[1,0])]:
            encloses(source[k],v,k)
        encloses(ev["gram"]["G"],G/(lam*z)**2,"independent Gram not enclosed")
        l,h=interval(ev["gram"]["G"])
        lo,hi=map(point,[ev["endpoints"]["lower"],ev["endpoints"]["upper"]])
        need(h>=0 and hi>=lo>=0 and hi>=delta,"final domain")
        need((hi-delta)**2>=h and (lo==0 or (lo+delta)**2<=max(A(0),l)),"inward rounding")
    return dict(model=model,inputs=d,verified=True,stages=STAGES,
                endpoint_display=[float(lo),float(hi)],generator_sha256=record["generator_sha256"],
                verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                trust="Local reviewed source, CPython and FLINT/Arb. Independent numerical replay of contour,"
                      " Lobatto/DCT-I, source and Gram; analytic theorems rely on the written proof.")


def check(record,expected):
    from common import dy
    from fractions import Fraction
    proof=verify(record,expected)
    ev=record['obligations']['endpoints']['evidence']
    l,u=map(dy,(ev['lower'],ev['upper']))
    ok=u-l<=Fraction(1,100000) and l>0 and 2*(u-l)<=Fraction(1,100)*(u+l)
    need(record.get('target_met') is ok,'target flag')
    proof.update(target_met=ok)
    return proof

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('certificate',type=Path)
    parser.add_argument('expected',type=Path)
    opts=parser.parse_args()
    print(json.dumps(check(json.loads(opts.certificate.read_text()),json.loads(opts.expected.read_text()))))
