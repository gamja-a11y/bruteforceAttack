"""Continuous disk RMS certificate for the carrier-stripped scalar residual."""
import argparse
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from math import comb
from pathlib import Path

import flint
from flint import acb as C, arb as A, acb_poly, arb_poly, ctx


def demand(condition, reason):
    if not condition:
        raise ValueError(reason)


def scalar(value, name):
    demand(type(value) in (str, int, float, Decimal), f"{name}: expected a real scalar")
    try:
        d = Decimal(value)
    except (ValueError, InvalidOperation) as exc:
        raise ValueError(f"{name}: invalid numeric input") from exc
    demand(d.is_finite() and d > 0, f"{name}: must be positive and finite")
    return str(d)


def model_name(model):
    demand(type(model) is str and model in ("Pade", "Fresnel"), "model must be Pade or Fresnel")
    return model


def positive(value, label):
    demand(value.is_finite() and value.lower() > 0, label+" is not provably finite and positive")
    return value


def dyadic(value):
    demand(value.is_finite() and value.is_exact(), "expected a finite dyadic endpoint")
    m, e = value.man_exp()
    return [str(m), int(e)]


def box(value):
    demand(value.is_finite(), "nonfinite interval")
    return [dyadic(value.lower()), dyadic(value.upper())]


def geometry(p, nf, theta, ar, radius, requested=None):
    z = positive(10/theta, "z")
    lam = positive(10*theta/nf, "wavelength")
    K = positive(2*A.pi()*z/lam, "K")
    ay = positive((25*A.pi()*(1+2/p).gamma()/(ar*(1+1/p).gamma()**2)).sqrt(), "ay")
    ax = positive(ay*ar, "ax")
    if p <= 2:
        reach = ax.max(ay)
    elif p > 2:
        exponent = 2*p/(p-2)
        reach = (ax**exponent+ay**exponent)**(1/exponent)
    else:
        raise ValueError("p interval straddles 2; increase precision")
    qmax = positive(((radius+positive(reach,"source radius"))/z)**2, "q maximum")

    if requested is None:
        Q = (qmax.upper()*1024).ceil()/1024
    else:
        Q = positive(requested, "requested Q").lower()
    positive(Q, "Q")
    demand(Q.is_exact() and qmax.upper()<=Q.lower(), "continuous q maximum exceeds Q")
    return ax,ay,z,lam,K,qmax,Q


def guards(q, model):
    values = {"branch": (1+q).real}
    if model=="Pade":
        values.update(phase=(1+3*q/4+q*q/16).real, amplitude=(1+7*q/4+35*q*q/48).real)
    for name,v in values.items():
        positive(v, name+" on complex domain")
    return values


def residual(q,k,kind,complex_type=C):
    model_name(kind)
    guards(q,kind)
    i = complex_type(0,1)
    t = (1+q).sqrt()
    invcube = (1/t)**3
    amp = 1/(1+q)+i/k*invcube
    if kind=="Fresnel":
        phase = q/2
        da = -q/(1+q)+i/k*invcube
        gap = -q*q/(2*(1+t)**2)
    elif kind=="Pade":
        phase = (q/2+q*q/4)/(1+3*q/4+q*q/16)
        da = i/k*(invcube-(1+q/4-q*q/48)/(1+7*q/4+35*q*q/48))
        gap = q/(1+t)-phase
    value = (i*k*phase).exp()*(da+amp*(i*k*gap).expm1())
    demand(value.is_finite(), "nonfinite residual")
    return value


def contour(K,Q,model,rho):
    rho_value=positive(A(rho),"rho")
    demand(rho_value>1,"rho must exceed one")
    major=Q*(rho_value+1/rho_value)/4
    minor=Q*(rho_value-1/rho_value)/4
    h=Q/2
    rect=C(h+A(0,1)*major,A(0,1)*minor)
    margins={k:box(v) for k,v in guards(rect,model).items()}
    arcs=[]
    edge=A(0)
    M=A(0)
    for j in range(512):
        end=2*A.pi()*(j+1)/512
        theta=edge.union(end)
        demand(theta.contains(edge) and theta.contains(end), "Bernstein arc gap")
        q=C(h+major*theta.cos(),minor*theta.sin())
        bound=residual(q,K,model).abs_upper()
        demand(bound.is_finite() and bound>=0, "invalid contour upper bound")
        arcs.append(dyadic(bound))
        M=max(M,bound)
        edge=end
    demand(edge.contains(2*A.pi()) and len(arcs)==512,"incomplete contour")
    return margins,dict(rho=rho,panels=512,arc_uppers=arcs,M=dyadic(M))


def spectrum(K,Q,n,model,rho):
    margins,record=contour(K,Q,model,rho)
    M=A((int(record["M"][0]),record["M"][1]))
    nodes=[(1+(A.pi()*j/n).cos())/2 for j in range(n+1)]
    gap=min((x-y).lower() for x,y in zip(nodes,nodes[1:]))
    positive(gap,"Lobatto separation")
    vals=[residual(C(Q*x),K,model) for x in nodes]
    for j in range(1,n+1):
        for i in range(n,j-1,-1):
            vals[i]=(vals[i]-vals[i-1])/(nodes[i]-nodes[i-j])
    poly=acb_poly([vals[-1]])
    for i in reversed(range(n)):
        poly=poly*acb_poly([-nodes[i],1])+acb_poly([vals[i]])
    coeff=[poly[i] for i in range(n+1)]
    demand(all(c.is_finite() for c in coeff),"invalid interpolant coefficients")
    r=A(rho)
    eps=(4*M/(r**n*(r-1))).upper()
    return coeff,eps,margins,record,dict(
        nodes=n+1,minimum_gap=dyadic(gap),epsilon=dyadic(eps))


def source_transform(coeff,p,ax,ay,z,Q,radius):
    n=len(coeff)-1
    scale=z*z*Q
    gamma=[positive((A(2*j+1)/p).gamma(),"source gamma") for j in range(n+1)]
    den=[positive((1+A(2*j+2)/p).gamma(),"source denominator") for j in range(n+1)]
    xp=[(ax*ax/scale)**j for j in range(n+1)]
    yp=[(ay*ay/scale)**j for j in range(n+1)]
    wp=[(radius**2/scale)**j for j in range(n+1)]
    mu=[[positive(4*ax*ay/p**2*xp[i]*yp[j]*gamma[i]*gamma[j]/den[i+j],"source moment")
         for j in range(n-i+1)] for i in range(n+1)]
    expected_area=4*ax*ay*(1+1/p).gamma()**2/(1+2/p).gamma()
    demand(mu[0][0].overlaps(expected_area),"source zeroth moment mismatch")
    bc=[[comb(2*k,2*i) for i in range(k+1)] for k in range(n+1)]
    radial=[[coeff[k+l]*comb(k+l,k) for l in range(n-k+1)] for k in range(n+1)]
    from math import factorial
    fac=[factorial(2*j) for j in range(n+1)]
    moments=[arb_poly([mu[a][b]/fac[b] for b in range(n-a+1)]) for a in range(n+1)]
    terms={(i,j):C(0) for i in range(n+1) for j in range(n-i+1)}
    for k in range(n+1):
        top=n-k
        source=acb_poly([radial[k][l]*fac[l] for l in range(top,-1,-1)])
        for i in range(k+1):
            conv=source*moments[k-i]
            for j in range(top+1):
                terms[i,j]+=conv[top-j]*bc[k][i]
    for (i,j),v in terms.items():
        terms[i,j]=v*wp[i+j]/fac[j]
    demand(all(c.is_finite() for c in terms.values()),"nonfinite source polynomial")
    return terms,dict(count=len(terms),mu00=box(mu[0][0]),mu10_scaled=box(mu[1][0]),
                      mu01_scaled=box(mu[0][1]))


def disk_gram(terms,n):
    stride=2*n+1
    re=[A(0)]*(n*stride+n+1)
    im=re.copy()
    for (i,j),c in terms.items():
        re[i*stride+j],im[i*stride+j]=c.real,c.imag
    power=[A(1)]
    fac=[1]
    for i in range(1,2*n+2):
        power.append(power[-1]*(A(i)-A(1)/2))
        fac.append(fac[-1]*i)
    G=A(0)
    for index,c in enumerate((arb_poly(re)**2+arb_poly(im)**2).coeffs()):
        i,j=divmod(index,stride)
        if i+j>2*n:
            demand(c.is_zero(),"Kronecker alias")
        else:
            G+=c*power[i]*power[j]/fac[i+j+1]
    return G


def check_endpoints(lo,hi):
    demand(lo.is_finite() and hi.is_finite() and hi>=lo>=0, "invalid final endpoint order or sign")


def endpoints(G,eps,slack):
    demand(G.is_finite() and G.lower()<=G.upper() and G.upper()>=0,"invalid Gram interval")
    for name,value in [("interpolation",eps),("field",slack)]:
        demand(value.is_finite() and value.lower()>=0,name+" remainder invalid")
    lower=max(A(0),(max(A(0),G.lower()).sqrt()-slack.upper()).lower())
    upper=(G.upper().sqrt()+slack.upper()).upper()
    check_endpoints(lower,upper)
    return lower,upper


def certify(*,p="12",nf="10",theta="0.3",ar="1",radius="5",degree=96,bits=768,model="Pade",Q=None,rho="2"):
    rho=scalar(rho,"rho")
    demand(Decimal(rho)>1,"rho must exceed one")
    model_name(model)
    demand(type(degree) is int and degree>=1,"degree must be an integer >= 1")
    demand(type(bits) is int and bits>=64,"precision must be an integer >= 64 bits")
    raw={k:scalar(v,k) for k,v in dict(p=p,nf=nf,theta=theta,ar=ar,radius=radius).items()}
    requested=None if Q is None else scalar(Q,"Q")
    with ctx.workprec(bits):
        pp,nf,theta,ar,r=[positive(A(s),k) for k,s in raw.items()]
        ax,ay,z,lam,K,qmax,q=geometry(pp,nf,theta,ar,r,None if requested is None else A(requested))
        area=4*ax*ay*(1+1/pp).gamma()**2/(1+2/pp).gamma()
        demand(area.overlaps(100*A.pi()),"aperture area mismatch")
        coeff,eps,margins,contour,lobatto=spectrum(K,q,degree,model,rho)
        m,e=contour["M"]
        demand(eps.is_finite() and eps>= (4*A((int(m),e))/(A(rho)**degree*(A(rho)-1))).upper(),"invalid interpolation budget")
        terms,moments=source_transform(coeff,pp,ax,ay,z,q,r)
        G=disk_gram(terms,degree)/(lam*z)**2
        slack=(100*A.pi()*eps/(lam*z)).upper()
        lo,hi=endpoints(G,eps,slack)
        check_endpoints(lo,hi)
        with ctx.extraprec(2*bits+64):
            demand(hi>=slack and (hi-slack)**2>=G.upper() and
                   (lo==0 or (lo+slack)**2<=max(A(0),G.lower())), "final endpoint rounded inward")
        evidence=dict(physical=dict(z=box(z),wavelength=box(lam),K=box(K)),
            geometry=dict(ax=box(ax),ay=box(ay),qmax=box(qmax),Q=dyadic(q)),
            identity=dict(model=model,illumination=1,carrier="removed",measure="disk area average"),
            analyticity=margins,contour=contour,interpolation=lobatto,source=moments,
            gram=dict(G=box(G),stride=2*degree+1),remainder=dict(delta=dyadic(slack)),
            endpoints=dict(lower=dyadic(lo),upper=dyadic(hi)))
        return dict(schema=1,status="interval-certified",inputs=dict(**raw,degree=degree,bits=bits,
                    model=model,Q=requested,rho=rho),flint=flint.__version__,native_flint=flint.__FLINT_VERSION__,
                    generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    obligations={k:dict(passed=True,evidence=v) for k,v in evidence.items()})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for key,value in dict(p="12",nf="10",theta="0.3",ar="1",radius="5",model="Pade",Q=None,rho="2").items():
        ap.add_argument("--"+key,default=value)
    ap.add_argument("--degree",type=int,default=96)
    ap.add_argument("--bits",type=int,default=768)
    ap.add_argument("--out",type=Path,required=True)
    options=vars(ap.parse_args())
    dest=options.pop("out")
    record=certify(**options)
    with dest.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2)

