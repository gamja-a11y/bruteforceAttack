"""Reproduce the two extended, separately executable source copies."""
from common import *
def change(s,a,b):
    if s.count(a)!=1: raise RuntimeError(a)
    return s.replace(a,b)
s=(OLD/'repaired_certification.py').read_text(encoding='utf8')
s=change(s,'    demand(qmax.upper()<8, "rho=2 ellipse cannot exclude the square-root cut")','')
s=change(s,'def spectrum(K,Q,n,model):','''def contour(K,Q,model,rho):
    rho_value=positive(A(rho),"rho")
    demand(rho_value>1,"rho must exceed one")
    major=Q*(rho_value+1/rho_value)/4
    minor=Q*(rho_value-1/rho_value)/4''')
s=s.replace('A(0,1)*5*h/4','A(0,1)*major').replace('A(0,1)*3*h/4','A(0,1)*minor').replace('5*h*theta.cos()/4','major*theta.cos()').replace('3*h*theta.sin()/4','minor*theta.sin()')
s=change(s,'    nodes=[','''    return margins,dict(rho=rho,panels=512,arc_uppers=arcs,M=dyadic(M))


def spectrum(K,Q,n,model,rho):
    margins,record=contour(K,Q,model,rho)
    M=A((int(record["M"][0]),record["M"][1]))
    nodes=[''')
s=change(s,'eps=(4*M/2**n).upper()','r=A(rho)\n    eps=(4*M/(r**n*(r-1))).upper()')
s=change(s,'dict(rho=2,panels=512,arc_uppers=arcs,M=dyadic(M)),dict(','record,dict(')
s=change(s,'model="Pade",Q=None):','model="Pade",Q=None,rho="2"):\n    rho=scalar(rho,"rho")\n    demand(Decimal(rho)>1,"rho must exceed one")')
s=s.replace('spectrum(K,q,degree,model)','spectrum(K,q,degree,model,rho)').replace('eps>=4*A((int(m),e))/2**degree','eps>= (4*A((int(m),e))/(A(rho)**degree*(A(rho)-1))).upper()')
s=s.replace('model=model,Q=requested)','model=model,Q=requested,rho=rho)').replace('radius="5",model="Pade",Q=None).items()', 'radius="5",model="Pade",Q=None,rho="2").items()')
# Per-row correlation. Reversing the source y index makes the required sum
# an ordinary real/complex polynomial product. The x index remains explicit.
a=s.index('    terms={}')
b=s.index('    demand(all(c.is_finite()',a)
s=s[:a]+'''    from math import factorial
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
'''+s[b:]
(D/'src/rho_certifier.py').write_text(s,encoding='utf8')
s=(OLD/'repaired_independent_verification.py').read_text(encoding='utf8')
s=s.replace('bits model Q".split()', 'bits model Q rho".split()').replace('    for k in ("p","nf"', '    need(type(d["rho"]) is str and Decimal(d["rho"]).is_finite() and Decimal(d["rho"])>1,"rho")\n    for k in ("p","nf"')
s=s.replace('def contour_check(ev,Q,K,model):','''def contour_check(ev,Q,K,model,rho):
    r=A(rho)
    major=Q*(r+1/r)/4
    minor=Q*(r-1/r)/4''').replace('A(0,1)*5*h/4','A(0,1)*major').replace('A(0,1)*3*h/4','A(0,1)*minor').replace('5*h*angle.cos()/4','major*angle.cos()').replace('3*h*angle.sin()/4','minor*angle.sin()').replace('cont["rho"]==2','cont["rho"]==rho')
s=s.replace('def verify(record,expected=None):','def verify(record,expected):\n    need(isinstance(expected,dict),"external specification required")').replace('ROOT/"repaired_certification.py"','ROOT/"rho_certifier.py"').replace('contour_check(ev,Q,K,model)','contour_check(ev,Q,K,model,d["rho"])').replace('eps>=4*M/2**n','eps>=(4*M/(A(d["rho"])**n*(A(d["rho"])-1))).upper()')
# Sum by aperture x exponent and output x exponent, using a polynomial
# product along the x (rather than y) coordinate by symmetry/transposition.
a=s.index('    for a in range(n+1):\n        for b in range(n+1-a):')
b=s.index('    re,im=[],[]',a)
s=s[:a]+'''    from math import factorial
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
'''+s[b:]
s=s[:s.index('if __name__=="__main__":')]+'''def check(record,expected):
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
'''
(D/'src/rho_independent_verifier.py').write_text(s,encoding='utf8')

