"""Common degree/precision scheduler shared by fixed- and variable-rho policies."""
from fractions import Fraction as F
def choose(scans,model,theta,done,previous=None,joint=True):
    valid=[r for r in scans if r.get('analytic')]
    if not valid:return None
    # At development scales small-angle Pade residuals need a tighter first guess.
    threshold=F('2e-12') if model=='Pade' and F(theta)<=F('0.12') else F('1e-8') if model=='Pade' else F('4e-6')
    if previous and F(previous['L'])>0:
        threshold=min(F('4e-6'),F(previous['L'])/250)
    for n in (64,80,96,128):
        best=min((r for r in valid if r['n']==n),key=lambda r:F(r['delta']))
        bits=(512 if n<=80 else 768 if n==96 else 1024) if joint else 1024
        if n not in done and (F(best['delta'])<=threshold or n==128):
            return n,bits,best['rho']
    return None

