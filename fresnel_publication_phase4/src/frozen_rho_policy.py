from higher_degree_baseline import choose
from compute import RHOS,scan
from common import ROOT,sys
sys.path.append(str(ROOT/'fresnel_publication_phase3'))
import frozen_selection_rule as phase3
def candidates(params,method):
    if method=='A':
        n,b=phase3.start(params)
        raw=yield n,b,'2',[]
        while not raw['met']:
            from fractions import Fraction as F
            from common import D,read,dy
            ev=read(D/raw['file'])['certificate']['obligations']
            g=list(map(dy,ev['gram']['evidence']['G']))
            q=phase3.advance(n,b,dy(ev['endpoints']['evidence']['lower']),g[1]-g[0],dy(ev['remainder']['evidence']['delta']),dy(ev['contour']['evidence']['M']),F(params['nf']),F('1e-5'),F('.01'))
            if q is None:return
            n,b=q
            raw=yield n,b,'2',[]
        return
    scans=scan(params,512,['2'] if method=='B' else RHOS)
    used=set()
    last=None
    while True:
        choice=choose(scans,params['model'],params['theta'],used,last,joint=method!='Cfixedbits')
        if choice is None:return
        n,b,rho=choice
        last=yield n,b,rho,scans if not used else []
        used.add(n)
        if last['met'] or n==128:return
        # Avoid returning to a lower degree after trying a higher degree.
        used.update(k for k in (64,80,96,128) if k<n)


