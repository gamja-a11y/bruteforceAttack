from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compute import *
cases=[('rho_at_one',A(1),'1','Fresnel','rho must exceed'),('rho_below_one',A(1),'0.9','Pade','rho must exceed'),('rectangle_meets_branch',A(8),'2','Fresnel','branch'),('pade_rectangle_guard',A(6),'2','Pade','complex domain')]
rows=[]
for name,Q,rho,model,expected in cases:
    try:gen.contour(A(100),Q,model,rho);reason='accepted'
    except ValueError as exc:reason=str(exc)
    rows.append(dict(test=name,Q=str(Q),rho=rho,model=model,expected=expected,observed=reason,passed=expected in reason))
writecsv(D/'audits/analyticity_counterexamples.csv',rows)
if any(not r['passed'] for r in rows):raise ValueError(rows)

