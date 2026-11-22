from copy import deepcopy
from compute import *

def main():
    path=D/'records/probe.json'
    bundle=read(path)
    expected=dict(p='4',nf='10',ar='1',theta='0.1',radius='5',model='Pade',Q=None,n=64)
    expected['degree']=expected.pop('n')
    expected.update(bits=512,rho='1.75')
    rec=bundle['certificate']
    if not verify.check(rec,expected)['verified']: raise RuntimeError('positive control')
    ev=rec['obligations']
    tests=[('Q',['obligations','geometry','evidence','Q'],['1',-20],'continuous q'),
           ('M',['obligations','contour','evidence','M'],['0',0],'contour panel'),
           ('rho',['inputs','rho'],'2','requested inputs differ'),
           ('arc',['obligations','contour','evidence','arc_uppers'],ev['contour']['evidence']['arc_uppers'][:-1],'incomplete contour'),
           ('endpoints',['obligations','endpoints','evidence','upper'],['0',0],'final domain'),
           ('model',['inputs','model'],'Fresnel','requested inputs differ'),
           ('forged success',['target_met'],True,'target flag'),
           ('unissued',['status'],'unverified','no issued certificate'),
           ('epsilon',['obligations','interpolation','evidence','epsilon'],['0',0],'remainder understated')]
    rows=[dict(test='positive control',expected='verified',observed='verified',passed=True)]
    for name,keys,value,reason in tests:
        obj=deepcopy(rec);v=obj
        for k in keys[:-1]:v=v[k]
        v[keys[-1]]=value
        try:verify.check(obj,expected);msg='accepted'
        except ValueError as exc:msg=str(exc)
        rows.append(dict(test=name,expected=reason,observed=msg,passed=reason in msg))
    writecsv(D/'audits/negative_integrity_tests.csv',rows)
    if not all(r['passed'] for r in rows):raise RuntimeError(rows)
if __name__=='__main__':main()

