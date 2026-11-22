import sys,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compute import *
p=dict(p='4',nf='10',ar='1',theta='0.1',radius='5',model='Pade')
target=D/'audits/wsl_generated_certificate.json'
result=evaluate(p,80,512,'3',target)
def ends(path):
    ev=read(path)['certificate']['obligations']['endpoints']['evidence']
    return [dy(ev[k]) for k in ('lower','upper')]
old=ends(D/'records/dev_9_Pade_80_512_3.json')
new=ends(target)
if max(old[0],new[0])>min(old[1],new[1]) or not result['met']:raise ValueError('reproduction failure')
dump(D/'audits/wsl_regeneration.json',dict(verified=True,met=True,overlap=True,identical_endpoints=old==new,platform=platform.platform(),windows_source='records/dev_9_Pade_80_512_3.json',rho='3',degree=80,bits=512))

