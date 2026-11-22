import sys,platform,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import D,read,dump
from rho_independent_verifier import check
import flint
expected=dict(p='4',nf='10',theta='0.1',ar='1',radius='5',model='Pade',degree=64,bits=512,rho='1.75',Q=None)
record=read(D/'records/probe.json')['certificate']
proof=check(record,expected)
dump(D/'audits/wsl_probe.json',dict(verified=proof['verified'],met=proof['target_met'],platform=platform.platform(),flint=flint.__version__,native=flint.__FLINT_VERSION__,endpoints=record['obligations']['endpoints']['evidence']))

