from pathlib import Path
import sys,hashlib,json,csv,os
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
D=Path(__file__).resolve().parents[1]
ROOT=D.parent
OLD=ROOT/'theorem1_certification_repaired'
sys.path.insert(0,str(OLD))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for d in ('theorem1_certification_repaired','fresnel_publication_phase2','fresnel_publication_phase3','fresnel_publication_phase3_audit_repaired') for p in sorted((ROOT/d).rglob('*')) if p.is_file()}
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf8')
def dy(t): return F(int(t[0]))*F(2)**t[1]
def dec(x,upper=True):
    with localcontext() as c:
        c.prec=24
        c.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        return str(Decimal(x.numerator)/Decimal(x.denominator))
def writecsv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
if __name__=='__main__':
    p=D/'audits/source_hash_audit.json'
    if sys.argv[1]=='before':
        with p.open('x') as f: json.dump({'before':hashes()},f)
    else:
        obj=read(p);obj['after']=hashes();obj['unchanged']=obj['before']==obj['after'];dump(p,obj)
        if not obj['unchanged']: raise ValueError('protected file changed')
