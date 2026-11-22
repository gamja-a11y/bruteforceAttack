"""Read-only audit of completed Phase 3 pilot-budget runs (CPython 3.10+)."""
import csv,json,math,re,sys
from decimal import Decimal,localcontext,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction as F
from source_hash import D,ROOT,PH,sha
sys.path.insert(0,str(ROOT/'theorem1_certification_repaired'))
import repaired_independent_verification as verifier

class AuditError(ValueError):
    def __init__(self,code,detail=''):
        self.code=code
        super().__init__(f'{code}: {detail}')

def need(ok,code,detail=''):
    if not ok: raise AuditError(code,detail)

def read(p):
    def pairs(items):
        result=dict(items)
        need(len(result)==len(items),'duplicate_json_key',p)
        return result
    try: return json.loads(p.read_text(encoding='utf-8-sig'),object_pairs_hook=pairs)
    except (ValueError,OSError) as exc: raise AuditError('malformed_json',p) from exc

def decimal_input(s):
    need(type(s) is str and re.fullmatch(r'[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',s) and Decimal(s).is_finite() and Decimal(s)>0,'physical_input',s)
    return str(Decimal(s))

def manifest():
    m=read(PH/'holdout_manifest.json')
    for name,h in m['sha256'].items():
        need(sha(ROOT/name.replace('\\','/'))==h,'source_hash',name)
    for name in ('repaired_certification.py','repaired_independent_verification.py'):
        need(sha(ROOT/'theorem1_certification_repaired'/name)==sha(ROOT/'fresnel_publication_phase2'/'frozen'/name),'source_hash',name)
    need(m['order']==['p','nf','ar','theta'] and m['models']==['Pade','Fresnel'] and (F(m['absolute']),F(m['relative']))==(F(1,100000),F(1,100)),'manifest_schema')
    unique={tuple(map(F,c)) for c in m['cases']}
    need(len(unique)==len(m['cases'])==24 and not unique.intersection(tuple(map(F,c)) for c in m['excluded']),'manifest_cohort')
    return m

def dy(pair):
    return F(int(pair[0]))*F(2)**pair[1]

def dec(x,up):
    with localcontext() as c:
        c.prec=22
        c.rounding=ROUND_CEILING if up else ROUND_FLOOR
        return str(Decimal(x.numerator)/Decimal(x.denominator))

def sqrt_bounds(x):
    if x==0: return F(0),F(0)
    exponent=(x.numerator.bit_length()-x.denominator.bit_length())//2-192
    unit=F(2)**exponent
    scaled=x/unit**2
    k=math.isqrt(scaled.numerator//scaled.denominator)
    return k*unit,(k if (k*unit)**2==x else k+1)*unit

def gram_width(a,b):
    need(a<=b and b>=0,'gram_domain')
    a=max(F(0),a)
    if a==b: return F(0),F(0)
    al,au=sqrt_bounds(a)
    bl,bu=sqrt_bounds(b)
    return (b-a)/(au+bu),(b-a)/(al+bl)

def overlap(intervals):
    need(intervals and max(i[0] for i in intervals)<=min(i[1] for i in intervals),'disjoint_intervals')

def inventory(expected,actual):
    need(not expected-actual,'missing_certificate',sorted(expected-actual))
    need(not actual-expected,'unreferenced_file',sorted(actual-expected))

def verify_candidate(path,expected):
    payload=read(path)
    need(isinstance(payload,dict) and isinstance(payload.get('certificate'),dict),'certificate_schema')
    rec=payload['certificate']
    need(rec.get('inputs')==expected,'candidate_inputs',path)
    try: fresh=verifier.verify(rec,expected)
    except Exception as exc: raise AuditError('independent_verification_failure',str(exc)) from exc
    need(fresh.get('verified') is True,'independent_verification_failure')
    need(fresh==payload.get('verification'),'saved_verification_mismatch',path)
    return rec

def inspect_run(folder,m,case,model,method):
    import frozen_selection_rule as rule
    params=dict(zip(m['order'],map(decimal_input,m['cases'][case-1])),radius=decimal_input(m['radius']),model=model)
    need(folder.name==f'{case:02}_{model}_{method}','run_path')
    s=read(folder/'search.json')
    need(all(s.get(k)==v for k,v in params.items()),'run_inputs')
    if method!='B':
        need((s['maximum_degree'],s['maximum_bits'],s['timeout_seconds'])==(max(m['degrees']),max(m['bits']),m['seconds_per_method']) and (F(s['target_absolute_width']),F(s['target_relative_width']))==(F(m['absolute']),F(m['relative'])),'run_budget')
    attempts=s['attempts']
    need(isinstance(attempts,list) and attempts,'no_candidates')
    refs=[r.get('certificate') for r in attempts]
    need(all(isinstance(v,str) for v in refs),'missing_certificate_reference')
    need(len(refs)==len(set(refs)),'duplicate_candidate')
    inventory(set(refs)|{'search.json'},{p.name for p in folder.iterdir()})
    expected_pair={'A':(m['degrees'][0],m['bits'][0]),'B':rule.start(params),'C':tuple(m['fixed'])}[method]
    rows=[]
    goals=[]
    intervals=[]
    ranks=[]
    rt,at=F(m['relative']),F(m['absolute'])
    for r in attempts:
        n,b=r['degree'],r['bits']
        need(type(n) is int and type(b) is int and (n,b)==expected_pair and n in m['degrees'] and b in m['bits'],'candidate_schedule',(expected_pair,n,b))
        name=f'n{n}_b{b}.json'
        need(r['certificate']==name and r['status']=='verified','candidate_filename')
        rec=verify_candidate(folder/name,dict(**params,bits=b,degree=n,Q=None))
        ev={k:v['evidence'] for k,v in rec['obligations'].items()}
        l,u=[dy(ev['endpoints'][k]) for k in ('lower','upper')]
        w=u-l
        rel=2*w/(u+l) if u+l else None
        rok=l>0 and rel is not None and rel<=rt
        aok=w<=at
        need(r['absolute_met'] is aok and r['relative_met'] is rok,'candidate_target_flag')
        for k,val,up in [('lower',l,False),('upper',u,True),('absolute_width',w,True),('relative_width',rel,True)]:
            need(r[k]==(None if val is None else dec(val,up)),'candidate_display',k)
        a,h=map(dy,ev['gram']['G'])
        gl,gu=gram_width(a,h)
        delta=dy(ev['remainder']['delta'])
        M=dy(ev['contour']['M'])
        rows.append(dict(case=case,model=model,method=method,degree=n,bits=b,file=name,gram_lower=a,gram_upper=h,gram_squared_width=h-a,gram_norm_width_lower=dec(gl,False),gram_norm_width_upper=dec(gu,True),kernel_epsilon=dy(ev['interpolation']['epsilon']),field_delta=delta,final_width=w,lower_zero=l==0,L=l,U=u,relative_width=rel,absolute_met=aok,relative_met=rok))
        goals.append(aok and rok)
        intervals.append((l,u))
        ranks.append((max(w/at,(rel if l else F(2))/rt),w) if method=='A' else (w,))
        expected_pair=None
        if not goals[-1]:
            if method=='B':
                expected_pair=rule.advance(n,b,l,h-a,delta,M,F(params['nf']),at,rt)
            elif method=='A':
                if w>4*delta and b<m['bits'][-1]: expected_pair=(n,m['bits'][m['bits'].index(b)+1])
                elif n<m['degrees'][-1]: expected_pair=(m['degrees'][m['degrees'].index(n)+1],m['bits'][0])
    met=any(goals)
    need(s['status']==('target_met' if met else 'verified_target_unmet'),'run_target_flag')
    need(expected_pair is None,'incomplete_search',expected_pair)
    best=goals.index(True) if met else min(range(len(ranks)),key=ranks.__getitem__)
    need(s['best']==attempts[best],'best_mismatch')
    overlap(intervals)
    return rows,best,s

sys.path.append(str(PH))
def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys())
        w.writeheader()
        w.writerows(rows)

