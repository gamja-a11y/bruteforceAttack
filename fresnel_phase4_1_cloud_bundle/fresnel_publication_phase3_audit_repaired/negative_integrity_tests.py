"""Mutations use temporary copies only. Compare specific rejection codes."""
import shutil,subprocess,tempfile
from pathlib import Path
from repaired_analyze import *
from repaired_resume_guard import check_resume,policy
from repaired_worker_wrapper import checked_worker

def run_tests():
    m=manifest()
    results=[]
    def test(name,expected,call):
        try: observed=call()
        except AuditError as exc: observed=exc.code
        except Exception as exc: observed=f'unexpected {type(exc).__name__}: {exc}'
        results.append(dict(test=name,expected=expected,observed=observed,passed=observed==expected))
    def fixture(change=lambda s,p:None,nf10=False):
        with tempfile.TemporaryDirectory(dir=D) as tmp:
            case=9 if nf10 else 1
            name=f'{case:02}_Pade_C'
            d=Path(tmp)/name
            shutil.copytree(PH/'runs'/name,d)
            s=read(d/'search.json')
            p=d/s['attempts'][0]['certificate']
            change(s,p)
            (d/'search.json').write_text(json.dumps(s))
            inspect_run(d,m,case,'Pade','C')
            return 'verified'
    def mutate(path,keys,value):
        obj=read(path)
        parent=obj
        for k in keys[:-1]: parent=parent[k]
        parent[keys[-1]]=value
        path.write_text(json.dumps(obj))
    def patch(keys,value): return lambda s,p:mutate(p,keys,value)
    test('T1 normal complete run','verified',fixture)
    def empty(s,p):
        p.unlink()
        s['attempts']=[]
    test('T2 empty success','no_candidates',lambda:fixture(empty))
    for name,key,value in [('T3','model','Fresnel'),('T4p','p','12'),('T4nf','nf','3'),('T4ar','ar','2'),('T4theta','theta','0.2'),('T5n','degree',16),('T5b','bits',256)]:
        test(name,'candidate_inputs',lambda:fixture(patch(['certificate','inputs',key],value)))
    test('T6 saved flag changed','saved_verification_mismatch',lambda:fixture(patch(['verification','verified'],False)))
    test('T7 bad JSON','malformed_json',lambda:fixture(lambda s,p:p.write_text('{')))
    test('T7 missing proof despite verified=True','independent_verification_failure',lambda:fixture(patch(['certificate','obligations'],{})))
    test('T8 forged success','run_target_flag',lambda:fixture(lambda s,p:s.update(status='target_met'),True))
    test('T9 inward endpoint','independent_verification_failure',lambda:fixture(patch(['certificate','obligations','endpoints','evidence','upper'],['0',0])))
    def resume(tamper):
        with tempfile.TemporaryDirectory(dir=D) as tmp:
            receipt=Path(tmp)/'trusted.json'
            d=PH/'runs/01_Pade_C'
            config=policy(m,1,'Pade','C')
            if tamper: config[tamper]='wrong'
            receipt.write_text(json.dumps(dict(policy=config,files={p.name:sha(p) for p in d.iterdir()})))
            check_resume(d,1,'Pade','C',receipt,sha(receipt))
            return 'verified'
    test('T10 matching receipt','verified',lambda:resume(None))
    for k in ('manifest_sha256','params','rule_sha256','budget','issuer','verifier'):
        test('T10 '+k,'resume_policy',lambda:resume(k))
    for code,expected in [('raise SystemExit(7)','nonzero_exit_without_stderr'),('raise ValueError(5)','nonzero_exit_with_stderr'),('print("oops")','malformed_worker_output'),('print("{}")','certificate_integrity_failure'),('import time;time.sleep(5)','timeout')]:
        test('T11 '+expected,expected,lambda:checked_worker([sys.executable,'-B','-c',code],{},0.25 if expected=='timeout' else 5)[0])
    def tiny():
        a,b=F(1),F(1)+F(1,2**200)
        l,u=gram_width(a,b)
        # (1+w)^2=b gives independent rational bracket test.
        return 'positive_enclosed' if float(a)==float(b) and 0<l<=u and (1+l)**2<=b<=(1+u)**2 and gram_width(-1,F(4))==(2,2) and gram_width(-1,F(0))==(0,0) else 'bad'
    test('T12 stable Gram width','positive_enclosed',tiny)
    test('T13 disjoint','disjoint_intervals',lambda:overlap([(F(0),F(1)),(F(2),F(3))]))
    test('T14 missing','missing_certificate',lambda:fixture(lambda s,p:p.unlink()))
    test('T14 duplicate','duplicate_candidate',lambda:fixture(lambda s,p:s['attempts'].extend(s['attempts'][:1])))
    test('T15 unregistered','unreferenced_file',lambda:fixture(lambda s,p:shutil.copy(p,p.with_name('extra.json'))))
    table(D/'integrity_test_results.csv',results)
    print([r for r in results if not r['passed']] or f'{len(results)} passed')
    if not all(r['passed'] for r in results): raise SystemExit(1)

if __name__=='__main__': run_tests()

