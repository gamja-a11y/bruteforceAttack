from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent
ROOT=D.parent
PH=ROOT/'fresnel_publication_phase3'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot():
    paths=[ROOT/'phase3_wsl_independent_verification.json']
    for d in (PH,ROOT/'theorem1_certification_repaired',ROOT/'fresnel_publication_phase2'):
        paths.extend(p for p in d.rglob('*') if p.is_file())
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
if __name__=='__main__':
    path=D/'source_hash_audit.json'
    now=snapshot()
    if sys.argv[1]=='before':
        with path.open('x') as f: json.dump({'before':now},f)
    elif sys.argv[1]=='after':
        obj=json.loads(path.read_text())
        obj.update(after=now,unchanged=now==obj['before'])
        path.write_text(json.dumps(obj,indent=2))
        if not obj['unchanged']: raise SystemExit('source changed')
    else: raise ValueError('before or after required')
