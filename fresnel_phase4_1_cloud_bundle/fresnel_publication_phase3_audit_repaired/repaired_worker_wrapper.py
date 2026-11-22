"""Checked subprocess adapter. Does not issue certificates in this audit."""
import subprocess,json
from repaired_analyze import verifier

def checked_worker(argv,expected,seconds):
    try:
        job=subprocess.run(argv,input=json.dumps({k:v for k,v in expected.items() if k!='Q'}),capture_output=True,text=True,timeout=seconds)
    except subprocess.TimeoutExpired: return 'timeout',None
    if job.returncode:
        msg=job.stderr.strip()
        return ('nonzero_exit_with_stderr' if msg else 'nonzero_exit_without_stderr'),(job.returncode,msg)
    try:
        obj=json.loads(job.stdout)
    except ValueError as exc: return 'malformed_worker_output',str(exc)
    if not isinstance(obj,dict) or not isinstance(obj.get('certificate'),dict) or obj['certificate'].get('inputs')!=expected:
        return 'certificate_integrity_failure',None
    try:
        proof=verifier.verify(obj['certificate'],expected)
    except Exception as exc: return 'independent_verification_failure',str(exc)
    if proof.get('verified') is not True: return 'independent_verification_failure',None
    if obj.get('verification')!=proof: return 'certificate_integrity_failure','saved evidence mismatch'
    return 'verified',proof

