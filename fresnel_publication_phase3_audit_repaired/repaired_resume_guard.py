"""Prospective resume only. Keep the externally pinned receipt outside the run."""
from repaired_analyze import *

def policy(m,case,model,method):
    need(type(case) is int and 1<=case<=len(m['cases']) and model in m['models'] and method in ('A','B','C'),'requested_policy')
    rules={'A':ROOT/'fresnel_publication_phase2/adaptive_certification.py','B':PH/'frozen_selection_rule.py','C':PH/'experiment.py'}
    return dict(case=case,params=dict(zip(m['order'],m['cases'][case-1]),radius=m['radius'],model=model),method=method,rule_sha256=sha(rules[method]),manifest_sha256=sha(PH/'holdout_manifest.json'),driver_sha256=sha(PH/'experiment.py'),issuer=sha(verifier.ROOT/'repaired_certification.py'),verifier=sha(verifier.ROOT/'repaired_independent_verification.py'),budget={k:m[k] for k in ('degrees','bits','seconds_per_method','process_mib','absolute','relative','fixed')})

def check_resume(folder,case,model,method,receipt,pinned_digest):
    m=manifest()
    need(receipt.is_file() and sha(receipt)==pinned_digest,'receipt_hash')
    prior=read(receipt)
    need(prior.get('policy')==policy(m,case,model,method),'resume_policy')
    need(prior.get('files')=={p.name:sha(p) for p in folder.iterdir() if p.is_file()},'resume_files')
    return inspect_run(folder,m,case,model,method)

