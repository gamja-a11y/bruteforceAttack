"""Matched single-condition v4 validated boundary quadrature and fixed moment control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from adaptive_certification import HERE, peak_mib, search, decimal_out, rational


def main():
    if sys.argv[1:]==["--worker"]:
        sys.path.insert(0,str(HERE/"legacy"))
        import fresnel_certified_bound_v4 as v
        _,A,C,ctx=v.backend()
        ctx.prec=128
        v.symbolic_proof()
        nodes,_=v.gauss(A,12)
        orig,geo=v.continuous(A)
        other,rho=v.geometric(A)
        if not orig.overlaps(other) or not all(0<t<1 for t in nodes):
            raise ValueError("legacy prerequisites")
        derivative,_=v.derivative_bound(A)
        cells,_,_,upper=v.certify_grid(A,v.make_center_solver(A,C,12,24),1,1,derivative,rho.upper())
        def render(x):
            m,e=x.upper().man_exp()
            return decimal_out(rational([str(m),int(e)]),True)
        print(json.dumps(dict(lower="0",upper=render(upper),geometric_upper=render(geo),
                              peak_worker_mib=peak_mib(),cells=cells,gl_order=12,panels=24,bits=128)))
        return
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    dest=ap.parse_args().out
    dest.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    result=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker"],capture_output=True,text=True,check=True)
    row=json.loads(result.stdout)
    row["seconds"]=time.perf_counter()-start
    (dest/"boundary.json").write_text(json.dumps(row,indent=2),encoding="utf8")
    search(dict(p="12",nf="10",ar="1",theta="0.3",radius="5",model="Pade"),dest/"fixed",
           fixed=(96,768),maximum_degree=96,maximum_bits=768,timeout_seconds=60,
           target_relative_width="0.01",target_absolute_width="1e-5")


if __name__=="__main__":
    main()

