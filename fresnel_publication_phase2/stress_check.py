"""Isolate the degree bottleneck in G6 after the capped adaptive pilot."""
import argparse
from pathlib import Path
from adaptive_certification import search
from run_pilot import params,GEOMETRIES,TARGET

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True,type=Path)
    out=ap.parse_args().out
    out.mkdir(parents=True,exist_ok=False)
    for model in ("Pade","Fresnel"):
        opts=dict(TARGET,maximum_degree=128,maximum_bits=1024,timeout_seconds=90)
        x=search(params(GEOMETRIES[5],model),out/model,fixed=(128,1024),**opts)
        print(model,x["status"],round(x["seconds"],2),flush=True)

