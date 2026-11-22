"""Prespecified after inspecting the 12-case pilot. Store separately."""
import argparse
from pathlib import Path
from adaptive_certification import search
from run_pilot import GEOMETRIES,TARGET,params


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    out=ap.parse_args().out
    out.mkdir(parents=True,exist_ok=False)
    for rep in range(1,4):
        for job in (["mid","oracle"] if rep%2 else ["oracle","mid"]):
            opts=dict(TARGET)
            if job=="oracle":
                opts["fixed"]=(24,256)
            else:
                opts["target_absolute_width"]=None
            search(params(GEOMETRIES[0],"Fresnel"),out/f"bench_{job}_{rep}",**opts)
    for i,model in [(5,"Pade"),(6,"Pade"),(6,"Fresnel")]:
        options=dict(TARGET,maximum_degree=96,maximum_bits=768,timeout_seconds=60)
        x=search(params(GEOMETRIES[i-1],model),out/f"follow_{i}_{model}",**options)
        print(i,model,x["status"],x["reason"],round(x["seconds"],2),flush=True)


if __name__=="__main__":
    main()

