"""Six geometries, both residuals, followed by serial repeated baselines."""
import argparse
from pathlib import Path
from adaptive_certification import search

GEOMETRIES=[("2","1","1","0.1"),("8","1","1","0.1"),("16","1","1","0.1"),
            ("8","3","2","0.3"),("2","10","1","0.1"),("16","10","4","0.3")]
TARGET=dict(target_relative_width="0.01",target_absolute_width="1e-5",maximum_degree=64,
            maximum_bits=512,timeout_seconds=30)


def params(geometry,model):
    p,nf,ar,theta=geometry
    return dict(p=p,nf=nf,ar=ar,theta=theta,radius="5",model=model)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    for i,geometry in enumerate(GEOMETRIES,1):
        for model in ("Pade","Fresnel"):
            name=f"pilot_{i}_{model}"
            result=search(params(geometry,model),args.out/name,**TARGET)
            print(name,result["status"],round(result["seconds"],2),flush=True)
    # Three paired, interleaved repetitions for a common circular Fresnel target.
    # Relative-only tolerance sweep isolates one control; primary tolerance uses both.
    for rep in range(1,4):
        jobs=["fixed","adaptive","loose","tight"]
        jobs=jobs[rep-1:]+jobs[:rep-1]
        for job in jobs:
            opts=dict(TARGET)
            if job=="fixed":
                opts["fixed"]=(64,512)
            elif job in ("loose","tight"):
                opts.update(target_relative_width="0.1" if job=="loose" else "1e-8",target_absolute_width=None)
            name=f"bench_{job}_{rep}"
            result=search(params(GEOMETRIES[0],"Fresnel"),args.out/name,**opts)
            print(name,result["status"],round(result["seconds"],2),flush=True)


if __name__=="__main__":
    main()

