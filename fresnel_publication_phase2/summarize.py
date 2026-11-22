"""Build publication tables and figures from saved, independently checked trials."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
import platform
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from adaptive_certification import HERE, HASHES, check_sources, rational, decimal_out
from run_pilot import GEOMETRIES

MODELS=("Pade","Fresnel")
COLORS=("#335e89","#bd5329")
META="experiment_manifest.json"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def bounds(base,x):
    entry=read(base/x["best"]["certificate"])
    if not entry["verification"]["verified"]:
        raise ValueError("unverified candidate")
    ends=entry["certificate"]["obligations"]["endpoints"]["evidence"]
    return [rational(ends[k]) for k in ("lower","upper")]


def write_csv(path,rows):
    with path.open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def common(row):
    return {k:row[k] for k in ("model","p","nf","theta","ar","radius","target_relative_width",
                 "target_absolute_width","maximum_degree","maximum_bits","timeout_seconds")}


def summary(path,x):
    b=x["best"] or {}
    return dict(**common(x),status=x["status"],failure_reason=x["reason"] if x["status"]!="target_met" else "",
        **{k:b.get(k) for k in ("degree","bits","lower","upper","relative_width","absolute_width","relative_met","absolute_met")},
        runtime_s=x["seconds"],peak_worker_mib=x["peak_worker_mib"],attempts=len(x["attempts"]),
        certificate=str(path/b["certificate"]) if b else None,manifest=META)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("."))
    args=ap.parse_args()
    root,out=args.data,args.out
    out.mkdir(exist_ok=True,parents=True)
    check_sources()
    physical={"model":"Fresnel","p":"2","nf":"1","theta":"0.1","ar":"1","radius":"5"}
    rows,bench=[],[]
    trials={}
    intervals={}
    for cohort,pattern in [("runs","pilot_*"),("controls","follow_*")]:
        for path in sorted((root/cohort).glob(pattern)):
            _,i,m=path.name.split("_")
            x=read(path/"search.json")
            key=(cohort,int(i),m)
            trials[key]=x
            if x["best"]:
                intervals[key]=bounds(path,x)
            rows.append(dict(cohort="pilot" if cohort=="runs" else "followup",geometry="G"+i,
                             **summary(path.relative_to(root),x)))
    for row in rows:
        i=int(row["geometry"][1:])
        def interval(m):
            extended=("controls",i,m)
            return intervals.get(extended) if row["cohort"]=="followup" and extended in intervals else intervals.get(("runs",i,m))
        a,b=interval("Fresnel"),interval("Pade")
        if a and b and b[0]>0:
            row.update(ratio_lower=decimal_out(a[0]/b[1],False),ratio_upper=decimal_out(a[1]/b[0],True),
                       ratio_status="bounded",ratio_basis="followup where available, otherwise pilot" if row["cohort"]=="followup" else "pilot pair")
        else:
            row.update(ratio_status="no finite certified ratio; missing positive denominator")
    benchtrials={}
    for cohort in ("runs","controls"):
        for path in sorted((root/cohort).glob("bench_*")):
            _,kind,rep=path.name.split("_")
            x=read(path/"search.json")
            benchtrials.setdefault(kind,[]).append(x)
            bench.append(dict(method=kind,rep=rep,geometry="G1",measurement="independently verified",**summary(path.relative_to(root),x)))
    ref=read(root/"reference_runs.json")
    for row in ref:
        v=dict(method=row["method"],rep=row["rep"],geometry="G1",**physical,
               measurement=row["status"],runtime_s=row["seconds"],manifest=META,
               **{k:v for k,v in row.items() if k not in ("method","rep","seconds","status")})
        v.update(target_relative_width="0.01",target_absolute_width="1e-5")
        if row["method"]=="range":
            v.update(status="verified_target_unmet",relative_met=False,absolute_met=False,relative_width="2")
            if float(row["lower"])!=0 or float(row["upper"])<=1e-5:
                raise ValueError("review range target labels")
        else:
            v.update(status="empirical; no certified stopping test")
        bench.append(v)
    leg=read(root/"legacy_run"/"boundary.json")
    fixed_path=root/"legacy_run"/"fixed"
    fixed=read(fixed_path/"search.json")
    bench.append(dict(method="legacy v4 boundary GL",rep=1,geometry="legacy matched",
                **common(fixed),measurement="legacy upper bound; separate v4 proof",status="verified_target_unmet",
                lower=leg["lower"],upper=leg["upper"],geometric_upper=leg["geometric_upper"],
                absolute_width=leg["upper"],relative_width="2",relative_met=False,absolute_met=False,
                peak_worker_mib=leg["peak_worker_mib"],runtime_s=leg["seconds"],bits=128,gl_order=12,panels=24,manifest=META))
    bench.append(dict(method="fixed96",rep=1,geometry="legacy matched",measurement="independently verified",
                      **summary(fixed_path.relative_to(root),fixed)))
    stress_records={}
    for model in MODELS:
        stress_path=root/"stress128"/model
        test=read(stress_path/"search.json")
        stress_records[model]=test
        bench.append(dict(method="fixed128 diagnostic",rep=1,geometry="G6",measurement=test["status"],
                          **summary(stress_path.relative_to(root),test)))
    if all(stress_records[m]["best"] for m in MODELS):
        p,f=[bounds(root/"stress128"/m,stress_records[m]) for m in MODELS]
        if p[0]>0:
            for b in bench[-2:]:
                b.update(ratio_lower=decimal_out(f[0]/p[1],False),ratio_upper=decimal_out(f[1]/p[0],True))
    write_csv(out/"adaptive_results.csv",rows)
    write_csv(out/"benchmark_results.csv",bench)

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    def figsave(fig,name,caption):
        fig.savefig(out/name,bbox_inches="tight",metadata={"Description":caption+"; "+META,"Software":"summarize.py"})
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.0),layout="constrained")
    for ax,m in zip(axes,MODELS):
        for i,color in [(1,"#335e89"),(5,"#23877c"),(6,"#b6592a")]:
            data=trials.get(("controls",i,m),trials["runs",i,m])
            pts=[a for a in data["attempts"] if a["status"]=="verified"]
            ax.semilogy([a["degree"] for a in pts],[float(a["absolute_width"]) for a in pts],"o-",c=color,
                        ms=4,label=f"G{i}")
        ax.axhline(1e-5,color="0.3",ls=":",label="Absolute target")
        ax.set(xlabel="Lobatto degree",title=m,ylabel="Certified RMS interval width")
        ax.legend(fontsize=8)
    figsave(fig,"error_vs_degree.png","Width of the physical residual enclosure; vertical segments are precision retries, follow-up included")
    fig,ax=plt.subplots(figsize=(6.7,3.8),layout="constrained")
    for kind in ("loose","mid","tight"):
        data=benchtrials[kind]
        tol=float(data[0]["target_relative_width"])
        ys=[x["seconds"] for x in data]
        ax.scatter([tol]*len(ys),ys,color=COLORS[0],s=22,alpha=.7)
        ax.plot(tol,statistics.median(ys),marker="_",color="black",ms=18)
        ax.annotate("n="+str(data[0]["best"]["degree"]),(tol,max(ys)),xytext=(0,6),textcoords="offset points",ha="center")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.margins(x=.15,y=.24)
    ax.set(xlabel="Requested relative width (absolute target disabled)",ylabel="Wall time including verification (s)",
           title="Circular Fresnel residual · three fresh runs per tolerance")
    figsave(fig,"runtime_vs_tolerance.png","G1, dots are runs; black marks are medians")
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.7),layout="constrained")
    for ax,m,c in zip(axes,MODELS,COLORS):
        for x,(a,b) in enumerate(((1,2),(2,3))):
            left,right=intervals["runs",a,m],intervals["runs",b,m]
            step=int(GEOMETRIES[b-1][0])-int(GEOMETRIES[a-1][0])
            l=float((right[0]-left[1])/step)
            h=float((right[1]-left[0])/step)
            ax.vlines(x,l,h,color=c,lw=3)
            ax.plot([x,x],[l,h],"_",color=c,ms=18)
        ax.set_xticks([0,1],["2 to 8","8 to 16"])
        ax.set(xlabel="Shape-exponent interval",ylabel="RMS change per unit p",title=m)
        ax.margins(x=.6,y=.25)
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0),useOffset=False)
    figsave(fig,"shape_sensitivity.png","Certified secant intervals, not derivatives; NF=1, theta=0.1, AR=1")
    fig,ax=plt.subplots(figsize=(7.4,3.9),layout="constrained")
    grid=np.zeros((6,6))
    for i in range(1,7):
        for j,m in enumerate(MODELS):
            for phase in range(3):
                x=(stress_records[m] if i==6 else None) if phase==2 else trials.get(("controls" if phase else "runs",i,m))
                code=0 if x is None else 2 if x["status"]=="target_met" else 1 if x["best"] else 3
                grid[i-1,j+2*phase]=code
                ax.text(j+2*phase,i-1,["—","Unmet","Met","No cert."][code],ha="center",va="center")
    ax.imshow(grid,cmap=ListedColormap(["#e5e8ea","#f5c8a7","#abdad0","#e598a1"]),vmin=-.5,vmax=3.5,aspect="auto")
    ax.set_xticks(range(6),["Padé","Fresnel"]*3)
    ax.set_yticks(range(6),[f"G{i}   ({p}, {nf}, {ar}, {t})" for i,(p,nf,ar,t) in enumerate(GEOMETRIES,1)])
    ax.set(xlabel="Adaptive ≤64          Adaptive ≤96           Fixed 128",ylabel="(p, NF, AR, theta)",title="Joint width targets by tested geometry")
    figsave(fig,"accuracy_failure_map.png","Blank means no follow-up; unmet retains valid certified interval")
    versions={k:version(k) for k in ("python-flint","numpy","scipy","matplotlib","sympy","mpmath")}
    import flint
    manifest=dict(python=sys.version,platform=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
                  packages=versions,flint=flint.__FLINT_VERSION__,frozen_sha256=HASHES,
                  geometry_order=["p","NF","AR","theta"],geometries=GEOMETRIES,area="100*pi",radius=5,
                  z="10/theta",wavelength="10*theta/NF",raw_norm="sqrt(int_disk |U_RS-U_model|^2 dA/(25*pi))",
                  timing="Serial cold child per candidate, parent wall clock; includes import, verification, IPC and trial persistence; excludes final report writing. Peak is child working set/high-water RSS, not sum across processes.",
                  commands=["python -B run_pilot.py --out runs","python -B run_controls.py --out controls",
                            "python -B reference_benchmarks.py --out reference_runs.json",
                            "python -B legacy_control.py --out legacy_run","python -B stress_check.py --out stress128",
                            "python -B summarize.py --data . --out ."],
                  artifact_status="Norm enclosures/ratios/secants mathematically enclosed; resource measurements and direct/Taylor point estimates empirical",
                  seed="deterministic; three repetitions, no random sampling")
    (out/META).write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("pilot",sum(r["status"]=="target_met" for r in rows[:12]),"of",12)
    for kind,values in benchtrials.items():
        print(kind,round(statistics.median(x["seconds"] for x in values),4))
    print("range16",next(x["upper"] for x in ref if x["method"]=="range" and x["quadrature_order"]==16))
    print("verified attempts",sum(a["status"]=="verified" for p in root.glob("*/*/search.json")
                                    for a in read(p)["attempts"]))


if __name__=="__main__":
    main()

