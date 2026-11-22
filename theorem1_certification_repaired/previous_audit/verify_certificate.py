"""Independent read-only replay of the exact dyadic endpoint inequalities."""
from pathlib import Path
import argparse
import hashlib
import json
from flint import arb, ctx


def require(ok, name):
    if not ok:
        raise ValueError(name)


def dyadic(pair):
    return arb((int(pair[0]), int(pair[1])))


def verify(folder):
    base = Path(__file__).resolve().parent
    report = json.loads((folder/"final_certificate.json").read_text(encoding="utf-8"))
    for prefix,key in [(base,"sources_sha256"),(folder,"evidence_sha256")]:
        for file,digest in report[key].items():
            require(hashlib.sha256((prefix/file).read_bytes()).hexdigest()==digest, file)
    runs = json.loads((folder/"certification_reproduction.json").read_text(encoding="utf-8"))["runs"]
    for r in runs:
        ctx.prec = 4*r["bits"]+256
        glo, ghi = [dyadic(r["G"][side]) for side in ("lower","upper")]
        d = dyadic(r["delta"]["upper"])
        eps = dyadic(r["epsilon"]["upper"])
        M = dyadic(r["M"]["upper"])
        require(ghi>=0 and ghi>=glo and d>=0 and eps>=4*M/(2**r["degree"]), "invalid Gram/remainder")
        require(dyadic(r["q_max"]["upper"])<dyadic(r["Q"]["lower"]), "q outside")
        # Work in exact dyadics and use a squared comparison to avoid any
        # uncertainty at a square-root endpoint (all quantities nonnegative).
        for name,ends in [("original",r["original_dyadics"]),("audit",[r["lower"],r["upper"]])]:
            lo,hi=map(dyadic,ends)
            require(hi>=lo>=0 and hi>=d, name+" endpoint order")
            require((hi-d)**2>=ghi, name+" upper endpoint rounded inward")
            require(lo==0 or (lo+d)**2<=max(arb(0),glo),name+" lower endpoint rounded inward")
        print(r["engine"],r["kind"],r["degree"],r["bits"],"outward endpoints verified")
    print(len(runs),"replays checked against hashed audit evidence")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",nargs="?",type=Path,default=Path(__file__).resolve().parent)
    verify(parser.parse_args().folder)

