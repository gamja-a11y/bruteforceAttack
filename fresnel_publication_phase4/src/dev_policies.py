from experiment import trial,ThreadPoolExecutor,E,writecsv
from development import case
m=dict(seconds=90,process_mib=768)
jobs=[('devpolicy',i,case(i,model),method,1,m) for i in (9,10,11,12,21,22,23,24) for model in ('Pade','Fresnel') for method in ('A','B','Cfixedbits','C')]
if __name__=='__main__':
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for r in pool.map(trial,jobs):
            rows.append(r);writecsv(E/'development_policy_comparison.csv',rows)

