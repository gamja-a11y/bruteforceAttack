"""Reverify complete or partial registered results; smoke is excluded."""
from run import HERE,frozen,original_check,jobs,saved,load,save,dy,F,dec
from common import writecsv

def exact(folder):
    ev=load(folder/'certificate.json')['obligations']
    e=ev['endpoints']['evidence']
    return (dy(ev['remainder']['evidence']['delta']),dy(e['upper'])-dy(e['lower']))

def main():
    m=frozen();original_check()
    rows=[];pairs={}
    for label,p,method in jobs(m,False):
        folder=HERE/'runs'/f'{label}_{method}'
        base=dict(case=label,**p,method=method)
        if (folder/'receipt').exists():
            r=dict(saved(folder,p,method))
            r.pop('params')
            rows.append(dict(base,**{k:v for k,v in r.items() if k!='method'}))
            pairs.setdefault(label,{})[method]=(r,exact(folder) if r['verified'] else None)
        else:
            rows.append(dict(base,verified=False,target_met=None,reason='incomplete' if folder.exists() else 'not_run'))
    writecsv(HERE/'results.csv',rows)
    comparison=[]
    for label in dict.fromkeys(r['case'] for r in rows):
        pair=pairs.get(label,{})
        row=dict(case=label,paired=len(pair)==2)
        if len(pair)==2:
            (b,x),(c,y)=pair['B'],pair['C']
            row.update(B_success=b['target_met'],C_success=c['target_met'],added=c['target_met'] and not b['target_met'],lost=b['target_met'] and not c['target_met'])
            if x and y:
                for i,k in enumerate(('delta','width')):
                    row[k+'_reduction_percent']=dec(100*(1-y[i]/x[i])) if x[i] else None
            if b['target_met'] and c['target_met']:
                row.update(B_common_s=b['wall_s'],C_common_s=c['wall_s'])
        comparison.append(row)
    writecsv(HERE/'paired.csv',comparison)
    totals=dict(completed=sum(len(p) for p in pairs.values()),planned=96,paired=sum(r['paired'] for r in comparison),added=sum(bool(r.get('added')) for r in comparison),lost=sum(bool(r.get('lost')) for r in comparison),common_success=sum('B_common_s' in r for r in comparison),B_common_s=sum(r.get('B_common_s',0) for r in comparison),C_common_s=sum(r.get('C_common_s',0) for r in comparison))
    save(HERE/'summary.json',totals);print(totals)
if __name__=='__main__':main()

