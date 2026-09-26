"""Daily-level model zoo. Objective = hourly WAPE with common class shape. Train spring stretch, test autumn (and reverse)."""
from core import *
import warnings; warnings.filterwarnings('ignore')
EXCL={(28,'2025-09-22'),(28,'2025-09-23'),(28,'2025-09-24'),(50,'2025-10-01')}
OKR=np.array([[OK[t] and (r,str(DATES[t].date())) not in EXCL for t in range(T)] for r in ROUTES])
R=len(ROUTES); D=Y.sum(2).astype(float)
D[:,di('2025-10-27'):di('2025-10-31')+1]/=0.985
def cls4(dw): return 0 if dw<4 else dw-3
PER={'spring':(di('2025-01-13'),di('2025-03-30')),'autumn':(di('2025-09-01'),di('2025-10-31'))}
def origins(per): a,b=PER[per]; return list(range(a+21,b-6,3))
HMAX=28
def lagfeat(i,o,t,start):
    """features for route i, origin o, target t using data in [start,o)"""
    past=np.array([u for u in range(start,o) if OKR[i,u]])
    dw=DOW[t]; sd=past[DOW[past]==dw][::-1]
    cl=past[np.array([cls4(DOW[p])==cls4(dw) for p in past])][::-1]
    wd=past[DOW[past]<5][::-1]
    f=[D[i,sd[:k]].mean() for k in (1,2,3,4)]+[D[i,cl[:4]].mean(),D[i,cl[:12]].mean(),D[i,wd[:5]].mean(),D[i,wd[:10]].mean(),D[i,wd[:20]].mean()]
    return f
def dataset(per):
    rows=[]; a,b=PER[per]
    for o in origins(per):
        for t in range(o,min(o+HMAX,b+1)):
            if not OK[t]: continue
            for i in range(R):
                if not OKR[i,t]: continue
                f=lagfeat(i,o,t,a)
                rows.append([i,DOW[t],t-o]+f+[D[i,t],o,t])
    cols=['route','dow','h','sd1','sd2','sd3','sd4','cl4','cl12','wd5','wd10','wd20','y','o','t']
    return pd.DataFrame(rows,columns=cols)
SH={}
def shape(i,o,start,c):
    k=(i,o,c)
    if k not in SH:
        p=[u for u in range(max(start,o-56),o) if OKR[i,u] and cls4(DOW[u])==c]; s=Y[i,p].sum(0); SH[k]=s/max(s.sum(),1)
    return SH[k]
def hourly_score(df,pred,per):
    a,_=PER[per]; num=den=0
    for (i,o,t),p in zip(df[['route','o','t']].values,pred):
        y=Y[int(i),int(t)]; P=p*shape(int(i),int(o),a,cls4(DOW[int(t)])); num+=np.abs(y-P).sum(); den+=y.sum()
    return 1-num/den
if __name__=='__main__':
    import pickle
    DS={p:dataset(p) for p in PER}; pickle.dump(DS,open('dzoo.pkl','wb'))
    for p in PER: print(p,len(DS[p]))
