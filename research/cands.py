"""Candidate forecast library + stable-regime test suite."""
from core import *
EXCL={(28,'2025-09-22'),(28,'2025-09-23'),(28,'2025-09-24'),(50,'2025-10-01')}
OKR=np.array([[OK[t] and (r,str(DATES[t].date())) not in EXCL for t in range(T)] for r in ROUTES])
def cls4(dw): return 0 if dw<4 else dw-3
def cands(o, tdays, start=None):
    """returns dict name -> array R x len(tdays) x 24 using data in [start, o)"""
    lo=0 if start is None else start
    C={}
    R=len(ROUTES)
    for i in range(R):
        past=np.array([t for t in range(lo,o) if OKR[i,t]])
        D=Y[i].sum(1)
        for j,t in enumerate(tdays):
            dw=DOW[t]; c=cls4(dw)
            sd=past[DOW[past]==dw][::-1]
            cl=past[np.array([cls4(DOW[p])==c for p in past])][::-1]
            def put(n,v):
                C.setdefault(n,np.zeros((R,len(tdays),24)))[i,j]=v
            for k in (1,2,3,4,6,8):
                put(f'sd_mean{k}',Y[i,sd[:k]].mean(0))
            for k in (3,4,6,8):
                put(f'sd_med{k}',np.median(Y[i,sd[:k]],0))
            # level x shape
            nk=4 if c==0 else 1
            for hl in (7,14,28):
                w=0.5**((o-1-cl[:12*nk])/hl); Lc=np.sum(D[cl[:12*nk]]*w)/np.sum(w)
                for ks in (4,8):
                    sh=Y[i,cl[:ks*nk]].sum(0); sh=sh/max(sh.sum(),1)
                    put(f'ls_hl{hl}_s{ks}',Lc*sh)
            # dow-level x class shape
            for k in (2,4):
                sh=Y[i,cl[:8*nk]].sum(0); sh=sh/max(sh.sum(),1)
                put(f'dl{k}_s8',D[sd[:k]].mean()*sh)
    return C
# stable-regime suite: (origin, last target day, history start)
SUITE=[('2025-02-03','2025-03-30','2025-01-11'),('2025-02-17','2025-03-30','2025-01-11'),('2025-03-03','2025-03-30','2025-01-11'),
       ('2025-09-22','2025-10-31','2025-09-01'),('2025-10-01','2025-10-31','2025-09-01'),('2025-10-06','2025-10-31','2025-09-01'),('2025-10-13','2025-10-31','2025-09-01')]
def suite_data():
    out=[]
    for o,e,s in SUITE:
        o_,e_,s_=di(o),di(e),di(s)
        td=[t for t in range(o_,e_+1) if OK[t]]
        C=cands(o_,td,s_); y=Y[:,td]; m=OKR[:,td]          # R x n
        out.append((o,C,y,m))
    return out
