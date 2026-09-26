from cands import *
def cls4(dw): return 0 if dw<4 else dw-3
WIN=('2025-02-03','2025-03-28')
def shape_for(i,o,t,kappa,wk_sparse,alpha):
    dw=DOW[t]; c=cls4(dw); nw=wk_sparse if c>0 else 4
    u=[q for q in range(o-7*nw,o) if OKR[i,q] and cls4(DOW[q])==c]
    s=Y[i,u].sum(0); s=s/max(s.sum(),1)
    if kappa>0 and c==0:
        ud=[q for q in range(o-28,o) if OKR[i,q] and DOW[q]==dw]; sd=Y[i,ud].sum(0)
        if sd.sum()>0: s=(1-kappa)*s+kappa*sd/sd.sum()
    if alpha>0:
        uw=[q for q in range(di(WIN[0]),min(di(WIN[1]),o-1)+1) if OKR[i,q] and cls4(DOW[q])==c]
        if uw: sw=Y[i,uw].sum(0); s=(1-alpha)*s+alpha*sw/sw.sum()
    return s/s.sum()
def test(ref_end,a,b,kappa=0,wk_sparse=4,alpha=0.25):
    o=di(ref_end)+1; num=den=0
    for i in range(len(ROUTES)):
        for t in range(di(a),di(b)+1):
            if OKR[i,t]: P=Y[i,t].sum()*shape_for(i,o,t,kappa,wk_sparse,alpha); num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
    return 1-num/den
S=[('2025-09-30','2025-10-01','2025-10-31'),('2025-10-12','2025-10-13','2025-10-31'),('2025-09-14','2025-09-15','2025-10-31'),('2025-05-04','2025-05-12','2025-05-25'),('2025-03-02','2025-03-03','2025-03-30')]
V={'base a.25':dict(),'dow k.3':dict(kappa=0.3),'dow k.5':dict(kappa=0.5),'sparse 6wk':dict(wk_sparse=6),'sparse 8wk':dict(wk_sparse=8),'a.5':dict(alpha=0.5),'a.5+sparse6':dict(alpha=0.5,wk_sparse=6)}
for n,v in V.items():
    r=[test(*s,**v) for s in S]; print('%-12s'%n,' '.join('%.4f'%x for x in r),' mean %.4f'%np.mean(r))
