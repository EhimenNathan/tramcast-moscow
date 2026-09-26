from cands import *
D=Y.sum(2)
HOLS={'May1-4':['2025-05-01','2025-05-02','2025-05-03','2025-05-04'],'May8-11':['2025-05-08','2025-05-09','2025-05-10','2025-05-11'],'Jun12-15':['2025-06-12','2025-06-13','2025-06-14','2025-06-15'],'Jan2-8':['2025-01-0%d'%k for k in range(2,9)]}
def sh(i,o,dws,weeks=4):
    u=[t for t in range(o-7*weeks,o) if OKR[i,t] and DOW[t] in dws]; s=Y[i,u].sum(0); return s/max(s.sum(),1)
def holshape(i,o,excl):
    # average normalized shape of earlier holiday days (before o), excluding the block itself
    u=[di(d) for b,ds in HOLS.items() for d in ds if di(d)<o and b!=excl]
    if not u: return None
    s=Y[i,u].sum(0); return s/max(s.sum(),1)
print('oracle daily totals; shape variants on holiday days')
for b,ds in HOLS.items():
    tt=[di(d) for d in ds]; o=tt[0]
    res={}
    for name in ['sun','sat','sun+sat','hol-hist','0.5sun+0.5hol']:
        num=den=0
        for i in range(len(ROUTES)):
            if D[i,tt].sum()==0: continue
            s_sun=sh(i,o,[6]); s_sat=sh(i,o,[5]); hh=holshape(i,o,b)
            s={'sun':s_sun,'sat':s_sat,'sun+sat':(s_sun+s_sat)/2,'hol-hist':hh if hh is not None else s_sun,'0.5sun+0.5hol':(s_sun+(hh if hh is not None else s_sun))/2}[name]
            for t in tt: P=D[i,t]*s; num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
        res[name]=round(1-num/den,4)
    print(b,res)
# level factor vs Sunday level (4 prior Sundays), per block
print('level ratio holiday/Sunday (all routes w/o 7,50):')
for b,ds in HOLS.items():
    tt=[di(d) for d in ds]; o=tt[0]; idx=[k for k,r in enumerate(ROUTES) if r not in (7,50)]
    sun=[t for t in range(o-28,o) if DOW[t]==6 and OKR[0,t]]
    print(b,[round(D[idx,t].sum()/D[idx][:,sun].sum(1).mean()/1,3) for t in tt])
