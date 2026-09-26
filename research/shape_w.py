from cands import *
def cls4(dw): return 0 if dw<4 else dw-3
def shp(i,a,b,c):
    u=[t for t in range(di(a),di(b)+1) if OKR[i,t] and cls4(DOW[t])==c]; s=Y[i,u].sum(0); return s/s.sum()
def test(o,e,alpha,wwin=('2025-02-03','2025-03-28')):
    num=den=0
    for i in range(len(ROUTES)):
        for c in range(4):
            s4=shp(i,str((DATES[di(o)]-pd.Timedelta(days=28)).date()),str((DATES[di(o)]-pd.Timedelta(days=1)).date()),c)
            sw=shp(i,*wwin,c) if not (ROUTES[i] in (7,50) and c>=2) else s4
            sh=(1-alpha)*s4+alpha*sw
            for t in range(di(o),di(e)+1):
                if OKR[i,t] and cls4(DOW[t])==c: P=Y[i,t].sum()*sh; num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
    return 1-num/den
for o,e in [('2025-10-01','2025-10-24'),('2025-10-13','2025-10-24'),('2025-10-06','2025-10-24'),('2025-10-27','2025-10-31')]:
    print(o,e,' '.join('a%.1f=%.4f'%(a,test(o,e,a)) for a in [0,0.1,0.2,0.3,0.5]))
