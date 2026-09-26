"""Daylight-driven hourly shape: share_{r,c}(w)[h] = a_{r,c}[h] + b_c[h]*sunset_w + g_c[h]*school_w  (b,g pooled over routes)
Forecast shape = recent 4-week shape + b_c * (sunset_target - sunset_recent) (+ g_c * school delta). Strict: fit only on data <= ref_end.
Evaluation with ORACLE daily totals (isolates shape quality)."""
from cands import *
DL=pd.read_csv('daylight_2025.csv',parse_dates=['time']).set_index('time')
SUN=DL.sunset_h.reindex(DATES).values
SCHOOL=np.zeros(T,bool)
for a,b in [('2025-01-01','2025-01-12'),('2025-03-22','2025-03-30'),('2025-05-26','2025-08-31'),('2025-10-25','2025-10-31')]:
    SCHOOL[di(a):di(b)+1]=True
def cls4(dw): return 0 if dw<4 else dw-3
def fit_beta(ref_end, ridge=1.0):
    """weekly class shapes per route -> pooled regression of (share - route mean) on (sunset, school)"""
    e=di(ref_end); B={}
    for c in range(4):
        rows=[];X=[]
        for i in range(len(ROUTES)):
            if ROUTES[i] in (7,50) and c>=2: continue
            wk={}
            for t in range(di('2025-01-13'),e+1):
                if OKR[i,t] and cls4(DOW[t])==c: wk.setdefault(t-DOW[t],[]).append(t)
            S=[];x=[]
            for w,ts in wk.items():
                s=Y[i,ts].sum(0)
                if s.sum()<500: continue
                S.append(s/s.sum()); x.append([SUN[ts].mean(), SCHOOL[ts].mean()])
            if len(S)<6: continue
            S=np.array(S); x=np.array(x); rows.append(S-S.mean(0)); X.append(x-x.mean(0))
        R=np.vstack(rows); Xm=np.vstack(X)
        beta=np.linalg.solve(Xm.T@Xm+ridge*np.eye(2),Xm.T@R)       # 2 x 24
        B[c]=beta
    return B
def test(ref_end,a,b,mode,B=None,alpha=0.25):
    o=di(ref_end)+1; num=den=0
    for i in range(len(ROUTES)):
        for c in range(4):
            u=[t for t in range(o-28,o) if OKR[i,t] and cls4(DOW[t])==c and not SCHOOL[t]]
            if not u: u=[t for t in range(o-28,o) if OKR[i,t] and cls4(DOW[t])==c]
            s=Y[i,u].sum(0); s4=s/max(s.sum(),1)
            for t in range(di(a),di(b)+1):
                if not (OKR[i,t] and cls4(DOW[t])==c): continue
                if mode=='raw': sh=s4
                elif mode=='winter':
                    uw=[q for q in range(di('2025-02-03'),di('2025-03-28')+1) if OKR[i,q] and cls4(DOW[q])==c]; sw=Y[i,uw].sum(0); sw=sw/sw.sum(); sh=(1-alpha)*s4+alpha*sw
                else:
                    dx=np.array([SUN[t]-SUN[u].mean(), SCHOOL[t]-SCHOOL[u].mean()])
                    sh=np.maximum(s4+dx@B[c],0); sh=sh/sh.sum()
                P=Y[i,t].sum()*sh; num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
    return 1-num/den
for ref_end,a,b in [('2025-09-30','2025-10-01','2025-10-31'),('2025-10-12','2025-10-13','2025-10-31'),('2025-09-14','2025-09-15','2025-10-31'),('2025-04-30','2025-05-12','2025-05-25')]:
    B=fit_beta(ref_end)
    print(ref_end,'->',a,b,' raw %.4f  winter-blend %.4f  daylight %.4f'%(test(ref_end,a,b,'raw'),test(ref_end,a,b,'winter'),test(ref_end,a,b,'daylight',B)))
B=fit_beta('2025-10-31')
print('beta (Mon-Thu) per hour, share change per +1h sunset:',np.round(B[0][0]*100,3))
print('--- control: what reference shape makes the blend work?')
def test_ref(ref_end,a,b,ref,alpha):
    o=di(ref_end)+1; num=den=0
    ra,rb=ref
    for i in range(len(ROUTES)):
        for c in range(4):
            u=[t for t in range(o-28,o) if OKR[i,t] and cls4(DOW[t])==c]
            s=Y[i,u].sum(0); s4=s/max(s.sum(),1)
            uw=[q for q in range(di(ra),min(di(rb),o-1)+1) if OKR[i,q] and cls4(DOW[q])==c and not SCHOOL[q]]
            if not uw: uw=u
            sw=Y[i,uw].sum(0); sw=sw/sw.sum(); sh=(1-alpha)*s4+alpha*sw
            for t in range(di(a),di(b)+1):
                if OKR[i,t] and cls4(DOW[t])==c: P=Y[i,t].sum()*sh; num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
    return 1-num/den
refs={'winter Feb-Mar':('2025-02-03','2025-03-28'),'all-year Jan13-Sep':('2025-01-13','2025-09-30'),'summer Jun-Aug':('2025-06-01','2025-08-31'),'spring Apr-May':('2025-04-07','2025-05-25')}
for ref_end,a,b in [('2025-09-30','2025-10-01','2025-10-31'),('2025-10-12','2025-10-13','2025-10-31'),('2025-09-14','2025-09-15','2025-10-31')]:
    print(ref_end, '  '.join('%s a.25=%.4f a.5=%.4f'%(k,test_ref(ref_end,a,b,v,0.25),test_ref(ref_end,a,b,v,0.5)) for k,v in refs.items()))
