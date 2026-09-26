"""Hierarchical local-level filter: one signal/noise ratio q per day-group shared across routes (joint likelihood).
Scalar Kalman filter implemented directly (fast): y_t = mu_t + e_t (var s2), mu_t = mu_{t-1} + n_t (var q*s2)."""
from core import *
EXCL={(28,'2025-09-22'),(28,'2025-09-23'),(28,'2025-09-24'),(50,'2025-10-01')}
OKR=np.array([[OK[t] and (r,str(DATES[t].date())) not in EXCL for t in range(T)] for r in ROUTES])
SCHOOL=[(di('2025-10-27'),di('2025-10-31'),0.985)]
def series(i,start,o,dws):
    D=Y[i].sum(1).astype(float).copy()
    for a,b,f in SCHOOL: D[a:b+1]/=f
    tt=np.array([t for t in range(start,o) if OKR[i,t] and DOW[t] in dws])
    y=np.log(np.maximum(D[tt],50.0)); x=(DOW[tt]==4).astype(float)
    return tt,y,x
def kf(y,q,fri=None):
    """concentrated-likelihood scalar KF with diffuse start; optional fixed Friday offset (estimated by OLS)"""
    if fri is not None: y=y-fri[0]*fri[1]
    mu=y[0]; P=1e6; ll=0; v2=[]; F=[]
    for t in range(1,len(y)):
        P=P+q; Fv=P+1; v=y[t]-mu; K=P/Fv; mu=mu+K*v; P=P*(1-K); v2.append(v*v/Fv); F.append(Fv)
    n=len(v2); s2=np.sum(v2)/n; ll=-0.5*(n*np.log(s2)+np.sum(np.log(F)))
    return mu, ll
def fit_q(start,o,group,grid=np.exp(np.linspace(np.log(1e-4),np.log(3),40))):
    best=None
    for q in grid:
        tot=0
        for i,r in enumerate(ROUTES):
            tt,y,x=series(i,start,o,group)
            if len(y)<4: continue
            fri=None
            if 4 in group: b=np.linalg.lstsq(np.c_[np.ones(len(y)),x],y,rcond=None)[0][1]; fri=(b,x)
            tot+=kf(y,q,fri)[1]
        if best is None or tot>best[1]: best=(q,tot)
    return best[0]
def levels(o,start,qs=None):
    L=np.zeros((len(ROUTES),7)); used={}
    for g,dws in {'wd':[0,1,2,3,4],'sat':[5],'sun':[6]}.items():
        q=qs[g] if qs else fit_q(start,o,dws); used[g]=q
        for i in range(len(ROUTES)):
            tt,y,x=series(i,start,o,dws)
            if g=='wd':
                b=np.linalg.lstsq(np.c_[np.ones(len(y)),x],y,rcond=None)[0][1]
                mu,_=kf(y,q,(b,x)); L[i,:4]=np.exp(mu); L[i,4]=np.exp(mu+b)
            else:
                mu,_=kf(y,q); L[i,dws[0]]=np.exp(mu)
    return L,used
if __name__=='__main__':
    for start in ['2025-09-01','2025-09-08']:
        L,q=levels(T,di(start)); print(start,q); print(np.round(L).astype(int))
    # also fit q on long winter-spring stable stretch for a better-identified prior
    L,q=levels(di('2025-03-30'),di('2025-01-13')); print('Jan13-Mar29 q:',q)
    import kalman2 as K
    K.ROUTES_Q=[0,2,3,4,5,6,7]
