"""Nov-Dec 2025 hourly boardings forecast: level x shape x calendar x weather."""
from core import *
import sys, json
Wd=pd.read_csv('weather_daily.csv',parse_dates=['date']).set_index('date')
EXCL={(28,'2025-09-22'),(28,'2025-09-23'),(28,'2025-09-24'),(50,'2025-10-01')}
def cls4(dw): return 0 if dw<4 else dw-3        # Mon-Thu, Fri, Sat, Sun

def wmean(x,w): return float(np.sum(x*w)/np.sum(w))

def fit(cfg):
    BETA=cfg['beta']
    M={}
    for i,r in enumerate(ROUTES):
        def days(a,b,pred):
            return np.array([t for t in range(di(a),di(b)+1) if OK[t] and pred(DOW[t]) and (r,str(DATES[t].date())) not in EXCL])
        def lvl(a,b,pred,hl,end):
            u=days(a,b,pred); D=Y[i,u].sum(1)*np.exp(BETA*Wd.p.reindex(DATES[u]).values)
            if cfg.get('school_fix'):  # Oct27-31 autumn school break
                D=D*np.where((DATES[u]>=pd.Timestamp('2025-10-25'))&(DATES[u]<=pd.Timestamp('2025-11-02'))&(DOW[u]<5),1/cfg['school_f'],1)
            w=0.5**((di(end)-u)/hl); return wmean(D,w)
        a,b=cfg['wd_win']; L=np.zeros(7)
        L_mt=lvl(a,b,lambda d:d<4,cfg['wd_hl'],b)
        # Friday ratio from longer window (Sep-Oct)
        fr=lvl('2025-09-01','2025-10-31',lambda d:d==4,1e9,b)/lvl('2025-09-01','2025-10-31',lambda d:d<4,1e9,b)
        if cfg.get('dowf'):
            f=np.array(cfg['dowf']); L[:4]=L_mt*f[:4]/f[:4].mean(); L[4]=L_mt*(fr*(1-cfg.get('fri_shrink',0))+cfg['dowf'][4]/f[:4].mean()*cfg.get('fri_shrink',0))
        else: L[:4]=L_mt; L[4]=L_mt*fr
        a2,b2=cfg['we_win']
        L[5]=lvl(a2,b2,lambda d:d==5,cfg['we_hl'],b2); L[6]=lvl(a2,b2,lambda d:d==6,cfg['we_hl'],b2)
        for k in (5,6):   # closed-service weekends: WAPE-optimal median instead of mean
            if L[k]<0.1*L_mt:
                u=days(a2,b2,lambda d,k=k:d==k); L[k]=float(np.median(Y[i,u].sum(1)))
        S=np.zeros((4,24)); a3,b3=cfg['sh_win']
        for c in range(4):
            u=days(a3,b3,lambda d:cls4(d)==c); w=0.5**((di(b3)-u)/cfg['sh_hl'])
            s=(Y[i,u]*w[:,None]).sum(0); S[c]=s/max(s.sum(),1)
        # "restored" weekend profile for routes 7/50 from Feb-Mar weekends, rescaled by weekday level ratio
        Lr=None; Sr=None
        if r in (7,50):
            u_wd=days('2025-02-03','2025-03-28',lambda d:d<5); u_sa=days('2025-02-01','2025-03-29',lambda d:d==5); u_su=days('2025-02-02','2025-03-30',lambda d:d==6)
            k=L_mt/Y[i,u_wd].sum(1).mean()
            Lr=(Y[i,u_sa].sum(1).mean()*k, Y[i,u_su].sum(1).mean()*k)
            Sr=[Y[i,u].sum(0)/Y[i,u].sum() for u in (u_sa,u_su)]
        SW=np.zeros((4,24))
        for c in range(4):
            u=days('2025-02-03','2025-03-28',lambda d:cls4(d)==c); s=Y[i,u].sum(0); SW[c]=s/max(s.sum(),1)
        M[r]=dict(L=L,S=S,Lr=Lr,Sr=Sr,SW=SW)
    return M

def day_spec(d, cfg):
    """calendar rules -> (kind, dow_for_level, factor)"""
    s=str(d.date()); dw=d.dayofweek; C=cfg['cal']; mi=d.month-11
    mwd=cfg['month_wd'][mi]; mwe=cfg['month_we'][mi]
    if s=='2025-11-01': return ('nov1',4,C['nov1']*mwd)
    if s=='2025-11-02': return ('we',6,C['nov2']*mwe)
    if s=='2025-11-03': return ('we',6,C['nov3']*mwe)
    if s=='2025-11-04': return ('we',6,C['nov4']*mwe)
    if s=='2025-12-31': return ('dec31',5,C['dec31']*mwe)
    if s=='2025-12-29': return ('wd',dw,C['dec29']*mwd)
    if s=='2025-12-30': return ('wd',dw,C['dec30']*mwd)
    if dw>=5: return ('we',dw,mwe*(C['dec27_28'] if s in ('2025-12-27','2025-12-28') else 1))
    return ('wd',dw,mwd*(C['dec22_26'] if '2025-12-22'<=s<='2025-12-26' else 1))

def predict(M, cfg, dates):
    out=np.zeros((len(ROUTES),len(dates),24)); BETA=cfg['beta']; pref=cfg['p_ref']
    for i,r in enumerate(ROUTES):
        m=M[r]; L,S=m['L'],m['S']
        for j,d in enumerate(dates):
            kind,dl,f=day_spec(d,cfg)
            restored = r in (7,50) and str(r) in cfg['restore'] and d>=pd.Timestamp(cfg['restore'][str(r)])
            if kind=='wd': lev,sh=L[dl],S[cls4(dl)]
            elif kind in ('we','dec31'):
                if restored: lev,sh=m['Lr'][dl-5],m['Sr'][dl-5]
                else: lev,sh=L[dl],S[cls4(dl)]
                if kind=='dec31': sh=sh.copy(); sh[21:]*=0.6; sh=sh/sh.sum()
            elif kind=='nov1':
                if r in (7,50) and cfg.get('nov1_closed'): lev,sh=L[5],S[2]; f=f/cfg['cal']['nov1']
                else: lev,sh=L[4],0.6*S[1]+0.4*S[2]
            a=cfg.get('winter_alpha',[0,0])[d.month-11]
            closed_we = r in (7,50) and kind in ('we','dec31') and not restored
            if a>0 and not closed_we and not (kind in ('we','dec31') and restored):
                cc = cls4(dl) if kind in ('wd','we','dec31') else 1
                blend=(1-a)*sh+a*m['SW'][cc]
                if kind=='nov1': blend=(1-a)*sh+a*(0.6*m['SW'][1]+0.4*m['SW'][2])
                if kind=='dec31': blend=blend.copy(); blend[21:]*=0.6
                sh=blend/blend.sum()
            wf=np.exp(-BETA*(Wd.p.get(d,pref)-pref))
            out[i,j]=lev*sh*f*wf
    return out

CFG=dict(beta=0.004,p_ref=1.0,school_fix=True,school_f=0.98,
    wd_win=('2025-09-15','2025-10-31'),wd_hl=14, we_win=('2025-09-06','2025-10-26'),we_hl=21,
    sh_win=('2025-09-08','2025-10-31'),sh_hl=28,
    month_wd=[1.0,1.0], month_we=[0.98,0.97],
    cal=dict(nov1=0.75,nov2=0.88,nov3=0.97,nov4=1.0,dec22_26=0.98,dec27_28=1.0,dec29=0.90,dec30=0.85,dec31=0.85),
    restore={'50':'2025-12-01','7':'2025-12-01'}, nov1_closed=False)

def write_sub(P, dates, path):
    rows=[]
    for i,r in enumerate(ROUTES):
        for j,d in enumerate(dates):
            for h in range(24): rows.append((r,str(d.date()),h,P[i,j,h]))
    df=pd.DataFrame(rows,columns=['route','date','hour','prediction'])
    base=pd.read_csv('test_submission.csv',sep=';')[['route','date','hour']]
    out=base.merge(df,on=['route','date','hour'],how='left').fillna({'prediction':0.0})
    out['prediction']=out.prediction.clip(lower=0).round(2)
    assert len(out)==14640 and out.prediction.notna().all()
    out.to_csv(path,sep=';',index=False); return out
