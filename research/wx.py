from core import *
import json
w=json.load(open('weather_2025.json'))['hourly']
W=pd.DataFrame(w); W['time']=pd.to_datetime(W.time); W['date']=W.time.dt.normalize()
Wd=W[W.time.dt.hour.between(6,22)].groupby('date').agg(t=('temperature_2m','mean'),p=('precipitation','sum'),sn=('snowfall','sum'),sd=('snow_depth','mean'),ws=('wind_speed_10m','mean'))
print(Wd.loc['2025-11-01':].resample('MS').mean())
print(Wd.loc['2025-01-01':'2025-10-31'].resample('MS').mean())
st=[ROUTES.index(r) for r in [1,11,12,17,25,26,28]]
D=Y[st].sum(2).sum(0)   # stable total per day
s=pd.Series(D,index=DATES)
# local level: median of same dow within +-21 days excluding self and special
res=[]
for t in range(T):
    if not OK[t]: continue
    nb=[u for u in range(t-21,t+22,7) if 0<=u<T and u!=t and OK[u]]
    if len(nb)<4: continue
    res.append((DATES[t], np.log(D[t]/np.median(D[nb])), DOW[t]))
R=pd.DataFrame(res,columns=['date','lr','dow']).set_index('date').join(Wd)
R['we']=(R.dow>=5).astype(int)
import statsmodels.formula.api as smf
for f in ['lr ~ p + sn','lr ~ p*we + sn*we','lr ~ np.log1p(p) + np.log1p(sn)+ t']:
    m=smf.ols(f,R).fit(); print(m.summary().tables[1])
print(R.lr.std())
import statsmodels.api as sm
R['snwe']=R.sn*R.we; R['snwd']=R.sn*(1-R.we)
for name,sub in [('all',R),('H1',R.loc[:'2025-05-31']),('H2',R.loc['2025-06-01':]),('winter',R.loc[:'2025-04-15'])]:
    X=sm.add_constant(sub[['p','snwd','snwe']]); m=sm.RLM(sub.lr,X,M=sm.robust.norms.HuberT()).fit()
    print(name,len(sub),m.params.round(4).to_dict(), m.tvalues.round(1).to_dict())
Wd.to_csv('weather_daily.csv')
print(Wd.loc['2025-11-01':].to_string())
print(R[R.sn>0.3][['lr','dow','t','p','sn']].round(3).to_string())
