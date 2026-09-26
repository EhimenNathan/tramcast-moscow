"""Prophet daily-level forecasts on the dzoo protocol (per route x origin fits)."""
from dzoo import *
import pickle, logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR); logging.getLogger('prophet').setLevel(logging.ERROR)
from prophet import Prophet
DS=pickle.load(open('dzoo.pkl','rb'))
out={}
for per in ['autumn','spring']:
    df=DS[per]; a,_=PER[per]
    for growth in ['flat','linear']:
        pred=np.zeros(len(df))
        for (i,o),g in df.groupby(['route','o']):
            i,o=int(i),int(o)
            tt=[u for u in range(a,o) if OKR[i,u]]
            h=pd.DataFrame({'ds':DATES[tt],'y':D[i,tt]})
            m=Prophet(growth=growth,weekly_seasonality=True,yearly_seasonality=False,daily_seasonality=False,
                      seasonality_mode='multiplicative' if ROUTES[i]!=50 else 'additive',changepoint_prior_scale=0.05,uncertainty_samples=0)
            m.fit(h)
            f=m.predict(pd.DataFrame({'ds':DATES[g.t.astype(int).values]}))
            pred[g.index]=np.maximum(f.yhat.values,0)
        out[(per,growth)]=pred
        print(per,growth,'daily %.4f'%(1-np.abs(df.y-pred).sum()/df.y.sum()),flush=True)
pickle.dump(out,open('pred_prophet.pkl','wb'))
