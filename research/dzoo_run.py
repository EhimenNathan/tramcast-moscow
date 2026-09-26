from dzoo import *
import pickle, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
DS=pickle.load(open('dzoo.pkl','rb'))
F=['dow','h','r1','r2','r3','r4','rc4','rw5','rw10']
def prep(df):
    df=df.copy(); b=df.cl12
    for k in (1,2,3,4): df[f'r{k}']=df[f'sd{k}']/b
    df['rc4']=df.cl4/b; df['rw5']=df.wd5/df.wd20; df['rw10']=df.wd10/df.wd20
    return df
res={}
for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
    tr=prep(DS[tr_p]); te=prep(DS[te_p]); out={}
    for n in ['sd1','sd2','sd3','sd4','cl4','cl12']: out[n]=te[n].values
    w=tr.cl12.values; yr=tr.y.values/w
    m=lgb.train(dict(objective='l1',learning_rate=0.03,num_leaves=7,min_data_in_leaf=50,lambda_l2=5,verbose=-1,seed=0),lgb.Dataset(tr[F],yr,weight=w),300)
    out['lgbm']=m.predict(te[F])*te.cl12.values
    x=xgb.XGBRegressor(objective='reg:absoluteerror',n_estimators=300,learning_rate=0.03,max_depth=3,subsample=0.8,reg_lambda=5)
    x.fit(tr[F],yr,sample_weight=w); out['xgb']=x.predict(te[F])*te.cl12.values
    c=CatBoostRegressor(loss_function='MAE',iterations=500,learning_rate=0.03,depth=4,verbose=0,random_seed=0)
    c.fit(tr[F],yr,sample_weight=w); out['catboost']=c.predict(te[F])*te.cl12.values
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    a=AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),n_estimators=100,learning_rate=0.05,loss='linear',random_state=0)
    a.fit(tr[F],yr,sample_weight=w); out['adaboost']=a.predict(te[F])*te.cl12.values
    try:
        from ngboost import NGBRegressor
        ng=NGBRegressor(n_estimators=300,learning_rate=0.03,verbose=False,random_state=0); ng.fit(tr[F].values,yr,sample_weight=w/w.mean())
        dist=ng.pred_dist(te[F].values); out['ngboost_median']=dist.ppf(0.5)*te.cl12.values
    except Exception as e: print('ngboost fail',e)
    out['ens_trees']=np.mean([out[k] for k in ['lgbm','xgb','catboost']],0)
    out['blend_sd4_cl4']=0.5*out['sd4']+0.5*out['cl4']
    for n,p in out.items():
        res.setdefault(n,{})[te_p]=(round(hourly_score(te,p,te_p),4), round(1-np.abs(te.y-p).sum()/te.y.sum(),4))
print('%-16s %-22s %-22s'%('model','test=autumn (hourly,daily)','test=spring (hourly,daily)'))
for n,v in res.items(): print('%-16s %-22s %-22s'%(n,v.get('autumn'),v.get('spring')))
