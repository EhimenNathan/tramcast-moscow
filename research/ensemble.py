"""Ensemble tournament on daily levels (cross-period), raw and drift-neutralised.
Drift-neutralised: per origin, each model is rescaled so its total equals the structural forecast's total ->
measures skill in day/route pattern only (the part that can transfer to Nov-Dec)."""
from dzoo import *
import pickle, os, lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
DS=pickle.load(open('dzoo.pkl','rb'))
F=['dow','h','r1','r2','r3','r4','rc4','rw5','rw10']
def prep(df):
    df=df.copy(); b=df.cl12
    for k in (1,2,3,4): df[f'r{k}']=df[f'sd{k}']/b
    df['rc4']=df.cl4/b; df['rw5']=df.wd5/df.wd20; df['rw10']=df.wd10/df.wd20
    return df
P={p:prep(DS[p]) for p in DS}
PR=pickle.load(open('pred_prophet.pkl','rb')); GR=pickle.load(open('pred_gru.pkl','rb')) if os.path.exists('pred_gru.pkl') else {}
res={}
for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
    tr,te=P[tr_p],P[te_p]; w=tr.cl12.values; yr=tr.y/w; M={}
    M['struct_cl4']=te.cl4.values; M['struct_cl12']=te.cl12.values; M['sd4']=te.sd4.values
    M['adaboost']=np.mean([AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),n_estimators=50,learning_rate=0.05,random_state=s).fit(tr[F],yr,sample_weight=w).predict(te[F]) for s in range(3)],0)*te.cl12.values
    M['lgbm']=lgb.train(dict(objective='l1',learning_rate=0.03,num_leaves=7,min_data_in_leaf=50,lambda_l2=5,verbose=-1),lgb.Dataset(tr[F],yr,weight=w),300).predict(te[F])*te.cl12.values
    M['xgb']=xgb.XGBRegressor(objective='reg:absoluteerror',n_estimators=300,learning_rate=0.03,max_depth=3,subsample=0.8).fit(tr[F],yr,sample_weight=w).predict(te[F])*te.cl12.values
    M['prophet_flat']=PR[(te_p,'flat')]
    for k in ('gru_big','gru_big_aug'):
        if (te_p,k) in GR: M[k]=GR[(te_p,k)]
    base=['struct_cl4','adaboost','lgbm','xgb','prophet_flat']+[k for k in ('gru_big','gru_big_aug') if k in M]
    M['ens_mean']=np.mean([M[k] for k in base],0)
    M['ens_median']=np.median([M[k] for k in base],0)          # WAPE-consistent robust combiner
    M['ens_struct_ada_gru']=np.mean([M[k] for k in ['struct_cl4','adaboost']+[k for k in ('gru_big_aug','gru_big') if k in M][:1]],0)
    for n,p in M.items():
        raw=1-np.abs(te.y-p).sum()/te.y.sum()
        pn=p.copy()
        for o,g in te.groupby('o'): ix=g.index.values; pn[ix]=p[ix]*M['struct_cl4'][ix].sum()/max(p[ix].sum(),1)
        neu=1-np.abs(te.y-pn).sum()/te.y.sum()
        res.setdefault(n,{})[te_p]=(round(raw,4),round(neu,4))
print('%-20s %-24s %-24s'%('model','autumn (raw, neutral)','spring (raw, neutral)'))
for n,v in res.items(): print('%-20s %-24s %-24s'%(n,v['autumn'],v['spring']))
