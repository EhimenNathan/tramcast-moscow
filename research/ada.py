from dzoo import *
import pickle
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
for depth in (2,3,4):
  for n in (50,100,200):
    for lr in (0.02,0.05,0.1):
      row=[]
      for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
        tr,te=P[tr_p],P[te_p]; w=tr.cl12.values
        sc=[]
        for seed in (0,1,2):
            a=AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth),n_estimators=n,learning_rate=lr,random_state=seed).fit(tr[F],tr.y/w,sample_weight=w)
            p=a.predict(te[F])*te.cl12.values; sc.append(1-np.abs(te.y-p).sum()/te.y.sum())
        row.append(np.mean(sc))
      print(depth,n,lr,'daily autumn %.4f spring %.4f'%tuple(row))
# reference daily cl4
for te_p in ['autumn','spring']:
    te=P[te_p]; print(te_p,'cl4 daily %.4f'%(1-np.abs(te.y-te.cl4).sum()/te.y.sum()))
