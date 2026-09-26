from ada import P, F
import numpy as np, pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
    tr,te=P[tr_p],P[te_p]
    def sc(p): return round(1-np.abs(te.y-p).sum()/te.y.sum(),4)
    # (1) cl12 x per-dow weighted-median ratio
    tr['ratio']=tr.y/tr.cl12
    med=tr.groupby('dow').ratio.median(); print(tr_p,'->',te_p,'dow ratios',med.round(3).to_dict())
    p1=te.cl12*te.dow.map(med)
    # (2) cl4 x dow ratio vs cl4
    tr['r_c4']=tr.y/tr.cl4; med2=tr.groupby('dow').r_c4.median(); p2=te.cl4*te.dow.map(med2)
    # (3) ada with only dow
    a=AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),n_estimators=50,learning_rate=0.05,random_state=0).fit(tr[['dow']],tr.y/tr.cl12,sample_weight=tr.cl12)
    p3=a.predict(te[['dow']])*te.cl12
    a=AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),n_estimators=50,learning_rate=0.05,random_state=0).fit(tr[F],tr.y/tr.cl12,sample_weight=tr.cl12)
    p4=a.predict(te[F])*te.cl12
    imp=dict(zip(F,np.round(a.feature_importances_,3)))
    print('  cl12 %.4f cl4 %.4f | cl12*dowmed %.4f  cl4*dowmed %.4f  ada(dow) %.4f  ada(all) %.4f'%(sc(te.cl12),sc(te.cl4),sc(p1),sc(p2),sc(p3),sc(p4)), imp)
