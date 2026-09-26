import forecast2 as F, copy, numpy as np, pandas as pd
from core import *
CFG2=copy.deepcopy(F.CFG)
CFG2.update(wd_win=('2025-09-15','2025-10-31'),wd_hl=10,we_win=('2025-09-06','2025-10-26'),we_hl=14,
            sh_win=('2025-09-29','2025-10-26'),sh_hl=1e9,dowf=[0.970,1.004,1.006,1.014,0.99],fri_shrink=0.5,
            winter_alpha=[0.3,0.4], month_wd=[1.005,1.0])
dates=pd.date_range('2025-11-01','2025-12-31')
M=F.fit(CFG2); P=F.predict(M,CFG2,dates)
out=F.write_sub(P,dates,'../submissions/sub_v2.csv')
P1=np.load('P_v1.npy'); print('total v2 %.0f  v1 %.0f  |v2-v1|/v1 %.4f'%(P.sum(),P1.sum(),np.abs(P-P1).sum()/P1.sum()))
for r in ROUTES: print(r,np.round(M[r]['L']).astype(int))
np.save('P_v2.npy',P)
import json; json.dump({k:(v if not isinstance(v,tuple) else list(v)) for k,v in CFG2.items()},open('cfg_v2.json','w'),indent=1,default=str)
