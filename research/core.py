import pandas as pd, numpy as np
g=pd.read_parquet('grid.parquet')
ROUTES=[1,7,11,12,17,25,26,28,50]
DATES=pd.date_range('2025-01-01','2025-10-31')
Y=np.stack([g[g.route==r].pivot(index='date',columns='hour',values='boardings').reindex(DATES).values for r in ROUTES]).astype(float) # R x T x 24
T=len(DATES); DOW=DATES.dayofweek.values
HOL=set(pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05','2025-01-06','2025-01-07','2025-01-08',
 '2025-05-01','2025-05-02','2025-05-03','2025-05-04','2025-05-08','2025-05-09','2025-05-10','2025-05-11',
 '2025-06-12','2025-06-13','2025-06-14','2025-06-15']))
SPEC=HOL|set(pd.to_datetime(['2025-01-09','2025-01-10','2025-04-30','2025-05-07','2025-06-11']))
OK=np.array([d not in SPEC for d in DATES])
def di(s): return DATES.get_loc(pd.Timestamp(s))
def run(fc, origins, H=61, verbose=False, bydim=False):
    num=den=0; per=[]
    for o in origins:
        t0=di(o); tt=np.array([t for t in range(t0,min(t0+H,T)) if OK[t]])
        P=fc(t0,tt)                       # R x len(tt) x 24
        y=Y[:,tt]
        n=np.abs(y-P).sum(); d=y.sum(); per.append(1-n/d); num+=n; den+=d
    if verbose: print(np.round(per,3))
    return round(1-num/den,4)
ORIG=[str(d.date()) for d in pd.date_range('2025-02-15','2025-09-01',freq='14D')]
LATE=[str(d.date()) for d in pd.date_range('2025-08-18','2025-10-20',freq='7D')]
