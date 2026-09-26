from core import *
t0=di('2025-09-08'); t1=di('2025-10-31')
tt=np.arange(t0,t1+1)
D=Y.sum(2)
# daily-level in-sample: per route x dow median
tot=0;err=0; errh=0; err_sh=0
for i in range(len(ROUTES)):
    for dw in range(7):
        s=tt[DOW[tt]==dw]
        m=np.median(D[i,s]); err+=np.abs(D[i,s]-m).sum(); tot+=D[i,s].sum()
        # hourly in-sample median by dow
        mh=np.median(Y[i,s],0); errh+=np.abs(Y[i,s]-mh).sum()
        mean_sh=Y[i,s].sum(0)/Y[i,s].sum()
        err_sh+=np.abs(Y[i,s]-D[i,s][:,None]*mean_sh).sum()
print('daily WAPE in-sample (route,dow median):',err/tot)
print('hourly in-sample median by dow:',1-errh/tot)
print('oracle daily total x in-sample dow shape:',1-err_sh/tot)
for i,r in enumerate(ROUTES):
    s=tt; print(r, [int(np.median(D[i,s[DOW[s]==dw]])) for dw in range(7)], np.round(np.std(D[i,s[DOW[s]<5]])/np.mean(D[i,s[DOW[s]<5]]),3))
