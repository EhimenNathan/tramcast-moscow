from core import *
# Irreducible hourly noise: for stable routes/weeks, compare each day to the mean of the SAME dow in the other weeks,
# and extrapolate the estimation error away (K -> infinity) using WAPE(K) ~ a + b/K.
t0=di('2025-09-01'); W=8
YY=np.stack([Y[:,t0+7*k:t0+7*k+7] for k in range(W)],1)   # R x W x 7 x 24
res=[]
for K in [1,2,3,4,5,6,7]:
    num=den=0
    rng=np.random.default_rng(0)
    for k in range(W):
        for rep in range(6):
            others=rng.choice([j for j in range(W) if j!=k],K,replace=False)
            P=YY[:,others].mean(1); y=YY[:,k]
            num+=np.abs(y-P).sum(); den+=y.sum()
    res.append((K,1-num/den))
K=np.array([r[0] for r in res]); s=np.array([r[1] for r in res])
A=np.c_[np.ones_like(K),1/np.sqrt(K)*0+1/K]; a,b=np.linalg.lstsq(A,1-s,rcond=None)[0]
print(res); print('extrapolated floor error (K->inf, same regime, perfect expectation): %.4f  => max score ~ %.4f'%(a,1-a))
# Poisson-only floor for comparison
lam=YY.mean(1); sim=np.random.default_rng(1).poisson(np.repeat(lam[:,None],W,1))
print('pure-Poisson floor score: %.4f'%(1-np.abs(sim-lam[:,None]).sum()/sim.sum()))
