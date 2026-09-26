from cands import *
def cls4(dw): return 0 if dw<4 else dw-3
def test(o,e,weeks=4,rank=None,lam=0.0):
    o,e=di(o),di(e); num=den=0
    tt=[t for t in range(o,e+1) if OK[t]]
    for c in range(4):
        # profiles matrix over last 8 weeks, all routes, class c
        rows=[];meta=[]
        for i in range(len(ROUTES)):
            for u in range(o-56,o):
                if OKR[i,u] and cls4(DOW[u])==c and Y[i,u].sum()>1000:
                    rows.append(Y[i,u]/Y[i,u].sum()); meta.append((i,u))
        X=np.array(rows)
        if rank:
            mu=X.mean(0); U,s,Vt=np.linalg.svd(X-mu,full_matrices=False); Xr=mu+(U[:,:rank]*s[:rank])@Vt[:rank]
        else: Xr=X
        for i in range(len(ROUTES)):
            idx=[k for k,(ii,u) in enumerate(meta) if ii==i and u>=o-7*weeks]
            if not idx: continue
            Dw=np.array([Y[i,meta[k][1]].sum() for k in idx])
            sh=(Xr[idx]*Dw[:,None]).sum(0)/Dw.sum()
            if lam>0:  # shrink toward all-route pooled shape of this class
                pool=(Xr*np.array([Y[ii,u].sum() for ii,u in meta])[:,None]).sum(0); pool/=pool.sum(); sh=(1-lam)*sh+lam*pool
            for t in tt:
                if cls4(DOW[t])!=c or not OKR[i,t]: continue
                P=Y[i,t].sum()*sh; num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
    return 1-num/den
for (o,e) in [('2025-10-01','2025-10-31'),('2025-10-13','2025-10-31'),('2025-03-03','2025-03-30'),('2025-02-17','2025-03-30')]:
    r=[('raw4',test(o,e,4)),('raw8',test(o,e,8)),('raw2',test(o,e,2))]
    for k in (2,4,6,10): r.append((f'svd{k}_w4',test(o,e,4,k)))
    r.append(('raw4_shrink0.1',test(o,e,4,None,0.1)))
    print(o,' '.join('%s=%.4f'%x for x in r))
