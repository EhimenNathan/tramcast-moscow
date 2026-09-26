from cands import *
from scipy.optimize import linprog
import pickle
SD=pickle.load(open('suite.pkl','rb'))
names=list(SD[0][1].keys())
def flat(C,y,m,names):
    X=np.stack([C[n][m] for n in names],-1).reshape(-1,len(names)); return X, y[m].reshape(-1)
def score(p,y): return 1-np.abs(y-p).sum()/y.sum()
print('%-14s'%'cand', ' '.join('%10s'%o[5:] for o,_,_,_ in SD), '  pooled')
tab={}
for n in names:
    num=den=0; row=[]
    for o,C,y,m in SD:
        X,yy=flat(C,y,m,[n]); p=X[:,0]; row.append(score(p,yy)); num+=np.abs(yy-p).sum(); den+=yy.sum()
    tab[n]=1-num/den; print('%-14s'%n,' '.join('%10.4f'%v for v in row),'  %.4f'%(1-num/den))
def lp_weights(X,y):
    """min sum|y - Xw| s.t. w>=0, sum w=1  (exact WAPE-optimal convex blend)"""
    n,k=X.shape
    # vars: w(k), e+(n), e-(n); X w + e+ - e- = y
    c=np.r_[np.zeros(k),np.ones(2*n)]
    from scipy.sparse import hstack, identity, csr_matrix
    A_eq=hstack([csr_matrix(X),identity(n),-identity(n)]).tocsr()
    A_eq=__import__('scipy.sparse',fromlist=['vstack']).vstack([A_eq,csr_matrix(np.r_[np.ones(k),np.zeros(2*n)])])
    b_eq=np.r_[y,1.0]
    r=linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=[(0,None)]*(k+2*n),method='highs'); return r.x[:k]
# leave-one-origin-out LP stacking (subsample rows for speed)
rng=np.random.default_rng(0)
num=den=0; rows=[]
for k,(o,C,y,m) in enumerate(SD):
    Xs=[];ys=[]
    for j,(o2,C2,y2,m2) in enumerate(SD):
        if j==k or abs(j-k)<=0 and False: continue
        if (k>=3)==(j>=3) and abs(j-k)<=1: continue   # drop adjacent overlapping origin
        X,yy=flat(C2,y2,m2,names); idx=rng.choice(len(yy),min(len(yy),700),replace=False); Xs.append(X[idx]); ys.append(yy[idx])
    w=lp_weights(np.vstack(Xs),np.concatenate(ys))
    X,yy=flat(C,y,m,names); p=X@w; num+=np.abs(yy-p).sum(); den+=yy.sum(); rows.append(round(score(p,yy),4))
    print(o,'LP-stack %.4f'%score(p,yy),{names[i]:round(w[i],2) for i in np.argsort(-w)[:5]})
print('LP stack pooled %.4f'%(1-num/den), rows)
