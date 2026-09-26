from core import *
t0=di('2025-09-01'); W=8
weeks=[np.arange(t0+7*k,t0+7*k+7) for k in range(W)]   # each Mon..Sun
YY=np.stack([Y[:,w] for w in weeks],1)   # R x W x 7 x 24
def ev(est, folds='lowo'):
    num=den=0
    if folds=='lowo':
        for k in range(W):
            tr=[j for j in range(W) if j!=k]
            P=est(YY[:,tr]); y=YY[:,k]
            num+=np.abs(y-P).sum(); den+=y.sum()
    else:  # forward: train first 4 predict last 4
        P=est(YY[:,:4]); y=YY[:,4:]
        num=np.abs(y-P[:,None]).sum(); den=y.sum()
    return round(1-num/den,4)
def e_med(X): return np.median(X,1)
def e_mean(X): return X.mean(1)
def e_struct(classes, lev='median', shp='mean', lam=0.0):
    def f(X):  # X: R x w x 7 x 24
        D=X.sum(3)        # R x w x 7
        L=np.median(D,1) if lev=='median' else D.mean(1)   # R x 7
        out=np.zeros(X.shape[:1]+X.shape[2:])
        for c in set(classes):
            ds=[d for d in range(7) if classes[d]==c]
            if shp=='mean':
                S=X[:,:,ds].sum((1,2)); S=S/S.sum(1,keepdims=True)
            else:
                N=X[:,:,ds]/np.maximum(D[:,:,ds][...,None],1)
                S=np.median(N.reshape(N.shape[0],-1,24),1); S=S/S.sum(1,keepdims=True)
            for d in ds:
                Sd=X[:,:,d].sum(1); Sd=Sd/Sd.sum(1,keepdims=True)
                out[:,d]=L[:,d][:,None]*((1-lam)*S+lam*Sd)
        return out
    return f
print('median',ev(e_med),ev(e_med,'fwd'))
print('mean',ev(e_mean),ev(e_mean,'fwd'))
for name,cl in [('7cls',[0,1,2,3,4,5,6]),('M-F',[0,0,0,0,0,5,6]),('M-Th,F',[0,0,0,0,4,5,6]),('Mon,TWT,F',[0,1,1,1,4,5,6])]:
    for lev in ['median','mean']:
        for shp in ['mean','median']:
            print(name,lev,shp,ev(e_struct(cl,lev,shp)),ev(e_struct(cl,lev,shp),'fwd'))
f=e_struct([0,0,0,0,4,5,6],'mean','mean')
for c in [0.97,0.99,1.0,1.01,1.03]:
    print('scale',c, ev(lambda X: c*f(X)))
