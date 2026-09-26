"""High-capacity GRU with drift-symmetric synthetic augmentation (block-bootstrap residuals)."""
from dzoo import *
import torch, torch.nn as nn, pickle, sys
torch.set_num_threads(4)
DS=pickle.load(open('dzoo.pkl','rb')); L=42
def window(Dser,okser,o,t,cl12,a):
    idx=np.arange(o-L,o); m=np.array([1.0 if (u>=a and okser[u]) else 0.0 for u in idx])
    return np.c_[Dser[idx]/cl12*m,m,np.eye(7)[DOW[idx]]], np.r_[np.eye(7)[DOW[t]],(t-o)/28]
def real_set(df,per):
    a,_=PER[per]; X=[];Z=[];Yv=[];B=[]
    for i,o,t,y,cl12 in df[['route','o','t','y','cl12']].values:
        x,z=window(D[int(i)],OKR[int(i)],int(o),int(t),cl12,a); X.append(x);Z.append(z);Yv.append(y/cl12);B.append(cl12)
    return X,Z,Yv,B
def synth_set(per,n_series=60,seed=0):
    """synthetic daily series: symmetric random drift x dow pattern x block-bootstrapped weekly residuals"""
    rng=np.random.default_rng(seed); a,b=PER[per]; X=[];Z=[];Yv=[];B=[]
    for s in range(n_series):
        i=rng.integers(len(ROUTES)); tt=np.array([u for u in range(a,b+1) if OKR[i,u]])
        dp=np.array([np.median(D[i,tt[DOW[tt]==d]]) for d in range(7)]); dp=np.maximum(dp,1)
        res=np.log(np.maximum(D[i,tt],1)/dp[DOW[tt]]); res=res-np.median(res)
        n=len(tt); drift=rng.uniform(-0.004,0.004); lvl=np.cumsum(np.r_[0,rng.normal(drift,0.004,T-1)])
        blocks=[]; 
        while len(blocks)<T: st=rng.integers(0,max(1,n-7)); blocks.extend(res[st:st+7])
        Dsyn=dp[DOW]*np.exp(lvl-lvl[a+14]+np.array(blocks[:T]))
        oks=np.ones(T,bool)
        for _ in range(12):
            o=rng.integers(a+14,b-27)
            p=np.array([u for u in range(max(a,o-84),o) if DOW[u]<4 or cls4(DOW[u])==cls4(DOW[u])])
            for t in rng.choice(np.arange(o,o+28),6,replace=False):
                c=cls4(DOW[t]); cl=[u for u in range(o-1,a-1,-1) if cls4(DOW[u])==c][:12*(4 if c==0 else 1)]
                cl12=Dsyn[cl].mean(); x,z=window(Dsyn,oks,o,t,cl12,a)
                X.append(x);Z.append(z);Yv.append(Dsyn[t]/cl12);B.append(cl12)
    return X,Z,Yv,B
T_=lambda a_: torch.tensor(np.array(a_),dtype=torch.float32)
class Net(nn.Module):
    def __init__(s,h=48,layers=2,drop=0.2):
        super().__init__(); s.g=nn.GRU(9,h,num_layers=layers,batch_first=True,dropout=drop)
        s.f=nn.Sequential(nn.Linear(h+8,32),nn.GELU(),nn.Dropout(drop),nn.Linear(32,1))
    def forward(s,x,z): _,hN=s.g(x); return 1+s.f(torch.cat([hN[-1],z],1)).squeeze(1)
def train_predict(tr,te,seed,epochs=40):
    torch.manual_seed(seed); Xtr,Ztr,Ytr,Btr=map(T_,tr); Xte,Zte,_,Bte=map(T_,te)
    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Ytr))
        for k in range(0,len(perm),256):
            b=perm[k:k+256]; loss=(Btr[b]*(net(Xtr[b],Ztr[b])-Ytr[b]).abs()).sum()/Btr[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad(): return (net(Xte,Zte)*Bte).numpy()
out={}
for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
    real=real_set(DS[tr_p],tr_p); te=real_set(DS[te_p],te_p)
    syn=synth_set(tr_p,n_series=120,seed=1)
    aug=tuple(r+s for r,s in zip(real,syn))
    for name,trs in [('gru_big',real),('gru_big_aug',aug)]:
        p=np.mean([train_predict(trs,te,seed) for seed in (0,1)],0); out[(te_p,name)]=p
        df=DS[te_p]; print(te_p,name,'daily %.4f'%(1-np.abs(df.y-p).sum()/df.y.sum()),flush=True)
pickle.dump(out,open('pred_gru.pkl','wb'))
