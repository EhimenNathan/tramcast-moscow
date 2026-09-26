"""LSTM / temporal CNN (dilated) / Transformer encoder on the same cross-period daily-level protocol as the GRU.
Exact WAPE loss; raw and drift-neutral scores; 2 seeds each."""
from dzoo import *
import torch, torch.nn as nn, pickle, math
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
class Head(nn.Module):
    def __init__(s,h): super().__init__(); s.f=nn.Sequential(nn.Linear(h+8,32),nn.GELU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(s,hN,z): return 1+s.f(torch.cat([hN,z],1)).squeeze(1)
class LSTMNet(nn.Module):
    def __init__(s,h=48): super().__init__(); s.r=nn.LSTM(9,h,2,batch_first=True,dropout=0.2); s.h=Head(h)
    def forward(s,x,z): o,_=s.r(x); return s.h(o[:,-1],z)
class TCN(nn.Module):
    def __init__(s,h=32):
        super().__init__(); L=[]; c=9
        for d in (1,2,4,8,16): L+=[nn.Conv1d(c,h,3,padding=d,dilation=d),nn.GELU(),nn.Dropout(0.1)]; c=h
        s.net=nn.Sequential(*L); s.h=Head(h)
    def forward(s,x,z): o=s.net(x.transpose(1,2)); return s.h(o[:,:,-7:].mean(2),z)
class Transf(nn.Module):
    def __init__(s,d=32):
        super().__init__(); s.inp=nn.Linear(9,d); s.pos=nn.Parameter(torch.randn(1,42,d)*0.02)
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,64,0.1,batch_first=True),2); s.h=Head(d)
    def forward(s,x,z): o=s.enc(s.inp(x)+s.pos); return s.h(o.mean(1),z)
def fit_pred(cls,tr,te,seed,epochs=40):
    torch.manual_seed(seed); Xtr,Ztr,Ytr,Btr=map(T_,tr); Xte,Zte,_,Bte=map(T_,te)
    net=cls(); opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Ytr))
        for k in range(0,len(perm),256):
            b=perm[k:k+256]; loss=(Btr[b]*(net(Xtr[b],Ztr[b])-Ytr[b]).abs()).sum()/Btr[b].sum(); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad(): return (net(Xte,Zte)*Bte).numpy()
out={}
for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
    tr=real_set(DS[tr_p],tr_p); te=real_set(DS[te_p],te_p); df=DS[te_p]
    for name,cls in [('lstm',LSTMNet),('tcn_cnn',TCN),('transformer',Transf)]:
        p=np.mean([fit_pred(cls,tr,te,s) for s in (0,1)],0); out[(te_p,name)]=p
        pn=p.copy()
        for o,g in df.groupby('o'): ix=g.index.values; pn[ix]=p[ix]*df.cl4.values[ix].sum()/max(p[ix].sum(),1)
        print(te_p,name,'raw %.4f  drift-neutral %.4f  (structural %.4f)'%(1-np.abs(df.y-p).sum()/df.y.sum(),1-np.abs(df.y-pn).sum()/df.y.sum(),1-np.abs(df.y-df.cl4).sum()/df.y.sum()),flush=True)
pickle.dump(out,open('pred_deep.pkl','wb'))
