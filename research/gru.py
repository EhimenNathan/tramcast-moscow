from dzoo import *
import torch, torch.nn as nn, pickle
torch.manual_seed(0); np.random.seed(0)
DS=pickle.load(open('dzoo.pkl','rb'))
L=42
def seqs(df,per):
    a,_=PER[per]; X=[];Z=[];Yv=[];B=[]
    for i,o,t,y,cl12 in df[['route','o','t','y','cl12']].values:
        i,o,t=int(i),int(o),int(t)
        idx=np.arange(o-L,o); v=D[i,idx]/cl12; m=np.array([1.0 if (u>=a and OKR[i,u]) else 0.0 for u in idx]); v=v*m
        X.append(np.c_[v,m,np.eye(7)[DOW[idx]]]); Z.append(np.r_[np.eye(7)[DOW[t]],(t-o)/28]); Yv.append(y/cl12); B.append(cl12)
    return [torch.tensor(np.array(a_),dtype=torch.float32) for a_ in (X,Z,Yv,B)]
class Net(nn.Module):
    def __init__(s,h=16):
        super().__init__(); s.g=nn.GRU(9,h,batch_first=True); s.f=nn.Sequential(nn.Linear(h+8,16),nn.ReLU(),nn.Linear(16,1))
    def forward(s,x,z): _,hN=s.g(x); return 1+s.f(torch.cat([hN[0],z],1)).squeeze(1)
for tr_p,te_p in [('spring','autumn'),('autumn','spring')]:
    Xtr,Ztr,Ytr,Btr=seqs(DS[tr_p],tr_p); Xte,Zte,Yte,Bte=seqs(DS[te_p],te_p)
    preds=[]
    for seed in range(3):
        torch.manual_seed(seed); net=Net(); opt=torch.optim.Adam(net.parameters(),lr=3e-3,weight_decay=1e-4)
        for ep in range(60):
            perm=torch.randperm(len(Ytr))
            for k in range(0,len(perm),256):
                b=perm[k:k+256]; loss=(Btr[b]*(net(Xtr[b],Ztr[b])-Ytr[b]).abs()).sum()/Btr[b].sum()   # exact WAPE loss
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): preds.append((net(Xte,Zte)*Bte).numpy())
    p=np.mean(preds,0); te=DS[te_p]
    print(tr_p,'->',te_p,'GRU daily %.4f  hourly %.4f   (cl4 daily %.4f)'%(1-np.abs(te.y-p).sum()/te.y.sum(), hourly_score(te,p,te_p), 1-np.abs(te.y-te.cl4).sum()/te.y.sum()))
