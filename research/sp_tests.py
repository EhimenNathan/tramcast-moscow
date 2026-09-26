from cands import *
def cls4(dw): return 0 if dw<4 else dw-3
def trunc(sh,K):
    F=np.fft.rfft(sh); F[K+1:]=0; s=np.maximum(np.fft.irfft(F,24),0); return s/s.sum()
def shp(i,o,c,wk=4):
    u=[t for t in range(o-7*wk,o) if OKR[i,t] and cls4(DOW[t])==c]; s=Y[i,u].sum(0); return s/s.sum()
def test_shape(o,e,K):
    o,e=di(o),di(e); num=den=0
    for i in range(len(ROUTES)):
        for c in range(4):
            sh=shp(i,o,c); sh=trunc(sh,K) if K else sh
            for t in range(o,e+1):
                if OKR[i,t] and cls4(DOW[t])==c: P=Y[i,t].sum()*sh; num+=np.abs(Y[i,t]-P).sum(); den+=Y[i,t].sum()
    return 1-num/den
print('Harmonic truncation of hourly shape (oracle daily totals):')
for o,e in [('2025-10-01','2025-10-31'),('2025-03-03','2025-03-30')]:
    print(' ',o,' '.join('K%s=%.4f'%(K if K else 'raw',test_shape(o,e,K)) for K in [None,11,10,8,6]))
# Hampel-filtered robust level vs plain recency level (daily-level WAPE on stable suite)
D=Y.sum(2)
def level(i,o,c,hampel):
    u=np.array([t for t in range(o-42,o) if OKR[i,t] and cls4(DOW[t])==c]); x=D[i,u].astype(float)
    if hampel and len(x)>=4:
        med=np.median(x); mad=1.4826*np.median(np.abs(x-med))+1e-9; x=np.where(np.abs(x-med)>3*mad,med,x)
    w=0.5**((o-1-u)/10); return np.sum(x*w)/np.sum(w)
for hp in [False,True]:
    num=den=0
    for o,e in [('2025-02-03','2025-03-30'),('2025-03-03','2025-03-30'),('2025-10-01','2025-10-31'),('2025-10-13','2025-10-31')]:
        o_,e_=di(o),di(e)
        for i in range(len(ROUTES)):
            for t in range(o_,e_+1):
                if OKR[i,t]: p=level(i,o_,cls4(DOW[t]),hp); num+=abs(D[i,t]-p); den+=D[i,t]
    print('Hampel' if hp else 'plain ','level daily score %.4f'%(1-num/den))
