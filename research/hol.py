from core import *
D=Y.sum(2)
def near(t,dw,excl=SPEC,k=3):
    c=[u for u in range(t-28,t+29) if 0<=u<T and DOW[u]==dw and DATES[u] not in excl and u!=t]
    return np.median(D[:,c],1)
rows=[]
for s in ['2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05','2025-01-06','2025-01-07','2025-01-08','2025-01-09','2025-01-10','2025-01-11','2025-01-12','2025-04-30','2025-05-01','2025-05-02','2025-05-03','2025-05-04','2025-05-07','2025-05-08','2025-05-09','2025-05-10','2025-05-11','2025-06-11','2025-06-12','2025-06-13','2025-06-14','2025-06-15','2025-02-23','2025-03-08','2025-03-07']:
    t=di(s); sun=near(t,6); sat=near(t,5); own=near(t,DOW[t])
    st=[0,2,3,4,5,6,7]
    rows.append((s,DOW[t],D[st,t].sum()/sun[st].sum(),D[st,t].sum()/sat[st].sum(),D[st,t].sum()/own[st].sum()))
print(pd.DataFrame(rows,columns=['date','dow','vsSun','vsSat','vsOwnDow']).round(3).to_string())
