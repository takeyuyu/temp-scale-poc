import numpy as np, pandas as pd
df=pd.read_csv("scale_timeseries.csv",parse_dates=["date"])
cyc=pd.read_csv("scale_cycles.csv")
clean=np.load("clean_days.npy")

print("===== 記事の各記述 vs 実データ 突き合わせ =====\n")

# 1) 清掃回数
print(f"[清掃回数] 実データ: {len(clean)}回 / 記事図キャプション: 赤破線=清掃")

# 2) サイクル相関
r=np.corrcoef(cyc["cold_degree_days_per_day"],cyc["slope_current_per_day"])[0,1]
print(f"[サイクル相関] 実データ r={r:.3f} / 記事: 0.63 → {'OK' if abs(r-0.63)<0.005 else 'NG'}")

# 3) 素朴な相関
nr=np.corrcoef(df["daily_temp"].iloc[1:],df["pump_current"].diff().iloc[1:])[0,1]
print(f"[素朴相関] 実データ r={nr:.3f} / 記事: r≒0.0 → {'OK' if abs(nr)<0.02 else 'NG'}")

# 4) 「冷えると溜まりが速い」=寒いサイクルほど傾き大、を確認（符号）
slope_temp_corr=np.corrcoef(cyc["mean_temp"],cyc["slope_current_per_day"])[0,1]
print(f"[向きの整合] 平均気温 vs 傾き r={slope_temp_corr:.3f} (負なら『冷えると速い』と整合) → {'OK' if slope_temp_corr<0 else 'NG'}")

# 5) 冬場(1-4月)の清掃間隔 vs 夏場
df["month"]=df["date"].dt.month
winter_cleans=[c for c in clean if df['date'].iloc[c].month in (1,2,3,4)]
summer_cleans=[c for c in clean if df['date'].iloc[c].month in (6,7,8,9)]
print(f"[冬の清掃] 1-4月: {len(winter_cleans)}回 / [夏の清掃] 6-9月: {len(summer_cleans)}回")
print(f"  → 記事『冬は間隔短い・夏は長く保つ』と整合: {'OK' if len(winter_cleans)>len(summer_cleans) else 'NG'}")

# 6) 夏の最長サイクル日数
print(f"[夏の長期サイクル] 最長サイクル: {cyc['n_days'].max()}日 (記事『夏は長く保つ』の根拠)")

# 7) ベース電流と最大電流（図の軸と整合）
print(f"[電流レンジ] min={df['pump_current'].min():.1f}A max={df['pump_current'].max():.1f}A (記事本文に具体値の言及なし=矛盾なし)")

# 8) 10シード頑健性の再確認
def run(seed):
    rng=np.random.default_rng(seed); days=365; t=np.arange(days)
    ts=15-12*np.cos(2*np.pi*(t-20)/365); tn=rng.normal(0,2.0,days)
    for _ in range(8):
        s=rng.integers(0,days-5); tn[s:s+rng.integers(2,5)]+=rng.normal(0,4)
    dt=ts+tn; lt=np.zeros(days); lt[0]=dt[0]
    for i in range(1,days):
        tg=30.0+0.45*(dt[i]-15); lt[i]=lt[i-1]+(tg-lt[i-1])/7.0
    ss=np.clip(30.0-lt,0,None); rt=0.020*ss
    lot=rng.normal(1.0,0.25,4).clip(0.5,1.6); lf=np.repeat(lot,days//4+1)[:days]
    rate=rt+0.015*lf+np.abs(rng.normal(0,0.004,days))
    sc=np.zeros(days); cum=0;cid=0;cy=np.zeros(days,int);cl=[]
    for i in range(days):
        cum+=rate[i]
        if cum>=2.2: cl.append(i);cum=0;cid+=1
        sc[i]=cum;cy[i]=cid
    cur=18.0+3.5*sc+rng.normal(0,0.15,days)+rng.normal(0,0.08,days)
    d=pd.DataFrame({"day":t,"temp":dt,"cyc":cy,"cur":cur}); rec=[]
    for c,g in d.groupby("cyc"):
        if len(g)<5:continue
        sl=np.polyfit(g["day"]-g["day"].mean(),g["cur"],1)[0]
        cdd=np.clip(18.0-g["temp"],0,None).sum()/len(g); rec.append((sl,cdd))
    rr=pd.DataFrame(rec,columns=["s","c"])
    return np.corrcoef(rr["c"],rr["s"])[0,1]
cs=[run(s) for s in range(10)]
print(f"\n[10シード] 平均={np.mean(cs):.3f} 最小={np.min(cs):.3f} 最大={np.max(cs):.3f}")
print(f"  → 記事『0.6〜0.85(平均0.72)』と整合: {'OK' if abs(np.mean(cs)-0.72)<0.03 and min(cs)>0.58 else '要確認'}")
