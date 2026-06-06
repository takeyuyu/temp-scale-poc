"""
気温 × 配管スケール堆積 PoC（合成データ）
主題: 気温という外部要因を、清掃時期の現場判断に組み込めるか

設計方針（結論ありきにしない）:
- 堆積速度は「気温で決まる成分」+「気温と無関係な交絡（原料ロット）」+ ノイズ で作る
- 薬注は検証期間中固定（=堆積に一定の抑制をかける定数として吸収）
- 清掃でリセットされるノコギリ波。各サイクルの「登る傾き」を取り出して気温と突き合わせる
- 観測は流量一定下のポンプ電流。電流には測定ノイズと、堆積と無関係な日々変動も乗せる
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---- 1. 1年分の日付と気象庁的な気温プロファイル ----
days = 365
t = np.arange(days)
# 日平均気温: 年周期（夏高・冬低）+ 日々のゆらぎ。日本の平野部を想定（年平均~15℃, 振幅~12℃）
temp_seasonal = 15 - 12 * np.cos(2 * np.pi * (t - 20) / 365)   # 1月末が最低
temp_noise = rng.normal(0, 2.0, days)                          # 日々のばらつき
# 寒波・暖かい日などの不規則イベント
for _ in range(8):
    s = rng.integers(0, days - 5)
    temp_noise[s:s+rng.integers(2,5)] += rng.normal(0, 4)
daily_temp = temp_seasonal + temp_noise

# ---- 2. 液温: 外気の影響を「少なからず受ける」が、遅れて鈍る（保温の有無を問わず季節では効く） ----
# 一次遅れフィルタで外気を平滑化（保温=時定数が大きい、と解釈）
liquid_temp = np.zeros(days)
liquid_temp[0] = daily_temp[0]
tau = 7.0  # 時定数(日): 保温やプロセス熱でならされる
base_liquid = 30.0  # プロセス自身の発熱で押し上げられたベース液温
for i in range(1, days):
    # 液温はベース液温と外気の間で平衡。外気が下がると液温も少し下がる
    target = base_liquid + 0.45 * (daily_temp[i] - 15)  # 外気感度0.45（完全追従でない）
    liquid_temp[i] = liquid_temp[i-1] + (target - liquid_temp[i-1]) / tau

# ---- 3. 堆積速度(日ごとの増分) ----
# (A) 気温(液温)成分: ここでは「低温ほど析出が進む」向きを採用（通常溶解度の物質）
#     液温が基準より低いほど過飽和→堆積速い。向きは物質次第、というのは記事で両論併記する。
ref_temp = 30.0
super_sat = np.clip(ref_temp - liquid_temp, 0, None)  # 基準より冷えた分だけ過飽和
rate_temp = 0.020 * super_sat                          # 気温由来の堆積速度

# (B) 交絡: 原料ロット変動（気温と無関係に堆積性が変わる）。四半期ごとにロットが変わる想定
lot_levels = rng.normal(1.0, 0.25, 4).clip(0.5, 1.6)
lot_factor = np.repeat(lot_levels, days // 4 + 1)[:days]
rate_lot = 0.015 * lot_factor                          # ロット由来のベース堆積（気温と無相関）

# (C) ランダムな日々ノイズ
rate_noise = np.abs(rng.normal(0, 0.004, days))

daily_deposit_rate = rate_temp + rate_lot + rate_noise  # mm/day 相当(無次元)

# ---- 4. 清掃でリセットされるノコギリ波の累積堆積 ----
# 清掃は「堆積がしきい値を超えたら実施」=現場の経験則的運用。間隔は一定でない。
clean_threshold = 2.2
scale = np.zeros(days)
clean_days = []
cum = 0.0
cycle_id = np.zeros(days, dtype=int)
cid = 0
for i in range(days):
    cum += daily_deposit_rate[i]
    if cum >= clean_threshold:
        clean_days.append(i)
        cum = 0.0
        cid += 1
    scale[i] = cum
    cycle_id[i] = cid

# ---- 5. 観測: ポンプ電流（流量一定）。電流 = ベース + 堆積による上昇 + 日々変動 + 測定ノイズ ----
base_current = 18.0  # A
current_from_scale = 3.5 * scale                       # 堆積で押し上げ
current_daily = rng.normal(0, 0.15, days)              # 堆積と無関係な運転ゆらぎ
current_meas_noise = rng.normal(0, 0.08, days)
pump_current = base_current + current_from_scale + current_daily + current_meas_noise

df = pd.DataFrame({
    "day": t,
    "date": pd.date_range("2024-01-01", periods=days, freq="D"),
    "daily_temp": daily_temp,
    "liquid_temp": liquid_temp,
    "deposit_rate": daily_deposit_rate,
    "scale": scale,
    "cycle": cycle_id,
    "pump_current": pump_current,
})
df["cleaned"] = df["day"].isin(clean_days)

# ---- 6. 解析: サイクルごとの「堆積速度(電流の上昇傾き)」を取り出し、気温と突き合わせる ----
# 各清掃サイクル内で電流を線形回帰し、その傾き(=実効堆積速度の代理)を出す
records = []
for c, g in df.groupby("cycle"):
    if len(g) < 5:   # 短すぎるサイクルは除外
        continue
    x = g["day"].values.astype(float)
    y = g["pump_current"].values
    slope = np.polyfit(x - x.mean(), y, 1)[0]   # A/day
    # そのサイクル期間の積算寒度（度日的指標）: 基準温度を上回る"冷え"の積算
    base_dd = 18.0  # この温度を下回った分を積算（暖房度日の発想の転用）
    cold_degree_days = np.clip(base_dd - g["daily_temp"].values, 0, None).sum()
    mean_temp = g["daily_temp"].mean()
    records.append({
        "cycle": c,
        "n_days": len(g),
        "slope_current_per_day": slope,
        "cold_degree_days_per_day": cold_degree_days / len(g),  # 1日あたりの冷度
        "mean_temp": mean_temp,
    })
cyc = pd.DataFrame(records)

# 相関: サイクルの「電流上昇の速さ」 vs 「1日あたり冷度(度日)」
corr = np.corrcoef(cyc["cold_degree_days_per_day"], cyc["slope_current_per_day"])[0, 1]

# 比較対象: 生の電流の日次変化を「上がった/下がった」で見ても季節は読めない、を示す
df["current_diff"] = df["pump_current"].diff()
naive_corr = np.corrcoef(
    df["daily_temp"].iloc[1:], df["current_diff"].iloc[1:]
)[0, 1]

print("=== 検証結果 ===")
print(f"清掃回数: {len(clean_days)} 回")
print(f"サイクル数(解析対象): {len(cyc)}")
print(f"[本命] サイクル堆積速度 vs 1日あたり冷度の相関 r = {corr:.3f}")
print(f"[比較] 生電流の日次変化 vs 当日気温の相関 r = {naive_corr:.3f}")
print()
print("サイクル別サマリ:")
print(cyc.round(3).to_string(index=False))

# 度日との回帰式
b, a = np.polyfit(cyc["cold_degree_days_per_day"], cyc["slope_current_per_day"], 1)
print(f"\n回帰: 電流上昇速度 = {b:.5f} × 冷度/日 + {a:.5f}")

# 保存
df.to_csv("/home/claude/scale_timeseries.csv", index=False)
cyc.to_csv("/home/claude/scale_cycles.csv", index=False)
np.save("/home/claude/clean_days.npy", np.array(clean_days))
print("\nsaved.")
