import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import os

# 日本語フォント探索
jp=None
for p in ["/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(p): jp=p; break
if jp:
    font_manager.fontManager.addfont(jp)
    rcParams["font.family"]=font_manager.FontProperties(fname=jp).get_name()
print("jp font:",jp)

# ---- 文字サイズの基準を全体的に底上げ ----
rcParams.update({
    "font.size":15,
    "axes.titlesize":17,
    "axes.labelsize":15,
    "xtick.labelsize":13,
    "ytick.labelsize":13,
    "legend.fontsize":14,
})

df=pd.read_csv("/home/claude/scale_timeseries.csv",parse_dates=["date"])
cyc=pd.read_csv("/home/claude/scale_cycles.csv")
clean_days=np.load("/home/claude/clean_days.npy")

INK="#1a1a2e"; ACC="#e94f37"; BLUE="#2a6f97"; GRAY="#8d99ae"; BG="#faf9f6"

# ===== 図1: 生の電流と気温 =====
fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,facecolor=BG)
for a in ax: a.set_facecolor(BG)
ax[0].plot(df["date"],df["pump_current"],color=INK,lw=1.1)
for cd in clean_days:
    ax[0].axvline(df["date"].iloc[cd],color=ACC,lw=1.0,alpha=0.55,ls="--")
ax[0].set_ylabel("ポンプ電流 (A)")
ax[0].set_title("① 生のポンプ電流：清掃のたびに下がるギザギザ（赤破線＝清掃）",
                loc="left",color=INK,pad=10)
ax[1].plot(df["date"],df["daily_temp"],color=BLUE,lw=1.1)
ax[1].set_ylabel("日平均気温 (℃)")
ax[1].set_title("② 気温：夏高・冬低。①と見比べても直接の対応は読み取りにくい",
                loc="left",color=INK,pad=10)
for a in ax:
    for s in ["top","right"]: a.spines[s].set_visible(False)
    a.grid(alpha=0.15)
plt.tight_layout(); plt.savefig("/home/claude/fig1_before.png",dpi=140,facecolor=BG,bbox_inches="tight"); plt.close()

# ===== 図2: サイクル散布図 =====
fig,ax=plt.subplots(figsize=(9,6.5),facecolor=BG); ax.set_facecolor(BG)
sc=ax.scatter(cyc["cold_degree_days_per_day"],cyc["slope_current_per_day"],
              s=160,c=cyc["mean_temp"],cmap="coolwarm_r",edgecolor=INK,lw=1.4,zorder=3)
b,a0=np.polyfit(cyc["cold_degree_days_per_day"],cyc["slope_current_per_day"],1)
xs=np.linspace(cyc["cold_degree_days_per_day"].min(),cyc["cold_degree_days_per_day"].max(),50)
ax.plot(xs,b*xs+a0,color=ACC,lw=2.4,zorder=2,label="回帰直線 (r=0.63)")
cb=plt.colorbar(sc); cb.set_label("サイクル平均気温 (℃)")
cb.ax.tick_params(labelsize=12)
ax.set_xlabel("1日あたりの「寒さの積み立て」")
ax.set_ylabel("電流上昇の速さ (A/日) ＝ 溜まりの速さ")
ax.set_title("③ 清掃サイクル単位で見ると：寒い時期ほど溜まりが速い",
             loc="left",color=INK,pad=10)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(alpha=0.15); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("/home/claude/fig2_after.png",dpi=140,facecolor=BG,bbox_inches="tight"); plt.close()

# ===== 図3: 対比 =====
fig,ax=plt.subplots(1,2,figsize=(14,5.8),facecolor=BG)
for a in ax: a.set_facecolor(BG)
ax[0].scatter(df["daily_temp"].iloc[1:],df["pump_current"].diff().iloc[1:],
              s=14,color=GRAY,alpha=0.55)
ax[0].set_title("素朴な見方：当日気温 vs 電流の日次変化\nr≒0.0（無相関に見える）",
                loc="left",color=INK,pad=10)
ax[0].set_xlabel("日平均気温 (℃)"); ax[0].set_ylabel("電流の日次変化 (A)")
ax[1].scatter(cyc["cold_degree_days_per_day"],cyc["slope_current_per_day"],
              s=150,c=cyc["mean_temp"],cmap="coolwarm_r",edgecolor=INK,lw=1.3)
ax[1].plot(xs,b*xs+a0,color=ACC,lw=2.4)
ax[1].set_title("粒度を変えた見方：溜まりの速さ vs 寒さの積み立て\nr=0.63（関係が立ち上がる）",
                loc="left",color=INK,pad=10)
ax[1].set_xlabel("1日あたりの「寒さの積み立て」"); ax[1].set_ylabel("電流上昇の速さ (A/日)")
for a in ax:
    for s in ["top","right"]: a.spines[s].set_visible(False)
    a.grid(alpha=0.15)
plt.tight_layout(); plt.savefig("/home/claude/fig3_contrast.png",dpi=140,facecolor=BG,bbox_inches="tight"); plt.close()
print("figs done")
