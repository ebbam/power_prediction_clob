import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ab = json.load(open("figdata_AB.json")); cd = json.load(open("figdata_CD.json")); fg = json.load(open("figdata_FGH.json")); sh = json.load(open("figdata_shock.json"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"; RAMP = ["#86b6ef", "#2a78d6", "#104281"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2})
dot = dict(marker="o", ms=7, mec=SURF, mew=1.2, ls="none"); sq = dict(dot, marker="s")
fig, ax = plt.subplots(3, 3, figsize=(15, 13)); ax = ax.ravel(); D, s0 = 0.2, 0.3

a = np.array(ab["A"]); r = np.linspace(0.05, 0.65, 50)
ax[0].plot(r, r*D, color=INK2, label=r"theory $\delta^*=\rho\Delta$")
ax[0].plot(a[a[:,1]==0][:,0], a[a[:,1]==0][:,2], color=BLUE, label="sim, no signal noise", **dot)
ax[0].plot(a[a[:,1]>0][:,0], a[a[:,1]>0][:,2], color=ORANGE, label="sim, signal s.d. 0.1", **sq)
ax[0].set(title="A  Whale distortion without herding", xlabel=r"whale wealth share $\rho$", ylabel=r"steady-state error $\delta^*$"); ax[0].legend(loc="upper left")

b = np.array(ab["B"]); hh = np.linspace(0, 1, 60)
for rho, c, off in [(0.2, BLUE, -11), (0.4, ORANGE, 4)]:
    Pi = rho*((1-hh)*(1-s0)+hh)/((1-hh)*(1-s0)+hh*rho); ax[1].plot(hh, Pi, color=c, alpha=0.55)
    sel = b[b[:,0]==rho]; ax[1].plot(sel[:,1], sel[:,2], color=c, **dot)
    ax[1].annotate(rf"$\rho={rho}$", (0.0, rho), xytext=(8, -14 if rho==0.2 else 8), textcoords="offset points", color=c, fontsize=9, fontweight="bold")
ax[1].plot([], [], color=INK2, alpha=0.6, label=r"theory $\Pi=\rho/(1-(1-\rho)\mathcal{H})$"); ax[1].plot([], [], color=INK2, label="sim", **dot)
ax[1].set(title=r"B  Herding amplifies pass-through ($s=0.3$)", xlabel=r"herding weight $h$", ylabel=r"pass-through $\Pi=\delta^*/\Delta$", ylim=(0, 1.08), xlim=(-0.03, 1.13)); ax[1].legend(loc="upper left")

for h, c in zip(["0", "0.5", "0.9"], RAMP):
    y = np.array(cd["C"][h]); hv = float(h); ax[2].plot(np.arange(len(y)), y, color=c, label=f"h = {h}")
    ax[2].axhline(0.3*D*((1-hv)*(1-s0)+hv)/((1-hv)*(1-s0)+hv*0.3), color=c, lw=1, ls="--")
ax[2].set(title=r"C  Error after a whale enters ($\rho=0.3$)", xlabel="period", ylabel=r"price error $m_t-\eta$")
ax[2].legend(loc="lower right", title=r"dashed = predicted $\delta^*$", title_fontsize=8)

for h, c in zip(["0", "0.5", "0.9"], RAMP):
    y = np.array(cd["D"][h]); hv = float(h); k0 = hv+(1-hv)*s0; t = np.arange(len(y))
    ax[3].plot(t, np.abs(y)+1e-4, color=c, label=rf"h = {h} ($\kappa$={k0:.2f})"); ax[3].plot(t, abs(y[0])*k0**t, color=c, lw=1, ls="--")
ax[3].axhline(0.01, color=INK2, lw=1, ls=":"); ax[3].text(39, 0.0112, "1 tick", ha="right", fontsize=8, color=INK2)
ax[3].set(yscale="log", ylim=(1e-3, 0.3), title="D  Recovery from a belief shock (no whale)", xlabel="period", ylabel=r"$|m_t-\eta|$ (log scale)")
ax[3].legend(loc="upper right", title=r"dashed = $\delta_0\,\kappa^t$", title_fontsize=8)

S = sh; tb = np.array([x["T_bench"] for x in S]); ts = np.array([x["T_sim"] for x in S]); wh = np.array([x["rho"] > 0 for x in S])
lim = 17; ax[4].plot([0, lim], [0, lim], color=INK2, lw=1, ls="--"); ax[4].text(16.5, 15.2, "sim = benchmark", color=INK2, fontsize=8, ha="right")
ax[4].plot(tb[~wh], ts[~wh], color=BLUE, label="no whale", **dot); ax[4].plot(tb[wh], ts[wh], color=ORANGE, label=r"whale, $\rho=0.3$", **sq)
ax[4].set(title=r"E  First passage vs fast-clearing benchmark $T_\epsilon$", xlabel=r"benchmark $T_\epsilon=\ln(\epsilon/|\delta_0-\delta^*|)/\ln\kappa$", ylabel="simulated first-passage time", xlim=(0, lim), ylim=(0, lim))
ax[4].legend(loc="upper left", title=r"belief shock $-0.15$; $\epsilon=0.02$", title_fontsize=8)

hx = np.linspace(0, 0.65, 40)
for s, c, lab in [(0.2, BLUE, r"$s=0.2<1-\rho$: herding slows"), (0.9, ORANGE, r"$s=0.9>1-\rho$: herding speeds")]:
    ax[5].plot(hx, (1-hx)*s + hx*0.7, color=c, alpha=0.55)
    sel = [x for x in S if x["rho"] == 0.3 and x["s"] == s and np.isfinite(x["kappa_emp"])]
    ax[5].plot([x["h"] for x in sel], [x["kappa_emp"] for x in sel], color=c, label=lab, **(dot if s == 0.2 else sq))
ax[5].set(title=r"F  $\partial\kappa/\partial h = 1-\rho-s$ ($\rho=0.3$)", xlabel=r"herding weight $h$", ylabel=r"convergence factor $\kappa$ (line theory, marks sim)", ylim=(0, 1)); ax[5].legend(loc="center right")

f = np.array(fg["F"]); hd = np.array(fg["H"])
ax[6].plot(f[:,1], f[:,2], color=BLUE, label=r"$h=0$, varying signal noise", **dot); ax[6].plot(hd[:,1], hd[:,2], color=ORANGE, label=r"signal s.d. 0.1, $h=0.25,0.5,0.75$", **sq)
ax[6].axvline(0.05, color=INK2, lw=1, ls=":"); ax[6].text(0.051, 0.002, r"$\epsilon_{\max}$", color=INK2, fontsize=8)
ax[6].set(ylim=(0, 0.055), title=r"G  Order-book noise $\sigma_\mu$ vs belief dispersion", xlabel="cross-sectional s.d. of valuations", ylabel=r"s.d. of $m_t-p^*_t$"); ax[6].legend(loc="upper left")

band = 2/np.sqrt(f[0,4]); ax[7].axhspan(-band, band, color=GRID, alpha=0.7, lw=0); ax[7].axhline(0, color=INK2, lw=1)
ax[7].plot(f[:,1], f[:,3], color=BLUE, label=r"$h=0$", **dot); ax[7].plot(hd[:,1], hd[:,3], color=ORANGE, label=r"$h>0$", **sq)
ax[7].text(0.012, band+0.012, r"$\pm 2/\sqrt{n}$", color=INK2, fontsize=8)
ax[7].set(title=r"H  Test of (A5): lag-1 autocorrelation of $\mu_t$", xlabel="cross-sectional s.d. of valuations", ylabel=r"corr$(\mu_t,\mu_{t-1})$", ylim=(-0.4, 0.4)); ax[7].legend(loc="lower left")

g = np.array(fg["G"]); ax[8].plot(g[:,0], g[:,2], color=INK2, label="theory (aggregation term)"); ax[8].plot(g[:,0], g[:,1], color=BLUE, label="sim", **dot)
ax[8].set(xscale="log", yscale="log", title="I  Signal-aggregation error vs market size", xlabel="number of agents N", ylabel=r"Var$(p^*_t-\eta)$"); ax[8].legend(loc="upper right")

fig.suptitle(r"CLOB theory vs simulation  (log utility $r_i=1$; 100 agents unless stated; $\eta=0.5$; whale $W=0.7$, $\Delta=0.2$)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("clob_theory_vs_sim.png", dpi=150); print("saved")
