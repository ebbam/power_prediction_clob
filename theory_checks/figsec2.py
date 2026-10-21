import sys, os, json
sys.path.insert(0, os.path.expanduser("~/mnt/power_prediction_clob"))
import numpy as np
from bettor import bettor, run_market_lob
D, eta, W, N = 0.2, 0.5, 0.7, 100
def first_passage(err, tol=0.02, window=3):
    ok = np.abs(err) <= tol
    for t in range(len(ok)-window+1):
        if ok[t:t+window].all(): return t
    return np.nan
def shock_run(rho, s, h, shock=-0.15, seed=5):
    g = np.random.default_rng(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    pop = [bettor(budget=g.uniform(100, 1000), market_valuation=eta, risk_av=1.0, stubbornness=s, expertise=1.0, market_imitation=h) for _ in range(N)]
    if rho > 0:
        B = sum(b.budget for b in pop); pop.append(bettor(budget=B*rho/(1-rho), market_valuation=W, risk_av=1.0, whale=True))
    r1 = run_market_lob(len(pop), 60, eta, 0.0, pop, gen_el=np.full(60, eta), rng=rng)
    m_settled = r1["price_history"][-1]
    for b in pop:
        if not b.whale: b.market_valuation += shock
    r2 = run_market_lob(len(pop), 60, m_settled, 0.0, pop, gen_el=np.full(60, eta), rng=rng)
    m = np.array(r2["price_history"][1:])
    Pi = rho*((1-h)*(1-s)+h)/((1-h)*(1-s)+h*rho) if rho > 0 else 0.0
    dstar = m_settled - eta                      # simulated steady state before the shock
    k = (1-h)*s + h*(1-rho)
    gap0 = (1-rho)*shock                          # fast-clearing price jump at the shock
    err = m - eta - dstar
    sel = np.abs(err) > 0.012                     # fit decay above the one-tick floor
    t = np.arange(len(err))
    k_emp = float(np.exp(np.polyfit(t[sel][:8], np.log(np.abs(err[sel][:8])), 1)[0])) if sel.sum() >= 3 else np.nan
    T_bench = float(np.log(0.02/abs(gap0))/np.log(k)) if 0 < k < 1 else 0.0
    return dict(rho=rho, s=s, h=h, kappa=k, kappa_emp=k_emp, T_sim=float(first_passage(err)), T_bench=T_bench,
                dstar_sim=float(dstar), dstar_pred=Pi*D, path=err[:30].tolist())
rows = []
for rho, s in [(0.0, 0.1), (0.0, 0.3), (0.0, 0.6), (0.3, 0.2), (0.3, 0.9)]:
    for h in [0, 0.2, 0.4, 0.6]:
        rows.append(shock_run(rho, s, h)); r = rows[-1]
        print(f"rho={rho} s={s} h={h}: kappa={r['kappa']:.2f} emp={r['kappa_emp']:.2f}  T sim={r['T_sim']} bench={r['T_bench']:.1f}  d* sim={r['dstar_sim']:.3f} pred={r['dstar_pred']:.3f}", flush=True)
json.dump(rows, open("figdata_shock.json", "w"))
