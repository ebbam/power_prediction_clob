import sys, json; sys.path.insert(0, sys.argv[1]); SEC = sys.argv[2]
import numpy as np
from bettor import bettor, run_market_lob
LOG = []
orig = bettor.update_belief
def logged(self, tv, mp, ph, k=1):
    LOG.append((self.market_valuation, self.budget, self.n_contracts)); return orig(self, tv, mp, ph, k)
bettor.update_belief = logged
def pop(N, g, s=0.3, noise=0.0, h=0.0, V0=0.5):
    return [bettor(budget=g.uniform(100, 1000), market_valuation=V0, risk_av=1.0, stubbornness=s,
                   expertise=1-noise, market_imitation=h) for _ in range(N)]
def add_whale(p, rho, W=0.7):
    B = sum(b.budget for b in p); p.append(bettor(budget=B*rho/(1-rho), market_valuation=W, risk_av=1.0, whale=True)); return p
def run(p, T, seed, eta=0.5):
    LOG.clear(); np.random.seed(seed); rng = np.random.default_rng(seed)
    r = run_market_lob(len(p), T, 0.5, 0.0, p, gen_el=np.full(T, eta), rng=rng)
    m = np.array(r["price_history"][1:]); L = np.array(LOG).reshape(T, len(p), 3)
    w = L[:, :, 1] + m[:, None]*L[:, :, 2]; peq = (w*L[:, :, 0]).sum(1)/w.sum(1)
    return m, peq, L
def first_passage(err, tol=0.02, window=3):
    ok = np.abs(err) <= tol
    for t in range(len(ok)-window+1):
        if ok[t:t+window].all(): return t
    return np.nan
N, D = 100, 0.2
out = {}
if SEC == "AB":
    out["A"] = [(rho, noise, float(np.mean([run(add_whale(pop(N, np.random.default_rng(sd), noise=noise), rho), 120, sd)[0][40:].mean()-0.5 for sd in range(2)])))
                for rho in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] for noise in [0.0, 0.1]]
    out["B"] = [(rho, h, float(run(add_whale(pop(N, np.random.default_rng(0), h=h), rho), 120, 0)[0][-40:].mean()-0.5)/D)
                for rho in [0.2, 0.4] for h in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
if SEC == "CD":
    out["C"] = {str(h): (run(add_whale(pop(N, np.random.default_rng(0), h=h), 0.3), 50, 0)[0]-0.5).tolist() for h in [0, 0.5, 0.9]}
    out["D"] = {str(h): (run(pop(N, np.random.default_rng(1), h=h, V0=0.7), 40, 1)[0]-0.5).tolist() for h in [0, 0.5, 0.9]}
if SEC == "EI":
    # first passage to |delta - delta*| <= 0.02 (3 periods) vs fast-clearing benchmark
    rows = []
    for h in [0, 0.25, 0.5, 0.75]:
        for s in [0.1, 0.3, 0.6]:                       # belief shock, no whale
            m = run(pop(N, np.random.default_rng(3), s=s, h=h, V0=0.8), 60, 3)[0]
            k = (1-h)*s + h; d0 = m[0]-0.5
            rows.append(("no whale", h, s, float(first_passage(m-0.5)), float(np.log(0.02/abs(d0))/np.log(k)) if k > 0 else 0.0))
        for s in [0.2, 0.9]:                            # belief shock with a whale, rho = 0.5
            rho = 0.5; Pi = rho*((1-h)*(1-s)+h)/((1-h)*(1-s)+h*rho); dstar = Pi*D
            m = run(add_whale(pop(N, np.random.default_rng(4), s=s, h=h, V0=0.2), rho), 80, 4)[0]
            k = (1-h)*s + h*(1-rho); d0 = m[0]-0.5-dstar
            rows.append((f"whale s={s}", h, s, float(first_passage(m-0.5-dstar)), float(np.log(0.02/abs(d0))/np.log(k))))
    out["E"] = rows
if SEC == "FGH":
    F = []
    for noise in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        m, peq, L = run(pop(N, np.random.default_rng(0), noise=noise), 220, 0)
        mu = (m-peq)[50:]
        F.append((noise, float(np.sqrt(L[50:, :, 0].var(1).mean())), float(np.std(mu)), float(np.corrcoef(mu[:-1], mu[1:])[0, 1]), len(mu)))
    out["F"] = F
    G = []
    for n in [25, 50, 100, 200, 400]:
        m, peq, L = run(pop(n, np.random.default_rng(0), noise=0.1), 180, 0)
        w = L[-1, :, 1] + m[-1]*L[-1, :, 2]; om = w/w.sum()
        G.append((n, float(np.var(peq[40:]-0.5)), float(0.7*np.sum(om**2)*0.01/1.3)))
    out["G"] = G
    Hh = []
    for h in [0.25, 0.5, 0.75]:
        m, peq, L = run(pop(N, np.random.default_rng(0), noise=0.1, h=h), 220, 0)
        mu = (m-peq)[50:]; Hh.append((h, float(np.sqrt(L[50:, :, 0].var(1).mean())), float(np.std(mu)), float(np.corrcoef(mu[:-1], mu[1:])[0, 1]), len(mu)))
    out["H"] = Hh
json.dump(out, open(f"figdata_{SEC}.json", "w")); print(SEC, "done")
