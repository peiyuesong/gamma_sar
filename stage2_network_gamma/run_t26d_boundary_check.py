"""T2.6d: 极简边界检查 — 找 γ Basin 边缘."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import *
from common.networks import make_ing
from common.params import PARAMS

N, SEED, DT = 100, 42, 0.025*ms
T_RUN = 3000*ms

for i_mu in [0.65, 0.70, 0.75, 0.80]:
    for gkv3 in [0, 20, 40]:
        start_scope(); defaultclock.dt = DT; seed(SEED)
        cell = PARAMS["cell"]
        saved = (cell.g_Kv3, cell.g_Kdr)
        cell.g_Kv3, cell.g_Kdr = gkv3, 20.0
        try:
            G, syn_ii, gap = make_ing(N, i_mu=i_mu, seed=SEED)
        finally:
            cell.g_Kv3, cell.g_Kdr = saved
        
        sp = SpikeMonitor(G)
        Network(G, syn_ii, gap, sp).run(T_RUN)
        
        t = np.array(sp.t/ms)
        t = t[t >= 1000] - 1000
        rate = len(t) / (N * 2.0)
        
        if len(t) > 10:
            bins = np.arange(0, 2001, 1.0)
            hist, _ = np.histogram(t, bins=bins)
            spec = np.abs(np.fft.rfft(hist-hist.mean(), 4096))**2
            freqs = np.fft.rfftfreq(4096, d=1e-3)
            band = (freqs>=20)&(freqs<=80)
            f0 = freqs[band][np.argmax(spec[band])] if spec[band].sum()>0 else 0
        else:
            f0 = 0
        
        gamma = "YES" if rate>=8 and f0>=25 else "NO"
        print(f"i={i_mu:.2f} gKv3={gkv3:2d}: rate={rate:5.1f} f0={f0:5.1f} [{gamma}]")
    print()