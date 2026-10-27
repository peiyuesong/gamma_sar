import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brian2 import (start_scope, run, defaultclock, seed, ms, mV, uA, cm2,
                    StateMonitor, SpikeMonitor)
from common.cells import make_pv_neuron

start_scope()
defaultclock.dt = 0.025 * ms
seed(0)
G = make_pv_neuron(1)
G.phi = 5.0
G.I_inj = 6.0 * (uA / cm2)
mon = StateMonitor(G, "v", record=True)
sp = SpikeMonitor(G)
run(2000 * ms, report=None)

v = np.array(mon.v[0] / mV)
print(f"rate = {len(sp.t) / 2.0:.1f} Hz, n_spikes = {len(sp.t)}")
print(f"v range: min={v.min():.1f}, max={v.max():.1f}, end={v[-1]:.1f}")
print(f"dt = 0.025 ms, total samples = {len(v)}")

# 打印稳态期 1000–1050 ms 的 50ms 迹线
t0_idx = int(1000.0 / 0.025)
seg = v[t0_idx:t0_idx + int(50.0 / 0.025)]
print("--- v trace 1000-1050 ms (every 0.5ms) ---")
for i in range(0, len(seg), 20):
    print(f"t={1000 + i*0.025:.1f}ms  v={seg[i]:.1f}mV")