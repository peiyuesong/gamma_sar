"""Is `cpp_standalone` available on this Windows box, and how much faster is it?

Why this exists
---------------
WP-B/A2/C cost 4-8 h of wall clock in Brian2's numpy runtime.  A GPU is not an
option: Brian2 2.7 has no GPU backend, and even a hypothetical one would lose to
the kernel-launch overhead on a 100-neuron, 0.025 ms-step problem.  The only real
lever is compiled code generation (`cpp_standalone`), which v0.1 abandoned
because of compilation errors on this host.  This script retests that in ~1 min.

Method
------
Run the same 2 s ING simulation (I_mu=0.85, gKv3=40, N=100, dt=0.025) twice:
once in the default numpy runtime, once with cpp_standalone.  Compare the
population rate (they must agree) and the wall clock.

Reading the output
------------------
  * rates agree within ~1% and cpp is >5x faster -> switch B/C/A2 to cpp.
  * cpp raises (no compiler / build error)    -> stay on numpy; the runtime is
    acceptable, and 4-8 h is a scheduling problem, not a blocker.
  * rates disagree                             -> do NOT switch; the compiled
    path changes numerics and the v0.2 numbers would no longer be comparable.

Usage:  python tools/bench_backend.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brian2
from brian2 import (start_scope, ms, seed, defaultclock, SpikeMonitor, Network)

from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.85
G_KV3 = 40.0
DT_MS = 0.025
DUR_MS = 2000.0
SEED = 42


def build_and_run(dur_ms):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(SEED)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = G_KV3, 20.0
    try:
        G, syn_ii, gap = make_ing(N, i_mu=I_MU, seed=SEED)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    sp = SpikeMonitor(G)
    net = Network(G, syn_ii, gap, sp)
    t0 = time.time()
    net.run(dur_ms * ms)
    wall = time.time() - t0
    n_spikes = int(sp.num_spikes)
    rate = n_spikes / (N * dur_ms / 1000.0)
    return rate, wall, n_spikes


def main():
    print(f"brian2 {brian2.__version__}, python {sys.version.split()[0]}")
    print(f"N={N}, dt={DT_MS} ms, duration={DUR_MS} ms\n")

    print("[1/2] numpy runtime ...")
    rate_np, wall_np, n_np = build_and_run(DUR_MS)
    print(f"      rate={rate_np:.2f} Hz, spikes={n_np}, wall={wall_np:.1f} s\n")

    print("[2/2] cpp_standalone ...")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "bench_cpp")
    try:
        brian2.set_device("cpp_standalone", directory=out_dir, clean=True,
                          build_on_run=True)
        rate_cpp, wall_cpp, n_cpp = build_and_run(DUR_MS)
        print(f"      rate={rate_cpp:.2f} Hz, spikes={n_cpp}, wall={wall_cpp:.1f} s")
        dev = (rate_cpp - rate_np) / rate_np * 100 if rate_np else float("nan")
        print(f"\n      rate difference: {dev:+.2f}%")
        print(f"      speedup: {wall_np / wall_cpp:.1f}x")
        if abs(dev) < 1.0 and wall_np / wall_cpp > 5:
            print("      VERDICT: switch to cpp_standalone for B/C/A2.")
        elif abs(dev) >= 1.0:
            print("      VERDICT: do NOT switch -- numerics differ.")
        else:
            print("      VERDICT: works, but not worth the switch.")
    except Exception:
        print("      FAILED (staying on numpy):")
        print("      " + traceback.format_exc().strip().replace("\n", "\n      "))
        print("\n      VERDICT: numpy runtime it is; cost is scheduling, not a blocker.")


if __name__ == "__main__":
    main()
