"""WP-C: phase-gated inhibitory plasticity, LTD vs LTP window, 2 arms x seeds.

Purpose
-------
Decide whether Fig 5 can carry a post-stimulus gamma claim, or must be demoted
to "weight dynamics".  Two questions, both answered only by seeds:
  1. Is there a genuine 30-50 Hz spectral *peak* after stimulation, or is the
     band power just noise/harmonic leakage?
  2. Does the effect reproduce across seeds?

Arms
----
  ltd : G(phi) = max(0,  cos(2*pi*(phase-0.25))),  A = -0.0003   (original)
  ltp : G(phi) = max(0, -cos(2*pi*(phase-0.25))),  A = -0.0003   (anti-phase;
        the inverted window pushes w *up*, i.e. LTP)

Direction note (read before interpreting): the synapse is inhibitory (GABA_A),
so weaker inhibition -> disinhibition -> more gamma, and stronger inhibition ->
less.  The LTD arm is the one that can *raise* post-stimulus gamma; the LTP arm
is expected to lower it.  A negative LTP result is a mechanism result, not a
failed run.

Implementation note: this reuses the T2.7 plasticity implementation verbatim
(w_syn multiplier, 1 ms network_operation, same update formula); only the sign
window, the seed count, and the post-stimulus PSD analysis are new.  The rule is
NOT pre/post spike-timing dependent -- it is a uniform, clock-driven update from
the tACS phase.  That is recorded in the meta JSON and must be reflected in the
Methods wording.

Windows
-------
  baseline   2000-7000 ms (before the tACS train)
  early post 0-20 s after offset   (guards against an effect that peaks early)
  late post  25-45 s after offset  (the pre-registered P2 window)

Usage
-----
  python stage2_network_gamma/run_c_stdp.py --time          # 1 run, wall clock
  python stage2_network_gamma/run_c_stdp.py --seeds 0-9     # full 2x10
  python stage2_network_gamma/run_c_stdp.py --seeds 0-4     # half, resumable
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brian2 import (start_scope, ms, mV, mS, cm, uA, SpikeMonitor, Network,
                    Synapses, network_operation, defaultclock, seed)
import brian2

from common.networks import make_ing
from common.params import PARAMS

N = 100
I_MU = 0.85
G_KV3 = 40.0
# NOTE: make_pv_neuron() hardcodes dt=0.025 ms on the NeuronGroup (common/cells.py
# line 88), so defaultclock.dt does NOT control the integration step for neurons.
# Setting 0.05 here only mismatches the synaptic clock and logs wrong numbers in
# the metadata.  Keep 0.025 so the recorded dt equals the effective dt.
DT_MS = 0.025

T_WARMUP = 2000.0
T_BASE = 5000.0
T_STIM = 30000.0
T_POST = 60000.0
T_TOTAL = T_WARMUP + T_BASE + T_STIM + T_POST

F_TACS = 40.0
AMP_TACS = 2.0
W_INIT = 1.0
W_MIN = 0.3
W_MAX = 2.0
A_MAG = 0.0003
TAU_DECAY = 120000.0

# Every window must be the SAME length or the power ratio is meaningless:
# |rfft|^2 of a periodic component scales with N^2, so a 20 s post window against
# a 5 s baseline gives a factor (20/5)^2 = 16 with no change in dynamics at all.
# That is exactly the x15.96 seen in the ctrl arm.  T_BASE is 5 s, so 5 s is the
# longest baseline this protocol can support; all windows are pinned to it.
MATCH_MS = 5000.0
POST_WIN = (25000.0, 30000.0)     # late post window, ms after stim offset

# When True, the raw spike train of every run is written to c_spikes/.  This is
# deliberately opt-in rather than always-on, but it is strongly recommended for
# the full run: every metric below is a summary, and the v0.1-vs-v0.2 discrepancy
# (x1.21 vs x3.89) can only be re-analysed -- at other window lengths, other band
# edges, other statistics -- if the spikes survive the run.  20 runs cost ~15 MB.
SAVE_SPIKES = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "c_stdp.csv")

# Single source of truth: run_one() must return exactly RUN_KEYS (asserted at the
# end of run_one), constants are added by the writer, and FIELDS is derived from
# both.  This is what prevents the v1 KeyError('dt_ms') class of bug: a column can
# no longer be added to the header without a matching key in the result dict.
RUN_KEYS = ["arm", "seed", "w_nadir", "w_final",
            "base_rate", "base_power", "base_peak", "base_prom",
            "early_rate", "early_power", "early_prom",
            "post_rate", "post_power", "post_peak", "post_prom"]
CONST_KEYS = ["dt_ms", "w_min", "w_max", "a_mag", "tau_decay_ms"]
FIELDS = RUN_KEYS + ["wall_s"] + CONST_KEYS


def psd_stats(t_spikes, t0, t1):
    seg = t_spikes[(t_spikes >= t0) & (t_spikes < t1)]
    dur_s = (t1 - t0) / 1000.0
    rate = len(seg) / (N * dur_s)
    out = {"rate": rate, "power": 0.0, "f_peak": float("nan"), "prom": 0.0}
    if len(seg) < 100:
        return out
    bins = np.arange(0.0, (t1 - t0) + 1.0, 1.0)
    hist, _ = np.histogram(seg - t0, bins=bins)
    if len(hist) < 16:
        return out
    r = hist - hist.mean()
    # NB: np.fft.rfft(r, n) TRUNCATES when n < len(r).  A 20 s window at 1 ms bins
    # is 20000 samples, so passing 16384 silently dropped 3.6 s.  Use the true
    # length; resolution is then 1/20 s = 0.05 Hz, plenty for a 1 Hz question.
    spec = np.abs(np.fft.rfft(r)) ** 2
    freqs = np.fft.rfftfreq(len(r), d=1e-3)
    band = (freqs >= 30.0) & (freqs <= 50.0)
    search = (freqs >= 20.0) & (freqs <= 80.0)
    out["power"] = float(spec[band].sum())
    idx = np.argmax(spec[search])
    out["f_peak"] = float(freqs[search][idx])
    med = float(np.median(spec[search]))
    out["prom"] = float(spec[search][idx] / med) if med > 0 else 0.0
    return out


def run_one(arm, seed_val):
    start_scope()
    defaultclock.dt = DT_MS * ms
    seed(seed_val)

    cell = PARAMS["cell"]
    saved = (cell.g_Kv3, cell.g_Kdr)
    cell.g_Kv3, cell.g_Kdr = G_KV3, 20.0
    try:
        G, _, gap = make_ing(N, i_mu=I_MU, seed=seed_val)
    finally:
        cell.g_Kv3, cell.g_Kdr = saved

    # --- T2.7 synapse variant (weighted GABA_A), unchanged ---
    syn_ii = Synapses(G, G, model="""
        dg/dt = -g / (10*ms) : 1 (clock-driven)
        I_syn_post = w_syn * g_syn_each * g * (-75*mV - v_post) : amp/meter**2 (summed)
        w_syn : 1
        g_syn_each : siemens/meter**2 (constant)
    """, on_pre="""g += 1
        g = clip(g, 0, 1)""")
    syn_ii.connect(condition="i != j", p=0.2)
    syn_ii.g_syn_each = (0.1 * mS / (cm ** 2)) / (N * 0.2)
    syn_ii.w_syn = W_INIT

    tacs_syn = Synapses(G, G, model="""
        I_GABA_B_post = I_stim : amp/meter**2 (summed)
        I_stim : amp/meter**2
    """)
    tacs_syn.connect(condition="i == j")
    tacs_syn.I_stim = 0.0 * uA / (cm ** 2)

    weight_hist = []
    sign = -1.0 if arm == "ltd" else 1.0

    @network_operation(dt=1.0 * ms)
    def dynamic_update(t):
        t_ms = float(t / ms)
        stim_on = (T_WARMUP + T_BASE) <= t_ms < (T_WARMUP + T_BASE + T_STIM)

        if stim_on:
            tacs_syn.I_stim = (AMP_TACS * np.sin(2 * np.pi * F_TACS * t_ms / 1000.0)
                               * uA / (cm ** 2))
        else:
            tacs_syn.I_stim = 0.0 * uA / (cm ** 2)

        w_now = syn_ii.w_syn[:]
        if stim_on:
            phase = (t_ms / 25.0) % 1.0
            if arm == "ltd":
                g_phi = max(0.0, np.cos(2 * np.pi * (phase - 0.25)))
            else:
                g_phi = max(0.0, -np.cos(2 * np.pi * (phase - 0.25)))
            if g_phi > 0:
                dw = sign * A_MAG * g_phi * (w_now - W_MIN) / (W_INIT - W_MIN)
                syn_ii.w_syn[:] = np.clip(w_now + dw, W_MIN, W_MAX)
        elif t_ms >= (T_WARMUP + T_BASE + T_STIM):
            w_new = W_INIT + (w_now - W_INIT) * np.exp(-1.0 / TAU_DECAY)
            syn_ii.w_syn[:] = w_new

        if int(t_ms) % 2000 == 0:
            weight_hist.append((t_ms, float(np.mean(syn_ii.w_syn[:]))))

    sp = SpikeMonitor(G)
    net = Network(G, syn_ii, gap, tacs_syn, sp, dynamic_update)
    net.run(T_TOTAL * ms)

    all_t = np.array(sp.t / ms)
    if SAVE_SPIKES:
        sp_dir = os.path.join(OUT_DIR, "c_spikes")
        os.makedirs(sp_dir, exist_ok=True)
        np.savez_compressed(os.path.join(sp_dir, f"{arm}_seed{seed_val}.npz"),
                            t_ms=all_t.astype(np.float32),
                            arm=np.array(arm), seed=np.array(seed_val),
                            dt_ms=np.array(DT_MS), n=np.array(N),
                            t_warmup=np.array(T_WARMUP), t_base=np.array(T_BASE),
                            t_stim=np.array(T_STIM), t_post=np.array(T_POST))
    stim_off = T_WARMUP + T_BASE + T_STIM
    stim_on = T_WARMUP + T_BASE
    base = psd_stats(all_t, stim_on - MATCH_MS, stim_on)
    early = psd_stats(all_t, stim_off, stim_off + MATCH_MS)
    late = psd_stats(all_t, stim_off + POST_WIN[0],
                     stim_off + POST_WIN[0] + MATCH_MS)

    w_hist = np.array([w for _, w in weight_hist]) if weight_hist else np.array([np.nan])
    out = dict(
        arm=arm, seed=seed_val,
        w_nadir=float(np.nanmin(w_hist)), w_final=float(w_hist[-1]),
        base_rate=base["rate"], base_power=base["power"], base_peak=base["f_peak"],
        base_prom=base["prom"],
        early_rate=early["rate"], early_power=early["power"], early_prom=early["prom"],
        post_rate=late["rate"], post_power=late["power"], post_peak=late["f_peak"],
        post_prom=late["prom"],
    )
    missing = [k for k in RUN_KEYS if k not in out]
    if missing:
        raise RuntimeError(f"run_one() is missing CSV columns: {missing}")
    return out


def parse_seeds(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def done_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        if not fh.readline():
            return keys
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) >= 2:
                try:
                    keys.add((p[0], int(p[1])))
                except ValueError:
                    continue
    return keys


def main():
    # declared before anything reads A_MAG (the argparse default reads it)
    global SAVE_SPIKES, A_MAG

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,1,2")
    ap.add_argument("--arms", default="ltd,ltp")
    ap.add_argument("--time", action="store_true", help="one run, report wall clock")
    ap.add_argument("--save-spikes", action="store_true",
                    help="write raw spike trains to c_spikes/ so metrics can be "
                         "recomputed without re-simulating")
    ap.add_argument("--amag", type=float, default=A_MAG,
                    help="learning rate; pass 0 with --arms ctrl for the "
                         "no-plasticity control (tACS only, weight fixed at 1.0)")
    args = ap.parse_args()

    SAVE_SPIKES = args.save_spikes
    A_MAG = args.amag

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = parse_seeds(args.seeds)
    if args.time:
        arms, seeds = ["ltd"], [42]

    done = done_keys(CSV_PATH)
    new_file = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", encoding="utf-8", newline="")
    if new_file:
        fh.write(",".join(FIELDS) + "\n")
        fh.flush()

    print("=" * 100)
    print(f"WP-C phase-gated plasticity | I_mu={I_MU}, gKv3={G_KV3}, dt={DT_MS} ms, "
          f"stim={T_STIM/1000:.0f}s, post={T_POST/1000:.0f}s")
    print(f"arms={arms}, seeds={seeds}, a_mag={A_MAG}, "
          f"brian2={brian2.__version__}, python={sys.version.split()[0]}")
    if done:
        print(f"resuming: {len(done)} runs already on disk")
    print("=" * 100)

    for arm in arms:
        for sd in seeds:
            if (arm, sd) in done:
                continue
            t0 = time.time()
            r = run_one(arm, sd)
            wall = time.time() - t0
            r["wall_s"] = round(wall, 1)
            # constant protocol fields are not returned by run_one()
            r.update(dt_ms=DT_MS, w_min=W_MIN, w_max=W_MAX,
                     a_mag=A_MAG, tau_decay_ms=TAU_DECAY)
            fh.write(",".join(str(r[k]) for k in FIELDS) + "\n")
            fh.flush()
            ratio_e = r["early_power"] / r["base_power"] if r["base_power"] > 0 else float("nan")
            ratio_l = r["post_power"] / r["base_power"] if r["base_power"] > 0 else float("nan")
            print(f"{arm} | seed={sd:2d} | w {r['w_nadir']:.3f}->{r['w_final']:.3f} | "
                  f"rate {r['base_rate']:5.1f}->{r['early_rate']:5.1f}/{r['post_rate']:5.1f} Hz | "
                  f"post peak {r['post_peak']:5.1f} Hz prom {r['post_prom']:6.2f} | "
                  f"P30_50 early x{ratio_e:.2f} late x{ratio_l:.2f} | {wall:.0f}s")
            if args.time:
                print(f"\nTIMING: one run = {wall:.0f} s. "
                      f"Full 2 arms x 10 seeds ~ {wall * 20 / 60:.0f} min.")

    fh.close()

    # ---- summary ----
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) == len(FIELDS):
                rows.append(dict(zip(FIELDS, p)))
    if not rows:
        return
    print("\n" + "=" * 100)
    print(f"{'arm':4s} {'n':>3s} {'early/P30_50':>18s} {'late/P30_50':>16s} "
          f"{'post peak':>10s} {'prom':>7s} {'rate b->p':>11s}")
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm and float(r["base_power"]) > 0]
        if not sel:
            continue
        e = np.array([float(r["early_power"]) / float(r["base_power"]) for r in sel])
        l = np.array([float(r["post_power"]) / float(r["base_power"]) for r in sel])
        pk = np.array([float(r["post_peak"]) for r in sel])
        pr = np.array([float(r["post_prom"]) for r in sel])
        rb = np.array([float(r["base_rate"]) for r in sel])
        rp = np.array([float(r["post_rate"]) for r in sel])
        print(f"{arm:4s} {len(sel):3d} {e.mean():8.2f}+/-{e.std():5.2f} "
              f"{l.mean():8.2f}+/-{l.std():5.2f} {np.nanmean(pk):10.1f} "
              f"{pr.mean():7.2f} {rb.mean():5.1f}->{rp.mean():5.1f}")

    meta_path = os.path.join(OUT_DIR, "c_stdp_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(dict(
            script="run_c_stdp.py", i_mu=I_MU, g_kv3=G_KV3, n_pv=N, dt_ms=DT_MS,
            f_tacs=F_TACS, amp_tacs=AMP_TACS, w_init=W_INIT, w_min=W_MIN,
            w_max=W_MAX, a_mag=A_MAG, tau_decay_ms=TAU_DECAY,
            stim_s=T_STIM / 1000.0, post_s=T_POST / 1000.0,
            save_spikes=SAVE_SPIKES,
            post_window_s=[POST_WIN[0] / 1000.0, POST_WIN[1] / 1000.0],
            arms={"ltd": "max(0, +cos)", "ltp": "max(0, -cos)"},
            rule=("uniform update every 1 ms from the tACS phase, "
                  "dw = sign*A*max(0,+/-cos(2*pi*(phase-0.25)))*(w-w_min)/(w_init-w_min), "
                  "clipped to [w_min,w_max]; NOT pre/post spike-timing dependent"),
            python=sys.version.split()[0], brian2=brian2.__version__,
        ), fh, ensure_ascii=False, indent=2)
    print(f"\nsaved CSV: {CSV_PATH}")
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()
