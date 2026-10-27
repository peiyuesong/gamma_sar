# gamma_sar

**A biophysical PV+ interneuron-network model of GABA_B-mediated gamma suppression
and 40 Hz tACS re-entrainment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Status: **v0.2, data frozen** at tag `data-freeze` (`1e1dd0f`). Preprint in preparation.
Every number below is reproduced by a script in this repository from the committed CSVs.

## Overview

A Brian2 model of `N = 100` Traub–Miles parvalbumin-positive (PV+) interneurons coupled
by recurrent GABA_A inhibition — an interneuron-network-gamma (ING) circuit — extended
with three mechanistic layers:

1. **Slow metabotropic inhibition** (Destexhe GABA_B kinetics, four cooperative binding
   sites), used as a surrogate for the synaptic footprint of iTBS.
2. **Exogenous 40 Hz tACS**, modelled as a sinusoidal current injection shared by all
   neurons, with the carrier amplitude and the residual inhibition as the two control
   parameters.
3. **Phase-gated inhibitory weight modulation** at the GABA_A synapses. This is *not*
   spike-timing-dependent plasticity: the update is a uniform, clock-driven function of
   the tACS phase, identical across all synapses (no synaptic specificity).

The model also reports a **prespecified hypothesis that it rejects**: the eyes-open /
eyes-closed (EO/EC) reversal predicted from a single Kv3-conductance state variable does
not reproduce the experimentally observed state dissociation.

## Headline results (frozen data)

| Result | Value | Source |
|---|---|---|
| GABA_B silencing of the ING rhythm | 16.79 ± 1.50 → 0.25 ± 0.21 Hz, **98.5%**, n = 10 | `stage2_network_gamma/b_stats.csv` |
| Spontaneous recovery after train offset | t50 = t90 ≈ **+2 s** | `stage2_network_gamma/a1_phase2_recovery.csv` |
| 40 Hz 1:1 locking threshold | **≥ 2.0 µA/cm²** (tACS applied +2 s after iTBS offset); ≥ 2.5 µA/cm² when applied concurrently | `a2_mode_map.csv`, `a1_amp_threshold.csv` |
| Re-entrainment, EO / EC | 39.84 ± 0.12 Hz (2.38×) / 40.00 ± 0.01 Hz (2.10×), own-state sham | `b_stats.csv` |
| Controls | tACS-only 39.96 / 40.00 Hz; iTBS-alone 15.91 / 18.68 Hz (already recovered) | `b_stats.csv` |
| Phase-gated plasticity, matched 5 s windows | depotentiation arm **1.28× [1.19–1.34]**, potentiation arm **0.65× [0.50–0.71]**, no-plasticity control **1.00×** | `c_reanalysis.csv` |
| Locked-mode map | EO: intrinsic 2:1 → 1:2 (1 µA/cm²) → 1:1 (≥ 2 µA/cm²) at every GABA_B level; EC: only partial entrainment at 1 µA/cm² | `a2_mode_map.csv` |

Two controls delimit the interpretation, and both are reported as such:

- the tACS-only arm locks just as well, so the ON state is **entrainment**, not rescue;
- the iTBS-alone arm has **already recovered spontaneously** by the tACS window, so the
  specific contribution of tACS is the *mode* of the reopened rhythm (1:1 at the carrier),
  not the recovery itself.

## Repository layout

```
common/                  # model core
  cells.py               #   Traub-Miles PV neuron (dt = 0.025 ms, hard-coded)
  networks.py            #   make_ing(): recurrent GABA_A + gap junctions
  params.py              #   all parameters (PARAMS)
  plasticity.py          #   phase-gated weight update
  stimuli.py             #   five-modality interface (placeholder eta values)
  metrics.py             #   time-window / dose-response / energy / decay / synergy
stage1_single_cell/      # f-I surface (t13_fI_surface.csv)
stage2_network_gamma/    # network experiments + frozen result CSVs
  run_*.py               #   one script per experiment (resumable, CSV-backed)
  *.csv                  #   frozen results: b_stats, a2_mode_map, c_stdp, ...
figures/                 # fig{1..5}_*_v2.png (built from the CSVs)
tools/                   # rendering + diagnostics
  make_fig{1,3,4,5}.py   #   figures from frozen CSVs only
  build_preprint.py      #   preprint.md -> HTML (pandoc) -> PDF (Edge headless)
  preprint.css           #   print stylesheet for the PDF
  reanalyze_spikes.py    #   recompute metrics from saved spike trains
  efield_conversion.py   #   J = sigma * E conversion for the tACS threshold
docs/                    # changelog, program roadmap, parameter tables
tests/                   # pytest smoke tests
```

## Reproduce

Tested with Python 3.11.17 and Brian2 2.7.1 (numpy runtime; `cpp_standalone` is not
available on the Windows host used here).

```bash
# smoke tests
python -m pytest tests/

# the two headline experiments (each writes its own CSV, both resumable)
python stage2_network_gamma/run_b_stats.py EO
python stage2_network_gamma/run_c_stdp.py --seeds 0-9 --save-spikes

# rebuild every figure from the committed CSVs (no simulation)
python tools/make_fig1.py
python tools/make_fig4.py
python tools/make_fig5.py

# rebuild the preprint PDF from preprint.md (pandoc + Edge headless, no LaTeX needed)
python tools/build_preprint.py
```

`common/cells.py` hard-codes the integration step at 0.025 ms; `defaultclock.dt` does not
override it. Figures are rebuilt from the CSVs, never re-simulated.

## Data and code availability

- Frozen result tables (`stage2_network_gamma/*.csv`) and the scripts that produced them
  are committed.
- Raw spike trains (`*_spikes/*.npz`) are excluded from git to keep the repository small;
  they are regenerated by the `--save-spikes` flag and archived with the release.
- Analysis-side conventions that matter for reading the tables are documented in
  `docs/v0.2_changelog.md` (matched-window power ratios, carrier-aware mode criteria,
  Wilson intervals).

## Citation

```bibtex
@software{gamma_sar_2026,
  title  = {gamma_sar: a biophysical PV+ interneuron-network model of
            GABA_B-mediated gamma suppression and 40 Hz tACS re-entrainment},
  author = {Song, Peiyue},
  year   = {2026},
  license = {MIT},
  url    = {https://github.com/peiyuesong/gamma_sar}
}
```

Please also cite the preprint once its DOI is assigned.

## Contact

Peiyue Song — Chengdu University of Traditional Chinese Medicine
peiyuesong@stu.cdutcm.edu.cn · ORCID [0009-0008-9131-7121](https://orcid.org/0009-0008-9131-7121)
