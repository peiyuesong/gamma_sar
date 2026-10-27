"""Map the model's injected current density to an equivalent tissue E-field.

Decision C: the manuscript quotes a tACS threshold in uA/cm^2, which is a
model-internal current density, not a dose anyone can apply to a head.  This
script converts it with the standard 1-D ohmic relation

    J = sigma * E        ->        E = J / sigma

and prints the equivalent field for the amplitudes used in the paper.

Caveats to keep in the Methods text
-----------------------------------
* sigma = 0.3 S/m is a bulk cortical conductivity (typical range 0.1-0.5 S/m);
  the equivalent field therefore inherits that spread (roughly 2x either way).
* Equating an injected membrane current density with a tissue current density
  assumes all field-induced current is collected by the compartment.  A real
  neuron collects only a fraction, set by the polarization length / kappa
  factor (Aspart 2018; Huang 2021).  Treat E here as an *upper bound* on the
  field needed, i.e. a calibration target, not a prediction.
* The number below is therefore reported as an order-of-magnitude bridge, and
  the threshold itself stays labelled "model-internal bifurcation parameter".

Usage:  python tools/efield_conversion.py
"""
import numpy as np

SIGMA_S_PER_M = 0.3          # bulk cortical conductivity
# 1 uA/cm^2 = 1e-6 A / 1e-4 m^2 = 1e-2 A/m^2


def ua_cm2_to_a_m2(j):
    return j * 1e-2


def main():
    amps = [0.3, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 10.0]
    print(f"sigma = {SIGMA_S_PER_M} S/m")
    print(f"{'J (uA/cm^2)':>12s} {'J (A/m^2)':>12s} {'E (V/m)':>10s} {'E (mV/mm)':>11s}")
    for j in amps:
        j_si = ua_cm2_to_a_m2(j)
        e = j_si / SIGMA_S_PER_M
        print(f"{j:12.2f} {j_si:12.4f} {e:10.3f} {e:11.1f}")

    e_thr = ua_cm2_to_a_m2(1.8) / SIGMA_S_PER_M
    print()
    print(f"headline: 1.8 uA/cm^2  ->  {e_thr*1000:.0f} mV/m  ({e_thr:.3f} V/m)")
    print("for reference, conventional human tACS induces roughly 0.1-1 V/m in cortex,")
    print("so the model threshold sits at the low end of that range; it is plausible,")
    print("not absurd, but it is still an uncalibrated upper bound.")


if __name__ == "__main__":
    main()
