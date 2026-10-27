"""相位门控 STDP：G(φ) = 1 + a·cos(φ − φ_peak)。

a ∈ [0.2, 0.8]；φ_peak=90°（H1 假设）；τ_trace ∈ {30, 60, 120} min。
"""
import numpy as np

from .params import PARAMS


def G_phi(phase_deg: np.ndarray, a: float = None, phi_peak: float = None) -> np.ndarray:
    """返回 G(φ) 值（无量纲，>=0）。"""
    a = a if a is not None else np.mean(PARAMS["stdp"].a_amp_range)
    phi_peak = phi_peak if phi_peak is not None else PARAMS["stdp"].phi_peak
    return 1.0 + a * np.cos(np.deg2rad(phase_deg - phi_peak))


def stdp_window(delta_t_ms: np.ndarray, tau_plus: float = 20.0,
                tau_minus: float = 20.0, A_plus: float = 0.005,
                A_minus: float = 0.00525) -> np.ndarray:
    """标准 STDP 窗（pair-based）。

    ΔW = A_plus · exp(−Δt/τ_plus)   if Δt > 0
        −A_minus · exp(Δt/τ_minus)  if Δt < 0
    """
    dt = np.asarray(delta_t_ms)
    out = np.zeros_like(dt)
    pos = dt > 0
    neg = dt < 0
    out[pos] = A_plus * np.exp(-dt[pos] / tau_plus)
    out[neg] = -A_minus * np.exp(dt[neg] / tau_minus)
    return out


def apply_phase_gating(delta_w: np.ndarray, post_phase_deg: np.ndarray,
                       a: float = None, phi_peak: float = None) -> np.ndarray:
    """将 G(φ) 调制施加到 STDP 输出。"""
    return delta_w * G_phi(post_phase_deg, a, phi_peak)