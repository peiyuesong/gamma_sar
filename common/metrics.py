"""构效四维度统一指标实现。

时窗/量效/能效/动效：跨模态同一口径，PBM 时窗记 N/A。
v3.1-fix2：normalize_window_to_period_fraction 去掉未使用的 stim_freq_hz 参数
（相位本就相对各自刺激周期定义，频率在分数归一化中相消）。
"""
import numpy as np


# ---------- 1. 时窗 ----------
def time_window(phase_curve: np.ndarray, phase_deg: np.ndarray) -> float:
    """相位扫描曲线 >50% 最大提升的相位宽度（度）。若 max≤0 返回 NaN。"""
    max_v = phase_curve.max()
    if max_v <= 0:
        return np.nan
    above = phase_curve > 0.5 * max_v
    if not above.any():
        return 0.0
    return float(phase_deg[above].max() - phase_deg[above].min())


def normalize_window_to_period_fraction(window_deg: float) -> float:
    """跨模态归一化：占一个刺激周期的分数（相位相对各自刺激周期定义，频率相消）。"""
    return window_deg / 360.0


# ---------- 2. 量效 ----------
def fit_hill(intensity: np.ndarray, response: np.ndarray):
    """Hill 函数拟合：response = R_max * I^n / (EC50^n + I^n)。返回 (R_max, EC50, n)。"""
    from scipy.optimize import curve_fit
    def hill(I, Rmax, EC50, n):
        return Rmax * I**n / (EC50**n + I**n)
    try:
        popt, _ = curve_fit(hill, intensity, response, p0=[1.0, np.median(intensity), 1.0])
        return popt
    except RuntimeError:
        return (np.nan, np.nan, np.nan)


# ---------- 3. 能效 ----------
def energy_efficiency(response: float, physical_energy: float) -> float:
    """归一化效应 / 归一化物理能量。

    能效防御句（v3.1-fix 写入，报告与 Discussion 沿用）：跨模态比较依赖能量归一化
    假设；PBM 高能效源于其不触发动作电位的纯调制机制，可比性限于"单位物理能量的
    网络调制幅度"层面。
    """
    if physical_energy <= 0:
        return np.nan
    return response / physical_energy


# ---------- 4. 动效 ----------
def decay_tau(time_ms: np.ndarray, response: np.ndarray,
              baseline: float = 0.0) -> float:
    """撤除后衰减指数拟合 τ_off：response(t) ≈ A·exp(−t/τ_off) + b。"""
    from scipy.optimize import curve_fit
    def exp_decay(t, A, tau, b):
        return A * np.exp(-t / tau) + b
    try:
        popt, _ = curve_fit(exp_decay, time_ms, response,
                            p0=[response[0] - baseline, 100.0, baseline])
        return float(popt[1])
    except RuntimeError:
        return np.nan


# ---------- 5. 协同指数（声光联合） ----------
def synergy_index(R_combined: np.ndarray,
                  R_light: np.ndarray, R_sound: np.ndarray) -> np.ndarray:
    """主指标：(R_combined − R_light − R_sound) / max(R_light, R_sound)；分母<ε → NaN。"""
    from .stats import EPS
    denom = np.maximum(R_light, R_sound)
    return np.where(denom < EPS, np.nan, (R_combined - R_light - R_sound) / denom)


def synergy_index_max(R_combined: np.ndarray,
                      R_light: np.ndarray, R_sound: np.ndarray) -> np.ndarray:
    """辅助指标：(R_combined − max(R_light, R_sound)) / max(R_light, R_sound)。"""
    from .stats import EPS
    denom = np.maximum(R_light, R_sound)
    return np.where(denom < EPS, np.nan, (R_combined - denom) / denom)


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
                 paired_with: np.ndarray = None, seed: int = 42):
    """配对 bootstrap 95% CI。协同判据：CI 下界 > 0。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if paired_with is not None:
            boots[i] = np.mean(x[idx] - paired_with[idx])
        else:
            boots[i] = np.mean(x[idx])
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi
