"""统计工具：Wilcoxon/Mann-Whitney + BH-FDR + 配对 bootstrap。

全文统一方案（详见路线图附录 B，v3.1-fix2）：
- 现象内比较：10 seeds；配对样本 Wilcoxon，非配对样本 Mann-Whitney U；BH-FDR
- 扫描曲线：逐点检验 vs 对照 + BH-FDR（族 = 曲线点数）
- 声光协同：配对 bootstrap n=1000，CI 下界 > 0
- 构效跨模态：Kruskal-Wallis + Dunn + BH-FDR
"""
import numpy as np
from scipy.stats import wilcoxon, kruskal, mannwhitneyu

EPS = 1e-12  # synergy_index 除零阈值


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。返回布尔数组，True = 拒绝。"""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1)) / n
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    k_max = np.max(np.where(passed)[0])
    cutoff = ranked[k_max]
    return p <= cutoff


def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney rank-biserial 效应量：r = 1 − 2U/(n_x·n_y)。

    fix2（B1 修复）：原实现误数"x 超过 mean(y) 的个数"，与 U 统计量无关，
    会把 r 系统性高估（实测 0.996 vs 真值 0.72）。现按定义计算。
    r > 0 表示 x 倾向于大于 y。
    """
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    U = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 1 - 2 * U / (nx * ny)


def compare_groups(groups: list, alpha: float = 0.05, paired: bool = False):
    """组间比较。

    - 两两比较：paired=True 用 Wilcoxon（要求等长）；否则 Mann-Whitney U（fix2 B2
      修复——原实现两分支都调 wilcoxon，非配对数据被裸 except 吞掉后 p 一律回填 1.0，
      产生假阴性）
    - ≥3 组：先 Kruskal-Wallis 总闸门，其 p 随结果返回（kruskal_p）；总闸门不显著
      时两两结果仍返回，但解读以总闸门为准
    - 多重比较：BH-FDR，族 = 全部两两比较
    """
    if len(groups) < 2:
        raise ValueError("至少两组")
    pvals, pairs = [], []
    names = [f"G{i}" for i in range(len(groups))]
    kruskal_p = None
    if len(groups) >= 3:
        kruskal_p = float(kruskal(*groups).pvalue)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = np.asarray(groups[i]), np.asarray(groups[j])
            if paired:
                if len(a) != len(b):
                    raise ValueError("配对比较要求等长样本")
                stat, p = wilcoxon(a, b)
            else:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            pvals.append(float(p))
            pairs.append((names[i], names[j]))
    reject = bh_fdr(np.array(pvals), q=alpha)
    return pairs, pvals, reject, kruskal_p
