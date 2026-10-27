"""Stats module tests: BH-FDR + bootstrap + synergy + rank_biserial + compare_groups."""
import numpy as np
import pytest
from scipy.stats import mannwhitneyu


def test_bh_fdr_basic():
    from common.stats import bh_fdr
    pvals = np.array([0.001, 0.01, 0.04, 0.5, 0.7])
    reject = bh_fdr(pvals, q=0.05)
    assert reject[0]
    assert reject[1]
    assert not reject[3]
    assert not reject[4]


def test_bh_fdr_empty():
    from common.stats import bh_fdr
    reject = bh_fdr(np.array([]))
    assert len(reject) == 0


def test_synergy_index_divzero():
    from common.metrics import synergy_index
    R_c = np.array([1.0, 1.0])
    R_l = np.array([0.0, 0.5])
    R_s = np.array([0.0, 0.5])
    out = synergy_index(R_c, R_l, R_s)
    assert np.isnan(out[0])
    assert not np.isnan(out[1])


def test_bootstrap_ci_paired():
    from common.metrics import bootstrap_ci
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 30)
    y = rng.normal(0.3, 0.1, 30)
    lo, hi = bootstrap_ci(x, paired_with=y, n_boot=500)
    assert lo > 0


def test_rank_biserial_r_value():
    from common.stats import rank_biserial_r
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 30)
    y = rng.normal(1, 1, 30)
    r = rank_biserial_r(x, y)
    U = mannwhitneyu(x, y).statistic
    r_true = 1 - 2 * U / (30 * 30)
    assert abs(r - r_true) < 1e-12
    assert 0.2 < r < 0.8


def test_compare_groups_paired():
    from common.stats import compare_groups
    rng = np.random.default_rng(0)
    g1 = rng.normal(0, 1, 20)
    g2 = rng.normal(1, 1, 20)
    g3 = rng.normal(2, 1, 20)
    pairs, pvals, reject, kruskal_p = compare_groups([g1, g2, g3], paired=False)
    assert len(pvals) == 3
    assert kruskal_p < 0.05


def test_compare_groups_unpaired_diff_n():
    from common.stats import compare_groups
    rng = np.random.default_rng(2)
    g1 = rng.normal(0, 1, 20)
    g2 = rng.normal(3, 1, 35)
    pairs, pvals, reject, _ = compare_groups([g1, g2], paired=False)
    assert pvals[0] < 0.05


def test_compare_groups_paired_requires_equal_length():
    from common.stats import compare_groups
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        compare_groups([rng.normal(0, 1, 20), rng.normal(1, 1, 35)], paired=True)