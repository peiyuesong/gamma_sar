"""单元验证脚本。

注：params 断言对应 v3.1-fix2（含 tau_p_slow 等新增参数，
make_pv_neuron 的首次 brian2 编译测试在装有 brian2 的环境执行，属 MS0 验收）。
"""
import importlib
import importlib.util
import pytest

HAS_BRIAN2 = importlib.util.find_spec("brian2") is not None


def test_imports_no_brian2():
    """不依赖 brian2 的模块可导入（任何环境都必须过）。"""
    for mod in ["common.params", "common.stimuli", "common.plasticity",
                "common.metrics", "common.stats"]:
        importlib.import_module(mod)


@pytest.mark.skipif(not HAS_BRIAN2, reason="brian2 not installed")
def test_imports_with_brian2():
    """brian2 依赖模块（装环境后取消跳过）。"""
    for mod in ["common.cells", "common.networks"]:
        importlib.import_module(mod)


def test_params_defaults():
    """参数锚定表 v3.1-fix 关键值。"""
    from common.params import PARAMS
    # g_Kv3 is the arousal-state scan axis: 40 = EO (the value used in every
    # network experiment), 0 = EC.  The dataclass default is the EO end.
    assert PARAMS["cell"].g_Kv3 == 40.0
    assert PARAMS["cell"].V12_p == 13.0     # Rudy & McBain 2001 克隆值
    assert PARAMS["gabab"].K1 == 0.09       # Destexhe 1998 修正后
    assert PARAMS["gabab"].K3 == 0.18       # Destexhe 1998 修正后
    assert PARAMS["gabab"].n_coop == 4      # Thomson & Destexhe 1999 实验签名
    assert PARAMS["stim"].mu_EC == pytest.approx(1.0 / 1.8)


def test_stimulus_entry_interface():
    """5 模态共享同一 StimulusEntry 结构。"""
    from common.stimuli import ENTRIES
    assert len(ENTRIES) == 5
    for k, e in ENTRIES.items():
        assert e.entry_type in ("ephaptic", "retino_thalamic", "cochlear",
                                "somatosensory", "pbm")


def test_G_phi_shape():
    from common.plasticity import G_phi
    import numpy as np
    phi = np.linspace(0, 360, 100)
    g = G_phi(phi, a=0.5, phi_peak=90.0)
    assert g.shape == (100,)
    # linspace(0,360,100) 步长 3.636°，90° 不在采样点上 → 用容差而非精确等值
    assert g.max() == pytest.approx(1.5, abs=0.01)
    assert g.min() == pytest.approx(0.5, abs=0.01)


def test_metrics_time_window():
    from common.metrics import time_window
    import numpy as np
    phi = np.linspace(0, 360, 73)
    # 构造 90° 峰，半高宽约 ±90°，故总宽 ~180°
    curve = 1 + np.cos(np.deg2rad(phi - 90))
    w = time_window(curve, phi)
    assert 150 <= w <= 210


def test_metrics_energy_efficiency():
    from common.metrics import energy_efficiency
    assert energy_efficiency(1.0, 2.0) == 0.5
    import math
    assert math.isnan(energy_efficiency(1.0, 0.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])