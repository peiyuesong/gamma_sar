"""真实 brian2 构建/运行的最小冒烟测试：5 神经元 × 10 ms。

守护 cells.py 的真实使用路径（构造阈值/重置/ summed 变量）——
import 测试过不了的方程错误（threshold 布尔性、summed target 缺失等）由本测试拦截。
"""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("brian2") is None,
                                reason="brian2 not installed")


def test_pv_neuron_smoke_run():
    from brian2 import start_scope, run, ms
    from common.cells import make_pv_neuron
    start_scope()
    G = make_pv_neuron(5)
    run(10 * ms)
    assert len(G.v) == 5  # 无异常且维度正确即通过
