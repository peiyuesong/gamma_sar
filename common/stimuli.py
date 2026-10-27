"""五种刺激入口的统一定义（StimulusEntry 接口）。

差异仅体现在四个结构化参数上：
    entry_type, eta, delay, adaptation, slow_mod
构效四维度（时窗/量效/能效/动效）由 metrics.py 提供统一口径。
"""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class StimulusEntry:
    """统一刺激入口。五模态共用此结构，差异只在参数值。"""
    entry_type: Literal["ephaptic", "retino_thalamic", "cochlear",
                        "somatosensory", "pbm"]
    eta: float                  # 转换效率：刺激物理量 → 网络有效驱动
    delay: float                # 入口传导延迟（ms）→ 时窗维度
    adaptation: float           # 感受器/突触适应时间常数（ms）→ 动效维度
    slow_mod: Optional[float] = None  # 仅 PBM：慢调制时间常数（s）


# 五个模态的占位参数（阶段 3 T1–T10 标定后冻结）
ENTRIES = {
    "ephaptic": StimulusEntry(
        entry_type="ephaptic",
        eta=1.0,                  # tACS κ=0.25 即为标定值
        delay=0.0,                # 电刺激即时（亚阈值作用于 V）
        adaptation=0.0,
    ),
    "retino_thalamic": StimulusEntry(
        entry_type="retino_thalamic",
        eta=0.5,                  # 占位：阶段 3 T1 标定
        delay=55.0,               # 30–80ms 中值
        adaptation=200.0,         # 秒级视网膜适应
    ),
    "cochlear": StimulusEntry(
        entry_type="cochlear",
        eta=0.5,                  # 占位
        delay=30.0,               # 耳蜗锁相到皮层延迟
        adaptation=150.0,
    ),
    "somatosensory": StimulusEntry(
        entry_type="somatosensory",
        eta=0.4,                  # 占位；Meissner 40Hz 调谐锚
        delay=40.0,
        adaptation=80.0,          # 预测级：动效最差（适应快）
    ),
    "pbm": StimulusEntry(
        entry_type="pbm",
        eta=0.3,                  # 占位
        delay=0.0,                # 无相位概念 → 时窗 N/A
        adaptation=0.0,
        slow_mod=180.0,           # τ_mod∈[60,600]s 中值，作用于 g_Kv3
    ),
}


def get_entry(name: str) -> StimulusEntry:
    return ENTRIES[name]


def all_entries():
    return list(ENTRIES.keys())