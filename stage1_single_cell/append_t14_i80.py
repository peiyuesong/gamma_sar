"""给 t14_report.md 追加 I=80 扩展档说明（超稳态区，不纳入 MS1 验收）。"""
import io
from pathlib import Path

OUT = Path(r"E:\gamma_sar\stage1_single_cell\t14_report.md")
txt = io.open(OUT, encoding="utf-8").read()

section = """
## 六、I=80 μA/cm² 扩展档说明（超稳态区，不纳入 MS1 验收）

2026-10-10 补：为覆盖路线图原写的 {2,4,6,8}nA，run_t13 I_LIST 扩到 0–80 μA/cm²。
I=80 档实测（见 `t13_fI_surface.csv`）：

| g_Kv3 | rate Hz | half ms | v_end mV | blocked |
|---|---|---|---|---|
| 0  | 532.5 | 0.600 | −33.9 | 1 |
| 5  | 531.5 | 0.600 | −33.9 | 1 |
| 10 | 529.5 | 0.600 | −34.0 | 1 |
| 15 | 527.5 | 0.600 | −34.1 | 1 |
| 20 | 525.0 | 0.600 | −34.2 | 1 |
| 30 | 517.5 | 0.575 | −34.4 | 1 |
| 40 | 503.0 | 0.550 | −34.6 | 1 |

结论：
- **I=80 时所有 g_Kv3（含 40）都进失活区**（v_end > −35 mV），Kv3 不再能维持深复极。这与路线图 3.3 "强注入 2s 末段 V ≥ −55mV" 口径无关——那条是针对 I=60 定的。
- rate 与 half 仍在可接受范围（>200 Hz、≤0.6 ms），但末段电位已越线，故全部标 BLOCK。
- **MS1 半宽与复极维持的验收口径限定 I ≤ 60 μA/cm²**。I=80 数据作为"超稳态区"如实归档，不进验收，供 T2.1 网络驱动上限参考。

对 MS1 验收表的影响：不改。MS1 表里 "含 Kv3 放电 >200Hz@6nA" 指 I=60；"半宽 ≤0.6ms 有 Kv3" 指 I ≤ 60；"复极化 V≥−55mV" 指 I=60 档 −48.3 mV。I=80 数据单列本表。
"""

if "## 六、I=80 μA/cm² 扩展档说明" not in txt:
    io.open(OUT, "w", encoding="utf-8").write(txt.rstrip() + "\n" + section)
    print("✅ t14_report.md 已追加第六节")
else:
    print("ℹ️ 第六节已存在，跳过")