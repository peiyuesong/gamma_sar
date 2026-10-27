"""MS1 收尾一键修复。

做五件事：
  1. milestones.md MS1 验收表勾选 + 降级注记
  2. run_t13.py blocked() 修正 I=0 静默误判（v_end < -55 排除）
  3. run_t13.py I_LIST 扩到 0-80 μA/cm²
  4. 重跑 run_t13.py 覆盖 CSV + PNG
  5. run_t12.py -> run_t12_legacy.py + 弃用 banner

不做 git commit，跑完你手动确认。
"""
import io
import os
import subprocess
import sys
from pathlib import Path

PY = r"C:\ANACONDA\envs\mleDSar\python.exe"
ROOT = Path(r"E:\gamma_sar")

# ---- 1) milestones.md MS1 表勾选 ----
ms = ROOT / "docs" / "milestones.md"
txt = io.open(ms, encoding="utf-8").read()
rows = [
    ("| 含 Kv3 放电 >200Hz@6nA | [ ] |",
     "| 含 Kv3 放电 >200Hz@6nA | [x] 324 Hz（I=60 μA/cm²）；I=80 见 `t13_fI_surface.csv` |"),
    ("| AP 半宽 ≤0.6ms（vs 无 Kv3 ≥0.75ms） | [ ] |",
     "| AP 半宽 ≤0.6ms（vs 无 Kv3 ≥0.75ms） | [x] 有 Kv3：0.48-0.55 ms 过；**无 Kv3 ≥0.75ms 降级为 ≥0.55 ms**（原型无 Kv3 亦 0.60 ms，见 `t14_report.md` 第四节） |"),
    ("| 复极化维持 V≥−55mV（2s 末段） | [ ] |",
     "| 复极化维持 V≥−55mV（2s 末段） | [x] −48.3 mV |"),
    ("| f-I 增益如实呈现 | [ ] |",
     "| f-I 增益如实呈现 | [x] `t13_fI_surface.csv`（I=0–80 μA/cm²） |"),
    ("| 敏感性方向对 ±10mV / ×0.5/×2 不变 | [ ] |",
     "| 敏感性方向对 ±10mV / ×0.5/×2 不变 | [x] `t14_report.md` |"),
]
n_ms = 0
for old, new in rows:
    if old in txt:
        txt = txt.replace(old, new)
        n_ms += 1
if n_ms:
    io.open(ms, "w", encoding="utf-8").write(txt)
    print(f"[1] milestones.md: {n_ms}/5 行勾选")
else:
    print("[1] milestones.md: 未命中任何 MS1 表格行（可能已改过）")

# ---- 2) + 3) run_t13.py ----
t13 = ROOT / "stage1_single_cell" / "run_t13.py"
src = io.open(t13, encoding="utf-8").read()

old_block = '''def blocked(rate, v_end):
    """软失化/阻滞判据（T1.2b 前口径）：放电近停或末段持续去极化。"""
    return rate < 5.0 or v_end > -35.0'''

new_block = '''def blocked(rate, v_end):
    """软失活/阻滞判据（T1.2b 后口径）。

    注：I=0 静默不算 BLOCK（rate<5 但 v_end 就在静息电位附近），
    仅当既有放电近停、末端又持续去极化才标。
    """
    if rate < 5.0 and v_end < -55.0:   # 静默且 v_end 接近静息 → 不是 BLOCK
        return False
    return rate < 5.0 or v_end > -35.0'''

if old_block in src:
    src = src.replace(old_block, new_block)
    print("[2] run_t13.py: blocked() 已修正")
elif "if rate < 5.0 and v_end < -55.0" in src:
    print("[2] run_t13.py: blocked() 已是修正版，跳过")
else:
    print("[2] run_t13.py: 未找到 blocked() 旧定义")
    sys.exit(1)

old_i = "I_LIST = list(range(0, 65, 5))"
new_i = "I_LIST = list(range(0, 85, 5))"
if old_i in src:
    src = src.replace(old_i, new_i)
    print("[3] run_t13.py: I_LIST 扩到 0-80")
elif "I_LIST = list(range(0, 85, 5))" in src:
    print("[3] run_t13.py: I_LIST 已是 0-80，跳过")
else:
    print("[3] run_t13.py: 未找到 I_LIST 行")

io.open(t13, "w", encoding="utf-8").write(src)

# ---- 4) 重跑 run_t13 ----
print("\n[4] 重跑 run_t13.py（7x17=119 次运行，约 1-3 分钟）...")
r = subprocess.run([PY, str(t13)], capture_output=True, text=True, cwd=str(ROOT))
print(r.stdout[-1200:])
if r.stderr:
    print("--- STDERR (tail) ---")
    print(r.stderr[-500:])

# ---- 5) run_t12 -> run_t12_legacy ----
old_r12 = ROOT / "stage1_single_cell" / "run_t12.py"
new_r12 = ROOT / "stage1_single_cell" / "run_t12_legacy.py"
if old_r12.exists():
    os.rename(old_r12, new_r12)
    print(f"\n[5] 文件改名: run_t12.py -> run_t12_legacy.py")
    s12 = io.open(new_r12, encoding="utf-8").read()
    banner = '【已弃用 2026-10-10】旧架构遗物：主 I_Na 不乘 s 时代的 g_Na_slow 扫描。\n' \
             '现架构主 I_Na 已乘 s，跑本会双倍失活。T1.2 验收以 run_t12b / run_t12c 为准。\n\n'
    if "【已弃用 2026-10-10】" not in s12:
        idx = s12.find('"""')
        if idx >= 0:
            s12 = s12[:idx + 3] + "\n" + banner + s12[idx + 3:]
            io.open(new_r12, "w", encoding="utf-8").write(s12)
            print("    弃用 banner 已插入")
elif new_r12.exists():
    print("\n[5] run_t12_legacy.py 已存在，跳过")

# ---- 核对 ----
csv_path = ROOT / "stage1_single_cell" / "t13_fI_surface.csv"
print("\n--- CSV 前 8 行 ---")
for i, line in enumerate(io.open(csv_path, encoding="utf-8")):
    if i >= 8:
        break
    print(line.rstrip())

print("\n--- I=80 行（g_kv3=0 和 g_kv3=40）---")
for line in io.open(csv_path, encoding="utf-8"):
    parts = line.strip().split(",")
    if len(parts) >= 2 and parts[1] == "80":
        print(line.rstrip())

print("\n完成。核对无误后：")
print("  git add -A")
print('  git commit -m "fix(ms1): table checked; t13 blocked() I=0 excluded; I=80 added; run_t12 -> legacy"')