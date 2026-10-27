"""重跑 run_t13.py，修 ms1_leftovers_fix.py [4] 步的 Windows GBK 管道编码崩溃。

不修改 run_t13.py 源码，用 PYTHONIOENCODING=utf-8 绕过 GBK 管道编码。
"""
import os
import subprocess
import sys
from pathlib import Path

# 父进程 stdout 保险：任何不可编码字符降级为 '?'
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PY = r"C:\ANACONDA\envs\mleDSar\python.exe"
ROOT = Path(r"E:\gamma_sar")
T13 = ROOT / "stage1_single_cell" / "run_t13.py"

# --- 检查 run_t13.py 已是修正版 ---
src = open(T13, encoding="utf-8").read()
ok_block = "if rate < 5.0 and v_end < -55.0" in src
ok_ilist = "I_LIST = list(range(0, 85, 5))" in src
print(f"[check] blocked() 修正版 = {ok_block}")
print(f"[check] I_LIST 0-80      = {ok_ilist}")
if not (ok_block and ok_ilist):
    print("!! run_t13.py 不是修正版，先重跑 ms1_leftovers_fix.py")
    raise SystemExit(1)

# --- UTF-8 管道重跑 ---
env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"
print("\n[run] 重跑 run_t13.py（7x17=119 次运行，约 1-3 分钟）...")
r = subprocess.run([PY, str(T13)], capture_output=True, cwd=str(ROOT),
                   env=env, encoding="utf-8", errors="replace")
# 只打印 ASCII 安全版 stdout 尾部
safe = r.stdout.encode("ascii", errors="replace").decode("ascii")
print(safe[-1200:])
if r.stderr:
    safe_err = r.stderr.encode("ascii", errors="replace").decode("ascii")
    print("--- STDERR (tail) ---")
    print(safe_err[-500:])

# --- 核对 ---
csv_path = ROOT / "stage1_single_cell" / "t13_fI_surface.csv"
print("\n--- CSV 前 8 行 ---")
for i, line in enumerate(open(csv_path, encoding="utf-8")):
    if i >= 8:
        break
    print(line.rstrip())

print("\n--- I=80 行 ---")
for line in open(csv_path, encoding="utf-8"):
    parts = line.strip().split(",")
    if len(parts) >= 2 and parts[1] == "80":
        print(line.rstrip())

print("\n完成。核对无误后：")
print("  git add -A")
print('  git commit -m "fix(ms1): t13 blocked() I=0 excluded, I=80 added; run_t12 -> legacy"')