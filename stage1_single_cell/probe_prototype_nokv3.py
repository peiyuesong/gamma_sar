"""MS1 前置：原型无 Kv3 半宽基准。

跑 E:\prototype_backup\prototype_kv3_pv.py，强制 g_Kv3=0，
逐 I 档报告 rate/half/v_end。

若原型不能直接改 g_Kv3 → 本脚本改源码写临时文件再跑。
"""
import io
import os
import re
import subprocess
from pathlib import Path

PY = r"C:\ANACONDA\envs\mleDSar\python.exe"
PROTO = Path(r"E:\prototype_backup\prototype_kv3_pv.py")
TMP = Path(r"E:\prototype_backup\_proto_kv3zero.py")

if not PROTO.exists():
    raise SystemExit(f"找不到原型: {PROTO}")

src = io.open(PROTO, encoding="utf-8").read()

print("=" * 70)
print("步骤 1：原型里所有含 g_Kv3 / g_kv3 的行")
print("=" * 70)
for i, line in enumerate(src.splitlines(), 1):
    if "g_Kv3" in line or "g_kv3" in line:
        print(f"{i:4} | {line.rstrip()}")
print()

print("=" * 70)
print("步骤 2：把 g_Kv3 强制为 0，写临时文件并跑")
print("=" * 70)

# 尝试多种写法
patterns = [
    (r"(g_Kv3\s*=\s*)([\d.]+)", r"\g<1>0"),
    (r"(g_kv3\s*=\s*)([\d.]+)", r"\g<1>0"),
    (r"g_Kv3\s*=\s*[^\n#]+", "g_Kv3 = 0"),
    (r"g_kv3\s*=\s*[^\n#]+", "g_kv3 = 0"),
]
patched = src
for pat, rep in patterns:
    new = re.sub(pat, rep, patched)
    if new != patched:
        patched = new
        print(f"  已替换: {pat}")
        break

if patched == src:
    raise SystemExit("⚠️ 找不到 g_Kv3 赋值，请手动打开原型改")

io.open(TMP, "w", encoding="utf-8").write(patched)
print(f"  临时文件: {TMP}")
print()

r = subprocess.run([PY, str(TMP)], capture_output=True, text=True,
                   cwd=str(PROTO.parent))
print("--- STDOUT ---")
print(r.stdout)
if r.stderr:
    print("--- STDERR (tail) ---")
    print(r.stderr[-2000:])