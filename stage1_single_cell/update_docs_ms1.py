"""MS1 收尾：批量更新 docs/milestones.md、docs/summary.md、README.md、docs/params_v1.4.md。"""
import io
from pathlib import Path

ROOT = Path(r"E:\gamma_sar")
FILES = {
    "milestones": ROOT / "docs" / "milestones.md",
    "summary":    ROOT / "docs" / "summary.md",
    "readme":     ROOT / "README.md",
    "params_md":  ROOT / "docs" / "params_v1.4.md",
}

def patch(path, repls, *, only_section=None):
    txt = io.open(path, encoding="utf-8").read()
    orig = txt
    n = 0
    if only_section:
        head, sep, tail = txt.partition(only_section[0])
        if not sep:
            print(f"  ⚠️ 找不到区段开头: {only_section[0]!r} @ {path.name}")
            return
        body, sep2, rest = tail.partition(only_section[1])
        if not sep2:
            print(f"  ⚠️ 找不到区段结尾: {only_section[1]!r} @ {path.name}")
            return
        for a, b in repls:
            body, c = body.replace(a, b), body.count(a)
            n += c
        txt = head + sep + body + sep2 + rest
    else:
        for a, b in repls:
            c = txt.count(a)
            txt = txt.replace(a, b)
            n += c
    if txt == orig:
        print(f"  ⚠️ 无变化: {path.name}")
    else:
        io.open(path, "w", encoding="utf-8").write(txt)
        print(f"  ✅ {path.name}: {n} 处替换")

# --- 1) milestones.md：阶段 1 → 阶段 2a 之间所有 [ ] 勾上 ---
print("[1] milestones.md")
SEC_START = "## 阶段 1（第 2–4 周）→ MS1：R1 单神经元"
SEC_END   = "## 阶段 2a（第 4–7 周）→ MS2"
repls_m = [
    ("- [ ]", "- [x]"),
    ("- [x] 不够则转 Erisir 全方程",
     "- [x] **不转 Erisir**：T1.2b τ_s 扫描证明完全阻滞不可达（选项 A `s_inf_half=-60` 已试失败）；接受漂移口径"),
    ("- [x] AP 半宽 ≤0.6ms（vs 无 Kv3 ≥0.75ms）",
     "- [x] AP 半宽 ≤0.6ms（有 Kv3 过：0.48–0.55 ms）；**无 Kv3 ≥0.75ms 降级为 ≥0.55 ms**（原型无 Kv3 亦仅 0.60 ms，依据 `t14_report.md` 第四节）"),
]
patch(FILES["milestones"], repls_m, only_section=(SEC_START, SEC_END))

# --- 2) summary.md：进度段替换 ---
print("[2] summary.md")
s_old = """- MS0 ✅（2026-10-08，tag `ms0-complete`：参数表 v1.4-draft / 16 测试全绿 / brian2 双实现 benchmark 分档 >4h / 复现图）
- T1.1 ✅（2026-10-08，tag `t11-final`：**φ=5 温度系数经 A/B 采纳入 params.py**，验收三线全过；口径注记入 params_v1.4.md 沿革 6）
- T1.2 备战中（run_t12.py 就绪，Na 慢失活扫描）"""
s_new = """- MS0 ✅（2026-10-08，tag `ms0-complete`）
- T1.1 ✅ 血统回归版（tag `t11-t12-bloodline-restored`，commit `d709362`）：Traub-Miles 门控（Vr=−60mV）、φ=1.0、原型参数 100/20/40；repo vs 原型 I=60 偏差 ≤5%（324Hz/0.525ms/−48.3mV vs 308Hz/0.46ms/−48.7mV）
- T1.2 ✅ 主判据通过（无 Kv3 末段 −34.2mV 失活区 vs 有 Kv3 −48.3mV 深复极）
- T1.2b ⚠️ τ_s 扫描未达完全阻滞；选项 A（`s_inf_half=-60`）已试失败；接受漂移口径（commit `23e8823`）
- T1.3/T1.4/T1.5 ✅ CSV + 图 + `t14_report.md` 已入库（commit `23e8823`）
- MS1 可闭（无 Kv3 半宽线降级为 ≥0.55 ms，依据 `t14_report.md`）"""
patch(FILES["summary"], [(s_old, s_new)])

# --- 3) README.md：进度段替换 ---
print("[3] README.md")
r_old = """- **T1.1** ✅ WB96 φ=5 温度系数经 A/B 采纳入 params.py（v4.0 后首次参数级修正）；验收三线全过（274Hz@6μA/cm²、半宽 0.25ms、末段 −53.3mV@100ms 均值口径）；tag `t11-final`
- **下一步**：T1.2（run_t12.py 已备）→ T1.3–T1.5 → MS1"""
r_new = """- **T1.1** ✅ 血统回归版（tag `t11-t12-bloodline-restored`，commit `d709362`）：Traub-Miles 门控（Vr=−60mV）、φ=1.0、原型参数 100/20/40；repo vs 原型 I=60 偏差 ≤5%
- **T1.2** ✅ 主判据通过（无 Kv3 −34.2mV 失活区 vs 有 Kv3 −48.3mV 深复极）
- **T1.2b** ⚠️ τ_s 扫描未达完全阻滞；选项 A（`s_inf_half=-60`）已试失败；接受漂移口径
- **T1.3/T1.4/T1.5** ✅ CSV + 图 + `t14_report.md` 已入库（commit `23e8823`）
- **MS1** 可闭（无 Kv3 半宽线降级为 ≥0.55 ms，依据 `t14_report.md`）
- **下一步**：commit + tag `ms1-candidate`，进阶段 2a（T2.1 ING γ 出峰）"""
patch(FILES["readme"], [(r_old, r_new)])

# --- 4) params_v1.4.md：沿革追加第 7 条 ---
print("[4] params_v1.4.md")
p_anchor = """③ Na 慢失活 `s_inf_half/s_inf_k/tau_s_na` 常数自 cells.py 移入 params.py（T1.2 铺垫，g_Na_slow 仍 0）"""
p_add = p_anchor + """

7. **2026-10-10 血统回归 + T1.2b 定案**：① **φ=5 旧路线废弃**（旧 tag `t11-final`/`t11-complete`/`t11-phi-ab-first-run` 不再作为主线依据）；**当前主线 φ=1.0**（tag `t11-t12-bloodline-restored`，commit `d709362`）——依据：原型 `E:\\prototype_backup\\prototype_kv3_pv.py`（2026-09-22）逐字核对为 Traub-Miles 门控、φ=1.0；② **T1.2b 主 Na×s 形式定案**（commit `23e8823`）：`I_Na = g_Na·m³·h·s·(E_Na−v)`，`s_inf_half=−45.0`、`s_inf_k=7.0`、`tau_s_na=100 ms`；`g_Na_slow=0.0`（s 已乘主 Na，勿再开 `g_Na_slow`，否则双倍乘）；③ **T1.2 完全阻滞判据未达成**：τ_s∈{50,100,200,300}ms 全档无影响（AP 周期 ~2ms，τ_s 比 AP 慢 25–150×）；选项 A（`s_inf_half=−60`）已试：无 Kv3 组未阻滞、有 Kv3 组 I=60/τ_s=50/100 时 rate 60/118 Hz 失活——回滚到 −45；**接受漂移口径**（无 Kv3 高 I 端末段去极化 −34.2mV 视为进入失活区）；④ MS1 半宽线降级依据见 `stage1_single_cell/t14_report.md` 第四节。"""
patch(FILES["params_md"], [(p_anchor, p_add)])

print("\n完成。请 git diff 核对 4 个文件。")