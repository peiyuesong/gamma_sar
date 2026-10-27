# 0.4b Benchmark 报告

## 环境

- 机器: Windows, Anaconda
- Python: 3.11.17 (mleDSar env)
- brian2: 2.7.1 + Cython 0.29.37
- 网络: 200 PV, GABA_A 电导突触, gap junction 线性近似, ±10% 异质性
- 生物时间: 2 s, dt = 0.025 ms

## 正式 benchmark（brian2 Cython 后端）

| 配置 | 2s 墙钟 | 外推 2h | f0 (Hz) | rate (Hz) | γ 占比 |
|---|---|---|---|---|---|
| N=200 | 7.96 s | 7.96 h | 32.2 | 64.0 | 0.274 |
| N=100 | 4.57 s | 4.57 h | 31.9 | 126.7 | 0.451 |

## 分档决策

- 单次 2h 墙钟（N=200 Cython）: **7.96 h** → 档位 **> 4h**
- 不降维 N=100（f0/rate 恶化）

## 执行策略（时程分离）

- T2.1-T2.6: 10s 生物时间/次（本机 ~40 s/次），全网格本机跑
- T2.7 (STDP 2h 维持): 2h/次 → 申请集群或分夜跑

## numpy 参考实现（教学参考；欧拉积分 + 随机图实例差异，rate 22 vs 64，不作对齐要求；分档以 brian2 为准）

- 脚本: benchmark/run_benchmark_numpy.py
- 用途: 无 brian2 环境下验证网络能起振
- 数值差异（rate 22 vs 64 Hz）源于占位参数下的实现敏感性，不作为 bug

## 待对齐项（T1.1 处理）

- f0 = 32.2 Hz（占位驱动下，T1.1 调 g_Kv3 × 驱动推至 ~40 Hz）
- 上述差异都是 T1.1 的调参任务，MS0 不调参凑

## 复现命令

& "C:\ANACONDA\envs\mleDSar\python.exe" benchmark\run_benchmark_brian2.py