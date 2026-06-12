"""
调度分析工具：诊断 Cmax 瓶颈，分析机器负荷、关键路径等
"""
import sys
sys.path.insert(0, '.')
from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from algorithms.ga import GA
from algorithms.rl_agent import RLController
from algorithms.alns import ALNS
from utils.scheduler import evaluate
import config
import numpy as np
from collections import defaultdict

# 加载数据
jobs_raw, nm, nj = load_fjsp_from_file('data/Brandimarte_Data/Mk01.fjs')
jobs = convert_to_zero_index(jobs_raw)
print(f"工件={nj}, 机器={nm}, 工序={sum(len(j) for j in jobs)}")

# 运行 GA
ga = GA(jobs, nm, config)
best = ga.run(rl_controller=RLController(config), alns=ALNS(jobs, nm, config))

# 解码
assign = ga.decode(best)
cmax, load_var, details = evaluate(jobs, assign, nm, return_details=True)

print(f"\n{'='*60}")
print(f"最优调度 Cmax = {cmax}")
print(f"{'='*60}")

# === 1. 机器负荷分析 ===
print(f"\n--- 1. 机器负荷分析 ---")
machine_load = details['machine_load']
total_load = sum(machine_load)
for m in range(nm):
    ratio = machine_load[m] / cmax * 100
    bar = '#' * int(ratio / 5) + '.' * (20 - int(ratio / 5))
    print(f"  机器 M{m}: 负荷={machine_load[m]:4.0f}, 利用率={ratio:5.1f}% [{bar}]")

avg_load = np.mean(machine_load)
load_cv = np.std(machine_load) / avg_load if avg_load > 0 else 0
print(f"\n  平均负荷: {avg_load:.1f}, 负荷变异系数: {load_cv:.3f}")
print(f"  负荷方差: {load_var:.1f}")

# === 2. 机器空闲时间分析 ===
print(f"\n--- 2. 机器空闲时间分析 ---")
schedule = details['schedule']
machine_ops = defaultdict(list)
for (jid, oid, mid, s, e) in schedule:
    machine_ops[mid].append((s, e, jid, oid))

for m in range(nm):
    ops = sorted(machine_ops[m], key=lambda x: x[0])
    total_idle = 0
    prev_end = 0
    for (s, e, jid, oid) in ops:
        if s > prev_end:
            total_idle += s - prev_end
        prev_end = max(prev_end, e)
    if prev_end < cmax:
        total_idle += cmax - prev_end
    print(f"  机器 M{m}: 总空闲={total_idle:.0f}, 繁忙={cmax - total_idle:.0f}")

# === 3. 瓶颈机器分析 ===
print(f"\n--- 3. 瓶颈机器分析 ---")
sorted_machines = sorted(range(nm), key=lambda m: machine_load[m], reverse=True)
print(f"  最高负荷: M{sorted_machines[0]} (负荷={machine_load[sorted_machines[0]]})")
print(f"  次高负荷: M{sorted_machines[1]} (负荷={machine_load[sorted_machines[1]]})")

# 瓶颈机器上的工序详情
bm = sorted_machines[0]
print(f"\n  瓶颈机器 M{bm} 上的工序序列:")
bm_ops = sorted(machine_ops[bm], key=lambda x: x[0])
for (s, e, jid, oid) in bm_ops:
    print(f"    J{jid} Op{oid}: [{s:.0f}-{e:.0f}] (时长={e-s:.0f})")

# === 4. 各工件加工路径 ===
print(f"\n--- 4. 各工件加工路径 ---")
for jid in range(nj):
    job_ops = [(oid, mid, s, e) for (jid2, oid, mid, s, e) in schedule if jid2 == jid]
    job_ops.sort(key=lambda x: x[2])  # 按开始时间
    path = " -> ".join([f"M{mid}[{s:.0f}-{e:.0f}]" for (oid, mid, s, e) in job_ops])
    total_dur = sum([e-s for (oid, mid, s, e) in job_ops])
    print(f"  J{jid}: {path} (总加工={total_dur:.0f})")

# === 5. 与 BKS 的差距分析 ===
print(f"\n--- 5. 与 BKS 差距分析 ---")
bks = 40
gap = (cmax - bks) / bks * 100
print(f"  当前 Cmax: {cmax}")
print(f"  BKS: {bks}")
print(f"  差距: {gap:.1f}%")
print(f"  需要缩短: {cmax - bks} 单位时间")

# 如果能将瓶颈机器的负荷降低
print(f"\n  如果瓶颈机器 M{bm} 的负荷降低 10%:")
reduction = machine_load[bm] * 0.1
print(f"    可减少约 {reduction:.0f} 单位时间")
print(f"    潜在 Cmax: {cmax - reduction:.0f}")

# === 6. 可选机器分析 ===
print(f"\n--- 6. 可选机器分析 ---")
# 统计每个工序有多少可选机器
total_alternatives = 0
total_ops = 0
for jid in range(nj):
    for op in jobs[jid]:
        total_alternatives += len(op["machines"])
        total_ops += 1
print(f"  平均每工序可选机器数: {total_alternatives/total_ops:.1f}")

# 瓶颈机器上各工序的可选机器
print(f"\n  瓶颈机器 M{bm} 上工序的可选方案:")
for (s, e, jid, oid) in bm_ops:
    op_data = jobs[jid][oid]
    machines_str = ", ".join([f"M{m}(t={t})" for m, t in zip(op_data["machines"], op_data["times"])])
    print(f"    J{jid} Op{oid}: {machines_str}")

print(f"\n{'='*60}")
print("分析完成")
