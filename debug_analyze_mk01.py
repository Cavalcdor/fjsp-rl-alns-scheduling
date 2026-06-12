"""Mk01 数据分析脚本"""
import sys
sys.path.insert(0, '.')
from core.instance_parser import load_fjsp_from_file, convert_to_zero_index

jobs, num_machines, num_jobs = load_fjsp_from_file('data/Brandimarte_Data/Mk01.fjs')
jobs = convert_to_zero_index(jobs)
print(f'工件: {num_jobs}, 机器: {num_machines}')

# 统计每台机器作为最优选择的次数和总时间
machine_shortest = [0] * num_machines
machine_count = [0] * num_machines
for job in jobs:
    for op in job:
        min_time = min(op['times'])
        min_idx = op['times'].index(min_time)
        mid = op['machines'][min_idx]
        machine_count[mid] += 1
        machine_shortest[mid] += min_time

print('各机器作为最优选择的次数:', machine_count)
print('各机器作为最优选择的总加工时间:', machine_shortest)
print(f'理论下界 LB = {max(max(machine_shortest), max(len(j) for j in jobs))}')

# 检查 M1(机器0) 的工序分布
print('\n=== M1 (机器0) 的工序分布 ===')
m1_ops = []
for jid, job in enumerate(jobs):
    for oid, op in enumerate(job):
        for i, mid in enumerate(op['machines']):
            if mid == 0:
                m1_ops.append((jid, oid, op['times'][i], min(op['times'])))
                break
print(f'M1 可加工工序数: {len(m1_ops)}')
print(f'M1 上最短加工时间占比: {sum(1 for x in m1_ops if x[2]==x[3])}/{len(m1_ops)}')

# 检查哪些工序只能在M1上加工
print('\n=== 只能在特定机器上加工的工序 ===')
for jid, job in enumerate(jobs):
    for oid, op in enumerate(job):
        if len(op['machines']) == 1:
            print(f'  工件{jid}工序{oid}: 只能在机器{op["machines"][0]}上加工, 时间={op["times"][0]}')
