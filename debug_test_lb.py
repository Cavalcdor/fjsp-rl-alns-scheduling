"""测试下界"""
import sys
sys.path.insert(0, '.')
from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from utils.scheduler import evaluate

jobs, nm, nj = load_fjsp_from_file('data/Brandimarte_Data/Mk01.fjs')
jobs = convert_to_zero_index(jobs)

# 构建一个简单的贪心调度来验证下界
machine_load = [0]*nm
assignment = []
for jid in range(nj):
    for oid in range(len(jobs[jid])):
        op = jobs[jid][oid]
        best_mid = 0
        best_val = float('inf')
        for i,mid in enumerate(op['machines']):
            val = max(machine_load[mid] + op['times'][i], max(machine_load))
            if val < best_val:
                best_val = val
                best_mid = i
        mid = op['machines'][best_mid]
        assignment.append((jid, oid, mid, op['times'][best_mid]))
        machine_load[mid] += op['times'][best_mid]

cmax, lv, det = evaluate(jobs, assignment, nm, return_details=True)
print(f"贪心+负载均衡 Cmax={cmax}, 负荷方差={lv}")
print(f"机器负荷: {det['machine_load']}")
