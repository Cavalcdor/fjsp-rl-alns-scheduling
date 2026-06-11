# utils/scheduler.py
import numpy as np

def evaluate(jobs, assignment, num_machines, return_details=False):
    """
    评估一个调度解
    
    参数:
        jobs: 解析后的工件数据（已转为0索引），格式:
              jobs[job_id][op_id] = {'machines': list, 'times': list}
        assignment: 每个工序的机器选择和加工时间，格式为 list of (job_id, op_id, machine_id, duration)
                    或者可以是由调度算法直接生成的分配方案，我们会模拟时间线
        num_machines: 机器总数
        return_details: 是否返回详细的机器时间线等信息
    
    返回:
        cmax: 最大完工时间
        load_variance: 设备负荷方差
        (可选) details: dict包含机器负荷列表、每个工序的开始/结束时间等
    """
    # 初始化机器可用时间
    machine_available_time = [0] * num_machines
    # 记录每个工件上一道工序的完成时间
    job_completion_time = [0] * len(jobs)
    
    # 记录每个工序的开始时间和结束时间，以及机器负荷累加
    machine_load = [0] * num_machines
    schedule = []  # 存储 (job_id, op_id, machine, start, end)
    
    # 按照工序顺序处理（assignment应当已经是拓扑序）
    # 注意：assignment需要按照工序先后顺序给出，通常遗传算法编码会保证顺序
    for (job_id, op_id, machine_id, duration) in assignment:
        # 工序的最早开始时间 = max(机器空闲时间, 同一工件上一道工序完成时间)
        start_time = max(machine_available_time[machine_id], job_completion_time[job_id])
        end_time = start_time + duration
        schedule.append((job_id, op_id, machine_id, start_time, end_time))
        # 更新机器可用时间和工件完成时间
        machine_available_time[machine_id] = end_time
        job_completion_time[job_id] = end_time
        # 累加机器负荷（实际加工时长）
        machine_load[machine_id] += duration
    
    cmax = max(job_completion_time)
    # 计算设备负荷方差
    avg_load = np.mean(machine_load)
    load_variance = np.var(machine_load)
    
    if return_details:
        details = {
            'machine_load': machine_load,
            'schedule': schedule,
            'cmax': cmax,
            'load_variance': load_variance
        }
        return cmax, load_variance, details
    else:
        return cmax, load_variance


if __name__ == "__main__":
    # 简单测试
    # 构造一个简单的2工件2机器示例
    test_jobs = [
        [   # 工件0
            {'machines': [0,1], 'times': [3,4]},  # 工序0
            {'machines': [0], 'times': [2]}        # 工序1
        ],
        [   # 工件1
            {'machines': [1], 'times': [5]}        # 工序0
        ]
    ]
    # 一个合法assignment（按拓扑序）
    assignment = [
        (0, 0, 0, 3),   # 工件0工序0在机器0加工3
        (1, 0, 1, 5),   # 工件1工序0在机器1加工5
        (0, 1, 0, 2)    # 工件0工序1在机器0加工2（要等机器0空闲）
    ]
    cmax, var, det = evaluate(test_jobs, assignment, 2, return_details=True)
    print("Cmax:", cmax)
    print("负荷方差:", var)
    print("调度详情:", det['schedule'])