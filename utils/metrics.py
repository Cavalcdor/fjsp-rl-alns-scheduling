# utils/metrics.py
"""
调度质量评估指标模块
提供多种评估调度方案质量的指标计算方法
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def calculate_makespan(schedule: List[Tuple], num_jobs: int) -> float:
    """
    计算最大完工时间（Makespan/Cmax）
    
    参数:
        schedule: 调度方案，格式为 [(job_id, op_id, machine_id, start, end), ...]
        num_jobs: 工件数量
    
    返回:
        makespan: float, 最大完工时间
    """
    if not schedule:
        return 0.0
    
    # 记录每个工件的完工时间
    job_completion = [0.0] * num_jobs
    for (job_id, op_id, machine_id, start, end) in schedule:
        if job_id < num_jobs:
            job_completion[job_id] = max(job_completion[job_id], end)
    
    return max(job_completion) if job_completion else 0.0


def calculate_load_variance(machine_loads: List[float]) -> float:
    """
    计算机器负荷方差
    
    参数:
        machine_loads: list, 各机器的总负荷
    
    返回:
        variance: float, 负荷方差
    """
    if not machine_loads:
        return 0.0
    return np.var(machine_loads)


def calculate_load_balance(machine_loads: List[float]) -> float:
    """
    计算机器负荷均衡度（标准差/均值，即变异系数）
    
    参数:
        machine_loads: list, 各机器的总负荷
    
    返回:
        cv: float, 变异系数（越小越均衡）
    """
    if not machine_loads:
        return 0.0
    mean_load = np.mean(machine_loads)
    if mean_load == 0:
        return 0.0
    return np.std(machine_loads) / mean_load


def calculate_machine_utilization(machine_loads: List[float], makespan: float) -> List[float]:
    """
    计算机器利用率
    
    参数:
        machine_loads: list, 各机器的总负荷
        makespan: float, 最大完工时间
    
    返回:
        utilizations: list, 各机器的利用率 (0~1)
    """
    if makespan <= 0:
        return [0.0] * len(machine_loads)
    return [load / makespan for load in machine_loads]


def calculate_average_utilization(machine_loads: List[float], makespan: float) -> float:
    """
    计算平均机器利用率
    
    参数:
        machine_loads: list, 各机器的总负荷
        makespan: float, 最大完工时间
    
    返回:
        avg_util: float, 平均利用率
    """
    utilizations = calculate_machine_utilization(machine_loads, makespan)
    return np.mean(utilizations) if utilizations else 0.0


def calculate_total_idle_time(machine_loads: List[float], makespan: float) -> float:
    """
    计算机器总空闲时间
    
    参数:
        machine_loads: list, 各机器的总负荷
        makespan: float, 最大完工时间
    
    返回:
        total_idle: float, 总空闲时间
    """
    if makespan <= 0:
        return 0.0
    total_available = makespan * len(machine_loads)
    total_used = sum(machine_loads)
    return total_available - total_used


def calculate_tardiness(schedule: List[Tuple], job_due_dates: List[float], num_jobs: int) -> float:
    """
    计算总拖期时间（如果有交货期）
    
    参数:
        schedule: 调度方案
        job_due_dates: list, 各工件的交货期
        num_jobs: 工件数量
    
    返回:
        total_tardiness: float, 总拖期时间
    """
    if not schedule or not job_due_dates:
        return 0.0
    
    # 计算每个工件的完工时间
    job_completion = [0.0] * num_jobs
    for (job_id, op_id, machine_id, start, end) in schedule:
        if job_id < num_jobs:
            job_completion[job_id] = max(job_completion[job_id], end)
    
    # 计算总拖期
    total_tardiness = 0.0
    for j in range(num_jobs):
        if j < len(job_due_dates):
            tardiness = max(0, job_completion[j] - job_due_dates[j])
            total_tardiness += tardiness
    
    return total_tardiness


def calculate_flow_time(schedule: List[Tuple], num_jobs: int) -> Tuple[float, float]:
    """
    计算工件流程时间（完工时间 - 到达时间，假设到达时间为0）
    
    参数:
        schedule: 调度方案
        num_jobs: 工件数量
    
    返回:
        (total_flow_time, average_flow_time)
    """
    if not schedule:
        return (0.0, 0.0)
    
    # 计算每个工件的完工时间
    job_completion = [0.0] * num_jobs
    for (job_id, op_id, machine_id, start, end) in schedule:
        if job_id < num_jobs:
            job_completion[job_id] = max(job_completion[job_id], end)
    
    total_flow = sum(job_completion)  # 假设到达时间为0
    avg_flow = total_flow / num_jobs if num_jobs > 0 else 0.0
    
    return (total_flow, avg_flow)


def evaluate_schedule(schedule: List[Tuple], num_jobs: int, num_machines: int) -> Dict[str, Any]:
    """
    综合评估调度方案的各项指标
    
    参数:
        schedule: 调度方案，格式为 [(job_id, op_id, machine_id, start, end), ...]
        num_jobs: 工件数量
        num_machines: 机器数量
    
    返回:
        metrics: dict, 包含各项指标的字典
    """
    if not schedule:
        return {
            "makespan": 0.0,
            "load_variance": 0.0,
            "load_balance_cv": 0.0,
            "avg_utilization": 0.0,
            "total_idle_time": 0.0,
            "machine_loads": [0.0] * num_machines,
            "machine_utilizations": [0.0] * num_machines
        }
    
    # 计算机器负荷
    machine_loads = [0.0] * num_machines
    for (job_id, op_id, machine_id, start, end) in schedule:
        if 0 <= machine_id < num_machines:
            duration = end - start
            machine_loads[machine_id] += duration
    
    # 计算makespan
    makespan = calculate_makespan(schedule, num_jobs)
    
    # 计算各项指标
    metrics = {
        "makespan": makespan,
        "load_variance": calculate_load_variance(machine_loads),
        "load_balance_cv": calculate_load_balance(machine_loads),
        "avg_utilization": calculate_average_utilization(machine_loads, makespan),
        "total_idle_time": calculate_total_idle_time(machine_loads, makespan),
        "machine_loads": machine_loads,
        "machine_utilizations": calculate_machine_utilization(machine_loads, makespan)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "调度评估结果"):
    """
    格式化打印评估指标
    
    参数:
        metrics: dict, 评估指标字典
        title: str, 标题
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  最大完工时间 (Cmax): {metrics['makespan']:.2f}")
    print(f"  负荷方差: {metrics['load_variance']:.2f}")
    print(f"  负荷均衡度 (CV): {metrics['load_balance_cv']:.4f}")
    print(f"  平均利用率: {metrics['avg_utilization']:.2%}")
    print(f"  总空闲时间: {metrics['total_idle_time']:.2f}")
    print(f"\n  各机器负荷: {[f'{x:.1f}' for x in metrics['machine_loads']]}")
    print(f"  各机器利用率: {[f'{x:.1%}' for x in metrics['machine_utilizations']]}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 测试代码
    test_schedule = [
        (0, 0, 0, 0, 3),
        (1, 0, 1, 0, 5),
        (0, 1, 0, 3, 5),
    ]
    
    metrics = evaluate_schedule(test_schedule, num_jobs=2, num_machines=2)
    print_metrics(metrics, "测试调度评估")
