# main.py
"""
融合强化学习与自适应大邻域搜索的滚动时域鲁棒优化算法
主程序入口
"""

import sys
import os
import numpy as np
import random

# 导入自定义模块
from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from utils.scheduler import evaluate
from utils.metrics import evaluate_schedule, print_metrics
from utils.visualization import plot_gantt_chart, plot_schedule_analysis
from algorithms.ga import GA
from algorithms.rl_agent import RLController
from algorithms.alns import ALNS
from core.rolling_horizon import RollingHorizon
import config

def set_seed(seed):
    """设置随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)

def main():
    print("=" * 60)
    print("融合强化学习与自适应大邻域搜索的柔性车间滚动调度系统")
    print("=" * 60)
    
    # 1. 加载数据集
    instance_path = config.INSTANCE_PATH
    print(f"\n加载数据集: {instance_path}")
    if not os.path.exists(instance_path):
        print(f"错误：数据集文件不存在 {instance_path}")
        print("请检查 config.py 中的 DATA_ROOT 和 DEFAULT_INSTANCE 设置")
        return
    
    jobs_raw, num_machines, num_jobs = load_fjsp_from_file(instance_path)
    jobs = convert_to_zero_index(jobs_raw)
    print(f"工件数: {num_jobs}, 机器数: {num_machines}")
    
    # 2. 初始化算法组件
    print("\n初始化算法组件...")
    
    # GA（先不传 RL 和 ALNS，稍后由 RollingHorizon 内部集成）
    ga = GA(jobs, num_machines, config)
    
    # RL 控制器（可选，如果配置中需要）
    rl = RLController(config)
    
    # ALNS 优化器（可选）
    alns = ALNS(jobs, num_machines, config)
    
    # 3. 选择运行模式：静态GA 或 滚动时域
    mode = "rolling_horizon"  # 可选 "static_ga" 或 "rolling_horizon"
    
    if mode == "static_ga":
        # 静态模式：不使用滚动时域机制，但仍使用GA+RL+ALNS混合优化求解整个问题
        print("\n=== 静态 GA+RL+ALNS 混合优化模式 ===")
        best = ga.run(rl_controller=rl, alns=alns)
        print(f"\n最优解 Cmax: {best['cmax']:.2f}")
        print(f"设备负荷方差: {best['load_var']:.2f}")
        
        # 解码得到 assignment (job_id, op_id, machine_id, duration)
        assignment = ga.decode(best)
        
        # 模拟调度，将 duration 转换为 (start, end) 时间
        mach_time = [0] * num_machines
        job_time = [0] * num_jobs
        standard_schedule = []
        for (job_id, op_id, machine_id, duration) in assignment:
            start = max(mach_time[machine_id], job_time[job_id])
            end = start + duration
            standard_schedule.append((job_id, op_id, machine_id, start, end))
            mach_time[machine_id] = end
            job_time[job_id] = end
        
        metrics = evaluate_schedule(standard_schedule, num_jobs, num_machines)
        print_metrics(metrics, "静态调度评估结果")
        
        # 绘制甘特图
        try:
            plot_schedule_analysis(standard_schedule, num_jobs, num_machines, 
                                   save_path="output/gantt_static.png", show=False)
            print("甘特图已保存至: output/gantt_static.png")
        except Exception as e:
            print(f"绘图失败: {e}")
    
    else:
        # 滚动时域模式（带扰动、重调度）
        print("\n=== 滚动时域模式 ===")
        rh = RollingHorizon(jobs, num_machines, config, ga, rl, alns)
        final_schedule = rh.run()
        print(f"\n最终调度完成，总完工时间: {rh.current_time:.2f}")
        
        # 评估最终调度
        if final_schedule:
            # 转换为标准格式
            standard_schedule = [(job_id, op_id, machine_id, start, end) 
                                 for (job_id, op_id, machine_id, start, end) in final_schedule]
            metrics = evaluate_schedule(standard_schedule, num_jobs, num_machines)
            print_metrics(metrics, "滚动时域调度评估结果")
            
            # 绘制甘特图
            try:
                plot_schedule_analysis(standard_schedule, num_jobs, num_machines,
                                       save_path="output/gantt_rolling.png", show=False)
                print("甘特图已保存至: output/gantt_rolling.png")
            except Exception as e:
                print(f"绘图失败: {e}")
        
        # 输出扰动记录
        if rh.disturbance_log:
            print(f"\n扰动记录（共 {len(rh.disturbance_log)} 次）:")
            for event in rh.disturbance_log:
                print(f"  时间 {event['time']:.2f}: {event['type']}")
    
    print("\n程序执行完毕。")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    set_seed(config.RANDOM_SEED)
    main()
