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
        # 静态模式：直接用 GA 求解整个问题（不引入扰动）
        print("\n=== 静态 GA 模式 ===")
        best = ga.run(rl_controller=rl, alns=alns)
        print(f"最优解 Cmax: {best['cmax']:.2f}")
        print(f"设备负荷方差: {best['load_var']:.2f}")
        # 可选：输出调度甘特图或详细结果
    else:
        # 滚动时域模式（带扰动、重调度）
        print("\n=== 滚动时域模式 ===")
        rh = RollingHorizon(jobs, num_machines, config, ga, rl, alns)
        final_schedule = rh.run()
        print(f"\n最终调度完成，总完工时间: {rh.current_time:.2f}")
    
    print("\n程序执行完毕。")

if __name__ == "__main__":
    set_seed(config.RANDOM_SEED)
    main()