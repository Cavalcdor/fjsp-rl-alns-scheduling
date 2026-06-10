# utils/visualization.py
"""
调度结果可视化模块
提供甘特图绘制和调度分析图表功能
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple, Dict, Optional


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_gantt_chart(schedule: List[Tuple], 
                     num_jobs: int, 
                     num_machines: int,
                     title: str = "调度甘特图",
                     save_path: Optional[str] = None,
                     show: bool = True):
    """
    绘制调度甘特图
    
    参数:
        schedule: 调度方案，格式为 [(job_id, op_id, machine_id, start, end), ...]
        num_jobs: 工件数量
        num_machines: 机器数量
        title: str, 图表标题
        save_path: str, 保存路径（可选）
        show: bool, 是否显示图表
    """
    if not schedule:
        print("调度方案为空，无法绘制甘特图")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, max(6, num_machines * 0.8)))
    
    # 为每个工件分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    
    # 绘制每个工序
    for (job_id, op_id, machine_id, start, end) in schedule:
        duration = end - start
        # 绘制矩形
        rect = Rectangle(
            (start, machine_id), 
            duration, 1, 
            facecolor=colors[job_id],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # 添加文本标签
        if duration > 0.5:  # 只在足够宽时显示文本
            ax.text(
                start + duration / 2, 
                machine_id + 0.5,
                f"J{job_id}\nO{op_id}",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold'
            )
    
    # 设置坐标轴
    ax.set_xlim(0, max(end for (_, _, _, _, end) in schedule) * 1.05)
    ax.set_ylim(0, num_machines)
    ax.set_yticks([i + 0.5 for i in range(num_machines)])
    ax.set_yticklabels([f"机器 {i}" for i in range(num_machines)])
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("机器", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加图例
    legend_patches = [mpatches.Patch(color=colors[i], label=f"工件 {i}") 
                      for i in range(num_jobs)]
    ax.legend(handles=legend_patches, 
              loc='upper right', 
              ncol=min(5, num_jobs),
              fontsize=8)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"甘特图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_machine_load(machine_loads: List[float], 
                      title: str = "机器负荷分布",
                      save_path: Optional[str] = None,
                      show: bool = True):
    """
    绘制机器负荷分布图
    
    参数:
        machine_loads: list, 各机器的总负荷
        title: str, 图表标题
        save_path: str, 保存路径（可选）
        show: bool, 是否显示图表
    """
    if not machine_loads:
        return
    
    num_machines = len(machine_loads)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    x = range(num_machines)
    bars = ax.bar(x, machine_loads, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, load in zip(bars, machine_loads):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{load:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # 添加平均线
    avg_load = np.mean(machine_loads)
    ax.axhline(y=avg_load, color='red', linestyle='--', linewidth=2, 
               label=f'平均负荷: {avg_load:.1f}')
    
    ax.set_xlabel("机器", fontsize=12)
    ax.set_ylabel("负荷（时间单位）", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i}" for i in range(num_machines)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"负荷分布图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_curve(history: List[float], 
                           title: str = "算法收敛曲线",
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    绘制算法收敛曲线
    
    参数:
        history: list, 每代的最优适应度值
        title: str, 图表标题
        save_path: str, 保存路径（可选）
        show: bool, 是否显示图表
    """
    if not history:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(1, len(history) + 1)
    ax.plot(generations, history, 'b-', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel("迭代次数", fontsize=12)
    ax.set_ylabel("最优 Cmax", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注最优值
    best_idx = np.argmin(history)
    best_val = history[best_idx]
    ax.annotate(f'最优: {best_val:.2f}', 
                xy=(best_idx + 1, best_val),
                xytext=(best_idx + 1 + len(history) * 0.1, best_val + (max(history) - min(history)) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"收敛曲线已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_schedule_analysis(schedule: List[Tuple], 
                           num_jobs: int, 
                           num_machines: int,
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    绘制调度分析综合图（甘特图 + 负荷分布 + 工件完工时间）
    
    参数:
        schedule: 调度方案
        num_jobs: 工件数量
        num_machines: 机器数量
        save_path: str, 保存路径（可选）
        show: bool, 是否显示图表
    """
    if not schedule:
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 甘特图（上方）
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    
    for (job_id, op_id, machine_id, start, end) in schedule:
        duration = end - start
        rect = Rectangle(
            (start, machine_id), 
            duration, 1, 
            facecolor=colors[job_id],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax1.add_patch(rect)
        
        if duration > 0.5:
            ax1.text(
                start + duration / 2, 
                machine_id + 0.5,
                f"J{job_id}",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold'
            )
    
    makespan = max(end for (_, _, _, _, end) in schedule)
    ax1.set_xlim(0, makespan * 1.05)
    ax1.set_ylim(0, num_machines)
    ax1.set_yticks([i + 0.5 for i in range(num_machines)])
    ax1.set_yticklabels([f"机器 {i}" for i in range(num_machines)])
    ax1.set_xlabel("时间", fontsize=11)
    ax1.set_ylabel("机器", fontsize=11)
    ax1.set_title("调度甘特图", fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 2. 机器负荷分布（左下）
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    machine_loads = [0] * num_machines
    for (job_id, op_id, machine_id, start, end) in schedule:
        machine_loads[machine_id] += (end - start)
    
    x = range(num_machines)
    ax2.bar(x, machine_loads, color='steelblue', alpha=0.7, edgecolor='black')
    avg_load = np.mean(machine_loads)
    ax2.axhline(y=avg_load, color='red', linestyle='--', linewidth=1.5, 
                label=f'平均: {avg_load:.1f}')
    ax2.set_xlabel("机器", fontsize=10)
    ax2.set_ylabel("负荷", fontsize=10)
    ax2.set_title("机器负荷分布", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"M{i}" for i in range(num_machines)], fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. 工件完工时间（右下）
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    job_completion = [0] * num_jobs
    for (job_id, op_id, machine_id, start, end) in schedule:
        job_completion[job_id] = max(job_completion[job_id], end)
    
    x_jobs = range(num_jobs)
    bars = ax3.bar(x_jobs, job_completion, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("工件", fontsize=10)
    ax3.set_ylabel("完工时间", fontsize=10)
    ax3.set_title("工件完工时间", fontsize=12, fontweight='bold')
    ax3.set_xticks(x_jobs)
    ax3.set_xticklabels([f"J{i}" for i in range(num_jobs)], fontsize=8)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, ct in zip(bars, job_completion):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{ct:.1f}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"调度分析图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 测试代码
    test_schedule = [
        (0, 0, 0, 0, 3),
        (0, 1, 0, 3, 5),
        (1, 0, 1, 0, 5),
        (2, 0, 1, 5, 8),
        (2, 1, 0, 5, 7),
    ]
    
    # 测试甘特图
    plot_gantt_chart(test_schedule, num_jobs=3, num_machines=2, show=False)
    
    # 测试负荷分布
    machine_loads = [10, 13, 8, 12]
    plot_machine_load(machine_loads, show=False)
    
    print("可视化模块测试完成")
