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


# ════════════════════════════════════════════════════════════
# 批量运行汇总可视化 (v1.4.0)
# ════════════════════════════════════════════════════════════

def plot_batch_summary_all(results_by_family, save_dir="output/summary"):
    """
    批量运行结束后，生成全套汇总图表 + CSV。

    results_by_family: dict，格式 {family_key: (title, [result_dict, ...])}
        其中 result_dict 包含:
            label, nj, nm, total_ops, cmax, bks_val, gap_str, load_var, runtime
    save_dir: 输出目录
    """
    import os, csv
    os.makedirs(save_dir, exist_ok=True)

    all_flat = []       # 用于 CSV
    family_stats = {}   # 用于 overall 对比

    # 1) 逐族画对比图
    for family_key, (title, results) in results_by_family.items():
        if not results:
            continue
        plot_per_family_comparison(results, family_key, title, save_dir)

        # 收集 flat 行
        for r in results:
            all_flat.append((family_key, title, r["label"], r["nj"], r["nm"],
                             r["total_ops"], r["cmax"], r["bks_val"],
                             r.get("gap_numeric", None), r["load_var"], r["runtime"]))

        # 族级统计
        bks_vals = [r["bks_val"] for r in results]
        cmax_vals = [r["cmax"] for r in results]
        reached = sum(1 for c, b in zip(cmax_vals, bks_vals)
                      if b is not None and c <= b)
        total_with_bks = sum(1 for b in bks_vals if b is not None)
        pos_gaps = [((c - b) / b * 100) for c, b in zip(cmax_vals, bks_vals)
                    if b is not None and b > 0 and c > b]
        avg_gap = sum(pos_gaps) / len(pos_gaps) if pos_gaps else 0.0
        total_runtime = sum(r["runtime"] for r in results)
        family_stats[family_key] = {
            "title": title,
            "total": len(results),
            "reached": reached,
            "total_bks": total_with_bks,
            "avg_gap": avg_gap,
            "runtime": total_runtime,
        }

    # 2) 跨族总览图
    plot_overall_comparison(family_stats, save_dir)

    # 3) CSV
    _save_results_csv(all_flat, save_dir)

    print(f"\n📊 汇总图表已输出至 {save_dir}/")
    return save_dir


def plot_per_family_comparison(results, family_key, family_title, save_dir):
    """
    单族对比图：GA_Cmax 柱状图 + BKS 菱形标记 + 差距标注
    柱色: 绿色=达BKS，红色=未达
    """
    import os
    n = len(results)
    if n == 0:
        return

    labels = [r["label"] for r in results]
    cmax_vals = [r["cmax"] for r in results]
    bks_vals = [r["bks_val"] for r in results]
    runtimes = [r["runtime"] for r in results]

    # 数值 gap
    gaps = []
    for c, b in zip(cmax_vals, bks_vals):
        if b is not None and b > 0:
            gaps.append((c - b) / b * 100)
        else:
            gaps.append(None)

    reached = sum(1 for c, b in zip(cmax_vals, bks_vals)
                  if b is not None and c <= b)
    total_bks = sum(1 for b in bks_vals if b is not None)
    pos_gaps = [g for g in gaps if g is not None and g > 0]
    avg_gap = sum(pos_gaps) / len(pos_gaps) if pos_gaps else 0.0

    # 自适应图宽
    fig_w = max(10, min(n * 0.55, 28))
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    x = range(n)
    bar_colors = []
    for c, b in zip(cmax_vals, bks_vals):
        bar_colors.append('#2ca02c' if (b is not None and c <= b) else '#d62728')

    bar_width = min(0.8, 12 / max(n, 1))
    bars = ax.bar(x, cmax_vals, width=bar_width, color=bar_colors,
                  alpha=0.85, edgecolor='black', linewidth=0.5, zorder=3)

    # BKS 菱形标记
    bks_x, bks_y = [], []
    for i, b in enumerate(bks_vals):
        if b is not None:
            bks_x.append(i)
            bks_y.append(b)
    if bks_x:
        ax.scatter(bks_x, bks_y, color='#1a3a5c', s=50, zorder=5,
                   marker='D', facecolors='none', linewidths=1.8,
                   label=f'BKS (最优已知解)')

    # 差距标注
    y_max = max(cmax_vals) if cmax_vals else 1
    for i, (c, g) in enumerate(zip(cmax_vals, gaps)):
        if g is not None:
            color = '#1b5e20' if g <= 0 else '#b71c1c'
            ax.text(i, c + y_max * 0.015, f'{g:+.1f}%',
                    ha='center', va='bottom', fontsize=6.5,
                    color=color, fontweight='bold', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Makespan (Cmax)', fontsize=11)
    ax.set_title(
        f'{family_title}  —  GA vs BKS 对比\n'
        f'  BKS 达成: {reached}/{total_bks}  |  '
        f'平均正偏差: {avg_gap:+.1f}%  |  '
        f'总耗时: {sum(runtimes):.0f}s  |  '
        f'共 {n} 算例',
        fontsize=12, fontweight='bold', linespacing=1.4)
    if bks_x:
        ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'family_{family_key}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  族级对比图: {save_path}")


def plot_overall_comparison(family_stats, save_dir):
    """
    跨族总览图：三面板 — BKS达标率 / 平均正偏差 / 总耗时
    """
    import os
    if not family_stats:
        return

    keys = list(family_stats.keys())
    labels = [_friendly_name(k) for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── 面板1: BKS 达标率 ──
    ax = axes[0]
    rates = []
    for k in keys:
        s = family_stats[k]
        rates.append(s["reached"] / max(s["total_bks"], 1) * 100)
    colors1 = ['#2ca02c' if r >= 80 else '#ffc107' if r >= 50 else '#d62728'
               for r in rates]
    bars = ax.bar(range(len(keys)), rates, color=colors1, alpha=0.85,
                  edgecolor='black', linewidth=0.5, width=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('BKS 达标率 (%)', fontsize=11)
    ax.set_title('BKS 达标率', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ── 面板2: 平均正偏差 ──
    ax = axes[1]
    avg_gaps = [family_stats[k]["avg_gap"] for k in keys]
    colors2 = ['#b71c1c' if g > 5 else '#ff8f00' if g > 0 else '#2ca02c'
               for g in avg_gaps]
    bars = ax.bar(range(len(keys)), avg_gaps, color=colors2, alpha=0.85,
                  edgecolor='black', linewidth=0.5, width=0.6)
    for bar, gap in zip(bars, avg_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{gap:+.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('平均正偏差 (%)', fontsize=11)
    ax.set_title('未达 BKS 算例的平均偏差', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # ── 面板3: 总耗时 ──
    ax = axes[2]
    runtimes = [family_stats[k]["runtime"] / 60 for k in keys]  # 分钟
    colors3 = plt.cm.Blues(np.linspace(0.4, 0.9, len(keys)))
    bars = ax.bar(range(len(keys)), runtimes, color=colors3, alpha=0.85,
                  edgecolor='black', linewidth=0.5, width=0.6)
    for bar, rt in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{rt:.1f}min', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('总耗时 (分钟)', fontsize=11)
    ax.set_title('各数据集运行耗时', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.suptitle('跨数据集汇总 — 算法性能总览', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'overall_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  跨族总览图: {save_path}")


def _save_results_csv(all_flat, save_dir):
    """保存全部结果为 CSV，方便贴进报告/Excel"""
    import os, csv
    filepath = os.path.join(save_dir, 'results.csv')
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["数据集族", "族名", "算例", "工件", "机器", "工序",
                     "GA_Cmax", "BKS", "偏差_%", "负荷方差", "耗时_s"])
        w.writerows(all_flat)
    print(f"  CSV 结果表: {filepath}")


def _friendly_name(key):
    """将内部 key 转为展示名"""
    mapping = {
        "mk": "Brandimarte Mk",
        "barnes": "Barnes",
        "dauzere": "Dauzère",
        "hurink_car": "Hurink car",
        "hurink_ft": "Hurink ft",
        "hurink_orb": "Hurink orb",
    }
    return mapping.get(key, key)


# ════════════════════════════════════════════════════════════
# 新增强化可视化 (v2.0)
# ════════════════════════════════════════════════════════════

def plot_convergence_curves_batch(results, title="收敛曲线", save_dir="output/summary"):
    """
    将同一批内所有算例的收敛曲线画在同一张图上（子图网格）。
    results: 每个元素包含 label, best_history, avg_history
    """
    import math, os

    valid = [r for r in results if r.get("best_history") and len(r["best_history"]) > 1]
    if not valid:
        return

    n = len(valid)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx, r in enumerate(valid):
        ax = axes_flat[idx]
        best_h = r["best_history"]
        avg_h = r["avg_history"]
        gens = range(1, len(best_h) + 1)

        ax.plot(gens, best_h, 'b-', linewidth=1.8, label='最优 Cmax')
        ax.plot(gens, avg_h, 'r--', linewidth=1.2, alpha=0.6, label='平均 Cmax')
        ax.fill_between(gens, best_h, avg_h, alpha=0.08, color='gray')

        # 标注最终值
        ax.axhline(y=best_h[-1], color='green', linestyle=':', linewidth=0.8)
        ax.text(len(best_h) * 0.7, best_h[-1] + (max(best_h) - min(best_h)) * 0.02,
                f'{best_h[-1]:.0f}', fontsize=8, color='green', fontweight='bold')

        ax.set_title(f'{r["label"]}  (Cmax={r["cmax"]})', fontsize=10, fontweight='bold')
        ax.set_xlabel('迭代次数', fontsize=8)
        ax.set_ylabel('Cmax', fontsize=8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=7)

    # 隐藏多余子图
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'convergence_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  收敛曲线图: {save_path}")


def plot_global_gap_scatter(results_by_family, save_dir="output/summary"):
    """
    全局 Gap 散点图：x=工序数, y=偏差%, 不同族用不同颜色
    直观看出算法在不同规模算例上的表现
    """
    import os
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_family)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    for idx, (family_key, (title, results)) in enumerate(results_by_family.items()):
        valid = [r for r in results
                 if r.get("total_ops", 0) > 0
                 and r.get("gap_numeric") is not None
                 and r.get("cmax", 0) > 0]
        if not valid:
            continue

        xs = [r["total_ops"] for r in valid]
        ys = [r["gap_numeric"] for r in valid]
        labels = [r["label"] for r in valid]

        marker = markers[idx % len(markers)]
        scatter = ax.scatter(xs, ys, c=[colors[idx]], s=80, marker=marker,
                             alpha=0.75, edgecolors='black', linewidth=0.5,
                             zorder=3, label=title)

        # 标签
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=6.5, alpha=0.8)

    ax.axhline(y=0, color='green', linestyle='-', linewidth=1.2, alpha=0.7, label='BKS 基准线')
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('工序数 (Total Operations)', fontsize=12)
    ax.set_ylabel('偏差 (Gap, %)', fontsize=12)
    ax.set_title('全局性能散点图 — 偏差 vs 算例规模', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'global_gap_scatter.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  全局 Gap 散点图: {save_path}")


def plot_scale_vs_runtime(results_by_family, save_dir="output/summary"):
    """
    规模-耗时散点图：x=工序数, y=耗时(秒), 点大小=Cmax
    直观看出算例复杂度与计算时间的关系
    """
    import os
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_by_family)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    for idx, (family_key, (title, results)) in enumerate(results_by_family.items()):
        valid = [r for r in results
                 if r.get("total_ops", 0) > 0
                 and r.get("runtime", 0) > 0
                 and r.get("cmax", 0) > 0]
        if not valid:
            continue

        xs = [r["total_ops"] for r in valid]
        ys = [r["runtime"] for r in valid]
        sizes = [max(20, r["cmax"] / 10) for r in valid]
        labels = [r["label"] for r in valid]

        marker = markers[idx % len(markers)]
        scatter = ax.scatter(xs, ys, s=sizes, c=[colors[idx]], marker=marker,
                             alpha=0.7, edgecolors='black', linewidth=0.5,
                             zorder=3, label=title)

        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=6.5, alpha=0.8)

    # 趋势线
    all_x = []
    all_y = []
    for _, (_, results) in results_by_family.items():
        for r in results:
            if r.get("total_ops", 0) > 0 and r.get("runtime", 0) > 0:
                all_x.append(r["total_ops"])
                all_y.append(r["runtime"])
    if len(all_x) >= 3:
        z = np.polyfit(all_x, all_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_x), max(all_x), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1, alpha=0.5,
                label=f'线性趋势 (斜率={z[0]:.2f}s/单位工序)')

    ax.set_xlabel('工序数 (Total Operations)', fontsize=12)
    ax.set_ylabel('运行时间 (秒)', fontsize=12)
    ax.set_title('算例规模 vs 运行时间 (点大小 = Cmax)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'scale_vs_runtime.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  规模-耗时散点图: {save_path}")


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
