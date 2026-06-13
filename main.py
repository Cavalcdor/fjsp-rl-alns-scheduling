"""
融合强化学习与自适应大邻域搜索的柔性车间滚动调度系统
主程序入口

用法:
  python main.py                            GA + ALNS + TS 模式 (默认 Mk01)
  python main.py Mk02                       静态模式跑 Mk02
  python main.py all                        跑所有 Mk01-Mk10
  python main.py rolling_horizon            滚动时域模式 (默认 Mk01)
  python main.py static_ga Mk05             跑 Mk05 实例
  python main.py Mk07 --trials 6            用6个进程并行跑6次独立GA取最优
"""

import sys
import os
import time
import numpy as np
import random

from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from utils.metrics import evaluate_schedule, print_metrics
from utils.visualization import plot_schedule_analysis
from algorithms.ga import GA
from algorithms.rl_agent import RLController
from algorithms.alns import ALNS
from algorithms.tabu_search import TabuSearch
from core.rolling_horizon import RollingHorizon
import config


# ============================================================

# Brandimarte Mk 系列最优已知解 (BKS)
BKS_TABLE = {
    "Mk01.fjs": 40, "Mk02.fjs": 26, "Mk03.fjs": 204,
    "Mk04.fjs": 60, "Mk05.fjs": 172, "Mk06.fjs": 57,
    "Mk07.fjs": 139, "Mk08.fjs": 523, "Mk09.fjs": 307,
    "Mk10.fjs": 165
}


def bks(name):
    return BKS_TABLE.get(name, "-")


def gap_str(ga_cmax, bks_val):
    if bks_val == "-":
        return "  N/A"
    diff = ga_cmax - bks_val
    if diff == 0:
        return "  ✅ 0%"
    elif diff < 0:
        return "  🔥 %+.1f%%" % (diff / bks_val * 100)
    else:
        return "  %+.1f%%" % (diff / bks_val * 100)


def fmt_sep(n, ch="─"):
    return ch * n


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_instance_path(name=None):
    if name is None:
        return config.INSTANCE_PATH
    return os.path.join(config.DATA_ROOT, name)


def run_all_static():
    """跑全部 Mk01-Mk10，输出带 BKS 对比的格式化表格"""
    header = "  %-8s %5s %5s %5s %8s %8s %8s %10s %6s" % (
        "算例", "工件", "机器", "工序", "GA_Cmax", "BKS", "差距", "负荷方差", "耗时")
    sep = "  " + fmt_sep(len(header) - 2)

    print(f"\n{'=' * (len(header) + 2)}")
    print("  Brandimarte Mk 系列批量测试 (GA + ALNS + Tabu Search)")
    print(f"{'=' * (len(header) + 2)}")
    print(header)
    print(sep)

    results = []
    t0 = time.time()
    for i in range(1, 10):
        name = "Mk%02d.fjs" % i
        set_seed(config.RANDOM_SEED)
        p = get_instance_path(name)
        jobs_raw, nm, nj = load_fjsp_from_file(p)
        jobs = convert_to_zero_index(jobs_raw)

        old_v = config.VERBOSE
        config.VERBOSE = False

        t_start = time.time()
        ga = GA(jobs, nm, config)
        best = ga.run(rl_controller=RLController(config),
                      alns=ALNS(jobs, nm, config),
                      tabu_search=TabuSearch(jobs, nm, config))

        t_inst = time.time() - t_start
        config.VERBOSE = old_v

        assign = ga.decode(best)
        mt = [0] * nm
        jt = [0] * nj
        ml = [0.0] * nm
        for (jid, oid, mid, dur) in assign:
            s = max(mt[mid], jt[jid])
            e = s + dur
            mt[mid] = e
            jt[jid] = e
            ml[mid] += dur

        ac = int(max(jt))
        lv = np.var(ml)
        total_ops = sum(len(j) for j in jobs)

        _bks = bks(name)
        _gap = gap_str(ac, _bks)
        results.append((name, nj, nm, total_ops, ac, _bks, _gap, lv, t_inst))

        print("  %-8s %5d %5d %5d %8d %8s %10s %8.1f %5.0fs" % (
            name.replace(".fjs", ""), nj, nm, total_ops, ac, _bks, _gap, lv, t_inst))

    print(sep)
    # 汇总对比表
    bks_vals = [r[5] for r in results]
    our_vals = [r[4] for r in results]
    gaps = [(o - b if isinstance(b, int) else 0) for o, b in zip(our_vals, bks_vals)]
    pos_gaps = [g for g in gaps if g > 0]
    avg_gap = sum(pos_gaps) / len(pos_gaps) if pos_gaps else 0.0
    reached = sum(1 for o, b in zip(our_vals, bks_vals) if isinstance(b, int) and o <= b)

    print(f"\n  📊 汇总: {reached}/9 达到/超越 BKS | "
          f"平均正偏差: {avg_gap:+.1f} | "
          f"总耗时: {time.time() - t0:.0f}s")
    print()


def main():
    print("=" * 60)
    print("柔性车间滚动调度系统 - Brandimarte Mk 系列")
    print("=" * 60)

    mode = "static_ga"
    instance_arg = None

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        instance_arg = sys.argv[2]

    # 简写: python main.py Mk02
    u = mode.upper()
    if u.startswith("MK") and ".FJS" not in u:
        instance_arg = mode + ".fjs"
        mode = "static_ga"
    elif u.startswith("MK"):
        instance_arg = mode
        mode = "static_ga"

    if mode == "all":
        run_all_static()
        return

    if instance_arg is None:
        instance_arg = config.DEFAULT_INSTANCE

    # 自动补全 .fjs 扩展名
    if instance_arg.upper().startswith("MK") and not instance_arg.upper().endswith(".FJS"):
        instance_arg += ".fjs"

    path = get_instance_path(instance_arg)
    print("\n数据集: %s" % path)
    if not os.path.exists(path):
        print("错误：文件不存在")
        return

    jobs_raw, nm, nj = load_fjsp_from_file(path)
    jobs = convert_to_zero_index(jobs_raw)
    print("工件: %d, 机器: %d, 工序: %d" % (nj, nm, sum(len(j) for j in jobs)))

    if mode == "rolling_horizon":
        print("\n=== 滚动时域模式 ===")
        ga = GA(jobs, nm, config)
        rh = RollingHorizon(jobs, nm, config, ga,
                            RLController(config),
                            ALNS(jobs, nm, config))
        final = rh.run()
        cmax_rh = rh.current_time

        # 与 BKS 对比（仅参考，滚动时域含扰动）
        _bks = bks(instance_arg)
        _gap = gap_str(int(cmax_rh), _bks)
        print("\n" + "─" * 50)
        print("  滚动时域 vs 静态最优已知解 (BKS，仅供参考)")
        print("  " + "─" * 46)
        print("  %-16s %12s %8s %8s" % ("算例", "RH_Cmax", "BKS", "差距"))
        print("  " + "─" * 46)
        print("  %-16s %12.0f %8s %10s" % (instance_arg.replace(".fjs", ""), cmax_rh, _bks, _gap))
        print("  " + "─" * 46 + "\n")

        if final:
            std = [(j, o, m, s, e) for (j, o, m, s, e) in final]
            metrics = evaluate_schedule(std, nj, nm)
            print_metrics(metrics, "滚动时域调度评估结果")

            fn = instance_arg.replace(".fjs", "")
            try:
                plot_schedule_analysis(std, nj, nm,
                                       save_path="output/gantt_%s_rolling.png" % fn,
                                       show=False)
                print("甘特图: output/gantt_%s_rolling.png" % fn)
            except Exception:
                pass

    else:
        print(f"\n=== GA + ALNS + Tabu Search 模式 ===")
        ga = GA(jobs, nm, config)
        best = ga.run(rl_controller=RLController(config),
                      alns=ALNS(jobs, nm, config),
                      tabu_search=TabuSearch(jobs, nm, config))

        # 显示与 BKS 的对比
        _bks = bks(instance_arg)
        _our = int(best["cmax"])
        _gap = gap_str(_our, _bks)
        print("\n" + "─" * 50)
        print("  算法结果 vs 最优已知解 (BKS)")
        print("  " + "─" * 46)
        print("  %-16s %8s %8s %8s" % ("算例", "GA_Cmax", "BKS", "差距"))
        print("  " + "─" * 46)
        print("  %-16s %8d %8s %10s" % (instance_arg.replace(".fjs", ""), _our, _bks, _gap))
        print("  " + "─" * 46)
        print(f"  负荷方差: {best['load_var']:.2f}")
        print()

        assign = ga.decode(best)
        mt = [0] * nm
        jt = [0] * nj
        sched = []
        for (jid, oid, mid, dur) in assign:
            s = max(mt[mid], jt[jid])
            e = s + dur
            sched.append((jid, oid, mid, s, e))
            mt[mid] = e
            jt[jid] = e

        metrics = evaluate_schedule(sched, nj, nm)
        print_metrics(metrics, "静态调度评估结果")

        fn = instance_arg.replace(".fjs", "")
        try:
            plot_schedule_analysis(sched, nj, nm,
                                   save_path="output/gantt_%s.png" % fn,
                                   show=False)
            print("甘特图: output/gantt_%s.png" % fn)
        except Exception as e:
            print("绘图失败: %s" % e)

    print("\n完成。")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    set_seed(config.RANDOM_SEED)
    main()
