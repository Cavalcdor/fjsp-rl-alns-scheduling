"""
融合强化学习与自适应大邻域搜索的柔性车间滚动调度系统
主程序入口

用法:
  python main.py                            静态 GA 模式 (默认 Mk01)
  python main.py Mk02                       静态模式跑 Mk02
  python main.py all                        跑所有 Mk01-Mk10
  python main.py rolling_horizon            滚动时域模式 (默认 Mk01)
  python main.py static_ga Mk05             静态模式跑 Mk05
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
from core.rolling_horizon import RollingHorizon
import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_instance_path(name=None):
    if name is None:
        return config.INSTANCE_PATH
    return os.path.join(config.DATA_ROOT, name)


def run_all_static():
    """跑全部 Mk01-Mk10"""
    print("=" * 75)
    print("Brandimarte Mk 系列批量测试（静态 GA）")
    print("=" * 75)
    print(" %-8s %4s %4s %5s %8s %9s %8s" % (
        "实例", "工件", "机器", "工序", "GA_Cmax", "实际Cmax", "负荷方差"))
    print("-" * 75)

    results = []
    t0 = time.time()
    for i in range(1, 11):
        name = "Mk%02d.fjs" % i
        set_seed(config.RANDOM_SEED)
        p = get_instance_path(name)
        jobs_raw, nm, nj = load_fjsp_from_file(p)
        jobs = convert_to_zero_index(jobs_raw)

        old_v = config.VERBOSE
        config.VERBOSE = False
        ga = GA(jobs, nm, config)
        best = ga.run(rl_controller=RLController(config),
                      alns=ALNS(jobs, nm, config))
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

        ac = max(jt)
        lv = np.var(ml)
        total_ops = sum(len(j) for j in jobs)
        results.append((name, nj, nm, total_ops, best["cmax"], ac, lv))
        print(" %-8s %4d %4d %5d %8d %9.0f %8.1f" % (
            name, nj, nm, total_ops, best["cmax"], ac, lv))

    print("-" * 75)
    # 汇总表
    if results:
        print("\n%-8s %8s %8s %8s" % ("实例", "GA_Cmax", "实际Cmax", "负荷方差"))
        print("-" * 36)
        for r in results:
            print("%-8s %8d %8.0f %8.1f" % (r[0], r[4], r[5], r[6]))
        print("-" * 36)
    print("总耗时: %.1fs" % (time.time() - t0))


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
        print("\n最终完工时间: %.2f" % rh.current_time)

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
        print("\n=== 静态 GA 模式 ===")
        ga = GA(jobs, nm, config)
        best = ga.run(rl_controller=RLController(config),
                      alns=ALNS(jobs, nm, config))
        print("\nGA 最优 Cmax: %.2f, 负荷方差: %.2f" % (best["cmax"], best["load_var"]))

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
