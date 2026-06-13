"""
融合强化学习与自适应大邻域搜索的柔性车间滚动调度系统
主程序入口 — 多数据集 CLI v1.4.0

用法:
  # ── 单例模式 ──
  python main.py run Mk01                    跑单个 Mk 算例
  python main.py run barnes mt10c1           跑单个 Barnes 算例
  python main.py run dauzere 01a             跑单个 Dauzère 算例
  python main.py run hurink_edata car1       跑单个 Hurink 算例 (指定变体)
  python main.py Mk01                        同上 (向后兼容)

  # ── 批量模式 ──
  python main.py batch mk                    Mk01-Mk09
  python main.py batch barnes                Barnes 全部 21 个
  python main.py batch dauzere               Dauzère 非 open 算例 (8个)
  python main.py batch hurink_car            Hurink car × edata/rdata/vdata (24个)
  python main.py batch hurink_ft             Hurink ft × edata/rdata/vdata (9个)
  python main.py batch hurink_orb            Hurink orb × edata/rdata/vdata (30个)
  python main.py batch all                   以上所有 101 个算例

  # ── 其他模式 ──
  python main.py all                         向后兼容，等价 batch mk
  python main.py rolling_horizon             滚动时域模式 (默认 Mk01)
  python main.py rolling_horizon Mk02        滚动时域模式 (Mk02)
  python main.py rolling_horizon barnes mt10c1  滚动时域模式 (Barnes)
  python main.py rolling_horizon hurink_edata car1  滚动时域 (Hurink)

参考基准: FJSPLib (https://github.com/FJSP/FJSPLib)
BKS 来源: OptalCP / CP Optimizer / Quintiq 等参考引擎
"""

import sys
import os
import time
import numpy as np
import random

from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from utils.metrics import evaluate_schedule, print_metrics
from utils.visualization import plot_schedule_analysis, plot_batch_summary_all
from algorithms.ga import GA
from algorithms.rl_agent import RLController
from algorithms.alns import ALNS
from algorithms.tabu_search import TabuSearch
from core.rolling_horizon import RollingHorizon
import config


# ============================================================
# BKS 表 — 来源于 FJSPLib (OptalCP 等参考引擎)
# 仅包含本地有文件、非 open 状态的算例
# ============================================================

MK_BKS = {
    "Mk01.fjs": 40, "Mk02.fjs": 26, "Mk03.fjs": 204,
    "Mk04.fjs": 60, "Mk05.fjs": 172, "Mk06.fjs": 57,
    "Mk07.fjs": 139, "Mk08.fjs": 523, "Mk09.fjs": 307,
}

BARNES_BKS = {
    "mt10c1.fjs": 927, "mt10cc.fjs": 908, "mt10x.fjs": 918,
    "mt10xx.fjs": 918, "mt10xxx.fjs": 918, "mt10xy.fjs": 905,
    "mt10xyz.fjs": 847,
    "setb4c9.fjs": 914, "setb4cc.fjs": 907, "setb4x.fjs": 925,
    "setb4xx.fjs": 925, "setb4xxx.fjs": 925, "setb4xy.fjs": 910,
    "setb4xyz.fjs": 902,
    "seti5c12.fjs": 1169, "seti5cc.fjs": 1135, "seti5x.fjs": 1198,
    "seti5xx.fjs": 1194, "seti5xxx.fjs": 1194, "seti5xy.fjs": 1135,
    "seti5xyz.fjs": 1125,
}

DAUZERE_BKS = {
    "01a.fjs": 2505, "02a.fjs": 2228, "03a.fjs": 2228, "04a.fjs": 2503,
    "08a.fjs": 2061, "09a.fjs": 2061,
    "14a.fjs": 2161, "15a.fjs": 2161,
}

# Hurink: edata / rdata / vdata 三组变体各自有不同 BKS
HURINK_CAR_BKS = {
    "car1": {"edata": 6176, "rdata": 5034, "vdata": 5005},
    "car2": {"edata": 6327, "rdata": 5985, "vdata": 5929},
    "car3": {"edata": 6856, "rdata": 5622, "vdata": 5597},
    "car4": {"edata": 7789, "rdata": 6514, "vdata": 6514},
    "car5": {"edata": 7229, "rdata": 5615, "vdata": 4909},
    "car6": {"edata": 7990, "rdata": 6147, "vdata": 5486},
    "car7": {"edata": 6123, "rdata": 4425, "vdata": 4281},
    "car8": {"edata": 7689, "rdata": 5692, "vdata": 4613},
}
HURINK_FT_BKS = {
    "mt06": {"edata": 55, "rdata": 47, "vdata": 47},
    "mt10": {"edata": 871, "rdata": 686, "vdata": 655},
    "mt20": {"edata": 1088, "rdata": 1022, "vdata": 1022},
}
HURINK_ORB_BKS = {
    "orb1": {"edata": 977, "rdata": 746, "vdata": 695},
    "orb2": {"edata": 865, "rdata": 696, "vdata": 620},
    "orb3": {"edata": 951, "rdata": 712, "vdata": 648},
    "orb4": {"edata": 984, "rdata": 753, "vdata": 753},
    "orb5": {"edata": 842, "rdata": 639, "vdata": 584},
    "orb6": {"edata": 958, "rdata": 754, "vdata": 715},
    "orb7": {"edata": 389, "rdata": 302, "vdata": 275},
    "orb8": {"edata": 894, "rdata": 639, "vdata": 573},
    "orb9": {"edata": 934, "rdata": 694, "vdata": 659},
    "orb10": {"edata": 933, "rdata": 742, "vdata": 681},
}


# ============================================================
# 数据集定义
# ============================================================

# 简单数据集：所有文件在同一个目录下
SIMPLE_DATASETS = {
    "mk": {
        "title": "Brandimarte Mk 系列",
        "path": "data/Brandimarte_Data",
        "files": [f"Mk{i:02d}.fjs" for i in range(1, 10)],
        "bks": MK_BKS,
    },
    "barnes": {
        "title": "Barnes (Chambers & Barnes 1996)",
        "path": "data/Barnes",
        "files": sorted(BARNES_BKS.keys()),
        "bks": BARNES_BKS,
    },
    "dauzere": {
        "title": "Dauzère (Dauzère-Pérès & Paulli 1994)",
        "path": "data/Dauzere_Data",
        "files": ["01a.fjs", "02a.fjs", "03a.fjs", "04a.fjs",
                   "08a.fjs", "09a.fjs", "14a.fjs", "15a.fjs"],
        "bks": DAUZERE_BKS,
    },
}

# Hurink 子族：多个变体子目录共享实例名，BKS 因变体而异
HURINK_FAMILIES = {
    "hurink_car": {
        "title": "Hurink car (Hurink et al. 1994)",
        "subdirs": ["edata", "rdata", "vdata"],
        "names": [f"car{i}" for i in range(1, 9)],
        "bks_map": HURINK_CAR_BKS,
    },
    "hurink_ft": {
        "title": "Hurink ft / mt (Fisher & Thompson / Hurink et al. 1994)",
        "subdirs": ["edata", "rdata", "vdata"],
        "names": ["mt06", "mt10", "mt20"],
        "bks_map": HURINK_FT_BKS,
    },
    "hurink_orb": {
        "title": "Hurink orb (Hurink et al. 1994)",
        "subdirs": ["edata", "rdata", "vdata"],
        "names": [f"orb{i}" for i in range(1, 11)],
        "bks_map": HURINK_ORB_BKS,
    },
}


# ============================================================
# 辅助函数
# ============================================================

def gap_str(ga_cmax, bks_val):
    """格式化差距百分比"""
    if bks_val is None or bks_val == "-":
        return "   N/A"
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


def run_single_instance(filepath, label, bks_val, show_gantt=True):
    """
    对单个算例文件执行 GA + ALNS + Tabu Search，返回结果字典。
    filepath: fjs 文件绝对/相对路径
    label: 算例显示名 (如 "Mk01")
    bks_val: BKS 值 (int) 或 None
    """
    jobs_raw, nm, nj = load_fjsp_from_file(filepath)
    jobs = convert_to_zero_index(jobs_raw)
    total_ops = sum(len(j) for j in jobs)
    print(f"\n数据集: {filepath}")
    print(f"工件: {nj}, 机器: {nm}, 工序: {total_ops}")

    print(f"\n=== GA + ALNS + Tabu Search 模式 ===")
    ga = GA(jobs, nm, config)
    best = ga.run(rl_controller=RLController(config),
                  alns=ALNS(jobs, nm, config),
                  tabu_search=TabuSearch(jobs, nm, config),
                  bks_value=bks_val)

    _our = int(best["cmax"])
    _gap = gap_str(_our, bks_val)
    _bks_display = bks_val if bks_val is not None else "-"

    print("\n" + "─" * 50)
    print("  算法结果 vs 最优已知解 (BKS)")
    print("  " + "─" * 46)
    print("  %-16s %8s %8s %8s" % ("算例", "GA_Cmax", "BKS", "差距"))
    print("  " + "─" * 46)
    print("  %-16s %8d %8s %10s" % (label, _our, _bks_display, _gap))
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

    if show_gantt:
        try:
            plot_schedule_analysis(sched, nj, nm,
                                   save_path="output/gantt_%s.png" % label,
                                   show=False)
            print("甘特图: output/gantt_%s.png" % label)
        except Exception as e:
            print("绘图失败: %s" % e)

    return {
        "label": label, "cmax": _our, "load_var": best["load_var"],
        "nj": nj, "nm": nm, "total_ops": total_ops,
        "bks": bks_val, "gap": _gap, "sched": sched,
    }


def run_batch(items, title):
    """
    批量运行一组算例，输出格式化结果表格。
    items: list of (label, filepath, bks_value_or_None)
    title: 表格标题
    """
    header = "  %-14s %5s %5s %5s %8s %8s %8s %10s %6s" % (
        "算例", "工件", "机器", "工序", "GA_Cmax", "BKS", "差距", "负荷方差", "耗时")
    sep = "  " + fmt_sep(len(header) - 2)

    print(f"\n{'=' * (len(header) + 2)}")
    print("  %s (GA + ALNS + Tabu Search)" % title)
    print(f"{'=' * (len(header) + 2)}")
    print(header)
    print(sep)

    results = []
    t0 = time.time()
    print()

    for idx, (label, filepath, bks_val) in enumerate(items):
        set_seed(config.RANDOM_SEED + idx)

        if not os.path.exists(filepath):
            print(f"\r  [{label}] 文件不存在: {filepath}")
            continue

        jobs_raw, nm, nj = load_fjsp_from_file(filepath)
        jobs = convert_to_zero_index(jobs_raw)
        total_ops = sum(len(j) for j in jobs)

        old_v = config.VERBOSE
        config.VERBOSE = False
        t_start = time.time()

        def make_progress(inst_label):
            def cb(gen, max_gen, best_cmax, avg_cmax, elapsed):
                bar_len = 20
                filled = int(bar_len * gen / max_gen)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{inst_label}] {bar} {gen:3d}/{max_gen} | best={best_cmax:.0f} avg={avg_cmax:.0f} | {elapsed:.0f}s", end="", flush=True)
            return cb

        ga = GA(jobs, nm, config)
        best = ga.run(rl_controller=RLController(config),
                      alns=ALNS(jobs, nm, config),
                      tabu_search=TabuSearch(jobs, nm, config),
                      progress_callback=make_progress(label),
                      bks_value=bks_val)

        # 清除进度条
        print("\r" + " " * 80 + "\r", end="", flush=True)

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
        _gap = gap_str(ac, bks_val)
        _bks_display = bks_val if bks_val is not None else "-"

        results.append((label, nj, nm, total_ops, ac, _bks_display, _gap, lv, t_inst))

        print("  %-14s %5d %5d %5d %8d %8s %10s %8.1f %5.0fs" % (
            label, nj, nm, total_ops, ac, _bks_display, _gap, lv, t_inst))

    print(sep)

    # 汇总
    if results:
        our_vals = [r[4] for r in results]
        bks_vals = [r[5] for r in results]
        gaps = [(o - b if isinstance(b, int) else 0) for o, b in zip(our_vals, bks_vals)]
        pos_gaps = [g for g in gaps if g > 0]
        avg_gap = sum(pos_gaps) / len(pos_gaps) if pos_gaps else 0.0
        reached = sum(1 for o, b in zip(our_vals, bks_vals) if isinstance(b, int) and o <= b)

        print(f"\n  📊 汇总: {reached}/{len(results)} 达到/超越 BKS | "
              f"平均正偏差: {avg_gap:+.1f} | "
              f"总耗时: {time.time() - t0:.0f}s")
    print()

    # 返回结构化结果（供图表使用）
    result_dicts = []
    for r in results:
        bks_raw = r[5]
        bks_val_int = bks_raw if isinstance(bks_raw, int) else None
        gap_num = None
        if bks_val_int is not None and bks_val_int > 0:
            gap_num = (r[4] - bks_val_int) / bks_val_int * 100
        result_dicts.append({
            "label": r[0],
            "nj": r[1], "nm": r[2], "total_ops": r[3],
            "cmax": r[4], "bks_val": bks_val_int,
            "gap_str": r[6], "gap_numeric": gap_num,
            "load_var": r[7], "runtime": r[8],
        })
    return result_dicts


# ============================================================
# 数据集调度
# ============================================================

def _simple_items(dataset_key):
    """生成简单数据集的 batch items"""
    info = SIMPLE_DATASETS[dataset_key]
    items = []
    for fname in info["files"]:
        label = fname.replace(".fjs", "")
        fp = os.path.join(info["path"], fname)
        bks_val = info["bks"].get(fname)
        if os.path.exists(fp):
            items.append((label, fp, bks_val))
    return items


def _hurink_items(family_key):
    """生成 Hurink 子族的 batch items，含变体后缀"""
    info = HURINK_FAMILIES[family_key]
    items = []
    for name in info["names"]:
        for subdir in info["subdirs"]:
            fname = f"{name}.fjs"
            fp = os.path.join("data/Hurink_Data", subdir, fname)
            if os.path.exists(fp):
                label = f"{name}_{subdir}"
                bks_val = info["bks_map"][name][subdir]
                items.append((label, fp, bks_val))
    return items


def _resolve_single_instance(dataset, name):
    """
    解析单例运行的文件路径和 BKS。
    dataset: "mk" / "barnes" / "dauzere" /
             "hurink_edata" / "hurink_rdata" / "hurink_vdata"
    name: 算例名 (如 "Mk01", "mt10c1", "car1")
    """
    if dataset in SIMPLE_DATASETS:
        info = SIMPLE_DATASETS[dataset]
        fname = name if name.endswith(".fjs") else name + ".fjs"
        fp = os.path.join(info["path"], fname)
        bks_val = info["bks"].get(fname)
        return fp, bks_val, name.replace(".fjs", "")

    # Hurink 单变体: python main.py run hurink_edata car1
    if dataset.startswith("hurink_") and "_" in dataset:
        subdir = dataset.split("_", 1)[1]
        fname = name if name.endswith(".fjs") else name + ".fjs"
        fp = os.path.join("data/Hurink_Data", subdir, fname)
        for family in HURINK_FAMILIES.values():
            if name in family["bks_map"]:
                bks_val = family["bks_map"][name].get(subdir)
                break
        else:
            bks_val = None
        label = f"{name}_{subdir}"
        return fp, bks_val, label

    return None, None, None


# ============================================================
# 主函数
# ============================================================

BATCH_ALL_ORDER = [
    ("mk", _simple_items, "mk"),
    ("barnes", _simple_items, "barnes"),
    ("dauzere", _simple_items, "dauzere"),
    ("hurink_car", _hurink_items, "hurink_car"),
    ("hurink_ft", _hurink_items, "hurink_ft"),
    ("hurink_orb", _hurink_items, "hurink_orb"),
]


def main():
    print("=" * 60)
    print("柔性车间滚动调度系统 - 多数据集支持 v1.4.0")
    print("=" * 60)

    if len(sys.argv) < 2:
        path = os.path.join("data/Brandimarte_Data", "Mk01.fjs")
        run_single_instance(path, "Mk01", MK_BKS.get("Mk01.fjs"))
        print("\n完成。")
        return

    cmd = sys.argv[1]

    # ── 向后兼容: python main.py all ──
    if cmd == "all":
        items = _simple_items("mk")
        run_batch(items, "Brandimarte Mk 系列")
        return

    # ── 滚动时域模式 ──
    if cmd == "rolling_horizon":
        _run_rolling_horizon(sys.argv[2:])
        return

    # ── 向后兼容: python main.py Mk01 ──
    u = cmd.upper()
    if u.startswith("MK") or cmd.startswith("Mk"):
        arg = cmd + ".fjs" if ".fjs" not in cmd else cmd
        path = os.path.join("data/Brandimarte_Data", arg)
        if os.path.exists(path):
            run_single_instance(path, arg.replace(".fjs", ""), MK_BKS.get(arg))
            print("\n完成。")
            return
        else:
            print("错误：文件不存在: %s" % path)
            return

    # ── run 模式 ──
    if cmd == "run":
        if len(sys.argv) < 3:
            print("用法: python main.py run <dataset> <instance>")
            print("数据集: mk, barnes, dauzere, hurink_edata, hurink_rdata, hurink_vdata")
            return
        dataset = sys.argv[2]
        name = sys.argv[3] if len(sys.argv) > 3 else None

        if name is None:
            name = dataset
            dataset = "mk"

        if dataset == "mk" and (name.upper().startswith("MK") or name.startswith("Mk")):
            arg = name + ".fjs" if ".fjs" not in name else name
            path = os.path.join("data/Brandimarte_Data", arg)
            if os.path.exists(path):
                run_single_instance(path, arg.replace(".fjs", ""), MK_BKS.get(arg))
                print("\n完成。")
                return

        fp, bks_val, label = _resolve_single_instance(dataset, name)
        if fp and os.path.exists(fp):
            run_single_instance(fp, label, bks_val)
            print("\n完成。")
        else:
            print("错误：未知的数据集 '%s' 或文件不存在。" % dataset)
        return

    # ── batch 模式 ──
    if cmd == "batch":
        if len(sys.argv) < 3:
            print("用法: python main.py batch <dataset>")
            print("数据集: mk, barnes, dauzere, hurink_car, hurink_ft, hurink_orb, all")
            return

        batch_target = sys.argv[2]

        if batch_target == "all":
            total_instances = 0
            t_all = time.time()
            results_by_family = {}
            for key, builder_fn, _ in BATCH_ALL_ORDER:
                if key in SIMPLE_DATASETS:
                    title = SIMPLE_DATASETS[key]["title"]
                else:
                    title = HURINK_FAMILIES[key]["title"]
                items = builder_fn(key)
                total_instances += len(items)
                res = run_batch(items, title)
                results_by_family[key] = (title, res)
            # 生成汇总图表
            plot_batch_summary_all(results_by_family)
            print(f"\n{'=' * 60}")
            print(f"  全部数据集完成！共 {total_instances} 个算例，"
                  f"总耗时: {time.time() - t_all:.0f}s")
            print(f"{'=' * 60}")
            return

        if batch_target in SIMPLE_DATASETS:
            items = _simple_items(batch_target)
            title = SIMPLE_DATASETS[batch_target]["title"]
            res = run_batch(items, title)
            plot_batch_summary_all({batch_target: (title, res)})
            return

        if batch_target in HURINK_FAMILIES:
            items = _hurink_items(batch_target)
            title = HURINK_FAMILIES[batch_target]["title"]
            res = run_batch(items, title)
            plot_batch_summary_all({batch_target: (title, res)})
            return

        print("错误：未知数据集 '%s'。" % batch_target)
        return

    # ── 未识别命令 ──
    print("错误：未知命令 '%s'" % cmd)
    print("用法: python main.py [run|batch|all|rolling_horizon|<instance>]")


def _run_rolling_horizon(args=None):
    """滚动时域模式 — 支持多数据集

    用法:
        python main.py rolling_horizon                  → Mk01 (默认)
        python main.py rolling_horizon Mk02              → Mk02 (向后兼容)
        python main.py rolling_horizon mk Mk02           → 显式指定 Mk02
        python main.py rolling_horizon barnes mt10c1     → Barnes 算例
        python main.py rolling_horizon hurink_edata car1 → Hurink 算例
    """
    dataset = "mk"
    name = config.DEFAULT_INSTANCE  # "Mk01.fjs"

    if args and len(args) >= 2:
        # python main.py rolling_horizon <dataset> <instance>
        dataset = args[0]
        name = args[1]
    elif args and len(args) == 1:
        arg = args[0]
        # python main.py rolling_horizon Mk02 (向后兼容)
        if arg.upper().startswith("MK") or arg.startswith("Mk"):
            dataset = "mk"
            name = arg + ".fjs" if ".fjs" not in arg else arg
        else:
            print("错误：单参数模式需要 Mk 系列算例名 (如 Mk02)")
            print("用法: python main.py rolling_horizon <dataset> <instance>")
            return

    # ── 解析文件路径和 BKS ──
    if dataset == "mk":
        arg = name + ".fjs" if ".fjs" not in name else name
        path = os.path.join("data/Brandimarte_Data", arg)
        bks_val = MK_BKS.get(arg)
        label = arg.replace(".fjs", "")
    else:
        fp, bks_val, label = _resolve_single_instance(dataset, name)
        if fp is None:
            print("错误：未知的数据集 '%s'。" % dataset)
            return
        path = fp

    print("\n数据集: %s" % path)
    if not os.path.exists(path):
        print("错误：文件不存在")
        return

    jobs_raw, nm, nj = load_fjsp_from_file(path)
    jobs = convert_to_zero_index(jobs_raw)
    print("工件: %d, 机器: %d, 工序: %d" % (nj, nm, sum(len(j) for j in jobs)))

    print("\n=== 滚动时域模式 ===")
    ga = GA(jobs, nm, config)
    rh = RollingHorizon(jobs, nm, config, ga,
                        RLController(config),
                        ALNS(jobs, nm, config))
    final = rh.run()
    cmax_rh = rh.current_time

    _bks_display = bks_val if bks_val is not None else "-"
    _gap = gap_str(int(cmax_rh), bks_val)
    print("\n" + "─" * 50)
    print("  滚动时域 vs 静态最优已知解 (BKS，仅供参考)")
    print("  " + "─" * 46)
    print("  %-16s %12s %8s %8s" % ("算例", "RH_Cmax", "BKS", "差距"))
    print("  " + "─" * 46)
    print("  %-16s %12.0f %8s %10s" % (label, cmax_rh, _bks_display, _gap))
    print("  " + "─" * 46 + "\n")

    if final:
        std = [(j, o, m, s, e) for (j, o, m, s, e) in final]
        metrics = evaluate_schedule(std, nj, nm)
        print_metrics(metrics, "滚动时域调度评估结果")

        try:
            plot_schedule_analysis(std, nj, nm,
                                   save_path="output/gantt_%s_rolling.png" % label,
                                   show=False)
            print("甘特图: output/gantt_%s_rolling.png" % label)
        except Exception:
            pass

    print("\n完成。")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    set_seed(config.RANDOM_SEED)
    main()
