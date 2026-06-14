"""
全量实验运行脚本 — 分段持久化版 (v2.0)
============================================
设计理念：
  ✅ 一键运行: python run_experiment.py
  ✅ 四段分明: Mk → Barnes → Dauzère → Hurink，段段有头有尾
  ✅ 每段独立: 每跑完一批立即写入磁盘，即使中间 crash 前面数据不丢
  ✅ 断点续跑: python run_experiment.py --resume，从上次断点继续
  ✅ 全程容错: 一个算例挂了不影响同批其他算例，一批挂了不影响下一批

输出位置:
  - output/experiment/    每批独⽴ CSV
  - output/summary/       族级对比图 + 总览图 + 主 CSV
  - output/summary/checkpoint.json  断点续跑用

算例构成（共 24 个）:
  1. Mk 系列 9 个 (Mk01-Mk09)
  2. Barnes 系列 5 个 (mt10c1/cc/x/xx/xxx)
  3. Dauzère 系列 5 个 (01a/02a/03a/04a/08a)
  4. Hurink 系列 5 个 (car1/car2/mt06/orb1/orb2 × edata)
"""

import sys
import os
import time
import json
import csv
import numpy as np
from datetime import datetime

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── 复用 main.py 的 batch 引擎 ───
from main import (
    run_batch, run_single_instance,
    SIMPLE_DATASETS, HURINK_FAMILIES,
    MK_BKS, BARNES_BKS, DAUZERE_BKS,
    HURINK_CAR_BKS, HURINK_FT_BKS, HURINK_ORB_BKS,
    set_seed,
)
from utils.visualization import (
    plot_batch_summary_all,
    plot_schedule_analysis,
    plot_convergence_curves_batch,
    plot_global_gap_scatter,
    plot_scale_vs_runtime,
)
import config

OUTPUT_DIR = "output/experiment"
SUMMARY_DIR = "output/summary"
CHECKPOINT_PATH = os.path.join(SUMMARY_DIR, "checkpoint.json")
MASTER_CSV = os.path.join(SUMMARY_DIR, "results.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
# 实验设计方案 (24 个算例)
# ════════════════════════════════════════════════════════════

EXPERIMENT_PLAN = [
    # (batch_key, 显示标题, 算例列表 or None(自动生成))
    ("mk",      "第1批 | Brandimarte Mk 系列 (全部 9 个)", None),
    ("barnes",  "第2批 | Barnes 系列 (前 5 个代表)", [
        ("mt10c1",  "data/Barnes/mt10c1.fjs",   BARNES_BKS["mt10c1.fjs"]),
        ("mt10cc",  "data/Barnes/mt10cc.fjs",   BARNES_BKS["mt10cc.fjs"]),
        ("mt10x",   "data/Barnes/mt10x.fjs",    BARNES_BKS["mt10x.fjs"]),
        ("mt10xx",  "data/Barnes/mt10xx.fjs",   BARNES_BKS["mt10xx.fjs"]),
        ("mt10xxx", "data/Barnes/mt10xxx.fjs",  BARNES_BKS["mt10xxx.fjs"]),
    ]),
    ("dauzere", "第3批 | Dauzère 系列 (前 5 个代表)", [
        ("01a", "data/Dauzere_Data/01a.fjs",  DAUZERE_BKS["01a.fjs"]),
        ("02a", "data/Dauzere_Data/02a.fjs",  DAUZERE_BKS["02a.fjs"]),
        ("03a", "data/Dauzere_Data/03a.fjs",  DAUZERE_BKS["03a.fjs"]),
        ("04a", "data/Dauzere_Data/04a.fjs",  DAUZERE_BKS["04a.fjs"]),
        ("08a", "data/Dauzere_Data/08a.fjs",  DAUZERE_BKS["08a.fjs"]),
    ]),
    ("hurink",  "第4批 | Hurink 系列 (三大子家族 5 个代表)", [
        ("car1_edata",  "data/Hurink_Data/edata/car1.fjs",  HURINK_CAR_BKS["car1"]["edata"]),
        ("car2_edata",  "data/Hurink_Data/edata/car2.fjs",  HURINK_CAR_BKS["car2"]["edata"]),
        ("mt06_edata",  "data/Hurink_Data/edata/mt06.fjs",  HURINK_FT_BKS["mt06"]["edata"]),
        ("orb1_edata",  "data/Hurink_Data/edata/orb1.fjs",  HURINK_ORB_BKS["orb1"]["edata"]),
        ("orb2_edata",  "data/Hurink_Data/edata/orb2.fjs",  HURINK_ORB_BKS["orb2"]["edata"]),
    ]),
]

# 批次执行顺序（与 EXPERIMENT_PLAN 对应）
BATCH_KEYS = [plan[0] for plan in EXPERIMENT_PLAN]


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_separator(title):
    """打印带时间戳的大分隔符"""
    now = timestamp()
    width = 66
    print()
    print("╔" + "═" * width + "╗")
    for line in title.split("\n"):
        print("║" + line.center(width) + "║")
    print("║" + f"  🕐 {now}".ljust(width) + "║")
    print("╚" + "═" * width + "╝")
    print()


def progress_bar_cb(label):
    """生成进度回调函数"""
    def cb(gen, max_gen, best_cmax, avg_cmax, elapsed):
        bar_len = 20
        filled = int(bar_len * gen / max_gen)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{label}] {bar} {gen:3d}/{max_gen} | best={best_cmax:.0f} avg={avg_cmax:.0f} | {elapsed:.0f}s", end="", flush=True)
    return cb


# ════════════════════════════════════════════════════════════
# 持久化 — 每批跑完立即写盘
# ════════════════════════════════════════════════════════════

def save_checkpoint(all_results_by_family):
    """
    将当前所有结果写入 checkpoint JSON + 主 CSV。
    每次调用都会覆盖，保证磁盘上的始终是最新完整数据。
    """
    # 排除大数据字段（schedule 仅运行时用，不持久化）
    # best_history/avg_history 保留以便断点续跑后仍可画收敛曲线
    _EXCLUDE_FROM_DISK = {"schedule"}

    def _to_native(v):
        """将 numpy 类型递归转为 Python 原生类型，确保 JSON 序列化安全"""
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return [_to_native(x) for x in v]
        if isinstance(v, dict):
            return {k: _to_native(val) for k, val in v.items()}
        return v

    def _clean(r):
        return {k: _to_native(v) for k, v in r.items() if k not in _EXCLUDE_FROM_DISK}

    # ── 序列化（bks_val 可能是 int/None → JSON 兼容） ──
    serializable = {}
    for batch_key, (title, results) in all_results_by_family.items():
        serializable[batch_key] = {
            "title": title,
            "results": [_clean(r) for r in results],
            "timestamp": timestamp(),
        }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

    # ── 主 CSV（追加模式，先清空重写以保证一致性） ──
    fieldnames = [
        "batch", "label", "nj", "nm", "total_ops",
        "cmax", "bks_val", "gap_str", "gap_numeric",
        "load_var", "runtime",
    ]
    with open(MASTER_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_key, (title, results) in all_results_by_family.items():
            for r in results:
                row = {"batch": batch_key}
                row.update(_clean(r))
                writer.writerow(row)

    # ── 同时写一份分批 CSV（便于单独查看每批） ──
    for batch_key, (title, results) in all_results_by_family.items():
        batch_csv = os.path.join(OUTPUT_DIR, f"{batch_key}.csv")
        with open(batch_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {"batch": batch_key}
                row.update(_clean(r))
                writer.writerow(row)

    print(f"  💾 已保存 → {MASTER_CSV}  |  断点 → {CHECKPOINT_PATH}")


def load_checkpoint():
    """
    从磁盘加载之前保存的结果。
    返回: {batch_key: (title, [results...]), ...}  或  {} (无断点)
    """
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        restored = {}
        for batch_key, data in raw.items():
            restored[batch_key] = (data["title"], data["results"])
        print(f"  🔄 检测到历史断点: {list(restored.keys())}")
        return restored
    except Exception as e:
        print(f"  ⚠ 断点文件读取失败 ({e})，从头运行")
        return {}


# ════════════════════════════════════════════════════════════
# 运行引擎
# ════════════════════════════════════════════════════════════

def _run_instances(items):
    """
    运行一组算例，返回结果列表。
    items: list of (label, filepath, bks_value)
    """
    from main import load_fjsp_from_file, convert_to_zero_index
    from algorithms.ga import GA
    from algorithms.rl_agent import RLController
    from algorithms.alns import ALNS
    from algorithms.tabu_search import TabuSearch

    results = []
    n_total = len(items)

    for idx, (label, filepath, bks_val) in enumerate(items):
        set_seed(config.RANDOM_SEED + idx)

        if not os.path.exists(filepath):
            print(f"\n  ⚠ [{label}] 文件不存在: {filepath}")
            results.append({
                "label": label, "nj": 0, "nm": 0, "total_ops": 0,
                "cmax": 0, "bks_val": bks_val,
                "gap_str": "  N/A", "gap_numeric": None,
                "load_var": 0.0, "runtime": 0.0,
                "best_history": [], "avg_history": [], "schedule": [],
                "error": "file_not_found",
            })
            continue

        try:
            jobs_raw, nm, nj = load_fjsp_from_file(filepath)
            jobs = convert_to_zero_index(jobs_raw)
            total_ops = sum(len(j) for j in jobs)

            old_v = config.VERBOSE
            config.VERBOSE = False
            t_start = time.time()

            ga = GA(jobs, nm, config)
            best = ga.run(
                rl_controller=RLController(config),
                alns=ALNS(jobs, nm, config),
                tabu_search=TabuSearch(jobs, nm, config),
                progress_callback=progress_bar_cb(label),
                bks_value=bks_val,
            )
            # 清进度行
            print("\r" + " " * 80 + "\r", end="", flush=True)

            t_inst = time.time() - t_start
            config.VERBOSE = old_v

            assign = ga.decode(best)
            mt = [0] * nm
            jt = [0] * nj
            ml = [0.0] * nm
            sched = []
            for (jid, oid, mid, dur) in assign:
                s = int(max(mt[mid], jt[jid]))
                e = int(s + dur)
                sched.append((int(jid), int(oid), int(mid), s, e))
                mt[mid] = e
                jt[jid] = e
                ml[mid] += dur

            ac = int(max(jt))
            lv = float(np.var(ml)) if len(ml) > 0 else 0.0
            gap_val = (ac - bks_val) / bks_val * 100 if bks_val and bks_val > 0 else None
            if bks_val and ac <= bks_val:
                # 精确 BKS 或超过 BKS（负偏差）→ 绿色标记
                gap_s = f" ✅ {gap_val:+.1f}%" if gap_val < 0 else " ✅ 0%"
            else:
                gap_s = f"{gap_val:+.1f}%" if gap_val is not None else "  N/A"

            results.append({
                "label": label, "nj": nj, "nm": nm, "total_ops": total_ops,
                "cmax": ac, "bks_val": bks_val,
                "gap_str": gap_s, "gap_numeric": gap_val,
                "load_var": lv, "runtime": t_inst,
                "best_history": best.get("best_history", []),
                "avg_history": best.get("avg_history", []),
                "schedule": sched,       # 直接存解码结果，画图用
            })

            print(f"  ✅ [{idx+1}/{n_total}] {label:15s}  Cmax={ac:5d}  BKS={str(bks_val):>5s}  {gap_s:>8s}  耗时={t_inst:.0f}s")

        except Exception as e:
            print(f"\r" + " " * 80 + "\r", end="", flush=True)
            print(f"  ❌ [{idx+1}/{n_total}] {label:15s}  失败: {e}")
            results.append({
                "label": label, "nj": 0, "nm": 0, "total_ops": 0,
                "cmax": 0, "bks_val": bks_val,
                "gap_str": "  FAIL", "gap_numeric": None,
                "load_var": 0.0, "runtime": 0.0,
                "best_history": [], "avg_history": [],
                "schedule": [],
                "error": str(e),
            })

    return results


def run_batch_plan(batch_key, title, items):
    """
    运行一个批次，返回结果列表。
    items=None 表示 Mk 系列，自动生成。
    """
    if items is None:
        from main import _simple_items
        items = _simple_items("mk")

    print_separator(f"📦 {title}")

    t0 = time.time()
    results = _run_instances(items)
    elapsed = time.time() - t0

    # 统计
    valid = [r for r in results if r.get("cmax", 0) > 0]
    reached = sum(1 for r in valid if r["bks_val"] and r["cmax"] <= r["bks_val"])
    if valid:
        print(f"\n  📊 {batch_key.upper()} 汇总: {reached}/{len(valid)} 达到 BKS  |  "
              f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    else:
        print(f"\n  ⚠ 该批次无有效结果")

    return results


def run_final_summary(all_results_by_family, t_global):
    """跨数据集总览 + 过程可视化"""
    elapsed_total = time.time() - t_global
    total_instances = sum(len(r) for _, r in all_results_by_family.values())

    print_separator(f"🏆 全部实验完成！\n"
                    f"总算例: {total_instances}  |  总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    # 总览图
    print(f"  📊 生成跨数据集总览图 ...")
    try:
        plot_batch_summary_all(all_results_by_family, save_dir=SUMMARY_DIR)
        print(f"  ✅ 总览图 → {SUMMARY_DIR}/")
    except Exception as e:
        print(f"  ⚠ 总览图生成失败: {e}")

    # 全局 Gap 散点图
    print(f"  📊 生成全局 Gap 散点图 ...")
    try:
        plot_global_gap_scatter(all_results_by_family, save_dir=SUMMARY_DIR)
    except Exception as e:
        print(f"  ⚠ Gap 散点图失败: {e}")

    # 规模-耗时散点图
    print(f"  📊 生成规模-耗时散点图 ...")
    try:
        plot_scale_vs_runtime(all_results_by_family, save_dir=SUMMARY_DIR)
    except Exception as e:
        print(f"  ⚠ 规模-耗时图失败: {e}")

    # 收敛曲线（所有族，这样断点续跑也能重建）
    print(f"\n  📈 生成各族的收敛曲线 ...")
    for family_key, (title, results) in all_results_by_family.items():
        try:
            has_history = any(r.get("best_history") and len(r["best_history"]) > 1 for r in results)
            if has_history:
                plot_convergence_curves_batch(
                    results,
                    title=f"收敛曲线 — {title}",
                    save_dir=SUMMARY_DIR,
                    save_name=f"convergence_{family_key}.png",
                )
        except Exception as e:
            print(f"    ⚠ {family_key} 收敛曲线失败: {e}")

    # 过程可视化
    print(f"\n  🎨 生成过程可视化（调度分析图）...")
    try:
        _plot_process_visualization(all_results_by_family)
    except Exception as e:
        print(f"  ⚠ 过程可视化失败: {e}")

    final_time = time.time() - t_global
    print()
    print("=" * 66)
    print(f"  ✅  全流程完成！          总耗时 {final_time:.0f}s ({final_time/60:.1f}min)")
    print(f"  📁  族级对比图            {SUMMARY_DIR}/")
    print(f"  📁  跨族总览图            {SUMMARY_DIR}/overall_comparison.png")
    print(f"  📁  收敛曲线图(逐批)     {SUMMARY_DIR}/convergence_*.png")
    print(f"  📁  全局 Gap 散点图       {SUMMARY_DIR}/global_gap_scatter.png")
    print(f"  📁  规模-耗时散点图       {SUMMARY_DIR}/scale_vs_runtime.png")
    print(f"  📁  分批 CSV              {OUTPUT_DIR}/")
    print(f"  📁  主 CSV                {MASTER_CSV}")
    print(f"  📁  断点文件              {CHECKPOINT_PATH}")
    print("=" * 66)
    print()


def _plot_process_visualization(all_results_by_family):
    """过程可视化：直接使用已存 schedule 画图（零额外 GA 运行）"""
    from utils.visualization import plot_schedule_analysis, plot_gantt_chart

    proc_dir = os.path.join(OUTPUT_DIR, "gantt_process")
    os.makedirs(proc_dir, exist_ok=True)

    print(f"  🎨 使用已存 schedule 生成调度分析图 ...")

    all_flat = []
    for family_key, (title, results) in all_results_by_family.items():
        for r in results:
            if r.get("schedule") and len(r["schedule"]) > 0:
                all_flat.append(r)

    for idx, r in enumerate(all_flat):
        label = r["label"]
        nj = r["nj"]
        nm = r["nm"]
        sched = r["schedule"]
        print(f"     [{idx+1}/{len(all_flat)}] {label} (Cmax={r['cmax']}) ...")

        try:
            save_path = os.path.join(proc_dir, f"{label}_analysis.png")
            plot_schedule_analysis(sched, nj, nm, save_path=save_path, show=False)

            gantt_path = os.path.join(proc_dir, f"{label}_gantt.png")
            plot_gantt_chart(
                sched, nj, nm,
                title=f"GA+RL+ALNS+TS — {label} (Cmax={r['cmax']})",
                save_path=gantt_path, show=False,
            )
        except Exception as e:
            print(f"        ⚠ 绘图失败: {e}")

    print(f"  ✅ 过程可视化完成！共 {len(all_flat)} 张图 → {proc_dir}")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 参数解析 ──
    resume_mode = "--resume" in sys.argv

    print()
    print("=" * 66)
    print("  融合强化学习与自适应大邻域搜索的柔性车间滚动调度系统")
    print("  ── 全量实验 (24 个代表性算例，分 4 批) ──")
    print("=" * 66)
    print(f"  启动时间: {timestamp()}")
    if resume_mode:
        print(f"  模式:     🔄 断点续跑")
    print(f"  算例:     Mk(9) + Barnes(5) + Dauzère(5) + Hurink(5) = 24")
    print(f"  输出:     {OUTPUT_DIR}/")
    print(f"  总览:     {SUMMARY_DIR}/")
    print("=" * 66)
    print()

    t_global = time.time()

    # ── 加载断点（续跑时用） ──
    existing = load_checkpoint() if resume_mode else {}

    # ── 逐批运行 ──
    all_results_by_family = dict(existing)  # 续跑时保留已有

    completed = set(all_results_by_family.keys())

    for batch_key, title, items in EXPERIMENT_PLAN:
        if batch_key in completed:
            print(f"  ⏭️  跳过已完成批次: {title}")
            continue

        try:
            results = run_batch_plan(batch_key, title, items)
            # 即使空结果也保存（标记完成）
            display_title = title.split("|")[1].strip() if "|" in title else title
            all_results_by_family[batch_key] = (display_title, results)

            # 关键: 每批跑完立即存盘!
            save_checkpoint(all_results_by_family)

            # 📊 每批跑完立即出族级对比图 + 累积跨族总览图
            print(f"  📊 生成 {batch_key} 族级对比图 + 累积跨族总览 ...")
            try:
                plot_batch_summary_all(
                    all_results_by_family,  # 传累积数据，跨族图逐步继承各批结果
                    save_dir=SUMMARY_DIR,
                )
                print(f"  ✅ 族级图 → {SUMMARY_DIR}/  |  跨族总览已累积\n")
            except Exception as plot_e:
                print(f"  ⚠ 族级图生成失败: {plot_e}\n")

            # 📈 每批跑完立即出收敛曲线（按批命名，避免覆盖）
            print(f"  📈 生成 {batch_key} 收敛曲线 ...")
            try:
                plot_convergence_curves_batch(
                    results,
                    title=f"收敛曲线 — {display_title}",
                    save_dir=SUMMARY_DIR,
                    save_name=f"convergence_{batch_key}.png",
                )
            except Exception as plot_e:
                print(f"  ⚠ 收敛曲线失败: {plot_e}\n")

        except Exception as e:
            print(f"\n  ❌ 批次 [{batch_key}] 崩溃: {e}")
            print(f"  💡 已有{len(all_results_by_family)} 批结果已保存到磁盘，修复后可用 --resume 续跑")
            # 保存已完成的
            if all_results_by_family:
                save_checkpoint(all_results_by_family)
            # 继续下一批，不中断
            print(f"  ➡️  继续下一批...\n")

    # ── 如果全部为空，说明全部失败 ──
    if not all_results_by_family:
        print("\n  ❌ 没有成功运行任何批次，请检查环境后重试。")
        sys.exit(1)

    # ── 总览 ──
    run_final_summary(all_results_by_family, t_global)
