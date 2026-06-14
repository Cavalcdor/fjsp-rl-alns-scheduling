# 项目变更日志

## 项目简介

**FJSP-RL-ALNS-Scheduling** — 基于遗传算法(GA) + 强化学习(RL) + 自适应大邻域搜索(ALNS) 的柔性作业车间调度问题(FJSP)求解器。

---

## [2.0.0] - 2026-06-15

### 报告定稿

#### 1. 全量实验报告定型
- **`output/report/main_report.md`** 完成全部 2051 行报告的撰写、审核与修订
- 涵盖 27 张图表 + 19 张表格，四大数据家族 24 个代表性算例的系统实验验证
- 完成全量检查三轮迭代，修复约束公式、奖励函数描述、符号表、附录版本同步等全部遗留问题

#### 2. 项目版本里程碑
- 代码、报告、变更日志三者版本统一为 **v2.0.0**
- README 新增 `Report-v2.0.0` 徽标

---

## [1.5.3] - 2026-06-14

### 新增

#### 1. 滚动时域批量模式 `batch_rh`
- **`main.py`** 新增 `batch_rh` 命令，一键运行 Mk01~Mk09 双场景（Baseline + Mild）滚动时域对比实验
- 自动生成双场景对比表格（近视代价 MC / 鲁棒性代价 RC / 退化损失率 DLR）
- 每算例输出场景标签甘特图（`gantt_Mk01_baseline.png` / `gantt_Mk01_mild.png`）
- 输出 `results_rh_comparison.csv` 汇总对比数据

#### 2. RollingHorizon 结构化结果返回
- **`core/rolling_horizon.py`** 新增 `_collect_results()` 方法，`run()` 返回包含 `schedule`、`cmax`、`load_var` 的结构化字典
- **`main.py`** `_run_rolling_horizon()` 适配新返回格式，解决甘特图数据丢失问题

#### 3. 退化参数默认启用
- **`config.py`** `DEGRADATION_COEFF` 从 1.0（禁用）调整为 **1.05**（启用指数退化），注释同步更新

---

## [1.5.2] - 2026-06-14

### 修复

#### 1. numpy JSON 序列化兼容性
- **问题**：`ga.run()` 中 `avg_history` 存储 `numpy.float64`，`save_checkpoint()` 用 `json.dump(..., default=str)` 序列化为字符串 `"87.5"` 而非浮点数 `87.5`
- **修复**：`save_checkpoint()` 新增 `_to_native()` 递归清洗函数，将 `np.integer`、`np.floating`、`np.ndarray` 全部转为 Python 原生类型；`_clean()` 中不再直接过滤 schedule，而是递归转换全部字段

#### 2. 死代码清理
- **删除**：`_find_filepath()` 函数（定义后从未被调用）

---

## [1.5.1] - 2026-06-14

### 修复

#### 1. 可视化图表覆盖 Bug 修复 (核心)
- **问题**：`plot_batch_summary_all()` 每批只传单族数据 `{batch_key: ...}`，导致 `overall_comparison.png` 反复被覆盖，跨族总览图始终只显示当前批次
- **修复**：改为传递累积的 `all_results_by_family`，使跨族总览图**逐步继承**已完成各批的数据

#### 2. 收敛曲线文件覆盖 Bug 修复
- **问题**：`plot_convergence_curves_batch()` 始终写入固定路径 `convergence_curves.png`，每批运行后前一批的收敛曲线被覆盖
- **修复**：函数签名新增 `save_name` 参数，按批命名如 `convergence_mk.png`、`convergence_barnes.png`

#### 3. 断点续跑收敛数据丢失 Bug 修复
- **问题**：`save_checkpoint()` 的 `_EXCLUDE_FROM_DISK` 排除了 `best_history` / `avg_history`，导致 `--resume` 续跑后无法重建收敛曲线
- **修复**：仅排除 `schedule`（太大不适合 JSON），保留历史收敛数据

#### 4. 超 BKS 时 gap_str 错误显示
- **问题**：当算法结果优于 BKS（负偏差）时，`gap_str` 错误显示为 `"  ✅ 0%"` 而非真实偏差值
- **修复**：负偏差时显示真实值如 `"  ✅ -2.5%"`

#### 5. CSV 文件覆盖 Bug 修复
- **问题**：`_save_results_csv()` 写入 `results.csv` 与 `save_checkpoint()` 写入的 `MASTER_CSV` 文件名相同，导致结果被覆盖
- **修复**：`_save_results_csv()` 输出文件名改为 `results_report.csv`

#### 6. 失败算例误判为达 BKS
- **问题**：算例运行失败时 `cmax=0`，使 `0 <= bks_val` 条件成立，被错误计入"BKS 达成"且柱状图显示绿色
- **修复**：所有绘图函数中增加 `cmax > 0` 过滤，排除失败算例

#### 7. 跨族总览图族名缺失
- **问题**：`_friendly_name()` 缺少 `"hurink"` 映射，跨族总览图中 Hurink 显示原始 key 而非可读名称
- **修复**：添加 `"hurink": "Hurink"` 映射

#### 8. 结尾引导信息错误
- **问题**：`run_final_summary()` 结尾打印 `convergence_curves.png`，但该文件已不再生成（改为逐批命名）
- **修复**：改为 `convergence_*.png`

### 优化

#### 9. 断点续跑收敛曲线重建
- `run_final_summary()` 新增遍历所有族的收敛曲线生成循环，确保断点续跑后所有批次的收敛图均可重建

---

## [1.5.0] - 2026-06-13

### 新增

#### 1. 全量实验运行脚本 `run_experiment.py`
- **一键运行 24 个代表性算例**：Mk(9) + Barnes(5) + Dauzère(5) + Hurink(5)，四段顺序执行
- **分段持久化**：每批跑完立即写入磁盘，中间 crash 前面数据不丢
- **断点续跑**：`python run_experiment.py --resume` 从上次断点继续
- **全程容错**：一个算例挂掉不影响同批其他算例，一批挂掉不影响下一批

#### 2. 分阶段可视化
- **每批结束后自动出图**：族内对比柱状图 + 收敛曲线子图网格
- **最终全量总览**：跨族 Cmax 对比 / 全局 Gap 散点 / 规模-耗时散点
- **过程可视化**：24 组 Schedule Analysis + Gantt Chart（**零额外 GA 重跑**）

#### 3. GA 收敛历史记录
- `best_history` / `avg_history` 逐代记录，支持收敛曲线绘制

### 优化

#### 4. Schedule 内存持久化
- 解码后的调度表 `(job, op, machine, start, end)` 存入 result dict
- `_plot_process_visualization()` 直接取用，消除原版 24 × 100 代冗余 GA 重算
- checkpoint/CSV 自动排除大数据字段，磁盘占用极小

#### 5. 可视化套件增强
- `plot_convergence_curves_batch()` — 分批收敛曲线子图网格
- `plot_global_gap_scatter()` — 全局 Gap% 散点图（颜色按族区分）
- `plot_scale_vs_runtime()` — 规模-耗时散点图（气泡大小=Cmax，含趋势线）

---

## [1.4.2] - 2026-06-13

### 变更

#### 1. 最大迭代次数提升
- `MAX_GEN`: 50 → **100**，给予算法更充分的搜索空间

#### 2. 早停策略参数等比调优（降低敏感度）
- `EARLY_STOP_MIN_GEN`: 20 → **35**（35% 总代数，起步阶段更稳固）
- `EARLY_STOP_PATIENCE`: 20 → **35**（等比例放大，容许更长的无改进期）
- `EARLY_STOP_DIVERSITY_THRESHOLD`: 0.05 → **0.03**（放宽 diversity 判定，避免正常波动误停）
- diversity 守卫条件自动变为 `max(5, 35//3) = 11`，停滞 11 代后才检查趋同

---

## [1.4.1] - 2026-06-13

### 修复

#### 1. 早停策略误停 Bug 修复
- **问题**：条件 3（种群趋同）之前使用 `else` 分支，每代都检查 diversity，导致算法在 10-15 代因 diversity 波动被误停
- **修复**：改为 `elif no_improve_gen >= max(5, PATIENCE//3)`，仅在至少停滞 6 代后才检查 diversity，避免正常进化阶段误停

#### 2. 早停参数调优
- `EARLY_STOP_MIN_GEN`: 10 → **20**（复杂算例需要更多初始搜索代）
- `EARLY_STOP_PATIENCE`: 15 → **20**（容许更长的停滞期，给算法更多跳出局部最优的机会）

### 新增

#### 3. 批量结果可视化
- **`plot_batch_summary_all()`** 批量运行结束后自动生成三套图表：
  - 各族内对比柱状图（绿色=达 BKS、红色=未达，菱形标记 BKS 位置，显示差距 %）
  - 全局 3 面板概览图（BKS 达标率 / 平均正差距 / 总运行时间）
  - `results.csv` CSV 汇总导出
- **集成到 `main.py`**：`batch all` 和 `batch <family>` 流程结束后自动调用可视化

---

## [1.4.0] - 2026-06-13

### 新增

#### 1. 多数据集 CLI 重构
- **`main.py`** 完全重写为 `run` / `batch` 双命令架构
  - `python main.py run Mk01` — 单例运行
  - `python main.py batch mk` — 批量 Mk 系列
  - `python main.py batch all` — 全部 101 个算例
  - 保留所有向后兼容语法（`python main.py Mk01`、`python main.py all`）
- **数据集定义**：`SIMPLE_DATASETS` + `HURINK_FAMILIES` 统一管理数据路径、文件列表、BKS

#### 2. 多数据集 BKS 支持
- **Brandimarte Mk**（9 个）：Mk01-Mk09，BKS 来自 OptalCP
- **Barnes**（21 个）：mt10 / setb4 / seti5 三子族各 7 个变体
- **Dauzère**（8 个）：01a-04a、08a、09a、14a、15a（排除 open 算例）
- **Hurink car**（24 个）：car1-car8 × edata/rdata/vdata
- **Hurink ft**（9 个）：mt06/mt10/mt20 × edata/rdata/vdata
- **Hurink orb**（30 个）：orb1-orb10 × edata/rdata/vdata
- 总计 **101 个 FJSP 算例**，覆盖 4 大经典数据集族

#### 3. 滚动时域模式独立化与多数据集支持
- `_run_rolling_horizon()` 独立函数，不再嵌入 `main()` 的条件分支
- **多数据集支持**：`python main.py rolling_horizon barnes mt10c1` 等语法，可在任意数据集上运行滚动时域调度
  - `rolling_horizon` + 两个参数 = `<dataset> <instance>`，底层复用 `_resolve_single_instance()`
  - `rolling_horizon Mk02` 单参数向后兼容
  - `rolling_horizon` 无参数 → 默认 Mk01
- 移除对 `config.INSTANCE_PATH` / `config.DATA_ROOT` 的耦合，路径硬编码为 `data/Brandimarte_Data`

### 变更
- **向后兼容**：`python main.py Mk01`、`python main.py all`、`python main.py rolling_horizon` 全部保留
- 移除旧版 `bks()` 函数和 `BKS_TABLE` 字典，替换为分族 BKS 常量
- 移除 `get_instance_path()` 和旧版 `run_all_static()`，统一为 `run_single_instance()` 和 `run_batch()`

---

## [1.3.4] - 2026-06-14

### 新增

#### 1. 三条件早停策略
- **`config.py`**：新增 4 个早停参数（`EARLY_STOP_ENABLED`、`PATIENCE=15`、`MIN_GEN=10`、`DIVERSITY_THRESHOLD=0.05`）
- **`algorithms/ga.py`**：`GA.run()` 新增 `bks_value` 参数，每代迭代结束判断三种早停条件
  - 🎯 **BKS 命中**：`best_cmax <= bks_value`，小算例秒停
  - ⏳ **收敛停滞**：连续 `PATIENCE` 代无改进，避免死循环
  - 📉 **种群趋同**：多样性低于阈值，种群已收敛
  - 配合 `MIN_GEN=10` 兜底，防止起步误停
- **`main.py`**：`run_single_instance()` 和 `run_batch()` 均传入 `bks_value` 启用早停

#### 2. 仓库结构规范化
- **`.gitignore`**：`data/*.fjs` / `data/**/*.fjs` 合并为 `data/`，统一忽略整个数据目录
- **`data/**/.gitkeep`**：4 个数据集子目录各放占位文件，确保空目录被 git 跟踪
- 用户 clone 后自行放置数据文件，符合开源仓库标准实践

### 性能

| 场景 | 原耗时 | 早停后 | 节省 |
|------|--------|--------|------|
| Mk01 收敛停滞（Cmax=50） | 100 代 ~21s | **24 代 ~5s** | **76%** |
| Mk01 命中 BKS=40 | 100 代 ~21s | ~10 代 ~2s | **~90%** |

---

## [1.3.3] - 2026-06-14

### 新增

#### 1. 批量测试进度条
- **`algorithms/ga.py`**：`GA.run()` 新增 `progress_callback` 参数，每代回调当前进度
- **`main.py`**：`run_all_static()` 集成进度条，`[Mk01] ████░░░░ 40/100 | best=42 avg=90 | 25s` 实时显示每个算例的运行进度

---

## [1.3.2] - 2026-06-14

### 修复

#### 1. 显示标签修正
- **`main.py`**：所有模式标签从 "静态 GA" 修正为 "GA + ALNS + Tabu Search"，准确反映算法组成（GA + ALNS + Tabu Search 三段式流程）

#### 2. BKS 表格修正
- **OptalCP 官方值更新**：修正 BKS_TABLE 中 4 个算例的标准值
  - Mk05: 170 → **172**
  - Mk06: 52 → **57**
  - Mk07: 214 → **139**
  - Mk09: 299 → **307**
- **批量测试范围**：`run_all_static()` 循环从 `range(1, 11)`（Mk01-Mk10）调整为 `range(1, 10)`（Mk01-Mk09），因 Mk10 超出当前配置上限

### 新增

#### 3. 单元测试套件
- **新增 `tests/` 测试目录**，包含 60 个测试用例，覆盖核心算法模块
  - `test_ga.py`（28 个测试）：解码、适应度评估（标量加权）、关键路径、交叉、变异、初始化、端到端收敛
  - `test_parser.py`（8 个测试）：FJSP 文件解析、工件/机器/工序计数
  - `test_scheduler.py`（6 个测试）：`evaluate()` 函数正确性
  - `test_tabu_search.py`（18 个测试）：TS 初始化、TS1/TS2 邻域、优化流程、禁忌期限、BKS 完整性
- **测试框架**：`pytest>=7.0.0` 已加入 `requirements.txt`
- **运行方式**：`python -m pytest tests/ -v`
- **测试结果**：60/60 测试通过（~77.5s）

### 清理

#### 4. 临时调试文件删除
- 移除 `debug_analyze.py`、`debug_analyze_mk01.py`、`debug_test_lb.py` 三个调试脚本

### 性能分析（Mk01）

| 指标 | v1.3.1 | v1.3.2 | 变化 |
|------|--------|--------|------|
| Cmax | 42 | 42 | — |
| 与 BKS 差距 | 5% | 5% | — |
| 显示标签 | 静态 GA | GA + ALNS + Tabu Search | 修复 |
| BKS 正确性 | 部分错误 | ✅ OptalCP 标准 | 修复 |
| 单元测试 | ❌ 无 | ✅ 60 个测试 | 新增 |

---

## [1.3.1] - 2026-06-13

### 新增

#### 1. BKS（Best Known Solution）对比输出
- **BKS_TABLE 字典**：存储 Brandimarte Mk01-Mk10 的已知最优解
- **`bks(name)` 函数**：根据文件名返回 BKS 值，未知返回 `"-"`
- **`gap_str(cmax, bks)` 函数**：计算并格式化差距百分比
  - `✅ 0%` — 达到 BKS
  - `🔥 -X%` — 超越 BKS（负数差距）
  - `+X%` — 未达到 BKS
- **`fmt_sep(n)` 函数**：生成表格分隔线

#### 2. 格式化输出表格
- **批量模式 (`python main.py all`)**：输出带 BKS 对比的完整表格，含汇总统计
  - 列：算例, 工件, 机器, 工序, GA_Cmax, BKS, 差距, 负荷方差, 耗时
  - 汇总行：达到/超越 BKS 的数量，平均正偏差
- **单例模式 (`python main.py Mk01`)**：输出 BKS 对比框
- **滚动时域模式 (`python main.py rolling_horizon`)**：输出 RH 结果与 BKS 参考对比

### 优化

#### 3. Batch 输出优化
- 批量测试时自动关闭详细日志（`VERBOSE = False`），仅输出结果表格
- 每行包含单个算例耗时统计

### 性能分析（Mk01）

| 指标 | v1.3.0 | v1.3.1 | 变化 |
|------|--------|--------|------|
| Cmax | 42 | 42 | — |
| 与 BKS 差距 | 5% | 5% | — |
| BKS 对比输出 | ❌ 无 | ✅ 格式化表格 | 新增 |
| 批量汇总 | ❌ 无 | ✅ 有 | 新增 |

---

## [1.3.0] - 2026-06-12

### 新增

#### 1. 局部搜索（Phase 1）
- **新增 `local_search_ms(individual)`**：对关键路径上的工序尝试替换为加工时间更短的机器，若 Cmax 改善则接受
- **新增 `_find_critical_path(individual)`**：通过正向调度 + 反向回溯追踪关键路径，识别瓶颈工序
- **进化集成**：`evolve()` 中每个子代以 `ls_prob` 概率执行局部搜索，概率随代数动态调整

#### 2. Tabu Search 禁忌搜索（Phase 2）
- **新增 `algorithms/tabu_search.py`**：完整的禁忌搜索模块
- **TS1 邻域（机器重分配）**：对关键路径上每个工序，尝试所有可选机器，评估 Cmax 变化
- **TS2 邻域（工序交换）**：对关键路径上同机器的相邻工序，交换 OS 编码中的位置
- **禁忌管理**：双禁忌表（TS1 记录 `(job_id, op_id, old_machine)`，TS2 记录 `("swap", idx1, idx2)`），禁忌期限 15 代
- **渴望准则**：若禁忌移动产生历史最优解，则破禁接受
- **扰动机制**：连续 10 代无改进时，随机扰动 10% 工序的机器分配
- **GA 集成**：Gen 0 对最优个体执行 TS，每 10 代对精英执行 TS，最终对最优执行 TS

#### 3. 配置参数
- 新增 `TS_ITERATIONS = 50`：Tabu Search 迭代次数
- 新增 `TABU_TENURE = 15`：禁忌期限

### 优化

#### 4. ALNS 频率提升
- ALNS 调用间隔从每 10 代缩短至每 5 代，增加局部搜索强度

### 性能分析（Mk01）

**Cmax 从 46 降至 42**（BKS=40，差距 5%），Tabu Search 成功突破原 GA 无法逾越的 46 瓶颈：

| 指标 | v1.2.0 | v1.3.0 | 变化 |
|------|--------|--------|------|
| Cmax | 46 | 42 | -8.7% |
| 与 BKS 差距 | 15% | 5% | -10% |
| 突破瓶颈代数 | - | Gen 10 | 快速收敛 |

**关键改进**：
- Tabu Search 在 Gen 10 即将 Cmax 从 46 降至 42，证明邻域搜索的有效性
- 关键路径追踪 + 机器重分配策略精准定位瓶颈工序
- 禁忌机制有效避免搜索回溯，保持探索方向

---

## [1.2.0] - 2026-06-12

### 新增

#### 1. 负载均衡机器选择策略
- **新增 `_least_load_ms()`**：替代原有的纯贪心 MS 选择，优先选择当前累计负荷最小的机器
- **新增 `_lb_ms_init()`**：70% 概率选最小负荷机器 + 30% 概率随机选择，作为新的 MS 初始化策略
- **初始化种群调整**：25% 纯贪心 → 25% 纯混沌 → 25% 纯负载均衡 → 25% 混合策略（50%贪心+30%负载均衡+20%随机）

#### 2. 适应度函数引入负载均衡权重
- 新增 `LOAD_BALANCE_WEIGHT = 0.3` 配置参数
- 适应度函数改为 `Cmax + 0.15 × 负荷方差 + 瓶颈惩罚项`
- 当某机器负荷超过 `Cmax × 0.85` 时，额外增加惩罚

#### 3. ALNS 新增负载均衡算子
- **`destroy_high_load_machine`**：优先破坏高负荷机器上的工序
- **`repair_least_load`**：修复时综合考虑负荷均衡（40%）和加工时间（60%）
- 算子池扩展为 3 种破坏 + 2 种修复算子

#### 4. 变异操作负载均衡导向
- `mutate()` 中 MS 变异以 50% 概率选择当前负荷最小的可选机器

### 优化

#### 5. 参数调优
| 参数 | v1.1.0 | v1.2.0 | 说明 |
|------|--------|--------|------|
| LOAD_BALANCE_WEIGHT | - | 0.3 | 新增负载均衡权重 |
| PM_BOUND | [0.15, 0.40] | [0.15, 0.40] | 维持高变异率 |
| DESTROY_SIZE_MIN | 4 | 4 | 维持 |
| DESTROY_SIZE_MAX | 18 | 18 | 维持 |

### 性能分析（Mk01）

**Cmax 从 50 降至 46**（BKS=40，差距 15%），负载均衡显著改善：

| 机器 | v1.1.0 负荷 | v1.2.0 负荷 | 变化 |
|------|------------|------------|------|
| M0 | 15 | 26 | +73% |
| M1 | 46 | 42 | -9% |
| M2 | 40 | 20 | -50% |
| M3 | 36 | 34 | -6% |
| M4 | 1 | 23 | +2200% |
| M5 | 15 | 33 | +120% |

**关键改进**：
- M4 利用率从 2% 提升至 50%，彻底解决闲置问题
- 负荷方差从 260+ 降至 55.6
- 负荷均衡度 CV 从 0.8+ 降至 0.25
- 各机器利用率分布在 43%~91%，不再有极端瓶颈

---

## [1.1.0] - 2026-06-12

### 修复

#### 1. GA 浅拷贝导致 Cmax 评估不一致（关键 Bug）
- **问题**：`crossover()` 中 `parent1.copy()` 是浅拷贝，子代的 `os`/`ms` 列表与父代共享同一引用。
  当 `mutate()` 修改子代时，父代也被连带修改，导致进化过程中 `evaluate_fitness` 存储的 Cmax 与重新解码评估的结果不一致。
- **表现**：GA 内部报告 Cmax=50，但外部重新评估为 Cmax=70。
- **修复**：将 `parent.copy()` 改为显式深拷贝 `{"os": parent["os"][:], "ms": parent["ms"][:]}`。

#### 2. 种群重启丢失全局最优解
- **问题**：`_restart_population()` 在重启时使用 `min(self.population, key=lambda ind: ind["fitness"])` 选取最优个体，
  但重启后新种群的个体普遍更差，导致全局最优解丢失。
- **修复**：增加 `global_best` 参数，由 `run()` 主循环传入已保存的全局最优个体。

#### 3. 多样性指标失效
- **问题**：`_compute_diversity()` 使用 OS 编码的海明距离，由于 OS 编码是工序排列（所有工件编号出现次数固定），
  任意两个个体间的海明距离始终在 0.89 左右，无法反映种群真实多样性。
- **修复**：改为基于 Cmax 值的变异系数(CV)，能真实反映适应度层面的种群分散程度。

### 优化

#### 4. 自适应变异机制
- 新增 `_adaptive_mutate()` 方法，在算法停滞时自动增强变异强度
- `evolve()` 后对种群中非精英个体施加额外扰动，帮助跳出局部最优

#### 5. 参数调优
| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| POP_SIZE | 200 | 400 | 增大种群多样性 |
| MAX_GEN | 300 | 800 | 增加搜索代数 |
| PC_BOUND | [0.8, 0.95] | [0.6, 0.95] | 降低交叉率下限，增加探索 |
| PM_BOUND | [0.08, 0.20] | [0.15, 0.40] | 大幅提高变异率 |
| TOURNAMENT_SIZE | 3 | 4 | 增大选择压力 |
| ELITE_RATIO | 0.05 | 0.02 | 减少精英占比 |
| DESTROY_SIZE_MAX | 6 | 18 | 扩大 ALNS 破坏范围 |
| T0 | 50 | 120 | 提高退火初温 |
| SA_ALPHA | 0.92 | 0.88 | 加快退火速率 |
| ALPHA (RL) | 0.1 | 0.15 | 提高 Q 学习率 |
| GAMMA (RL) | 0.9 | 0.85 | 降低折扣因子 |
| EPSILON_DECAY | 0.995 | 0.998 | 减慢探索率衰减 |

#### 6. 重启策略优化
- 重启间隔从 80 代缩短至 50 代
- 放宽重启条件：多样性 < 0.15 **或** 连续 100 代无改进即触发重启
- 重启时明确打印保留的全局最优 Cmax

#### 7. ALNS 迭代次数动态调整
- 从固定 50 次改为 `max(50, total_ops * 2)`，根据问题规模自适应

### 清理
- 删除 `utils/scheduler.py` 中未实现的 `decode_chromosome()` 函数
- 删除空的 `rolling_output.txt` 和 `static_output.txt`

### 性能分析（Mk01）
修复后 Mk01 达到 **Cmax=50**（BKS=40，差距 25%），评估结果真实可信。

**瓶颈诊断**：
| 机器 | 负荷 | 利用率 | 状态 |
|------|------|--------|------|
| M1 | 46 | 92% | 🔴 瓶颈，几乎满载 |
| M2 | 40 | 80% | 🟡 较高 |
| M3 | 36 | 72% | 🟢 正常 |
| M0 | 15 | 30% | 🟢 低负荷 |
| M5 | 15 | 30% | 🟢 低负荷 |
| M4 | 1 | 2% | 🔴 几乎闲置 |

**根因**：贪心初始化总是选最短加工时间，导致工序往 M1 集中。M4 几乎未被使用（仅 1 个工序），但部分工序可选 M4。适应度函数 `Cmax + 0.1 × 负荷方差` 对负载均衡的惩罚不足，需要加强负载均衡导向。

---

## [0.4.0] - MK01 原型 v1.3（最终迭代）

&emsp;&emsp;引入禁忌搜索（TS）双模式深度局部优化与滚动时域控制框架，Mk01 调度质量逼近已知最优解。

| 版本 | 最大完工时间 | 迭代要点 |
|------|-------------|---------|
| v1.3 | **42.0** | TS 双模式 + RHC 框架，逼近 BKS=40 |

![v1.3 调度甘特图](output/MK01/v1.3.png)

---

## [0.3.0] - MK01 原型 v1.2

&emsp;&emsp;集成 ALNS 多算子协同搜索与 Q-learning 参数自适应，调度质量取得突破性提升。

| 版本 | 最大完工时间 | 迭代要点 |
|------|-------------|---------|
| v1.2 | **46.0** | ALNS + Q-learning，较 v1.1 下降 22% |

![v1.2 调度甘特图](output/MK01/v1.2.png)

---

## [0.2.0] - MK01 原型 v1.1

&emsp;&emsp;引入混沌初始化策略与基础局部搜索，机器负荷差距开始缩小。

| 版本 | 最大完工时间 | 迭代要点 |
|------|-------------|---------|
| v1.1 | **59.0** | 混沌初始化 + 局部搜索，压缩关键路径 |

![v1.1 调度甘特图](output/MK01/v1.1.png)

---

## [0.1.0] - MK01 原型 v1.0（初始版本）

&emsp;&emsp;采用纯贪心初始化策略实现 Mk01 的基础调度求解。

| 版本 | 最大完工时间 | 迭代要点 |
|------|-------------|---------|
| v1.0 | **65.2** | 纯贪心初始化，M2 严重过载 |

![v1.0 调度甘特图](output/MK01/v1.0.png)

---

## [1.0.0] - 初始版本

### 功能实现

#### 核心算法
- **遗传算法 (GA)**
  - 两段式编码：工序序列(OS) + 机器选择(MS)
  - 混合初始化：30% 贪心 + 30% 混沌 + 40% 混合策略
  - POX 交叉（OS）+ 两点交叉（MS）
  - 锦标赛选择 + 精英保留
  - 适应度函数：Cmax + 0.1 × 负荷方差

- **强化学习控制器 (Q-learning)**
  - 9 种状态（3 收敛度 × 3 多样性）
  - 3 种动作（探索/中性/开发）
  - ε-贪心策略 + Bellman Q 表更新
  - 动态调整 GA 交叉/变异概率

- **自适应大邻域搜索 (ALNS)**
  - 3 种破坏算子：随机破坏、关键路径破坏、高负荷机器破坏
  - 2 种修复算子：贪心修复、最少负荷修复
  - 模拟退火接受准则
  - 轮盘赌算子选择 + 权重自适应更新

#### 滚动时域调度
- 事件驱动/周期驱动混合触发
- 机器退化模拟（加工时间增加）
- 动态重调度

#### 工具模块
- 调度评估：Cmax、负荷方差、机器利用率
- 甘特图可视化
- 收敛曲线绘制
- 机器负荷分布图

#### 算例支持
- Brandimarte Mk 标准算例集（Mk01-Mk10）
- 批量测试模式

### 初始性能
- Mk01 Cmax ≈ 58（修复前）
