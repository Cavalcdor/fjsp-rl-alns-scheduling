# 融合强化学习与自适应大邻域搜索的柔性车间滚动调度系统

## 📋 项目简介

本项目实现了一个**融合强化学习（RL）与自适应大邻域搜索（ALNS）的滚动时域鲁棒优化算法**，用于解决**柔性作业车间调度问题（FJSP）**。

### 核心特点

- 🎯 **三层融合架构**：GA全局搜索 + RL动态调参 + ALNS局部增强
- 🔄 **滚动时域机制**：应对动态扰动，实现持续重优化
- 📊 **多目标优化**：同时优化最大完工时间（Cmax）和设备负荷方差
- 🎲 **鲁棒性设计**：模拟设备退化、加工时间波动等真实环境扰动

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      滚动时域调度器                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  事件驱动    │  │  周期驱动    │  │  状态跟踪    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    遗传算法（GA）主框架                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  OS编码  │  │  MS编码  │  │  POX交叉 │  │  变异    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│  强化学习控制器      │    │   自适应大邻域搜索（ALNS）       │
│  ┌───────────────┐  │    │  ┌─────────┐  ┌─────────┐      │
│  │ Q-learning    │  │    │  │ 破坏算子│  │ 修复算子│      │
│  │ ε-greedy      │  │    │  └─────────┘  └─────────┘      │
│  │ 参数自适应    │  │    │  ┌─────────┐  ┌─────────┐      │
│  └───────────────┘  │    │  │ 模拟退火│  │ 权重更新│      │
└─────────────────────┘    │  └─────────┘  └─────────┘      │
                           └─────────────────────────────────┘
```

## 📁 项目结构

```
fjsp-rl-alns-scheduling/
├── algorithms/              # 算法模块
│   ├── ga.py               # 遗传算法
│   ├── rl_agent.py         # 强化学习控制器
│   └── alns.py             # 自适应大邻域搜索
├── core/                    # 核心数据模型
│   ├── instance_parser.py  # FJSP实例解析器
│   ├── job.py              # 工件类
│   ├── operation.py        # 工序类
│   ├── machine.py          # 机器类
│   └── rolling_horizon.py  # 滚动时域控制器
├── utils/                   # 工具模块
│   ├── scheduler.py        # 调度评估器
│   ├── metrics.py          # 评估指标
│   └── visualization.py    # 可视化模块
├── data/                    # 数据集
│   └── Brandimarte_Data/   # Brandimarte标准算例
├── output/                  # 输出结果（甘特图等）
├── config.py               # 全局配置
├── main.py                 # 主程序入口（单例/批量/滚动时域）
├── run_experiment.py       # 全量实验运行脚本（24算例分段持久化）
└── README.md               # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- NumPy
- Matplotlib（用于可视化）

### 安装依赖

```bash
pip install numpy matplotlib
```

### Git配置（首次使用）

如果项目中存在已跟踪的`.pyc`文件或`__pycache__`目录，需要先清理：

```bash
# 从Git索引中移除已跟踪的Python编译文件
git rm -r --cached **/__pycache__/
git commit -m "Remove tracked pycache files"
```

此后`.gitignore`规则将生效，自动忽略新生成的编译文件。

### 运行程序

```bash
# 单算例运行
python main.py Mk01

# 批量运行指定族
python main.py batch mk

# 全量 24 个代表性算例实验（分段持久化，支持断点续跑）
python run_experiment.py

# 从上次断点继续
python run_experiment.py --resume
```

### 配置参数

在 `config.py` 中修改参数：

```python
# 数据集配置
DEFAULT_INSTANCE = "Mk01.fjs"  # 测试算例

# GA参数
POP_SIZE = 100          # 种群规模
MAX_GEN = 100           # 最大迭代次数

# RL参数
ALPHA = 0.1             # 学习率
EPSILON_START = 0.9     # 初始探索率

# 滚动时域参数
DEGRADATION_COEFF = 1.05    # 设备退化系数
TIME_FLUCTUATION = 0.10     # 加工时间波动 ±10%
TRIGGER_TYPE = "event_driven"  # 触发方式
```

## 🧪 全量实验（run_experiment.py）

`run_experiment.py` 是专为 24 个代表性 FJSP 算例设计的全量实验运行脚本：

| 批次 | 算例族 | 数量 | 数据源 |
|------|--------|------|--------|
| 1 | Brandimarte Mk | 9 | `data/Brandimarte_Data/Mk01~Mk09.fjs` |
| 2 | Barnes | 5 | `data/Barnes/mt10c1~mt10xxx.fjs` |
| 3 | Dauzère | 5 | `data/Dauzere_Data/01a~08a.fjs` |
| 4 | Hurink | 5 | `data/Hurink_Data/{edata,rdata,vdata}/` |

### 核心特性

- **分段持久化**：每批结束后立即写入 checkpoint JSON + CSV
- **断点续跑**：`--resume` 跳过已完成批次
- **每批可视化**：族对比柱状图 + 收敛曲线子图
- **最终总览**：跨族对比 / Gap 散点 / 规模-耗时散点 / 24 组 Gantt + 分析图
- **零冗余计算**：schedule 内存持久化，画 Gantt 不重跑 GA

### 输出位置

```
output/
├── experiment/            # 分批 CSV + Gantt 过程图
│   ├── batch_1_mk.csv
│   └── gantt_process/     # 24 组分析图 + Gantt
└── summary/               # 总览图表
    ├── results.csv        # 主 CSV（utf-8-sig）
    ├── checkpoint.json    # 断点文件
    ├── family_comparison.png
    ├── convergence_curves.png
    ├── global_gap_scatter.png
    ├── scale_vs_runtime.png
    └── overview.png
```

## 📊 核心算法说明

### 1. 遗传算法（GA）

- **编码方式**：OS（工序顺序）+ MS（机器选择）双编码
- **交叉操作**：POX交叉（工序顺序）+ 两点交叉（机器选择）
- **变异操作**：位置交换变异 + 机器重选变异
- **选择策略**：锦标赛选择 + 精英保留

### 2. 强化学习控制器

- **状态空间**：9种离散状态（收敛度×多样性）
- **动作空间**：调整交叉/变异概率（±0.05或不变）
- **奖励函数**：基于Cmax改进和种群平均适应度提升
- **学习算法**：Q-learning + ε-greedy探索策略

### 3. 自适应大邻域搜索（ALNS）

**破坏算子**：
- 随机破坏
- 关键路径破坏（优先破坏决定makespan的工序）
- 高负荷机器破坏

**修复算子**：
- 贪心修复（保持原机器选择）
- 负荷均衡修复（重新选择负荷最小的机器）

### 4. 滚动时域调度

- **窗口化执行**：规划整个窗口，仅执行前半段
- **触发机制**：事件驱动 / 周期驱动 / 混合驱动
- **扰动模拟**：
  - 设备退化：加工时间 × 退化系数^(加工次数)
  - 随机波动：正态分布 ±10%

## 📈 输出结果

程序运行后会输出：

1. **调度评估指标**：
   - 最大完工时间（Cmax）
   - 负荷方差
   - 平均机器利用率
   - 各机器负荷分布

2. **可视化图表**（保存至 `output/` 目录）：
   - 调度甘特图
   - 机器负荷分布图
   - 工件完工时间图

## 🔧 自定义扩展

### 添加新的破坏算子

在 `algorithms/alns.py` 中添加：

```python
def destroy_custom(self, individual, destroy_size):
    # 你的破坏逻辑
    return new_individual, removed
```

然后在 `__init__` 中注册：

```python
self.destroy_operators.append(self.destroy_custom)
```

### 使用其他数据集

1. 将数据集放入 `data/` 目录
2. 修改 `config.py`：

```python
DATA_ROOT = "data/your_dataset"
DEFAULT_INSTANCE = "your_instance.fjs"
```

## 📝 数据集格式

支持 Brandimarte 格式（.fjs）：

```
第一行：工件数 机器数 [平均可选机器数]
后续每行：工序数 (可选机器数 机器ID 加工时间) ...

示例：
10 15 2
5  2 1 3 2 5  2 3 4 4 6  2 5 3 6 4  1 7 6  2 8 3 9 4
...
```

## 📚 参考文献


## 📄 许可证

MIT License

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

---

**作者**: 智能制造调度研究组  
**版本**: 1.5.0  
**最后更新**: 2026-06-13
