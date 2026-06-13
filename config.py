# config.py
# 融合强化学习与自适应大邻域搜索的滚动时域鲁棒优化算法
# 全局控制参数表

import os

# ========== 1. 文件路径配置 ==========
# 数据集根目录
DATA_ROOT = "data/Brandimarte_Data"

# 默认测试算例
DEFAULT_INSTANCE = "Mk01.fjs"

# 完整路径
INSTANCE_PATH = os.path.join(DATA_ROOT, DEFAULT_INSTANCE)

# ========== 2. 滚动时域与扰动参数 ==========
# 设备退化系数
# 如需禁用退化用于教学演示或原型验证，设为 1.0
DEGRADATION_COEFF = 1.0           # 设为1.0禁用退化
TIME_FLUCTUATION = 0.10           # 加工时间随机波动 ±10%
TRIGGER_TYPE = "event_driven"     # 重调度触发方式：event_driven / periodic
PERIODIC_INTERVAL = 100            # 周期驱动的时间间隔

# ========== 3. 遗传算法参数 ==========
POP_SIZE = 400                     # 种群规模
MAX_GEN = 50                       # 最大迭代次数
PC_BOUND = [0.6, 0.95]            # 交叉概率动态调整边界
PM_BOUND = [0.15, 0.40]           # 变异概率动态调整边界
TOURNAMENT_SIZE = 4                # 锦标赛选择规模
ELITE_RATIO = 0.02                 # 精英保留比例
ELITE_COUNT = int(POP_SIZE * ELITE_RATIO)  # 精英数量

# ========== 4. 强化学习参数 ==========
ALPHA = 0.15                       # 学习率
GAMMA = 0.85                       # 折扣因子
EPSILON_START = 0.9                # 初始探索率
EPSILON_END = 0.05                 # 最终探索率
EPSILON_DECAY = 0.998              # 探索率衰减系数

# 动作空间：3个动作，每个动作为 (delta_pc, delta_pm)
# 动作0: Pc增大, Pm减小
# 动作1: Pc/Pm不变
# 动作2: Pc减小, Pm增大
ACTION_SPACE = [(+0.08, -0.05), (0, 0), (-0.05, +0.08)]

# 状态空间定义
# 收敛度: 0=高收敛, 1=中收敛, 2=低收敛
# 多样性: 0=低多样性, 1=中多样性, 2=高多样性
STATE_SIZE = 9                     # 3x3 组合

# ========== 5. 自适应大邻域搜索参数 ==========
RHO = 0.4                          # 算子权重衰减系数
REWARD_GLOBAL_BEST = 15            # 找到全局最优解的奖励
REWARD_BETTER = 6                  # 找到比当前解更好的解的奖励
REWARD_ACCEPT = 2                  # 接受劣质解的奖励
DESTROY_SIZE_MIN = 4               # 破坏工序数最小值
DESTROY_SIZE_MAX = 18              # 破坏工序数最大值
T0 = 120                           # 模拟退火初始温度
SA_ALPHA = 0.88                    # 退火速率

# ========== 6. 禁忌搜索参数 (Tabu Search, 阶段二改进) ==========
TS_ITERATIONS = 80                 # 禁忌搜索迭代次数
TABU_TENURE = 15                   # 禁忌步长

# ========== 7. 随机性与日志 ==========
RANDOM_SEED = 42                   # 随机种子
VERBOSE = True                     # 是否打印详细日志

# ========== 8. 早停策略参数 ==========
# 三种早停条件（任一满足即停）：
#   1) BKS 命中: best_cmax <= bks_value（需外部传入 bks_value）
#   2) 收敛停滞: 连续 no_improve_gen >= PATIENCE 代无改进
#   3) 种群趋同: 多样性 diversity < DIVERSITY_THRESHOLD
EARLY_STOP_ENABLED = True                 # 总开关
EARLY_STOP_PATIENCE = 20                  # 停滞代数上限（超过此值触发早停）
EARLY_STOP_MIN_GEN = 20                   # 强制最小代数（防止起步阶段误停）
EARLY_STOP_DIVERSITY_THRESHOLD = 0.05     # 多样性阈值（低于此值视为趋同）

# ========== 9. 辅助函数 ==========
def update_instance_path(instance_name):
    """动态修改当前使用的算例"""
    global INSTANCE_PATH, DEFAULT_INSTANCE
    DEFAULT_INSTANCE = instance_name
    INSTANCE_PATH = os.path.join(DATA_ROOT, instance_name)