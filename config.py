# config.py
# 融合强化学习与自适应大邻域搜索的滚动时域鲁棒优化算法
# 全局控制参数表（依据设计文档与算法参数表）

import os

# ========== 1. 文件路径配置 ==========
# 数据集根目录
DATA_ROOT = "data/Brandimarte_Data"

# 默认测试算例（可改为 Mk02.fjs, Mk03.fjs, ...）
DEFAULT_INSTANCE = "Mk01.fjs"

# 完整路径（程序自动拼接）
INSTANCE_PATH = os.path.join(DATA_ROOT, DEFAULT_INSTANCE)

# ========== 2. 滚动时域与扰动参数 ==========
# 设备退化系数（温和退化，每次加工后耗时 *= 1.02，配合上限1.5倍）
# 如需禁用退化用于教学演示或原型验证，设为 1.0
DEGRADATION_COEFF = 1.0           # 设为1.0禁用退化（在GA基线验证后如需退化可改回1.02）
TIME_FLUCTUATION = 0.10           # 加工时间随机波动 ±10%（正态分布）
TRIGGER_TYPE = "event_driven"     # 重调度触发方式：event_driven / periodic
PERIODIC_INTERVAL = 100            # 周期驱动的时间间隔（若采用）

# ========== 3. 遗传算法参数 ==========
POP_SIZE = 200                     # 种群规模（原来150，增大多样性）
MAX_GEN = 300                      # 最大迭代次数（原来200）
PC_BOUND = [0.8, 0.95]            # 交叉概率动态调整边界（提高最低交叉率）
PM_BOUND = [0.08, 0.20]           # 变异概率动态调整边界（提高最低变异率防止早熟）
TOURNAMENT_SIZE = 3                # 锦标赛选择规模
ELITE_RATIO = 0.05                 # 精英保留比例（减少到5%，避免精英过快主导）
ELITE_COUNT = int(POP_SIZE * ELITE_RATIO)  # 精英数量（10）

# ========== 4. 强化学习参数（Q-learning） ==========
ALPHA = 0.1                        # 学习率
GAMMA = 0.9                        # 折扣因子
EPSILON_START = 0.9                # 初始探索率
EPSILON_END = 0.1                  # 最终探索率
EPSILON_DECAY = 0.995              # 探索率衰减系数（每代衰减）

# 动作空间：3个动作，每个动作为 (delta_pc, delta_pm)
# 动作0: Pc增大, Pm减小（加强搜索多样性，跳出局部最优）
# 动作1: Pc/Pm不变
# 动作2: Pc减小, Pm增大（加强局部开发）
ACTION_SPACE = [(+0.05, -0.05), (0, 0), (-0.05, +0.05)]

# 状态空间定义（9种状态，由收敛度+多样性组合）
# 收敛度: 0=高收敛, 1=中收敛, 2=低收敛
# 多样性: 0=低多样性, 1=中多样性, 2=高多样性（基于海明距离，值越大差异越大）
STATE_SIZE = 9                     # 3x3 组合

# ========== 5. 自适应大邻域搜索参数 ==========
RHO = 0.3                          # 算子权重衰减系数（原来0.5，加快权重自适应速度）
REWARD_GLOBAL_BEST = 10            # 找到全局最优解的奖励
REWARD_BETTER = 5                  # 找到比当前解更好的解的奖励
REWARD_ACCEPT = 1                  # 接受劣质解的奖励（原来2，降低劣质解权重）
DESTROY_SIZE_MIN = 2               # 破坏工序数最小值
DESTROY_SIZE_MAX = 6               # 破坏工序数最大值（原来5，扩大搜索范围）
T0 = 50                            # 模拟退火初始温度（原来100）
SA_ALPHA = 0.92                    # 退火速率（原来0.95，降温更快减少劣质接受）

# ========== 6. 随机性与日志 ==========
RANDOM_SEED = 42                   # 随机种子
VERBOSE = True                     # 是否打印详细日志

# ========== 7. 辅助函数 ==========
def update_instance_path(instance_name):
    """动态修改当前使用的算例"""
    global INSTANCE_PATH, DEFAULT_INSTANCE
    DEFAULT_INSTANCE = instance_name
    INSTANCE_PATH = os.path.join(DATA_ROOT, instance_name)