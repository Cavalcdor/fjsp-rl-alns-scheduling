# algorithms/rl_agent.py
import numpy as np
import random
import math

class RLController:
    def __init__(self, config):
        """
        初始化 Q-learning 控制器
        config: 配置对象，需包含以下属性:
            ALPHA: 学习率
            GAMMA: 折扣因子
            EPSILON_START: 初始探索率
            EPSILON_END: 最终探索率
            EPSILON_DECAY: 探索率衰减系数
            ACTION_SPACE: 动作列表，例如 [+0.05, 0, -0.05]
            STATE_SIZE: 状态数量（默认9）
            PC_BOUND, PM_BOUND: 交叉/变异概率边界
        """
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        self.actions = config.ACTION_SPACE  # 对概率的调整步长
        self.num_actions = len(self.actions)
        self.state_size = config.STATE_SIZE  # 9种离散状态
        
        # Q表: 状态数 x 动作数
        self.q_table = np.zeros((self.state_size, self.num_actions))
        
        # 概率边界
        self.pc_min, self.pc_max = config.PC_BOUND
        self.pm_min, self.pm_max = config.PM_BOUND
        
        # 当前状态和动作（用于更新）
        self.current_state = None
        self.current_action = None
        self.current_pc = (self.pc_min + self.pc_max) / 2
        self.current_pm = (self.pm_min + self.pm_max) / 2
        
        # 记录历史奖励（用于监控）
        self.reward_history = []
    
    def compute_state(self, population):
        """
        根据种群计算当前状态（9种离散状态之一）
        状态由两个维度构成：收敛度 和 多样性
        收敛度: 0=高收敛, 1=中收敛, 2=低收敛
        多样性: 0=低多样性, 1=中多样性, 2=高多样性（基于染色体海明距离）
        返回: state_id (0~8)
        """
        # 提取所有个体的 Cmax 值
        cmax_values = [ind["cmax"] for ind in population]
        if not cmax_values:
            return 4  # 默认中间状态
        
        best = min(cmax_values)
        worst = max(cmax_values)
        avg = np.mean(cmax_values)
        
        # ---- 收敛度：基于 (avg - best) / (worst - best + 1e-6) ----
        if worst - best < 1e-6:
            convergence = 0  # 完全收敛
        else:
            ratio = (avg - best) / (worst - best)
            if ratio < 0.2:
                convergence = 0   # 高收敛
            elif ratio < 0.6:
                convergence = 1   # 中收敛
            else:
                convergence = 2   # 低收敛
        
        # ---- 多样性：基于染色体 OS 编码的海明距离（采样估算） ----
        sample_size = min(20, len(population))
        sample_indices = random.sample(range(len(population)), sample_size)
        total_hamming = 0
        pair_count = 0
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                os_i = population[sample_indices[i]]["os"]
                os_j = population[sample_indices[j]]["os"]
                # 海明距离：不同位置的数量
                hamming = sum(1 for a, b in zip(os_i, os_j) if a != b)
                total_hamming += hamming
                pair_count += 1
        
        # 归一化海明距离 (0~1)
        chrom_len = max(len(population[0]["os"]), 1)
        avg_hamming = total_hamming / max(pair_count, 1)
        norm_hamming = avg_hamming / chrom_len
        
        if norm_hamming < 0.2:
            diversity = 0   # 低多样性（个体间差异小）
        elif norm_hamming < 0.5:
            diversity = 1   # 中多样性
        else:
            diversity = 2   # 高多样性（个体间差异大）
        
        # 状态编码：state = convergence * 3 + diversity
        state_id = convergence * 3 + diversity
        return min(state_id, self.state_size - 1)
    
    def select_action(self, state):
        """ε-greedy 策略选择动作，返回动作在 actions 列表中的索引"""
        self.current_state = state
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            # 利用：选择 Q 值最大的动作
            action_idx = np.argmax(self.q_table[state, :])
        self.current_action = action_idx
        return action_idx
    
    def get_action_value(self, action_idx):
        """根据动作索引返回实际调整步长 (delta_pc, delta_pm)"""
        return self.actions[action_idx]
    
    def update_q_table(self, reward, next_state):
        """
        更新 Q 表，基于贝尔曼方程
        reward: 获得的奖励
        next_state: 执行动作后的新状态
        """
        if self.current_state is None or self.current_action is None:
            return
        
        # 当前 Q 值
        q_current = self.q_table[self.current_state, self.current_action]
        # 下一状态的最大 Q 值
        q_next_max = np.max(self.q_table[next_state, :])
        # 贝尔曼更新
        new_q = q_current + self.alpha * (reward + self.gamma * q_next_max - q_current)
        self.q_table[self.current_state, self.current_action] = new_q
        
        # 记录奖励
        self.reward_history.append(reward)
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def compute_reward(self, old_best_cmax, new_best_cmax, old_avg_cmax, new_avg_cmax):
        """
        计算奖励值（严格遵循设计文档）
        R = +10, 若当前代最优解 Cmax < 上一代最优解 Cmax
        R =   0, 若 Cmax 相等
        R =  -5, 若种群发生退化（Cmax 变差）
        """
        cmax_improvement = old_best_cmax - new_best_cmax
        if cmax_improvement > 1e-6:
            reward = 10.0   # Cmax 改善
        elif cmax_improvement < -1e-6:
            reward = -5.0   # 种群退化
        else:
            reward = 0.0    # Cmax 无变化
        
        return reward
    
    def adjust_probabilities(self, pc, pm, action_idx):
        """
        根据动作索引调整交叉概率 Pc 和变异概率 Pm（反向调整）
        动作0: Pc += delta, Pm -= delta  (增强全局搜索)
        动作1: Pc 不变, Pm 不变
        动作2: Pc -= delta, Pm += delta  (增强局部开发)
        返回: (new_pc, new_pm)
        """
        delta_pc, delta_pm = self.actions[action_idx]
        
        new_pc = np.clip(pc + delta_pc, self.pc_min, self.pc_max)
        new_pm = np.clip(pm + delta_pm, self.pm_min, self.pm_max)
        
        self.current_pc = new_pc
        self.current_pm = new_pm
        return new_pc, new_pm
    
    def reset_episode(self):
        """重置当前状态和动作（用于新的一轮滚动时域）"""
        self.current_state = None
        self.current_action = None


# 测试代码（可选）
if __name__ == "__main__":
    # 模拟配置
    class Config:
        ALPHA = 0.1
        GAMMA = 0.9
        EPSILON_START = 0.9
        EPSILON_END = 0.1
        EPSILON_DECAY = 0.995
        ACTION_SPACE = [(+0.05, -0.05), (0, 0), (-0.05, +0.05)]
        STATE_SIZE = 9
        PC_BOUND = [0.6, 0.9]
        PM_BOUND = [0.01, 0.1]
    
    cfg = Config()
    rl = RLController(cfg)
    
    # 模拟种群
    class MockIndividual:
        def __init__(self, cmax):
            self.cmax = cmax
    pop = [MockIndividual(cmax=100), MockIndividual(cmax=120), MockIndividual(cmax=110)]
    state = rl.compute_state(pop)
    print("State:", state)
    action = rl.select_action(state)
    print("Action index:", action, "delta_pc/delta_pm:", rl.actions[action])
    
    old_best = 100
    new_best = 95
    old_avg = 110
    new_avg = 105
    reward = rl.compute_reward(old_best, new_best, old_avg, new_avg)
    print("Reward:", reward)
    
    next_state = state  # 简化
    rl.update_q_table(reward, next_state)
    print("Q-table updated")
