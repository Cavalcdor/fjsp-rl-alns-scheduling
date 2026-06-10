# algorithms/ga.py
import random
import numpy as np
from utils.scheduler import evaluate

class GA:
    def __init__(self, jobs, num_machines, config):
        """
        遗传算法初始化
        jobs: 解析后的工件数据 (已转为0索引)
        num_machines: 机器总数
        config: 配置对象（或字典）
        """
        self.jobs = jobs
        self.num_machines = num_machines
        self.num_jobs = len(jobs)
        # 计算每个工件的工序数
        self.ops_per_job = [len(job) for job in jobs]
        self.total_ops = sum(self.ops_per_job)
        
        # 参数
        self.pop_size = config.POP_SIZE
        self.max_gen = config.MAX_GEN
        self.pc_low, self.pc_high = config.PC_BOUND
        self.pm_low, self.pm_high = config.PM_BOUND
        self.tournament_size = config.TOURNAMENT_SIZE
        self.elite_count = config.ELITE_COUNT
        self.verbose = config.VERBOSE
        
        # 随机种子
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
    
    def _chaotic_sequence(self, length, x0=0.7):
        """
        生成 Logistic 混沌映射序列
        公式: x_{t+1} = 4 * x_t * (1 - x_t)
        用于种群初始化，比纯随机具有更好的遍历性和多样性
        """
        seq = np.zeros(length)
        seq[0] = x0  # 初始值（避免 0, 0.25, 0.5, 0.75, 1.0 等不动点）
        for i in range(1, length):
            seq[i] = 4.0 * seq[i - 1] * (1.0 - seq[i - 1])
        return seq
    
    def initialize_population(self):
        """初始化种群：使用 Logistic 混沌映射生成 OS 和 MS 编码"""
        population = []
        for _ in range(self.pop_size):
            # 1. OS 编码: 使用混沌序列辅助生成工序排列
            os_base = []
            for job_id, num_ops in enumerate(self.ops_per_job):
                os_base.extend([job_id] * num_ops)
            # 用混沌序列对 OS 进行排列（替代 random.shuffle）
            chaotic_os = self._chaotic_sequence(len(os_base), x0=random.random() * 0.8 + 0.1)
            os = [x for _, x in sorted(zip(chaotic_os, os_base), key=lambda pair: pair[0])]
            
            # 2. MS 编码: 使用混沌序列选择机器
            ms = []
            chaotic_ms = self._chaotic_sequence(self.total_ops, x0=random.random() * 0.8 + 0.1)
            idx = 0
            for job_id, job in enumerate(self.jobs):
                for op_id, op in enumerate(job):
                    # 混沌值映射到机器索引 [0, num_machines)
                    choice = int(chaotic_ms[idx] * len(op["machines"])) % len(op["machines"])
                    ms.append(choice)
                    idx += 1
            
            population.append({"os": os, "ms": ms, "fitness": None, "cmax": None, "load_var": None})
        return population
    
    def decode(self, individual):
        """
        解码：将 OS 和 MS 转换为 assignment 列表，同时得到每道工序实际使用的机器和加工时间
        返回: assignment = [(job_id, op_id, machine_id, duration), ...]
        """
        # 统计每个工件已经处理到的工序索引
        op_idx = [0] * self.num_jobs
        assignment = []
        # 对于 OS 中的每个工件 id
        for job_id in individual["os"]:
            # 当前工序索引
            current_op = op_idx[job_id]
            # 获取该工件该工序的可选机器和加工时间
            op_data = self.jobs[job_id][current_op]
            # 从 MS 中取出对应的机器选择（需要知道该工序在全局中的索引）
            # 为简化，我们预先构建一个全局索引映射：每个工件工序对应全局序号
            # 更好的方法：在解码时动态维护每个工件当前工序的全局索引
            # 这里先实现一个辅助函数 get_ms_index(job_id, op_id) 
            ms_index = self._get_ms_index(job_id, current_op)
            machine_choice = individual["ms"][ms_index]
            
            # 边界检查：确保机器选择索引在合法范围内
            num_available_machines = len(op_data["machines"])
            if machine_choice >= num_available_machines or machine_choice < 0:
                # 使用取模运算修正非法索引
                machine_choice = machine_choice % num_available_machines
            
            # 实际机器ID和加工时间
            machine_id = op_data["machines"][machine_choice]
            duration = op_data["times"][machine_choice]
            assignment.append((job_id, current_op, machine_id, duration))
            op_idx[job_id] += 1
        return assignment
    
    def _get_ms_index(self, job_id, op_id):
        """计算给定工件工序在 MS 编码中的全局索引（静态）"""
        # 预先计算每个工件起始索引
        if not hasattr(self, "_ms_start_idx"):
            self._ms_start_idx = []
            total = 0
            for j in range(self.num_jobs):
                self._ms_start_idx.append(total)
                total += self.ops_per_job[j]
        return self._ms_start_idx[job_id] + op_id
    
    def evaluate_fitness(self, individual):
        """评估个体适应度（Cmax 和 load_var），并存储到个体中"""
        assignment = self.decode(individual)
        # 使用调度器的 evaluate，需要传入 jobs 的适当格式
        # 注意：jobs 中的每个操作包含 machines 和 times，需要与 scheduler 中期望一致
        # 我们已经在 instance_parser 中规范为 {"machines":..., "times":...}
        cmax, load_var = evaluate(self.jobs, assignment, self.num_machines, return_details=False)
        individual["cmax"] = cmax
        individual["load_var"] = load_var
        # 适应度定义：Cmax 越小越好，负荷方差也越小越好，这里采用加权和或帕累托？
        # 由于双目标，我们采用简单的加权：fitness = cmax + w * load_var
        # 或者更常见的：主目标 Cmax，次目标负荷方差。为了选择，我们使用 Cmax 为主，当 Cmax 相同时比较负荷方差
        # 这里返回 Cmax 和 load_var 分别存储，比较时使用元组 (cmax, load_var)
        individual["fitness"] = (cmax, load_var)
        return cmax, load_var
    
    def selection_tournament(self, population):
        """锦标赛选择，返回选中的个体"""
        selected = random.sample(population, self.tournament_size)
        # 按适应度排序（Cmax 升序，然后 load_var 升序）
        selected.sort(key=lambda ind: (ind["cmax"], ind["load_var"]))
        return selected[0]
    
    def crossover(self, parent1, parent2, pc):
        """交叉操作：对 OS 和 MS 分别交叉，返回两个子代"""
        if random.random() > pc:
            # 不交叉，直接复制
            return parent1.copy(), parent2.copy()
        
        # 1. OS 交叉: 使用 POX (precedence operation crossover)
        # 随机划分工件集合为两个子集
        jobs_set = list(range(self.num_jobs))
        subset1 = set(random.sample(jobs_set, k=random.randint(1, self.num_jobs-1)))
        subset2 = set(jobs_set) - subset1
        
        child1_os = []
        child2_os = []
        # 子代1: 保留父代1中属于subset1的工件顺序，保留父代2中属于subset2的工件顺序
        for gene in parent1["os"]:
            if gene in subset1:
                child1_os.append(gene)
        for gene in parent2["os"]:
            if gene in subset2:
                child1_os.append(gene)
        # 子代2: 保留父代2中属于subset1，父代1中属于subset2
        for gene in parent2["os"]:
            if gene in subset1:
                child2_os.append(gene)
        for gene in parent1["os"]:
            if gene in subset2:
                child2_os.append(gene)
        
        # 2. MS 交叉: 两点交叉
        length = len(parent1["ms"])
        point1 = random.randint(1, length-2)
        point2 = random.randint(point1, length-1)
        child1_ms = parent1["ms"][:point1] + parent2["ms"][point1:point2] + parent1["ms"][point2:]
        child2_ms = parent2["ms"][:point1] + parent1["ms"][point1:point2] + parent2["ms"][point2:]
        
        child1 = {"os": child1_os, "ms": child1_ms}
        child2 = {"os": child2_os, "ms": child2_ms}
        return child1, child2
    
    def mutate(self, individual, pm):
        """变异操作：对 OS 和 MS 分别变异"""
        # OS 变异: 交换两个不同的基因
        if random.random() < pm:
            idx1, idx2 = random.sample(range(len(individual["os"])), 2)
            individual["os"][idx1], individual["os"][idx2] = individual["os"][idx2], individual["os"][idx1]
        
        # MS 变异: 每个位置以 pm 概率重新选择机器
        for i in range(len(individual["ms"])):
            if random.random() < pm:
                # 找出该位置对应的工件和工序
                # 需要从全局索引反推 (job_id, op_id)
                job_id, op_id = self._get_job_op_from_ms_index(i)
                op_data = self.jobs[job_id][op_id]
                new_choice = random.randint(0, len(op_data["machines"]) - 1)
                individual["ms"][i] = new_choice
        return individual
    
    def _get_job_op_from_ms_index(self, ms_idx):
        """根据 MS 编码索引反推 (job_id, op_id)"""
        if not hasattr(self, "_ms_start_idx"):
            self._ms_start_idx = []
            total = 0
            for j in range(self.num_jobs):
                self._ms_start_idx.append(total)
                total += self.ops_per_job[j]
        for job_id, start in enumerate(self._ms_start_idx):
            if ms_idx >= start and (job_id == self.num_jobs-1 or ms_idx < self._ms_start_idx[job_id+1]):
                op_id = ms_idx - start
                return job_id, op_id
        raise IndexError("Invalid ms index")
    
    def evolve(self, pc, pm):
        """执行一代进化，返回新一代种群"""
        # 选择、交叉、变异生成新种群
        new_pop = []
        # 保留精英（稍后加入，避免丢失最优）
        # 先按适应度排序
        sorted_pop = sorted(self.population, key=lambda ind: (ind["cmax"], ind["load_var"]))
        elites = sorted_pop[:self.elite_count]
        
        # 生成其余个体
        while len(new_pop) < self.pop_size - self.elite_count:
            parent1 = self.selection_tournament(self.population)
            parent2 = self.selection_tournament(self.population)
            child1, child2 = self.crossover(parent1, parent2, pc)
            child1 = self.mutate(child1, pm)
            child2 = self.mutate(child2, pm)
            # 评估新个体
            self.evaluate_fitness(child1)
            self.evaluate_fitness(child2)
            new_pop.append(child1)
            new_pop.append(child2)
        # 截断至所需数量
        new_pop = new_pop[:self.pop_size - self.elite_count]
        
        # 加入精英（需要深拷贝避免引用）
        # 关键安全检查：确保OS和MS编码长度一致
        for e in elites:
            os_len = len(e["os"])
            ms_len = len(e["ms"])
            if os_len != ms_len:
                # 如果检测到长度不匹配，发出警告并跳过该精英个体
                print(f"  [警告] 精英个体编码长度不一致: OS={os_len}, MS={ms_len}，跳过该个体")
                continue
            
            new_pop.append({
                "os": e["os"][:], 
                "ms": e["ms"][:], 
                "cmax": e["cmax"], 
                "load_var": e["load_var"], 
                "fitness": e["fitness"]
            })
        
        self.population = new_pop
        # 返回最优个体
        best = min(self.population, key=lambda ind: (ind["cmax"], ind["load_var"]))
        return best
    
    def run(self, rl_controller=None, alns=None):
        """
        主循环
        rl_controller: 可选的 RL 控制器，提供 get_actions 方法
        alns: 可选的 ALNS 优化器，用于优化精英个体
        """
        # 初始化种群
        self.population = self.initialize_population()
        # 评估初始种群
        for ind in self.population:
            self.evaluate_fitness(ind)
        
        best_individual = None
        best_cmax = float('inf')
        
        for gen in range(self.max_gen):
            # 计算当前代状态与指标，用于 RL 奖励更新
            if rl_controller is not None:
                old_best_cmax = min(ind["cmax"] for ind in self.population)
                old_avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                state = rl_controller.compute_state(self.population)
                action_idx = rl_controller.select_action(state)
                # 根据动作调整 pc, pm
                pc, pm = rl_controller.adjust_probabilities(rl_controller.current_pc, rl_controller.current_pm, action_idx)
            else:
                pc = self.pc_high  # 默认使用上限
                pm = self.pm_low
            
            # 进化一代
            current_best = self.evolve(pc, pm)
            if current_best["cmax"] < best_cmax:
                best_cmax = current_best["cmax"]
                best_individual = current_best
            
            # 可选：对精英执行 ALNS
            if alns is not None:
                # 提取当前种群最优的 elite_count 个个体
                sorted_pop = sorted(self.population, key=lambda ind: (ind["cmax"], ind["load_var"]))
                elites = sorted_pop[:self.elite_count]
                non_elites = sorted_pop[self.elite_count:]
                
                # 对精英个体执行 ALNS 局部深搜
                for i in range(len(elites)):
                    improved = alns.optimize(elites[i], self)
                    if improved["cmax"] < elites[i]["cmax"]:
                        elites[i] = improved
                
                # 精英替换策略：ALNS优化后的精英替换种群中最差的个体
                self.population = elites + non_elites

            if rl_controller is not None:
                new_best_cmax = min(ind["cmax"] for ind in self.population)
                new_avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                reward = rl_controller.compute_reward(old_best_cmax, new_best_cmax, old_avg_cmax, new_avg_cmax)
                next_state = rl_controller.compute_state(self.population)
                rl_controller.update_q_table(reward, next_state)
            
            if self.verbose and (gen+1) % 10 == 0:
                avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                print(f"Gen {gen+1}: best cmax={best_cmax}, avg cmax={avg_cmax:.2f}")
        
        return best_individual