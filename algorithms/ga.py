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
    
    def _greedy_ms(self, job_id, op_id):
        """贪婪 MS 选择：选择加工时间最短的机器"""
        op_data = self.jobs[job_id][op_id]
        times = op_data["times"]
        min_time = min(times)
        # 如果有多个相同最小时长，随机选一个
        candidates = [i for i, t in enumerate(times) if t == min_time]
        return random.choice(candidates)

    def initialize_population(self):
        """初始化种群：混合策略——部分贪心+部分混沌"""
        population = []
        # 策略分配：30%纯贪心, 30%纯混沌, 40%混沌OS+贪心MS
        for i in range(self.pop_size):
            # OS 编码: 混沌排列
            os_base = []
            for job_id, num_ops in enumerate(self.ops_per_job):
                os_base.extend([job_id] * num_ops)
            chaotic_os = self._chaotic_sequence(len(os_base), x0=random.random() * 0.8 + 0.1)
            os = [x for _, x in sorted(zip(chaotic_os, os_base), key=lambda pair: pair[0])]

            ms = []
            if i < self.pop_size * 0.3:
                # 策略1: 纯贪心 MS（全部选最短加工时间）
                for job_id, job in enumerate(self.jobs):
                    for op_id in range(len(job)):
                        ms.append(self._greedy_ms(job_id, op_id))
            elif i < self.pop_size * 0.6:
                # 策略2: 纯混沌 MS
                chaotic_ms = self._chaotic_sequence(self.total_ops, x0=random.random() * 0.8 + 0.1)
                idx = 0
                for job_id, job in enumerate(self.jobs):
                    for op in job:
                        choice = int(chaotic_ms[idx] * len(op["machines"])) % len(op["machines"])
                        ms.append(choice)
                        idx += 1
            else:
                # 策略3: 混合——70%概率贪心, 30%概率随机
                for job_id, job in enumerate(self.jobs):
                    for op_id in range(len(job)):
                        if random.random() < 0.7:
                            ms.append(self._greedy_ms(job_id, op_id))
                        else:
                            op = job[op_id]
                            ms.append(random.randint(0, len(op["machines"]) - 1))

            population.append({"os": os, "ms": ms, "fitness": None, "cmax": None, "load_var": None})
        return population
    
    def decode(self, individual):
        """
        解码：将 OS 和 MS 转换为 assignment 列表
        返回: assignment = [(job_id, op_id, machine_id, duration), ...]
        注意：decode 是确定性操作，无随机性，保证每次结果一致
        """
        op_idx = [0] * self.num_jobs
        assignment = []
        for job_id in individual["os"]:
            current_op = op_idx[job_id]
            op_data = self.jobs[job_id][current_op]
            ms_index = self._get_ms_index(job_id, current_op)
            machine_choice = individual["ms"][ms_index]
            # 确定性修正：取模确保在合法范围
            num_available = len(op_data["machines"])
            machine_choice = machine_choice % num_available
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
        assignment = self.decode(individual)
        cmax, load_var = evaluate(self.jobs, assignment, self.num_machines, return_details=False)
        individual["cmax"] = cmax
        individual["load_var"] = load_var
        # 适应度使用加权和：主目标 Cmax，辅以负荷方差（归一化权重0.1）
        # 这样在锦标赛选择时能同时考虑两个目标，避免只盯着Cmax导致负荷失衡
        individual["fitness"] = cmax + 0.1 * load_var
        return cmax, load_var
    
    def selection_tournament(self, population):
        selected = random.sample(population, self.tournament_size)
        # 按适应度排序（Cmax 升序，然后 load_var 升序）
        selected.sort(key=lambda ind: ind["fitness"])
        return selected[0]
    
    def crossover(self, parent1, parent2, pc):
        """交叉操作：对 OS 和 MS 分别交叉，返回两个子代"""
        if random.random() > pc:
            return parent1.copy(), parent2.copy()
        
        # OS 交叉: POX (precedence operation crossover)
        jobs_set = list(range(self.num_jobs))
        if self.num_jobs <= 1:
            child1_os = parent1["os"][:]
            child2_os = parent2["os"][:]
        else:
            subset1 = set(random.sample(jobs_set, k=random.randint(1, self.num_jobs-1)))
            subset2 = set(jobs_set) - subset1
            child1_os = []
            child2_os = []
            for gene in parent1["os"]:
                if gene in subset1:
                    child1_os.append(gene)
            for gene in parent2["os"]:
                if gene in subset2:
                    child1_os.append(gene)
            for gene in parent2["os"]:
                if gene in subset1:
                    child2_os.append(gene)
            for gene in parent1["os"]:
                if gene in subset2:
                    child2_os.append(gene)
        
        # 2. MS 交叉: 两点交叉
        length = len(parent1["ms"])
        # 边界保护：编码长度过短时无法执行两点交叉，直接交换或复制
        if length <= 1:
            child1_ms = parent1["ms"][:]
            child2_ms = parent2["ms"][:]
        elif length == 2:
            # 长度为2时直接交换两个基因
            child1_ms = [parent2["ms"][0], parent1["ms"][1]]
            child2_ms = [parent1["ms"][0], parent2["ms"][1]]
        else:
            point1 = random.randint(1, length-2)
            point2 = random.randint(point1, length-1)
            child1_ms = parent1["ms"][:point1] + parent2["ms"][point1:point2] + parent1["ms"][point2:]
            child2_ms = parent2["ms"][:point1] + parent1["ms"][point1:point2] + parent2["ms"][point2:]
        
        child1 = {"os": child1_os, "ms": child1_ms}
        child2 = {"os": child2_os, "ms": child2_ms}
        return child1, child2
    
    def mutate(self, individual, pm, gen_progress=0.0):
        """变异操作：OS 交换 + MS 随机重选"""
        os_len = len(individual["os"])
        if os_len >= 2 and random.random() < pm:
            num_swaps = random.randint(1, max(1, os_len // 10))
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(os_len), 2)
                individual["os"][idx1], individual["os"][idx2] = individual["os"][idx2], individual["os"][idx1]
        # MS 变异率略高于 pm，维持探索能力
        ms_pm = min(pm * 1.5, 0.5)
        for i in range(len(individual["ms"])):
            if random.random() < ms_pm:
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

    def _compute_diversity(self):
        """计算种群多样性（OS海明距离采样）"""
        if len(self.population) < 2:
            return 0.0
        sample_size = min(30, len(self.population))
        sample_indices = random.sample(range(len(self.population)), sample_size)
        total_hamming = 0
        count = 0
        chrom_len = max(len(self.population[0]["os"]), 1)
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                os_i = self.population[sample_indices[i]]["os"]
                os_j = self.population[sample_indices[j]]["os"]
                hamming = sum(1 for a, b in zip(os_i, os_j) if a != b)
                total_hamming += hamming
                count += 1
        avg_hamming = total_hamming / max(count, 1)
        return avg_hamming / chrom_len

    def _restart_population(self, keep_best=True):
        """重启种群：保留最优个体，其余重新初始化"""
        if keep_best:
            best = min(self.population, key=lambda ind: ind["fitness"])
            new_pop = [{
                "os": best["os"][:],
                "ms": best["ms"][:],
                "cmax": best["cmax"],
                "load_var": best["load_var"],
                "fitness": best["fitness"]
            }]
            rest = self.initialize_population()
            new_pop.extend(rest[:self.pop_size - 1])
        else:
            new_pop = self.initialize_population()
            best = None
        for ind in new_pop:
            if ind["fitness"] is None:
                self.evaluate_fitness(ind)
        self.population = new_pop
        if self.verbose and best is not None:
            print(f"  [重启] 种群已重启，保留最优 Cmax={best['cmax']}")

    def evolve(self, pc, pm, gen_progress=0.0):
        """执行一代进化，返回新一代种群"""
        new_pop = []
        sorted_pop = sorted(self.population, key=lambda ind: ind["fitness"])
        elites = sorted_pop[:self.elite_count]

        while len(new_pop) < self.pop_size - self.elite_count:
            parent1 = self.selection_tournament(self.population)
            parent2 = self.selection_tournament(self.population)
            child1, child2 = self.crossover(parent1, parent2, pc)
            child1 = self.mutate(child1, pm, gen_progress)
            child2 = self.mutate(child2, pm, gen_progress)
            self.evaluate_fitness(child1)
            self.evaluate_fitness(child2)
            new_pop.append(child1)
            new_pop.append(child2)
        new_pop = new_pop[:self.pop_size - self.elite_count]
        
        # 精英保留策略：用精英个体替换新种群中最差的个体（保持种群规模不变）
        # 先将新种群按适应度排序（最差在最后）
        new_pop.sort(key=lambda ind: ind["fitness"])
        # 从末尾开始替换最差的个体
        replace_count = min(self.elite_count, len(new_pop))
        for i in range(replace_count):
            e = elites[i]
            os_len = len(e["os"])
            ms_len = len(e["ms"])
            # 安全检查：确保编码长度一致
            if os_len != ms_len or os_len == 0:
                continue
            # 用精英替换新种群中最差的那个
            new_pop[-(i+1)] = {
                "os": e["os"][:], 
                "ms": e["ms"][:], 
                "cmax": e["cmax"], 
                "load_var": e["load_var"], 
                "fitness": e["fitness"]
            }
        
        self.population = new_pop
        # 返回最优个体
        best = min(self.population, key=lambda ind: ind["fitness"])
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
        
        best_individual = min(self.population, key=lambda ind: ind["fitness"])
        best_cmax = best_individual["cmax"]
        no_improve_gen = 0
        restart_interval = 80

        for gen in range(self.max_gen):
            gen_progress = gen / max(self.max_gen, 1)
            if rl_controller is not None:
                old_best_cmax = min(ind["cmax"] for ind in self.population)
                old_avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                state = rl_controller.compute_state(self.population)
                action_idx = rl_controller.select_action(state)
                pc, pm = rl_controller.adjust_probabilities(rl_controller.current_pc, rl_controller.current_pm, action_idx)
            else:
                pc = self.pc_high
                pm = self.pm_low

            self.evolve(pc, pm, gen_progress)

            current_best = min(self.population, key=lambda ind: ind["fitness"])
            if current_best["cmax"] < best_cmax:
                best_cmax = current_best["cmax"]
                best_individual = current_best
                no_improve_gen = 0
            else:
                no_improve_gen += 1

            # ALNS 每10代执行一次
            if alns is not None and (gen + 1) % 10 == 0:
                sorted_pop = sorted(self.population, key=lambda ind: ind["fitness"])
                elites = sorted_pop[:self.elite_count]
                non_elites = sorted_pop[self.elite_count:]
                for i in range(len(elites)):
                    improved = alns.optimize(elites[i], self)
                    if improved["cmax"] < elites[i]["cmax"]:
                        elites[i] = improved
                self.population = elites + non_elites

            if rl_controller is not None:
                new_best_cmax = min(ind["cmax"] for ind in self.population)
                new_avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                reward = rl_controller.compute_reward(old_best_cmax, new_best_cmax, old_avg_cmax, new_avg_cmax)
                next_state = rl_controller.compute_state(self.population)
                rl_controller.update_q_table(reward, next_state)

            # 多样性监控 + 重启
            if no_improve_gen > 0 and no_improve_gen % restart_interval == 0:
                diversity = self._compute_diversity()
                if diversity < 0.15:
                    self._restart_population(keep_best=True)
                    no_improve_gen = 0

            if self.verbose and (gen + 1) % 10 == 0:
                avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                diversity = self._compute_diversity()
                print(f"Gen {gen+1}: best cmax={best_cmax}, avg cmax={avg_cmax:.2f}, diversity={diversity:.3f}")

        return best_individual