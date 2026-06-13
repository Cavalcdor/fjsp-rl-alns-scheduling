# algorithms/ga.py
import random
import time
import numpy as np
from utils.scheduler import evaluate
import config


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
        """初始化种群：混合策略——部分贪心+部分混沌+部分负载均衡"""
        population = []
        # 策略分配：25%纯贪心, 25%纯混沌, 25%负载均衡, 25%混合
        for i in range(self.pop_size):
            # OS 编码: 混沌排列
            os_base = []
            for job_id, num_ops in enumerate(self.ops_per_job):
                os_base.extend([job_id] * num_ops)
            chaotic_os = self._chaotic_sequence(len(os_base), x0=random.random() * 0.8 + 0.1)
            os = [x for _, x in sorted(zip(chaotic_os, os_base), key=lambda pair: pair[0])]

            ms = []
            if i < self.pop_size * 0.25:
                # 策略1: 纯贪心 MS（全部选最短加工时间）
                for job_id, job in enumerate(self.jobs):
                    for op_id in range(len(job)):
                        ms.append(self._greedy_ms(job_id, op_id))
            elif i < self.pop_size * 0.50:
                # 策略2: 纯混沌 MS
                chaotic_ms = self._chaotic_sequence(self.total_ops, x0=random.random() * 0.8 + 0.1)
                idx = 0
                for job_id, job in enumerate(self.jobs):
                    for op in job:
                        choice = int(chaotic_ms[idx] * len(op["machines"])) % len(op["machines"])
                        ms.append(choice)
                        idx += 1
            elif i < self.pop_size * 0.75:
                # 策略3: 负载均衡 MS——选择当前累计负荷最小的机器
                machine_load = [0] * self.num_machines
                for job_id, job in enumerate(self.jobs):
                    for op_id in range(len(job)):
                        op = job[op_id]
                        min_load = float('inf')
                        best_choice = 0
                        for ci, machine_id in enumerate(op["machines"]):
                            effective_load = machine_load[machine_id] + op["times"][ci]
                            if effective_load < min_load:
                                min_load = effective_load
                                best_choice = ci
                        ms.append(best_choice)
                        machine_load[op["machines"][best_choice]] += op["times"][best_choice]
            else:
                # 策略4: 混合——50%概率贪心, 30%概率负载均衡, 20%概率随机
                machine_load = [0] * self.num_machines
                for job_id, job in enumerate(self.jobs):
                    for op_id in range(len(job)):
                        op = job[op_id]
                        r = random.random()
                        if r < 0.5:
                            ms.append(self._greedy_ms(job_id, op_id))
                        elif r < 0.8:
                            # 负载均衡选择
                            min_load = float('inf')
                            best_choice = 0
                            for ci, machine_id in enumerate(op["machines"]):
                                effective_load = machine_load[machine_id] + op["times"][ci]
                                if effective_load < min_load:
                                    min_load = effective_load
                                    best_choice = ci
                            ms.append(best_choice)
                            machine_load[op["machines"][best_choice]] += op["times"][best_choice]
                        else:
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
        cmax, load_var, details = evaluate(self.jobs, assignment, self.num_machines, return_details=True)
        individual["cmax"] = cmax
        individual["load_var"] = load_var
        # 适应度使用加权和：主目标 Cmax，辅以负荷方差
        # 负荷方差权重 0.15，在鼓励负载均衡的同时不压制 Cmax 优化
        # 注意：Cmax 是首要目标，负载均衡是次要目标
        machine_load = details['machine_load']
        max_load = max(machine_load)
        # 只有当 max_load 明显高于 cmax 时才惩罚（说明有机器成为瓶颈）
        load_penalty = 0.15 * load_var
        if max_load > cmax * 0.85:
            load_penalty += 0.1 * (max_load - cmax * 0.85)
        individual["fitness"] = cmax + load_penalty
        return cmax, load_var
    
    def selection_tournament(self, population):
        selected = random.sample(population, self.tournament_size)
        # 按适应度排序（Cmax 升序，然后 load_var 升序）
        selected.sort(key=lambda ind: ind["fitness"])
        return selected[0]
    
    def crossover(self, parent1, parent2, pc):
        """交叉操作：对 OS 和 MS 分别交叉，返回两个子代"""
        if random.random() > pc:
            # 深拷贝：避免子代与父代共享 os/ms 列表引用
            return {
                "os": parent1["os"][:],
                "ms": parent1["ms"][:]
            }, {
                "os": parent2["os"][:],
                "ms": parent2["ms"][:]
            }
        
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
        """变异操作：OS 交换 + MS 随机重选（含负载均衡导向）"""
        os_len = len(individual["os"])
        if os_len >= 2 and random.random() < pm:
            num_swaps = random.randint(1, max(1, os_len // 10))
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(os_len), 2)
                individual["os"][idx1], individual["os"][idx2] = individual["os"][idx2], individual["os"][idx1]
        # MS 变异率略高于 pm，维持探索能力
        ms_pm = min(pm * 1.5, 0.5)
        # 先计算机器当前负荷（用于负载均衡导向的变异）
        machine_load = [0] * self.num_machines
        job_op_counter = [0] * self.num_jobs
        for i, job_id in enumerate(individual["os"]):
            op_id = job_op_counter[job_id]
            ms_index = self._get_ms_index(job_id, op_id)
            machine_choice = individual["ms"][ms_index]
            op_data = self.jobs[job_id][op_id]
            if machine_choice < len(op_data["machines"]):
                machine_id = op_data["machines"][machine_choice]
                duration = op_data["times"][machine_choice]
                machine_load[machine_id] += duration
            job_op_counter[job_id] += 1
        
        for i in range(len(individual["ms"])):
            if random.random() < ms_pm:
                job_id, op_id = self._get_job_op_from_ms_index(i)
                op_data = self.jobs[job_id][op_id]
                # 负载均衡导向：以 50% 概率选择当前负荷最小的可选机器
                if random.random() < 0.5:
                    min_load = float('inf')
                    best_choice = 0
                    for ci, machine_id in enumerate(op_data["machines"]):
                        effective_load = machine_load[machine_id] + op_data["times"][ci]
                        if effective_load < min_load:
                            min_load = effective_load
                            best_choice = ci
                    new_choice = best_choice
                else:
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

    def _find_critical_path(self, individual):
        """
        找出个体的关键路径工序
        返回: list of dict, 每个元素包含 job_id, op_id, machine_id, start, end, duration
        关键路径 = 决定 makespan 的最长路径
        """
        machine_available = [0] * self.num_machines
        job_completion = [0] * self.num_jobs
        op_schedule = []
        
        job_op_counter = [0] * self.num_jobs
        for i, job_id in enumerate(individual["os"]):
            op_id = job_op_counter[job_id]
            ms_index = self._get_ms_index(job_id, op_id)
            machine_choice = individual["ms"][ms_index]
            op_data = self.jobs[job_id][op_id]
            
            num_available = len(op_data["machines"])
            machine_choice = machine_choice % num_available
            machine_id = op_data["machines"][machine_choice]
            duration = op_data["times"][machine_choice]
            
            start = max(machine_available[machine_id], job_completion[job_id])
            end = start + duration
            
            op_schedule.append({
                "os_idx": i,
                "job_id": job_id,
                "op_id": op_id,
                "machine_id": machine_id,
                "machine_choice": machine_choice,
                "start": start,
                "end": end,
                "duration": duration
            })
            machine_available[machine_id] = end
            job_completion[job_id] = end
            job_op_counter[job_id] += 1
        
        makespan = max(job_completion)
        
        # 反向追踪关键路径：从结束时间 == makespan 的工序开始
        critical_ops = []
        # 先找出所有结束时间等于 makespan 的工序
        end_candidates = [op for op in op_schedule if abs(op["end"] - makespan) < 1e-6]
        if not end_candidates:
            return op_schedule  # 如果没有明确的，返回全部
        
        visited = set()
        from collections import deque
        queue = deque(end_candidates)
        
        while queue:
            op = queue.popleft()
            if op["os_idx"] in visited:
                continue
            visited.add(op["os_idx"])
            critical_ops.append(op)
            
            # 找前驱工序：
            # 1. 同一机器上紧邻的前一道工序
            for other in op_schedule:
                if other["machine_id"] == op["machine_id"] and other["end"] <= op["start"] + 1e-6:
                    if other["os_idx"] not in visited:
                        # 找结束时间最接近当前工序开始的那个
                        if abs(other["end"] - op["start"]) < 1e-6 or (other["end"] <= op["start"] and other not in queue):
                            queue.append(other)
            
            # 2. 同一工件的前一道工序
            for other in op_schedule:
                if other["job_id"] == op["job_id"] and other["op_id"] == op["op_id"] - 1:
                    if other["os_idx"] not in visited:
                        queue.append(other)
        
        return critical_ops if critical_ops else op_schedule[:max(1, len(op_schedule)//3)]

    def local_search_ms(self, individual):
        """
        轻量级局部搜索：对关键路径上的工序，尝试所有可选机器
        只修改 MS（机器选择），不修改 OS（工序顺序）
        这是参考 HA_FJSP 的 TS1 思想，但简化版（无禁忌表）
        返回: 改进后的个体（原地修改）
        """
        # 确保个体已评估
        if individual.get("cmax") is None:
            self.evaluate_fitness(individual)
        
        # 找出关键路径
        critical_ops = self._find_critical_path(individual)
        
        # 对关键路径上的每道工序，尝试换机器
        improved = False
        for op_info in critical_ops:
            job_id = op_info["job_id"]
            op_id = op_info["op_id"]
            op_data = self.jobs[job_id][op_id]
            
            if len(op_data["machines"]) <= 1:
                continue  # 只有一台可选机器，跳过
            
            # 获取当前 MS 索引
            ms_idx = self._get_ms_index(job_id, op_id)
            old_choice = individual["ms"][ms_idx]
            
            # 尝试所有其他可选机器
            best_choice = old_choice
            best_cmax = individual["cmax"]
            
            for alt_idx in range(len(op_data["machines"])):
                if alt_idx == old_choice:
                    continue
                # 临时修改 MS
                individual["ms"][ms_idx] = alt_idx
                # 快速评估
                new_cmax, _, _ = evaluate(self.jobs, self.decode(individual), self.num_machines, return_details=True)
                if new_cmax < best_cmax:
                    best_cmax = new_cmax
                    best_choice = alt_idx
                    improved = True
            
            # 恢复或更新
            individual["ms"][ms_idx] = best_choice
        
        if improved:
            # 重新完整评估
            self.evaluate_fitness(individual)
        
        return individual

    def _compute_diversity(self):
        """计算种群多样性（基于适应度值的变异系数）"""
        if len(self.population) < 2:
            return 0.0
        cmax_values = [ind["cmax"] for ind in self.population]
        mean_cmax = np.mean(cmax_values)
        if mean_cmax < 1e-6:
            return 0.0
        std_cmax = np.std(cmax_values)
        cv = std_cmax / mean_cmax  # 变异系数
        # 归一化到 0~1 之间（通常 CV 在 0~0.5 之间）
        return min(cv * 2.0, 1.0)

    def _restart_population(self, keep_best=True, global_best=None):
        """重启种群：保留最优个体，其余重新初始化"""
        if keep_best:
            if global_best is not None:
                best = global_best
            else:
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
        # 只评估 fitness 为 None 的个体
        for ind in new_pop:
            if ind["fitness"] is None:
                self.evaluate_fitness(ind)
        self.population = new_pop
        if self.verbose and best is not None:
            print(f"  [重启] 种群已重启，保留最优 Cmax={best['cmax']}")

    def evolve(self, pc, pm, gen_progress=0.0, ls_prob=0.3):
        """
        执行一代进化，返回新一代种群
        ls_prob: 每个子代执行局部搜索的概率（阶段一改进）
        """
        new_pop = []
        sorted_pop = sorted(self.population, key=lambda ind: ind["fitness"])
        elites = sorted_pop[:self.elite_count]

        while len(new_pop) < self.pop_size - self.elite_count:
            parent1 = self.selection_tournament(self.population)
            parent2 = self.selection_tournament(self.population)
            child1, child2 = self.crossover(parent1, parent2, pc)
            child1 = self.mutate(child1, pm, gen_progress)
            child2 = self.mutate(child2, pm, gen_progress)
            
            # 阶段一改进：以概率 ls_prob 对子代执行轻量级局部搜索
            if random.random() < ls_prob:
                child1 = self.local_search_ms(child1)
            else:
                self.evaluate_fitness(child1)
            
            if random.random() < ls_prob:
                child2 = self.local_search_ms(child2)
            else:
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
    
    def _adaptive_mutate(self, individual, pm, gen_progress, no_improve_ratio):
        """
        自适应变异：根据停滞代数动态调整变异强度
        no_improve_ratio: 停滞代数 / max_gen，越大说明越需要强变异
        """
        # 基础变异
        self.mutate(individual, pm, gen_progress)
        # 如果长期停滞，额外增加扰动
        if no_improve_ratio > 0.15:
            extra_pm = min(pm * (1.0 + no_improve_ratio * 2), 0.6)
            os_len = len(individual["os"])
            if os_len >= 2 and random.random() < extra_pm:
                num_swaps = random.randint(1, max(2, os_len // 5))
                for _ in range(num_swaps):
                    idx1, idx2 = random.sample(range(os_len), 2)
                    individual["os"][idx1], individual["os"][idx2] = individual["os"][idx2], individual["os"][idx1]
            # MS 强变异
            ms_pm = min(extra_pm * 1.5, 0.6)
            for i in range(len(individual["ms"])):
                if random.random() < ms_pm:
                    job_id, op_id = self._get_job_op_from_ms_index(i)
                    op_data = self.jobs[job_id][op_id]
                    new_choice = random.randint(0, len(op_data["machines"]) - 1)
                    individual["ms"][i] = new_choice
        return individual

    def run(self, rl_controller=None, alns=None, tabu_search=None, progress_callback=None, bks_value=None):
        """
        主循环（集成早停策略）
        rl_controller: 可选的 RL 控制器，提供 get_actions 方法
        alns: 可选的 ALNS 优化器，用于优化精英个体
        tabu_search: 可选的 TabuSearch 优化器，用于深度局部搜索
        progress_callback: 可选的进度回调函数 fn(gen, max_gen, best_cmax, avg_cmax, elapsed)
        bks_value: 可选的 BKS 值，传入后启用 BKS 命中早停
        """
        # 初始化种群
        self.population = self.initialize_population()
        # 评估初始种群
        for ind in self.population:
            self.evaluate_fitness(ind)
        
        best_individual = min(self.population, key=lambda ind: ind["fitness"])
        best_cmax = best_individual["cmax"]
        no_improve_gen = 0
        _t0 = time.time()
        restart_interval = 50

        # 收敛曲线历史数据
        best_history = []
        avg_history = []

        # 阶段一改进：动态调整局部搜索概率
        base_ls_prob = 0.15

        for gen in range(self.max_gen):
            gen_progress = gen / max(self.max_gen, 1)
            no_improve_ratio = no_improve_gen / max(self.max_gen, 1)
            
            if rl_controller is not None:
                old_best_cmax = min(ind["cmax"] for ind in self.population)
                old_avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                state = rl_controller.compute_state(self.population)
                action_idx = rl_controller.select_action(state)
                pc, pm = rl_controller.adjust_probabilities(rl_controller.current_pc, rl_controller.current_pm, action_idx)
            else:
                pc = self.pc_high
                pm = self.pm_low

            # 阶段一改进：停滞时提高局部搜索概率
            ls_prob = base_ls_prob
            if no_improve_ratio > 0.1:
                ls_prob = min(base_ls_prob + no_improve_ratio * 0.5, 0.8)

            # 执行进化（集成局部搜索）
            self.evolve(pc, pm, gen_progress, ls_prob=ls_prob)
            
            # 对种群中部分个体施加额外自适应变异（跳出局部最优）
            if no_improve_ratio > 0.1:
                sorted_pop = sorted(self.population, key=lambda ind: ind["fitness"])
                perturb_count = max(1, len(sorted_pop) // 4)
                for i in range(perturb_count):
                    idx = -(i + 1)
                    self._adaptive_mutate(sorted_pop[idx], pm, gen_progress, no_improve_ratio)
                    self.evaluate_fitness(sorted_pop[idx])
                self.population = sorted_pop

            current_best = min(self.population, key=lambda ind: ind["fitness"])
            if current_best["cmax"] < best_cmax:
                best_cmax = current_best["cmax"]
                best_individual = current_best
                no_improve_gen = 0
            else:
                no_improve_gen += 1

            # 阶段二改进：Tabu Search + ALNS 协同优化
            # 参考 HA_FJSP：每代对部分个体执行 TS（但控制计算量）
            if tabu_search is not None:
                # 每代对最优个体执行 TS（早期迭代少，后期迭代多）
                ts_best = tabu_search.optimize(best_individual, gen=gen)
                if ts_best["cmax"] < best_individual["cmax"]:
                    best_individual = ts_best
                    best_cmax = best_individual["cmax"]
                    # 将改进后的个体放回种群
                    self.population[-1] = {
                        "os": best_individual["os"][:],
                        "ms": best_individual["ms"][:],
                        "cmax": best_individual["cmax"],
                        "load_var": best_individual["load_var"],
                        "fitness": best_individual["fitness"]
                    }

                # 每4代对多个精英执行完整 TS（平衡效率与优化深度）
                if (gen + 1) % 3 == 0:
                    sorted_pop = sorted(self.population, key=lambda ind: ind["fitness"])
                    elites = sorted_pop[:self.elite_count]
                    non_elites = sorted_pop[self.elite_count:]

                    for i in range(len(elites)):
                        # ALNS 优化
                        if alns is not None:
                            improved = alns.optimize(elites[i], self)
                            if improved["cmax"] < elites[i]["cmax"]:
                                elites[i] = improved
                        # Tabu Search 深度优化
                        ts_improved = tabu_search.optimize(elites[i], gen=gen)
                        if ts_improved["cmax"] < elites[i]["cmax"]:
                            elites[i] = ts_improved

                    self.population = elites + non_elites

            if rl_controller is not None:
                new_best_cmax = min(ind["cmax"] for ind in self.population)
                new_avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                reward = rl_controller.compute_reward(old_best_cmax, new_best_cmax, old_avg_cmax, new_avg_cmax)
                next_state = rl_controller.compute_state(self.population)
                rl_controller.update_q_table(reward, next_state)

            # 阶段一改进：更积极的停滞检测 + 重启
            if no_improve_gen > 0 and no_improve_gen % restart_interval == 0:
                diversity = self._compute_diversity()
                if diversity < 0.15 or no_improve_gen >= restart_interval * 2:
                    self._restart_population(keep_best=True, global_best=best_individual)
                    no_improve_gen = 0
                    if self.verbose:
                        print(f"  [重启] Gen {gen+1}: 种群重启，保留全局最优 Cmax={best_individual['cmax']}")

            if self.verbose and (gen + 1) % 10 == 0:
                avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                diversity = self._compute_diversity()
                print(f"Gen {gen+1}: best cmax={best_cmax}, avg cmax={avg_cmax:.2f}, diversity={diversity:.3f}")

            if progress_callback:
                avg_cmax = np.mean([ind["cmax"] for ind in self.population])
                elapsed = time.time() - _t0
                progress_callback(gen + 1, self.max_gen, best_cmax, avg_cmax, elapsed)

            # 记录收敛历史
            avg_cmax = np.mean([ind["cmax"] for ind in self.population])
            best_history.append(best_cmax)
            avg_history.append(avg_cmax)

            # ── 早停策略判断 ──
            # 设计原则：
            #   1) BKS 命中 → 立即停（最优）
            #   2) 长期无改进 → 停（已收敛）
            #   3) 种群趋同只在已有一定停滞时检查，避免每代误停
            if config.EARLY_STOP_ENABLED and (gen + 1) >= config.EARLY_STOP_MIN_GEN:
                stop_reason = None

                # 条件1: BKS 命中 → 立即停止
                if bks_value is not None and best_cmax <= bks_value:
                    stop_reason = f"🎯 BKS 命中: Cmax={best_cmax} == BKS={bks_value}"

                # 条件2: 长期收敛停滞
                elif no_improve_gen >= config.EARLY_STOP_PATIENCE:
                    stop_reason = (f"⏳ 收敛停滞: 连续 {no_improve_gen} 代无改进 "
                                   f"(阈值 {config.EARLY_STOP_PATIENCE})")

                # 条件3: 种群趋同（仅在已有一定停滞时检查）
                elif no_improve_gen >= max(5, config.EARLY_STOP_PATIENCE // 3):
                    diversity = self._compute_diversity()
                    if diversity < config.EARLY_STOP_DIVERSITY_THRESHOLD:
                        stop_reason = (f"📉 种群趋同: diversity={diversity:.4f} "
                                       f"< 阈值 {config.EARLY_STOP_DIVERSITY_THRESHOLD}")

                if stop_reason is not None:
                    if self.verbose or progress_callback:
                        print(f"\n  [早停] Gen {gen+1}/{self.max_gen}: {stop_reason}")
                    break

        # 最终：对最优个体执行一次深度 Tabu Search（gen=self.max_gen 确保全强度 80 次迭代）
        if tabu_search is not None:
            final_best = tabu_search.optimize(best_individual, gen=self.max_gen)
            if final_best["cmax"] < best_individual["cmax"]:
                best_individual = final_best
                best_cmax = best_individual["cmax"]

        # 将收敛历史附到返回值
        best_individual["best_history"] = best_history
        best_individual["avg_history"] = avg_history

        return best_individual