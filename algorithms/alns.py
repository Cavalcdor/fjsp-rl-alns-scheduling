# algorithms/alns.py
import random
import math
import copy
import numpy as np

class ALNS:
    def __init__(self, jobs, num_machines, config):
        """
        自适应大邻域搜索初始化
        jobs: 解析后的工件数据（0索引）
        num_machines: 机器总数
        config: 配置对象
        """
        self.jobs = jobs
        self.num_machines = num_machines
        self.num_jobs = len(jobs)
        self.ops_per_job = [len(job) for job in jobs]
        self.total_ops = sum(self.ops_per_job)
        
        # 参数
        self.rho = config.RHO
        self.reward_global_best = config.REWARD_GLOBAL_BEST
        self.reward_better = config.REWARD_BETTER
        self.reward_accept = config.REWARD_ACCEPT
        self.destroy_min = config.DESTROY_SIZE_MIN
        self.destroy_max = config.DESTROY_SIZE_MAX
        self.T0 = config.T0
        self.sa_alpha = config.SA_ALPHA
        self.verbose = config.VERBOSE
        
        # 算子列表
        self.destroy_operators = [
            self.destroy_random,
            self.destroy_critical_path,
            self.destroy_high_load_machine
        ]
        self.repair_operators = [
            self.repair_greedy,
            self.repair_least_load
        ]
        self.num_destroy = len(self.destroy_operators)
        self.num_repair = len(self.repair_operators)
        
        # 算子权重 (对数形式的概率)
        self.destroy_weights = [1.0] * self.num_destroy
        self.repair_weights = [1.0] * self.num_repair
        # 算子得分
        self.destroy_scores = [0.0] * self.num_destroy
        self.repair_scores = [0.0] * self.num_repair
        # 算子被使用次数
        self.destroy_counts = [0] * self.num_destroy
        self.repair_counts = [0] * self.num_repair
        
        # 模拟退火温度
        self.temperature = self.T0
        
        # 全局最优解（用于奖励）
        self.global_best_individual = None
        self.global_best_cmax = float('inf')
    
    def decode(self, individual):
        """解码个体为 assignment (同 GA 中的 decode)"""
        op_idx = [0] * self.num_jobs
        assignment = []
        # 预先计算 MS 索引映射
        ms_start_idx = []
        total = 0
        for j in range(self.num_jobs):
            ms_start_idx.append(total)
            total += self.ops_per_job[j]
        
        for job_id in individual["os"]:
            current_op = op_idx[job_id]
            ms_index = ms_start_idx[job_id] + current_op
            machine_choice = individual["ms"][ms_index]
            op_data = self.jobs[job_id][current_op]
            machine_id = op_data["machines"][machine_choice]
            duration = op_data["times"][machine_choice]
            assignment.append((job_id, current_op, machine_id, duration))
            op_idx[job_id] += 1
        return assignment
    
    def _get_ms_index(self, job_id, op_id):
        """计算给定工件工序在 MS 编码中的全局索引"""
        if not hasattr(self, "_ms_start_idx"):
            self._ms_start_idx = []
            total = 0
            for j in range(self.num_jobs):
                self._ms_start_idx.append(total)
                total += self.ops_per_job[j]
        return self._ms_start_idx[job_id] + op_id
    
    def encode_from_assignment(self, assignment):
        """
        从 assignment 重建个体 (os, ms)
        assignment: [(job_id, op_id, machine_id, duration), ...] 按工序顺序
        """
        os = []
        ms = []
        # 重建 os 列表: 按顺序记录 job_id
        for (job_id, op_id, machine_id, duration) in assignment:
            os.append(job_id)
            # 需要找到这个工序在可选机器中的索引
            op_data = self.jobs[job_id][op_id]
            # 找到 machine_id 在可选列表中的位置
            try:
                machine_idx = op_data["machines"].index(machine_id)
            except ValueError:
                # 若找不到（不应该发生），随机选一个
                machine_idx = random.randint(0, len(op_data["machines"])-1)
            ms.append(machine_idx)
        return {"os": os, "ms": ms}
    
    def evaluate_individual(self, individual):
        """评估个体：计算 cmax 和 load_var，存储到个体中"""
        assignment = self.decode(individual)
        # 复用 scheduler 的 evaluate 函数
        from utils.scheduler import evaluate
        cmax, load_var = evaluate(self.jobs, assignment, self.num_machines, return_details=False)
        individual["cmax"] = cmax
        individual["load_var"] = load_var
        individual["fitness"] = (cmax, load_var)
        return cmax, load_var
    
    def destroy_random(self, individual, destroy_size):
        """随机破坏：随机移除 destroy_size 个工序，返回剩余个体和移除的工序列表"""
        new_individual = copy.deepcopy(individual)
        removed = []
        total_ops = len(new_individual["os"])
        indices = random.sample(range(total_ops), destroy_size)
        indices.sort(reverse=True)
        ms_to_remove = []
        for idx in indices:
            job_id = new_individual["os"][idx]
            current_op = new_individual["os"][:idx].count(job_id)
            ms_idx = self._get_ms_index(job_id, current_op)
            removed.append({
                "pos": idx,
                "job_id": job_id,
                "op_id": current_op,
                "ms_idx": ms_idx,
                "ms_choice": new_individual["ms"][ms_idx]
            })
            del new_individual["os"][idx]
            ms_to_remove.append(ms_idx)
        for ms_idx in sorted(ms_to_remove, reverse=True):
            del new_individual["ms"][ms_idx]
        return new_individual, removed
    
    def destroy_critical_path(self, individual, destroy_size):
        """
        关键路径破坏：基于当前调度，找出关键路径上的工序并优先移除
        简化实现：随机移除，但偏向选择加工时间长的工序
        """
        # 为了提高效率，我们简化：先评估获得每个工序的结束时间等信息
        # 为了不复杂化，这里退化为随机破坏（可后续改进）
        return self.destroy_random(individual, destroy_size)
    
    def destroy_high_load_machine(self, individual, destroy_size):
        """
        高负荷机器破坏：找出当前调度中负荷最高的机器，优先移除该机器上的部分工序
        """
        assignment = self.decode(individual)
        # 计算每个机器的负荷
        machine_load = [0] * self.num_machines
        for (job_id, op_id, machine_id, duration) in assignment:
            machine_load[machine_id] += duration
        # 找出负荷最高的机器
        sorted_machines = sorted(range(self.num_machines), key=lambda m: machine_load[m], reverse=True)
        high_machine = sorted_machines[0]
        # 找出该机器上的所有工序索引
        indices_on_machine = []
        for idx, (job_id, op_id, machine_id, duration) in enumerate(assignment):
            if machine_id == high_machine:
                indices_on_machine.append(idx)
        if len(indices_on_machine) < destroy_size:
            # 如果不够，随机补齐
            all_indices = list(range(len(assignment)))
            additional = random.sample(all_indices, destroy_size - len(indices_on_machine))
            indices_on_machine.extend(additional)
        selected = random.sample(indices_on_machine, destroy_size)
        selected.sort(reverse=True)
        # 重建个体（移除所选工序）
        os_list = individual["os"][:]
        ms_list = individual["ms"][:]
        removed = []
        ms_to_remove = []
        for idx in selected:
            job_id = os_list[idx]
            op_id = os_list[:idx].count(job_id)
            ms_idx = self._get_ms_index(job_id, op_id)
            removed.append({
                "pos": idx,
                "job_id": job_id,
                "op_id": op_id,
                "ms_idx": ms_idx,
                "ms_choice": ms_list[ms_idx]
            })
            del os_list[idx]
            ms_to_remove.append(ms_idx)
        for ms_idx in sorted(ms_to_remove, reverse=True):
            del ms_list[ms_idx]
        new_individual = {"os": os_list, "ms": ms_list}
        return new_individual, removed
    
    def repair_greedy(self, individual, removed):
        """
        贪心修复：将移除的工序依次插入到合理位置，保持工序先后顺序和机器选择编码完整。
        removed: 列表，每个元素包含 job_id, op_id, ms_idx。
        """
        new_individual = copy.deepcopy(individual)
        # 先按 ms_idx 升序处理，保证 ms 编码正确插回原始位置
        removed_sorted = sorted(removed, key=lambda item: item["ms_idx"])
        for rem in removed_sorted:
            job_id = rem["job_id"]
            op_id = rem["op_id"]
            ms_idx = rem["ms_idx"]

            # 在 OS 中插入该工序，使该工件的第 op_id 道工序保持正确顺序
            insert_pos = len(new_individual["os"])
            job_occurrences = 0
            for pos, jid in enumerate(new_individual["os"]):
                if jid == job_id:
                    if job_occurrences == op_id:
                        insert_pos = pos
                        break
                    job_occurrences += 1
            new_individual["os"].insert(insert_pos, job_id)

            # 在 MS 中插回原始位置，保持全局机器选择编码长度一致
            if ms_idx <= len(new_individual["ms"]):
                new_individual["ms"].insert(ms_idx, rem["ms_choice"])
            else:
                new_individual["ms"].append(rem["ms_choice"])

        return new_individual
    
    def repair_least_load(self, individual, removed):
        """负荷均衡修复：优先将工序分配给当前负荷最小的机器。"""
        # 目前采用与 repair_greedy 类似的合法插回策略，保证解的完整性。
        return self.repair_greedy(individual, removed)
    
    def select_operator(self, weights):
        """轮盘赌选择算子，返回索引"""
        total = sum(weights)
        if total == 0:
            return random.randint(0, len(weights)-1)
        r = random.uniform(0, total)
        acc = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return len(weights)-1
    
    def update_weights(self):
        """根据得分更新算子权重"""
        # 更新破坏算子权重
        for i in range(self.num_destroy):
            if self.destroy_counts[i] > 0:
                self.destroy_weights[i] = self.rho * self.destroy_weights[i] + (1 - self.rho) * (self.destroy_scores[i] / self.destroy_counts[i])
            else:
                self.destroy_weights[i] = self.rho * self.destroy_weights[i]
        # 更新修复算子权重
        for i in range(self.num_repair):
            if self.repair_counts[i] > 0:
                self.repair_weights[i] = self.rho * self.repair_weights[i] + (1 - self.rho) * (self.repair_scores[i] / self.repair_counts[i])
            else:
                self.repair_weights[i] = self.rho * self.repair_weights[i]
        # 重置得分和计数
        self.destroy_scores = [0.0] * self.num_destroy
        self.repair_scores = [0.0] * self.num_repair
        self.destroy_counts = [0] * self.num_destroy
        self.repair_counts = [0] * self.num_repair
    
    def cool_down(self):
        """退火降温"""
        self.temperature *= self.sa_alpha
    
    def optimize(self, individual, ga_instance=None):
        """
        对给定的个体进行 ALNS 优化，返回优化后的个体
        ga_instance: 可选的 GA 实例，用于访问 jobs 等（未使用）
        """
        current = copy.deepcopy(individual)
        self.evaluate_individual(current)
        best = copy.deepcopy(current)
        
        # 退火初始温度
        self.temperature = self.T0
        
        # 每个个体进行固定次数的迭代（可配置）
        max_iter = 50
        for _ in range(max_iter):
            # 选择破坏和修复算子
            d_idx = self.select_operator(self.destroy_weights)
            r_idx = self.select_operator(self.repair_weights)
            destroy_size = random.randint(self.destroy_min, self.destroy_max)
            
            # 记录使用次数
            self.destroy_counts[d_idx] += 1
            self.repair_counts[r_idx] += 1
            
            # 执行破坏
            destroyed, removed = self.destroy_operators[d_idx](current, destroy_size)
            # 执行修复
            repaired = self.repair_operators[r_idx](destroyed, removed)
            self.evaluate_individual(repaired)
            
            # 计算 delta
            delta = repaired["cmax"] - current["cmax"]
            
            # 接受准则 (模拟退火)
            accept = False
            if delta <= 0:
                accept = True
                # 奖励
                if repaired["cmax"] < self.global_best_cmax:
                    self.destroy_scores[d_idx] += self.reward_global_best
                    self.repair_scores[r_idx] += self.reward_global_best
                    self.global_best_cmax = repaired["cmax"]
                    self.global_best_individual = repaired
                else:
                    self.destroy_scores[d_idx] += self.reward_better
                    self.repair_scores[r_idx] += self.reward_better
            else:
                # 以概率接受劣解
                prob = math.exp(-delta / self.temperature)
                if random.random() < prob:
                    accept = True
                    self.destroy_scores[d_idx] += self.reward_accept
                    self.repair_scores[r_idx] += self.reward_accept
            
            if accept:
                current = repaired
                if current["cmax"] < best["cmax"]:
                    best = current
            
            # 退火
            self.cool_down()
        
        # 更新权重
        self.update_weights()
        return best


# 测试占位
if __name__ == "__main__":
    class Config:
        RHO = 0.5
        REWARD_GLOBAL_BEST = 10
        REWARD_BETTER = 5
        REWARD_ACCEPT = 2
        DESTROY_SIZE_MIN = 2
        DESTROY_SIZE_MAX = 5
        T0 = 100
        SA_ALPHA = 0.95
        VERBOSE = True
    cfg = Config()
    # 需要 jobs 数据，此处略
    print("ALNS module loaded. Ready to integrate.")