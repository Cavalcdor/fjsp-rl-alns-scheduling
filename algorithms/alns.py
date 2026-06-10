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
        """将个体解码为调度方案"""
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
            
            # 边界检查：确保机器选择索引在合法范围内
            num_available_machines = len(op_data["machines"])
            if machine_choice >= num_available_machines or machine_choice < 0:
                # 使用取模运算修正非法索引
                machine_choice = machine_choice % num_available_machines
            
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

    def _rebuild_ms(self, os_list, original_ms):
        """
        根据剩余 OS 序列重建 MS 数组
        原理：OS 中每个 job_id 的第 k 次出现，对应该 job 的第 k 道工序，
        取原始 MS 中对应位置的机器选择值
        """
        job_op_counter = [0] * self.num_jobs
        new_ms = []
        for job_id in os_list:
            op_id = job_op_counter[job_id]
            ms_idx = self._get_ms_index(job_id, op_id)
            if ms_idx < len(original_ms):
                new_ms.append(original_ms[ms_idx])
            else:
                new_ms.append(0)  # 安全回退
            job_op_counter[job_id] += 1
        return new_ms
    
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
        
        # 安全检查：确保破坏规模不超过可用工序数
        actual_destroy_size = min(destroy_size, total_ops)
        if actual_destroy_size < destroy_size:
            print(f"  [ALNS警告] 请求破坏{destroy_size}个工序，但仅剩{total_ops}个，调整为{actual_destroy_size}")
        
        indices = random.sample(range(total_ops), actual_destroy_size)
        indices.sort(reverse=True)
        for idx in indices:
            job_id = new_individual["os"][idx]
            current_op = new_individual["os"][:idx].count(job_id)
            ms_idx = self._get_ms_index(job_id, current_op)
            
            # 边界检查：确保MS索引有效
            if ms_idx >= len(individual["ms"]):
                print(f"  [ALNS警告] MS索引越界: {ms_idx} >= {len(individual['ms'])}，跳过该工序")
                continue
            
            removed.append({
                "pos": idx,
                "job_id": job_id,
                "op_id": current_op,
                "ms_idx": ms_idx,
                "ms_choice": new_individual["ms"][ms_idx]
            })
            del new_individual["os"][idx]
        
        # 根据剩余 OS 重建 MS（避免索引错位）
        new_individual["ms"] = self._rebuild_ms(new_individual["os"], individual["ms"])
        return new_individual, removed
    
    def destroy_critical_path(self, individual, destroy_size):
        """
        关键路径破坏：基于当前调度，找出关键路径上的工序并优先移除
        
        关键路径是指决定最大完工时间的最长路径，破坏关键路径上的工序
        更有可能改善 makespan
        """
        # 直接基于 individual 计算调度，避免解码assignment
        machine_available = [0] * self.num_machines
        job_completion = [0] * self.num_jobs
        op_schedule = []  # 存储每个工序的调度信息
        
        job_op_counter = [0] * self.num_jobs
        for i, job_id in enumerate(individual["os"]):
            op_id = job_op_counter[job_id]
            machine_choice = individual["ms"][i]
            op_data = self.jobs[job_id][op_id]
            
            # 边界检查
            if machine_choice >= len(op_data["machines"]) or machine_choice < 0:
                machine_choice = machine_choice % len(op_data["machines"])
            
            machine_id = op_data["machines"][machine_choice]
            duration = op_data["times"][machine_choice]
            
            start = max(machine_available[machine_id], job_completion[job_id])
            end = start + duration
            op_schedule.append({
                "idx": i,  # 这里是OS中的索引，不是assignment中的索引
                "job_id": job_id,
                "op_id": op_id,
                "machine_id": machine_id,
                "start": start,
                "end": end,
                "duration": duration
            })
            machine_available[machine_id] = end
            job_completion[job_id] = end
            job_op_counter[job_id] += 1
        
        # 找出关键路径上的工序（结束时间等于 makespan 的工序，以及其前驱）
        makespan = max(job_completion)
        
        # 简化策略：优先选择结束时间接近 makespan 的工序
        # 计算每个工序的"关键度" = end / makespan
        for op_info in op_schedule:
            op_info["criticality"] = op_info["end"] / (makespan + 1e-6)
        
        # 按关键度排序，选择最关键的工序
        sorted_ops = sorted(op_schedule, key=lambda x: x["criticality"], reverse=True)
        
        # 选择要破坏的工序（优先选择关键度高的）
        selected_count = min(destroy_size, len(sorted_ops))
        selected_ops = sorted_ops[:selected_count]
        
        # 如果关键路径工序不够，补充其他长加工时间工序
        if selected_count < destroy_size:
            remaining_ops = [op for op in sorted_ops if op not in selected_ops]
            # 按加工时间降序排序
            remaining_ops.sort(key=lambda x: x["duration"], reverse=True)
            additional = remaining_ops[:destroy_size - selected_count]
            selected_ops.extend(additional)
        
        # 按索引降序排序，便于从后往前移除
        selected_ops.sort(key=lambda x: x["idx"], reverse=True)
        
        # 重建个体（移除所选工序）
        os_list = individual["os"][:]
        removed = []
        
        for op_info in selected_ops:
            idx = op_info["idx"]
            job_id = op_info["job_id"]
            op_id = op_info["op_id"]
            ms_idx = self._get_ms_index(job_id, op_id)
            
            removed.append({
                "pos": idx,
                "job_id": job_id,
                "op_id": op_id,
                "ms_idx": ms_idx,
                "ms_choice": individual["ms"][ms_idx]
            })
            del os_list[idx]
        
        # 根据剩余 OS 重建 MS（避免索引错位）
        new_ms = self._rebuild_ms(os_list, individual["ms"])
        new_individual = {"os": os_list, "ms": new_ms}
        return new_individual, removed
    
    def destroy_high_load_machine(self, individual, destroy_size):
        """
        高负荷机器破坏：找出当前调度中负荷最高的机器，优先移除该机器上的部分工序
        """
        # 安全边界检查：确保 destroy_size 不超过个体长度
        actual_destroy_size = min(destroy_size, len(individual["os"]))
        if actual_destroy_size == 0:
            return individual, []
        
        # 直接基于 individual["os"] 和 individual["ms"] 计算机器负荷，避免解码
        machine_load = [0] * self.num_machines
        job_op_counter = [0] * self.num_jobs
        
        for i, job_id in enumerate(individual["os"]):
            op_id = job_op_counter[job_id]
            machine_choice = individual["ms"][i]
            op_data = self.jobs[job_id][op_id]
            if machine_choice < len(op_data["machines"]):
                machine_id = op_data["machines"][machine_choice]
                duration = op_data["times"][machine_choice]
                machine_load[machine_id] += duration
            job_op_counter[job_id] += 1
        
        # 找出负荷最高的机器
        sorted_machines = sorted(range(self.num_machines), key=lambda m: machine_load[m], reverse=True)
        high_machine = sorted_machines[0]
        
        # 找出该机器上的所有工序在 OS 中的索引
        indices_on_machine = []
        job_op_counter = [0] * self.num_jobs
        for i, job_id in enumerate(individual["os"]):
            op_id = job_op_counter[job_id]
            machine_choice = individual["ms"][i]
            op_data = self.jobs[job_id][op_id]
            if machine_choice < len(op_data["machines"]):
                machine_id = op_data["machines"][machine_choice]
                if machine_id == high_machine:
                    indices_on_machine.append(i)
            job_op_counter[job_id] += 1
        
        if len(indices_on_machine) < actual_destroy_size:
            # 如果不够，随机补齐
            all_indices = list(range(len(individual["os"])))
            additional = random.sample(all_indices, actual_destroy_size - len(indices_on_machine))
            indices_on_machine.extend(additional)
        
        selected = random.sample(indices_on_machine, actual_destroy_size)
        selected.sort(reverse=True)
        
        # 重建个体（移除所选工序）
        os_list = individual["os"][:]
        ms_list = individual["ms"][:]
        removed = []
        
        # 关键修复：使用try-except保护，防止任何索引越界错误
        # 注意：selected中的索引是相对于原始individual["os"]的。
        # 由于我们要从os_list中删除元素，且selected已经按降序排列（在调用random.sample后未排序，但在后续逻辑中通常期望降序处理以维持索引有效性，
        # 或者我们需要重新排序。查看前文：selected = random.sample(...) 然后 selected.sort(reverse=True)。
        # 等等，前面的代码是：
        # selected = random.sample(indices_on_machine, actual_destroy_size)
        # selected.sort(reverse=True)
        # 所以 selected 已经是降序了。
        
        for idx in selected:
            try:
                # 1. 检查当前 os_list 的边界
                # 注意：因为是降序删除，idx 指的是原始列表的位置。
                # 如果我们直接操作 os_list (它是 original 的副本)，且按降序删除，
                # 那么对于当前的 os_list 来说，只要 idx < len(os_list) 即可？
                # 不，os_list 初始是完整副本。每次 del 后长度减小。
                # 因为是降序，之前删除的都是大于当前 idx 的位置，所以当前 idx 在剩余的 os_list 中依然有效（只要它没被之前的操作影响？不，降序删除互不影响低位索引）。
                # 但是，如果 selected 中有重复索引或者超出初始长度的索引，需要检查。
                
                if idx >= len(os_list) or idx < 0:
                    continue
                
                job_id = os_list[idx]
                
                # 2. 验证 job_id 是否合法
                if job_id < 0 or job_id >= self.num_jobs:
                    continue
                    
                op_id = os_list[:idx].count(job_id)
                
                # 3. 验证 op_id 是否在合理范围内
                if op_id >= len(self.jobs[job_id]):
                    continue
                    
                ms_idx = self._get_ms_index(job_id, op_id)
                
                # 4. 验证 ms_idx 边界
                if ms_idx >= len(ms_list) or ms_idx < 0:
                    continue
                
                removed.append({
                    "pos": idx,
                    "job_id": job_id,
                    "op_id": op_id,
                    "ms_idx": ms_idx,
                    "ms_choice": ms_list[ms_idx]
                })
                del os_list[idx]
                # 同步删除ms_list中的对应元素？不，MS会在后面重建，所以不需要操作 ms_list
            except (IndexError, ValueError, TypeError):
                # 任何索引错误或值错误都跳过，保证程序继续运行
                continue
        
        # 根据剩余 OS 重建 MS（避免索引错位）
        new_ms = self._rebuild_ms(os_list, individual["ms"])
        new_individual = {"os": os_list, "ms": new_ms}
        return new_individual, removed
    
    def repair_greedy(self, individual, removed):
        """
        贪心修复：将移除的工序依次插入到合理位置，保持工序先后顺序和机器选择编码完整。
        removed: 列表，每个元素包含 job_id, op_id, ms_idx。
        """
        new_individual = copy.deepcopy(individual)
        # 按 ms_idx 降序插入，避免先插入的工序推移后续插入位置
        removed_sorted = sorted(removed, key=lambda item: item["ms_idx"], reverse=True)
        
        for rem in removed_sorted:
            job_id = rem["job_id"]
            op_id = rem["op_id"]
            ms_idx = rem["ms_idx"]
            original_ms_choice = rem["ms_choice"]

            # 获取该工序的可选机器数量，确保机器选择索引有效
            op_data = self.jobs[job_id][op_id]
            num_available_machines = len(op_data["machines"])
            # 如果原始机器选择索引超出范围，修正为合法值
            safe_ms_choice = original_ms_choice % num_available_machines

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
                new_individual["ms"].insert(ms_idx, safe_ms_choice)
            else:
                new_individual["ms"].append(safe_ms_choice)

        return new_individual
    
    def repair_least_load(self, individual, removed):
        """
        负荷均衡修复：优先将工序分配给当前负荷最小的机器
        
        与贪心修复不同，该算子会重新选择机器，以平衡各机器负荷
        """
        new_individual = copy.deepcopy(individual)
        
        # 直接从 OS/MS 数组计算机器当前负荷（不使用 decode，避免索引越界）
        machine_current_load = [0] * self.num_machines
        job_op_counter = [0] * self.num_jobs
        for i, job_id in enumerate(new_individual["os"]):
            op_id = job_op_counter[job_id]
            machine_choice = new_individual["ms"][i]
            op_data = self.jobs[job_id][op_id]
            if machine_choice < len(op_data["machines"]):
                machine_id = op_data["machines"][machine_choice]
                duration = op_data["times"][machine_choice]
                machine_current_load[machine_id] += duration
            job_op_counter[job_id] += 1
        
        # 按 ms_idx 升序处理被移除的工序，保证正确插回
        removed_sorted = sorted(removed, key=lambda item: item["ms_idx"])
        
        for rem in removed_sorted:
            job_id = rem["job_id"]
            op_id = rem["op_id"]
            ms_idx = rem["ms_idx"]
            
            # 获取该工序的可选机器
            op_data = self.jobs[job_id][op_id]
            available_machines = op_data["machines"]
            available_times = op_data["times"]
            
            # 选择当前负荷最小的可选机器
            best_machine_idx = 0
            min_load = float('inf')
            
            for i, machine_id in enumerate(available_machines):
                # 考虑机器当前负荷 + 该工序在该机器上的加工时间
                effective_load = machine_current_load[machine_id] + available_times[i]
                if effective_load < min_load:
                    min_load = effective_load
                    best_machine_idx = i
            
            # 更新机器负荷
            chosen_machine = available_machines[best_machine_idx]
            machine_current_load[chosen_machine] += available_times[best_machine_idx]
            
            # 在 OS 中插入该工序，保持工件工序顺序
            insert_pos = len(new_individual["os"])
            job_occurrences = 0
            for pos, jid in enumerate(new_individual["os"]):
                if jid == job_id:
                    if job_occurrences == op_id:
                        insert_pos = pos
                        break
                    job_occurrences += 1
            new_individual["os"].insert(insert_pos, job_id)
            
            # 在 MS 中插回原始位置，使用新选择的合法机器索引
            if ms_idx <= len(new_individual["ms"]):
                new_individual["ms"].insert(ms_idx, best_machine_idx)
            else:
                new_individual["ms"].append(best_machine_idx)
        
        return new_individual

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
            
            # 关键修复：确保 destroy_size 不超过当前个体的工序数
            current_ops_count = len(current["os"])
            if current_ops_count <= 1:
                # 如果只剩1个或0个工序，无法破坏，直接跳过
                break
            
            # destroy_size 应该在 [min, min(max, current_ops_count-1)] 范围内
            safe_destroy_max = min(self.destroy_max, current_ops_count - 1)
            if safe_destroy_max < self.destroy_min:
                safe_destroy_size = 1  # 至少破坏1个
            else:
                safe_destroy_size = random.randint(self.destroy_min, safe_destroy_max)
            
            destroy_size = safe_destroy_size
            
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