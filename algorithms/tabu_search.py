"""
禁忌搜索 (Tabu Search) 模块
参考: "An effective hybrid genetic algorithm and tabu search for flexible job shop scheduling problem"
      (HA_FJSP) 中的 TS1 + TS2 邻域结构

核心思想:
  - TS1: 关键路径工序的机器重新分配 (Machine reassignment)
  - TS2: 关键路径工序的工序顺序交换 (Operation swap / interchanges)
  - 禁忌表防止循环，特赦准则允许突破

与 HA_FJSP 的区别:
  HA_FJSP 使用析取图模型，我们在 OS/MS 编码上直接操作，更轻量级。
"""

import random
import copy
import numpy as np
from utils.scheduler import evaluate


class TabuSearch:
    def __init__(self, jobs, num_machines, config):
        """
        禁忌搜索初始化
        jobs: 解析后的工件数据 (0索引)
        num_machines: 机器总数
        config: 配置对象
        """
        self.jobs = jobs
        self.num_machines = num_machines
        self.num_jobs = len(jobs)
        self.ops_per_job = [len(job) for job in jobs]
        self.total_ops = sum(self.ops_per_job)

        # 参数
        self.ts_iterations = getattr(config, 'TS_ITERATIONS', 80)
        self.tabu_tenure = getattr(config, 'TABU_TENURE', 15)
        self.verbose = getattr(config, 'VERBOSE', False)
        # 动态迭代：记录当前 gen，用于调整 TS 强度
        self.current_gen = 0
        self.max_gen = getattr(config, 'MAX_GEN', 800)

        # 预计算 MS 起始索引
        self._ms_start_idx = []
        total = 0
        for j in range(self.num_jobs):
            self._ms_start_idx.append(total)
            total += self.ops_per_job[j]

    def _get_ms_index(self, job_id, op_id):
        """计算给定工件工序在 MS 编码中的全局索引"""
        return self._ms_start_idx[job_id] + op_id

    def _get_job_op_from_ms_index(self, ms_idx):
        """根据 MS 编码索引反推 (job_id, op_id)"""
        for job_id, start in enumerate(self._ms_start_idx):
            if job_id == self.num_jobs - 1:
                if ms_idx >= start:
                    op_id = ms_idx - start
                    return job_id, op_id
            else:
                next_start = self._ms_start_idx[job_id + 1]
                if start <= ms_idx < next_start:
                    op_id = ms_idx - start
                    return job_id, op_id
        raise IndexError(f"Invalid ms index: {ms_idx}")

    def decode(self, individual):
        """解码：将 OS 和 MS 转换为 assignment 列表"""
        op_idx = [0] * self.num_jobs
        assignment = []
        for job_id in individual["os"]:
            current_op = op_idx[job_id]
            op_data = self.jobs[job_id][current_op]
            ms_index = self._get_ms_index(job_id, current_op)
            machine_choice = individual["ms"][ms_index]
            num_available = len(op_data["machines"])
            machine_choice = machine_choice % num_available
            machine_id = op_data["machines"][machine_choice]
            duration = op_data["times"][machine_choice]
            assignment.append((job_id, current_op, machine_id, duration))
            op_idx[job_id] += 1
        return assignment

    def _build_schedule(self, individual):
        """
        构建详细调度方案
        返回: (op_schedule, machine_available, job_completion, makespan)
          op_schedule: list of dict, 每个工序的调度详情
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
        return op_schedule, machine_available, job_completion, makespan

    def _find_critical_path(self, individual):
        """
        找出关键路径工序（参考 HA_FJSP 的反向追踪方法）
        从 makespan 终点开始，递归追踪前驱工序：
          1. 机器前驱：同一机器上结束时间 == 当前开始时间的工序
          2. 工序前驱：同一工件的前一道工序
        返回: list of dict, 按时间正序排列的关键路径工序
        """
        op_schedule, _, _, makespan = self._build_schedule(individual)

        # 反向追踪：从结束时间 == makespan 的工序开始
        end_candidates = [op for op in op_schedule if abs(op["end"] - makespan) < 1e-6]
        if not end_candidates:
            return []

        # 使用递归方式追踪（类似 HA_FJSP 的 findCriticalPath）
        def trace_back(op, visited_set):
            """递归追踪前驱工序"""
            if op["os_idx"] in visited_set:
                return
            visited_set.add(op["os_idx"])

            # 1. 找机器前驱：同一机器上结束时间 == 当前开始时间的工序
            #    如果有间隙（end < start），则没有机器前驱
            for other in op_schedule:
                if (other["machine_id"] == op["machine_id"]
                        and abs(other["end"] - op["start"]) < 1e-6):
                    trace_back(other, visited_set)

            # 2. 找工序前驱：同一工件的前一道工序
            for other in op_schedule:
                if other["job_id"] == op["job_id"] and other["op_id"] == op["op_id"] - 1:
                    trace_back(other, visited_set)

        visited = set()
        for end_op in end_candidates:
            trace_back(end_op, visited)

        # 按时间正序排列
        critical_ops = [op for op in op_schedule if op["os_idx"] in visited]
        critical_ops.sort(key=lambda x: x["start"])
        return critical_ops

    def _evaluate_individual(self, individual):
        """评估个体，计算 cmax 和 fitness"""
        assignment = self.decode(individual)
        cmax, load_var, details = evaluate(self.jobs, assignment, self.num_machines, return_details=True)
        individual["cmax"] = cmax
        individual["load_var"] = load_var
        machine_load = details['machine_load']
        max_load = max(machine_load)
        load_penalty = 0.15 * load_var
        if max_load > cmax * 0.85:
            load_penalty += 0.1 * (max_load - cmax * 0.85)
        individual["fitness"] = cmax + load_penalty
        return cmax

    # ========== TS1: 机器邻域搜索 ==========

    def _neighborhood_ts1(self, individual, critical_ops):
        """
        TS1 邻域：对关键路径上的每道工序，尝试所有可选机器
        返回: list of (neighbor, (job_id, op_id, new_machine_choice), cmax)
        """
        neighbors = []
        for op_info in critical_ops:
            job_id = op_info["job_id"]
            op_id = op_info["op_id"]
            op_data = self.jobs[job_id][op_id]

            if len(op_data["machines"]) <= 1:
                continue

            ms_idx = self._get_ms_index(job_id, op_id)
            old_choice = individual["ms"][ms_idx] % len(op_data["machines"])

            for alt_idx in range(len(op_data["machines"])):
                if alt_idx == old_choice:
                    continue

                # 创建邻居：复制个体并修改 MS
                neighbor = {
                    "os": individual["os"][:],
                    "ms": individual["ms"][:],
                    "cmax": None,
                    "load_var": None,
                    "fitness": None
                }
                neighbor["ms"][ms_idx] = alt_idx
                new_cmax = self._evaluate_individual(neighbor)
                neighbors.append((neighbor, (job_id, op_id, alt_idx), new_cmax))

        return neighbors

    # ========== TS2: 工序顺序邻域搜索 ==========

    def _neighborhood_ts2(self, individual, critical_ops):
        """
        TS2 邻域：对关键路径上的工序，尝试与同机器上的相邻工序交换 OS 位置
        参考 HA_FJSP 的关键块交换思想：
        - 只交换不同工件的工序（同一工件内交换会破坏工序顺序）
        - 只交换 OS 编码中相邻位置的工序（确保局部性）
        - 交换后检查 OS 可行性

        返回: list of (neighbor, ("swap", idx1, idx2), cmax)
        """
        neighbors = []

        # 构建 OS 位置到工序信息的映射
        job_op_counter = [0] * self.num_jobs
        os_info = []
        for i, job_id in enumerate(individual["os"]):
            op_id = job_op_counter[job_id]
            ms_idx = self._get_ms_index(job_id, op_id)
            machine_choice = individual["ms"][ms_idx]
            op_data = self.jobs[job_id][op_id]
            num_available = len(op_data["machines"])
            machine_choice = machine_choice % num_available
            machine_id = op_data["machines"][machine_choice]
            os_info.append({
                "os_idx": i,
                "job_id": job_id,
                "op_id": op_id,
                "machine_id": machine_id
            })
            job_op_counter[job_id] += 1

        # 构建机器 -> OS 位置列表的映射（按 OS 顺序）
        machine_os_indices = {}
        for info in os_info:
            mid = info["machine_id"]
            if mid not in machine_os_indices:
                machine_os_indices[mid] = []
            machine_os_indices[mid].append(info["os_idx"])

        # 对关键路径上的每道工序
        critical_os_set = {op["os_idx"] for op in critical_ops}

        for op_info in critical_ops:
            os_idx = op_info["os_idx"]
            machine_id = op_info["machine_id"]
            job_id = op_info["job_id"]

            # 找同机器上 OS 相邻的工序（在 OS 编码中位置相邻）
            machine_indices = machine_os_indices.get(machine_id, [])
            if os_idx in machine_indices:
                pos = machine_indices.index(os_idx)
                # 尝试与前一个工序交换
                if pos > 0:
                    other_idx = machine_indices[pos - 1]
                    other_info = os_info[other_idx]
                    # 只交换不同工件的工序
                    if other_info["job_id"] != job_id:
                        new_os = individual["os"][:]
                        new_os[os_idx], new_os[other_idx] = new_os[other_idx], new_os[os_idx]
                        if self._check_os_feasibility(new_os):
                            neighbor = {
                                "os": new_os,
                                "ms": individual["ms"][:],
                                "cmax": None, "load_var": None, "fitness": None
                            }
                            new_cmax = self._evaluate_individual(neighbor)
                            neighbors.append((neighbor, ("swap", os_idx, other_idx), new_cmax))
                # 尝试与后一个工序交换
                if pos < len(machine_indices) - 1:
                    other_idx = machine_indices[pos + 1]
                    other_info = os_info[other_idx]
                    if other_info["job_id"] != job_id:
                        new_os = individual["os"][:]
                        new_os[os_idx], new_os[other_idx] = new_os[other_idx], new_os[os_idx]
                        if self._check_os_feasibility(new_os):
                            neighbor = {
                                "os": new_os,
                                "ms": individual["ms"][:],
                                "cmax": None, "load_var": None, "fitness": None
                            }
                            new_cmax = self._evaluate_individual(neighbor)
                            neighbors.append((neighbor, ("swap", os_idx, other_idx), new_cmax))

        return neighbors

    def _check_os_feasibility(self, os_list):
        """
        检查 OS 编码的可行性：每个工件的工序必须按顺序出现
        即 job_id 的第 k 次出现对应第 k 道工序
        """
        counts = {}
        for job_id in os_list:
            counts[job_id] = counts.get(job_id, 0) + 1
            if counts[job_id] > self.ops_per_job[job_id]:
                return False
        # 还要检查每个工件的工序数是否正确
        for j in range(self.num_jobs):
            if counts.get(j, 0) != self.ops_per_job[j]:
                return False
        return True

    # ========== 主搜索过程 ==========

    def optimize(self, individual, ga_instance=None, gen=None):
        """
        对个体执行禁忌搜索

        参数:
            individual: 待优化的个体 (dict with os, ms)
            ga_instance: 可选的 GA 实例，用于获取 evaluate_fitness 等方法
            gen: 当前代数（用于动态调整迭代次数）

        返回:
            优化后的个体
        """
        # 动态调整迭代次数：参考 HA_FJSP，早期少迭代（探索），后期多迭代（利用）
        if gen is not None:
            self.current_gen = gen
        # 从 gen=0 时约 10 次线性增长到 gen=max_gen 时约 80 次
        effective_iterations = max(10, int(self.ts_iterations * self.current_gen / max(self.max_gen, 1)))

        # 深拷贝当前解
        current = {
            "os": individual["os"][:],
            "ms": individual["ms"][:],
            "cmax": None,
            "load_var": None,
            "fitness": None
        }
        self._evaluate_individual(current)
        best = {
            "os": current["os"][:],
            "ms": current["ms"][:],
            "cmax": current["cmax"],
            "load_var": current["load_var"],
            "fitness": current["fitness"]
        }

        # 禁忌表
        # TS1 禁忌: (job_id, op_id, machine_choice) -> 剩余禁忌代数
        tabu_ts1 = {}
        # TS2 禁忌: ("swap", os_idx1, os_idx2) -> 剩余禁忌代数
        tabu_ts2 = {}

        no_improve = 0

        for iteration in range(effective_iterations):
            # 找关键路径
            critical_ops = self._find_critical_path(current)
            if not critical_ops or len(critical_ops) <= 1:
                break

            # 生成 TS1 邻域
            ts1_neighbors = self._neighborhood_ts1(current, critical_ops)
            # 生成 TS2 邻域
            ts2_neighbors = self._neighborhood_ts2(current, critical_ops)

            all_neighbors = ts1_neighbors + ts2_neighbors

            if not all_neighbors:
                break

            # 按 cmax 排序
            all_neighbors.sort(key=lambda x: x[2])

            best_neighbor = None
            best_move = None
            best_neighbor_cmax = float('inf')

            for neighbor, move, cmax_val in all_neighbors:
                # 检查特赦准则：如果找到全局更优解，接受
                if cmax_val < best["cmax"]:
                    best_neighbor = neighbor
                    best_move = move
                    best_neighbor_cmax = cmax_val
                    break

                # 检查禁忌
                is_tabu = False
                if len(move) == 3 and isinstance(move[0], int):
                    # TS1 move: (job_id, op_id, machine_choice)
                    key = ("ts1", move[0], move[1], move[2])
                    if key in tabu_ts1 and tabu_ts1[key] > 0:
                        is_tabu = True
                elif move[0] == "swap":
                    # TS2 move: ("swap", idx1, idx2)
                    key1 = ("ts2", move[1], move[2])
                    key2 = ("ts2", move[2], move[1])
                    if (key1 in tabu_ts2 and tabu_ts2[key1] > 0) or \
                       (key2 in tabu_ts2 and tabu_ts2[key2] > 0):
                        is_tabu = True

                if not is_tabu:
                    best_neighbor = neighbor
                    best_move = move
                    best_neighbor_cmax = cmax_val
                    break

            if best_neighbor is None:
                # 所有邻域都被禁忌，选择第一个（特赦）
                if all_neighbors:
                    best_neighbor, best_move, best_neighbor_cmax = all_neighbors[0]

            if best_neighbor is None:
                break

            # 在更新 current 之前，记录旧机器选择（用于禁忌表）
            old_ms_choice = None
            if len(best_move) == 3 and isinstance(best_move[0], int):
                job_id, op_id, new_choice = best_move
                ms_idx = self._get_ms_index(job_id, op_id)
                old_ms_choice = current["ms"][ms_idx] % len(self.jobs[job_id][op_id]["machines"])

            # 更新当前解
            current = best_neighbor

            # 更新禁忌表
            if old_ms_choice is not None:
                # TS1: 禁忌旧机器选择（取模后的值，与邻域生成中的 alt_idx 一致）
                key = ("ts1", job_id, op_id, old_ms_choice)
                tabu_ts1[key] = self.tabu_tenure + random.randint(0, 5)
            elif best_move[0] == "swap":
                key1 = ("ts2", best_move[1], best_move[2])
                key2 = ("ts2", best_move[2], best_move[1])
                tabu_ts2[key1] = self.tabu_tenure + random.randint(0, 5)
                tabu_ts2[key2] = self.tabu_tenure + random.randint(0, 5)

            # 衰减禁忌表
            for k in list(tabu_ts1.keys()):
                tabu_ts1[k] -= 1
                if tabu_ts1[k] <= 0:
                    del tabu_ts1[k]
            for k in list(tabu_ts2.keys()):
                tabu_ts2[k] -= 1
                if tabu_ts2[k] <= 0:
                    del tabu_ts2[k]

            # 更新全局最优
            if current["cmax"] < best["cmax"]:
                best = {
                    "os": current["os"][:],
                    "ms": current["ms"][:],
                    "cmax": current["cmax"],
                    "load_var": current["load_var"],
                    "fitness": current["fitness"]
                }
                no_improve = 0
            else:
                no_improve += 1

            # 如果长时间无改善，执行扰动
            if no_improve >= max(10, self.ts_iterations // 5):
                # 对 MS 进行随机扰动
                current = self._perturb(current)
                self._evaluate_individual(current)
                no_improve = 0

        return best

    def _perturb(self, individual):
        """扰动：随机改变少量工序的机器分配"""
        perturbed = {
            "os": individual["os"][:],
            "ms": individual["ms"][:],
            "cmax": None,
            "load_var": None,
            "fitness": None
        }
        num_perturb = max(1, self.total_ops // 10)
        indices = random.sample(range(self.total_ops), min(num_perturb, self.total_ops))
        for idx in indices:
            job_id, op_id = self._get_job_op_from_ms_index(idx)
            op_data = self.jobs[job_id][op_id]
            perturbed["ms"][idx] = random.randint(0, len(op_data["machines"]) - 1)
        return perturbed
