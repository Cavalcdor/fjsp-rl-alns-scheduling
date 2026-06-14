# core/rolling_horizon.py
import copy
import random
import numpy as np
from utils.scheduler import evaluate

class RollingHorizon:
    def __init__(self, jobs, num_machines, config, ga, rl_controller=None, alns=None):
        """
        滚动时域控制器
        jobs: 工件数据（0索引）
        num_machines: 机器数
        config: 配置对象
        ga: GA 实例（需包含 run 方法）
        rl_controller: RL 控制器实例（可选）
        alns: ALNS 实例（可选）
        """
        self.jobs = jobs
        self.num_machines = num_machines
        self.config = config
        self.ga = ga
        self.rl = rl_controller
        self.alns = alns
        
        # 扰动参数
        self.degradation_coeff = config.DEGRADATION_COEFF
        self.time_fluctuation = config.TIME_FLUCTUATION
        
        # 重调度触发方式
        self.trigger_type = config.TRIGGER_TYPE
        self.periodic_interval = config.PERIODIC_INTERVAL
        
        # 车间状态
        self.current_time = 0  # 当前调度时间点
        self.schedule = []      # 已确定执行的调度片段 (job_id, op_id, machine_id, start, end)
        self.remaining_jobs = copy.deepcopy(jobs)   # 剩余未完成的工序（动态更新）
        self.machine_available_time = [0] * num_machines
        self.job_next_op = [0] * len(jobs)  # 每个工件下一个待加工工序索引
        
        # 记录设备累计加工次数（用于退化）
        self.machine_process_count = [0] * num_machines
        
        # 历史扰动记录（可选）
        self.disturbance_log = []
    
    def insert_new_job(self, new_job_operations, arrival_time):
        """
        处理紧急插单事件：插入高优先级新工件
        
        参数:
            new_job_operations: list, 新工件的工序列表
                格式: [{"machines": [...], "times": [...]}, ...]
            arrival_time: float, 新工件到达时间
        """
        new_job_id = len(self.jobs)
        # 将新工件加入工件列表
        self.jobs.append(new_job_operations)
        # 初始化新工件的进度跟踪
        self.job_next_op.append(0)
        # 记录插单事件
        self.disturbance_log.append({
            "time": arrival_time,
            "type": "new_job_arrival",
            "job_id": new_job_id
        })
        print(f"[插单事件] 时间 {arrival_time:.2f}: 新工件 J{new_job_id} 到达，"
              f"共 {len(new_job_operations)} 道工序")
    
    def apply_degradation(self, machine_id):
        """对指定机器的后续工序应用退化系数"""
        # 实际加工时间会在调度时动态调整，此处标记退化因子影响
        # 我们通过修改 jobs 中的加工时间来实现（注意：这会永久改变，如需保留原始数据，可保存原始副本）
        # 为简单，我们直接在获取加工时间时乘以退化系数，但需要知道当前机器已经加工的次数。
        # 此处作为一个接口，在每次机器完成一个工序后调用，增加计数，后续调度时动态计算退化时间。
        self.machine_process_count[machine_id] += 1
    
    def get_actual_processing_time(self, base_time, machine_id):
        """考虑退化系数和随机波动后的实际加工时间"""
        # 采用温和退化策略，避免指数级增长导致迭代失控
        # 参考记忆规范：使用局部窗口累积或设置退化上限
        
        # 方案：限制最大退化倍数，防止数值爆炸
        k = self.machine_process_count[machine_id] + 1
        max_degradation_factor = 1.5  # 最多退化到1.5倍
        actual_degradation = min(self.degradation_coeff ** (k-1), max_degradation_factor)
        
        degraded = base_time * actual_degradation
        
        # 随机波动：正态分布，均值为1，标准差为 time_fluctuation
        fluctuation = np.random.normal(1.0, self.time_fluctuation)
        actual = degraded * fluctuation
        
        return max(0.1, actual)  # 避免负时间
    
    def check_trigger(self, event_type=None, current_time=None):
        """
        检查是否需要触发重调度
        
        参数:
            event_type: str, 事件类型（可选）
                - "machine_breakdown": 机器故障
                - "new_job_arrival": 新工件到达
                - "process_delay": 加工延迟
                - None: 自动判断
            current_time: float, 当前时间（可选）
        
        返回:
            trigger: bool, 是否触发重调度
        """
        if self.trigger_type == "periodic":
            # 周期驱动：每隔固定时间触发
            if current_time is None:
                current_time = self.current_time
            if not hasattr(self, 'next_trigger_time'):
                self.next_trigger_time = self.periodic_interval
            if current_time >= self.next_trigger_time:
                self.next_trigger_time = current_time + self.periodic_interval
                return True
            return False
        
        elif self.trigger_type == "event_driven":
            # 事件驱动：检测到特定事件时触发
            if event_type is not None:
                # 记录事件
                self.disturbance_log.append({
                    "time": current_time or self.current_time,
                    "type": event_type
                })
                return True
            
            # 自动检测：检查是否有显著偏差
            # 例如：实际完工时间超过预期的一定比例
            if hasattr(self, 'expected_completion') and self.expected_completion:
                if current_time and current_time > self.expected_completion * 1.1:
                    return True
            
            return False
        
        else:
            # 混合驱动：周期 + 事件
            periodic_trigger = False
            if hasattr(self, 'next_trigger_time'):
                if current_time is None:
                    current_time = self.current_time
                if current_time >= self.next_trigger_time:
                    self.next_trigger_time = current_time + self.periodic_interval
                    periodic_trigger = True
            
            event_trigger = event_type is not None
            if event_trigger:
                self.disturbance_log.append({
                    "time": current_time or self.current_time,
                    "type": event_type
                })
            
            return periodic_trigger or event_trigger

    def get_remaining_subproblem(self):
        """
        获取当前剩余未调度的工序，构成新的 FJSP 子问题
        返回: remaining_jobs 结构（与原始 jobs 格式相同）
        """
        # 构建剩余工件工序列表（使用深拷贝避免修改原始数据）
        remaining = []
        for job_id, job in enumerate(self.jobs):
            next_op_idx = self.job_next_op[job_id]
            if next_op_idx >= len(job):
                remaining.append([])  # 该工件已完成
            else:
                # 使用深拷贝确保不会修改原始jobs数据
                remaining_job = []
                for op in job[next_op_idx:]:
                    # 深拷贝每个工序的字典
                    remaining_job.append({
                        "machines": op["machines"][:],  # 浅拷贝列表即可
                        "times": op["times"][:]
                    })
                remaining.append(remaining_job)
        return remaining
    
    def update_state(self, executed_segment):
        """
        执行当前窗口的前半段调度，更新车间状态
        executed_segment: 实际执行的工序列表 (job_id, local_op_id, machine_id, start, end, actual_duration)
            注意：op_id 是相对于剩余子问题的局部索引，需要转换为全局索引
        """
        # 关键修复：保存每个工件的原始 job_next_op，用于局部→全局索引转换
        # 根据项目规范第8条：global_op_id = current_job_next_operation + local_op_id
        # 必须在循环前快照，避免循环中更新 job_next_op 导致后续转换错误
        original_next_op = self.job_next_op[:]
        
        for (job_id, local_op_id, machine_id, start, end, actual_duration) in executed_segment:
            # 更新机器可用时间
            self.machine_available_time[machine_id] = end
            # 将局部索引转换为全局索引
            global_op_id = original_next_op[job_id] + local_op_id
            # 更新工件下一道工序时，取最大值避免回退
            self.job_next_op[job_id] = max(self.job_next_op[job_id], global_op_id + 1)
            # 记录已执行调度（使用全局 op_id）
            self.schedule.append((job_id, global_op_id, machine_id, start, end))
            # 更新机器加工计数（用于退化）
            self.machine_process_count[machine_id] += 1
        # 更新时间：确保 current_time 单调递增，不回退
        if executed_segment:
            segment_end = max(end for (_,_,_,_,end,_) in executed_segment)
            self.current_time = max(self.current_time, segment_end)
    
    def plan_horizon(self, horizon_length):
        """
        滚动时域规划：调用内层优化算法求解当前窗口的最优调度
        返回: 最优调度方案 (assignment 列表)
        """
        # 获取剩余子问题
        remaining = self.get_remaining_subproblem()
        # 如果所有工件完成，返回空
        if all(len(job) == 0 for job in remaining):
            return []
        
        # 关键修复：每次重调度必须完全重新初始化 GA 和 ALNS 实例
        # 确保其 jobs、ops_per_job、total_ops 等配置与当前剩余子问题完全同步
        # 这是项目规范的核心要求（记忆 ID: 2537c19a-38ef-423a-9484-4a7524c72612）
        from algorithms.ga import GA
        from algorithms.alns import ALNS
        
        # 基于当前剩余子问题创建新的 GA 实例
        sub_ga = GA(remaining, self.num_machines, self.config)
        
        # 基于当前剩余子问题创建新的 ALNS 实例（如果启用）
        sub_alns = None
        if self.alns is not None:
            sub_alns = ALNS(remaining, self.num_machines, self.config)
            print(f"  [ALNS重新初始化] 剩余工序数: {sub_alns.total_ops}, "
                  f"工件数: {len(remaining)}")
        
        # RL 状态重置（如果需要）
        if self.rl:
            self.rl.reset_episode()
        
        # 运行优化（传入新创建的子问题实例）
        best = sub_ga.run(rl_controller=self.rl, alns=sub_alns)
        # 解码得到 assignment
        assignment = sub_ga.decode(best)
        return assignment
    
    def execute_window(self, assignment, window_ratio=0.5):
        """
        执行当前窗口的前半段（或固定时间长度），返回实际执行的工序片段
        window_ratio: 执行窗口的比例（通常为0.5，即执行一半后重调度）
        """
        half = int(len(assignment) * window_ratio)
        if half == 0:
            half = 1
        planned_segment = assignment[:half]
        
        # 执行时应用扰动，记录实际开始/结束时间
        executed = []
        # 机器可用时间（临时模拟）
        mach_avail = self.machine_available_time[:]
        # 工件上一道工序的完成时间（关键：保证工序先后约束）
        job_avail = [0.0] * len(self.jobs)
        
        # 关键修复：对于已部分执行的工件，其可用时间应为已执行的最后一道工序的结束时间
        # 而不是简单的 current_time
        for job_id in range(len(self.jobs)):
            # 查找该工件已执行工序中的最大结束时间
            completed_end_times = [
                end for (jid, oid, mid, start, end) in self.schedule 
                if jid == job_id
            ]
            if completed_end_times:
                # 该工件已有工序执行完毕，可用时间为最后完成的工序结束时间
                job_avail[job_id] = max(completed_end_times)
            else:
                # 该工件尚未开始，可用时间为当前时间
                job_avail[job_id] = self.current_time
        
        for (job_id, op_id, machine_id, base_duration) in planned_segment:
            # 计算实际加工时间（含退化 + 随机波动）
            actual_duration = self.get_actual_processing_time(base_duration, machine_id)
            # 开始时间 = max(机器空闲时间, 该工件上一工序完成时间)
            start = max(mach_avail[machine_id], job_avail[job_id])
            end = start + actual_duration
            executed.append((job_id, op_id, machine_id, start, end, actual_duration))
            # 更新机器可用时间
            mach_avail[machine_id] = end
            # 更新该工件的完成时间（保证后续工序必须等本工序完成）
            job_avail[job_id] = end
            # 注意：machine_process_count 的递增由 update_state 统一管理，此处不重复计数
        
        return executed

    def _collect_results(self):
        """收集滚动时域调度的结果摘要"""
        import numpy as np
        nj = len(self.jobs)
        nm = self.num_machines
        total_ops = sum(len(job) for job in self.jobs) if self.jobs else 0
        cmax = self.current_time
        # 计算机器负荷方差
        ml = [0.0] * nm
        for (jid, oid, mid, start, end) in self.schedule:
            if 0 <= mid < nm:
                ml[mid] += (end - start)
        load_var = float(np.var(ml)) if ml else 0.0
        return {
            "schedule": self.schedule,
            "cmax": cmax,
            "load_var": load_var,
            "nj": nj,
            "nm": nm,
            "total_ops": total_ops,
        }

    def run(self, max_time=None):
        """
        滚动时域主循环
        max_time: 模拟的最大时间（可选，不设置则直到所有工件完成）
        """
        iteration = 0
        while True:
            # 1. 规划当前窗口（调用内层优化）
            print(f"\n=== 滚动时域迭代 {iteration+1} ===")
            print(f"当前时间: {self.current_time}")
            assignment = self.plan_horizon(horizon_length=self.config.PERIODIC_INTERVAL)
            if not assignment:
                print("所有工件已完成调度。")
                break
            
            # 2. 执行前半段（或固定数量工序）
            executed = self.execute_window(assignment, window_ratio=0.5)
            if not executed:
                break
            
            # 3. 更新车间状态
            self.update_state(executed)
            
            # 4. 检查是否所有工件完成
            if all(next_op >= len(self.jobs[job_id]) for job_id, next_op in enumerate(self.job_next_op)):
                print("所有工件加工完成！")
                break
            
            # 5. 检查是否触发重调度（事件驱动 + 周期驱动）
            # 在实际运行中，此处可检测外部事件（如插单、机器故障等）
            # 当前默认按固定周期（窗口执行后）自动触发重调度
            self.check_trigger(event_type=None, current_time=self.current_time)
            
            iteration += 1
            if max_time and self.current_time >= max_time:
                break
        
        # 输出最终调度甘特图信息（简化）
        print(f"\n调度完成，总完工时间: {self.current_time}")
        print(f"总执行工序数: {len(self.schedule)}")
        return self._collect_results()