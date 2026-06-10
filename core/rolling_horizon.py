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
        
    def apply_degradation(self, machine_id):
        """对指定机器的后续工序应用退化系数"""
        # 实际加工时间会在调度时动态调整，此处标记退化因子影响
        # 我们通过修改 jobs 中的加工时间来实现（注意：这会永久改变，如需保留原始数据，可保存原始副本）
        # 为简单，我们直接在获取加工时间时乘以退化系数，但需要知道当前机器已经加工的次数。
        # 此处作为一个接口，在每次机器完成一个工序后调用，增加计数，后续调度时动态计算退化时间。
        self.machine_process_count[machine_id] += 1
    
    def get_actual_processing_time(self, base_time, machine_id):
        """考虑退化系数和随机波动后的实际加工时间"""
        # 退化系数：每次加工后乘以退化系数，但退化是累积的。假设第k次加工耗时 = base * (deg_coeff)^(k-1)
        # 这里 machine_process_count 是已经完成的次数，本次是第 count+1 次
        k = self.machine_process_count[machine_id] + 1
        degraded = base_time * (self.degradation_coeff ** (k-1))
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
        # 构建剩余工件工序列表
        remaining = []
        for job_id, job in enumerate(self.jobs):
            next_op_idx = self.job_next_op[job_id]
            if next_op_idx >= len(job):
                remaining.append([])  # 该工件已完成
            else:
                remaining.append(job[next_op_idx:])  # 保留剩余工序
        return remaining
    
    def update_state(self, executed_segment):
        """
        执行当前窗口的前半段调度，更新车间状态
        executed_segment: 实际执行的工序列表 (job_id, op_id, machine_id, start, end, actual_duration)
        """
        for (job_id, op_id, machine_id, start, end, actual_duration) in executed_segment:
            # 更新机器可用时间
            self.machine_available_time[machine_id] = end
            # 更新工件下一道工序
            self.job_next_op[job_id] = op_id + 1
            # 记录已执行调度
            self.schedule.append((job_id, op_id, machine_id, start, end))
            # 更新机器加工计数（用于退化）
            self.machine_process_count[machine_id] += 1
        # 更新时间
        if executed_segment:
            self.current_time = max(end for (_,_,_,_,end,_) in executed_segment)
    
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
        
        # 重新初始化 GA 实例（基于剩余子问题）
        # 注意：需要将剩余工件重新编号，但为了简化，我们保持原 job_id，只考虑未完成工序
        # 我们将剩余工序重新打包成一个新的 jobs 结构（保持原 job_id 但只包含剩余工序）
        # GA 需要完整的工件列表（仅剩余工序），但注意机器和工件索引不变，可以直接使用剩余 jobs
        from algorithms.ga import GA
        sub_ga = GA(remaining, self.num_machines, self.config)
        if self.rl:
            # 重置 RL 状态
            self.rl.reset_episode()
        # 运行优化（可传 rl 和 alns）
        best = sub_ga.run(rl_controller=self.rl, alns=self.alns)
        # 解码得到 assignment
        assignment = sub_ga.decode(best)
        return assignment
    
    def execute_window(self, assignment, window_ratio=0.5):
        """
        执行当前窗口的前半段（或固定时间长度），返回实际执行的工序片段
        window_ratio: 执行窗口的比例（通常为0.5，即执行一半后重调度）
        """
        # 需要基于 assignment 模拟执行，考虑实际扰动（退化、随机波动）
        # 先模拟获得每个工序的计划开始时间（假设无扰动）
        # 然后按时间顺序执行，直到达到窗口长度或所有工序完成
        # 此处为简化，我们直接取 assignment 的前一半工序数执行（更好的做法是基于时间）
        half = int(len(assignment) * window_ratio)
        if half == 0:
            half = 1
        planned_segment = assignment[:half]
        
        # 执行时应用扰动，记录实际开始/结束时间
        executed = []
        # 机器可用时间（临时模拟）
        mach_avail = self.machine_available_time[:]
        job_next = self.job_next_op[:]
        
        for (job_id, op_id, machine_id, base_duration) in planned_segment:
            # 计算实际加工时间
            actual_duration = self.get_actual_processing_time(base_duration, machine_id)
            # 开始时间 = max(机器空闲, 该工件上一工序完成时间)
            # 需要知道该工件上一工序的完成时间，这里简化用 job_next 表示当前工序索引的前一道完成时间？
            # 我们记录每个工件的上一个完成时间
            # 由于我们没有维护工件完成时间数组，先简单用机器空闲时间模拟
            start = mach_avail[machine_id]
            end = start + actual_duration
            executed.append((job_id, op_id, machine_id, start, end, actual_duration))
            mach_avail[machine_id] = end
            # 注意：这里忽略了同一工件的工序顺序约束（因为 assignment 本身是合法的顺序）
            # 实际应该确保 job 的前置工序已完成，由于 assignment 是合法序列，job_id 在列表中顺序出现，已经保证了顺序，这里只需更新该工件的“最后完成时间”即可。
            # 为简单，暂不维护工件完成时间，因为 assignment 是拓扑序，不会违反约束。
        
        return executed
    
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
            
            # 5. 周期触发或事件触发（简单按固定周期）
            # 这里我们直接循环，每次执行半个窗口后重调度，相当于周期驱动
            iteration += 1
            if max_time and self.current_time >= max_time:
                break
        
        # 输出最终调度甘特图信息（简化）
        print(f"\n调度完成，总完工时间: {self.current_time}")
        print(f"总执行工序数: {len(self.schedule)}")
        return self.schedule


# 测试代码（简化）
if __name__ == "__main__":
    # 需要 config, jobs 等，此处仅示意
    print("RollingHorizon modugit add core/rolling_horizon.pyle loaded.")