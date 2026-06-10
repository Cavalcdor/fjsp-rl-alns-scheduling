# core/machine.py
"""
机器（Machine）数据模型
表示车间中的一台设备，跟踪其状态和负荷
"""

class Machine:
    def __init__(self, machine_id, name=None):
        """
        初始化机器对象
        
        参数:
            machine_id: int, 机器ID（0-based）
            name: str, 机器名称（可选）
        """
        self.machine_id = machine_id
        self.name = name if name else f"M{machine_id}"
        
        # 状态信息
        self.available_time = 0.0  # 下次可用时间
        self.total_load = 0.0      # 累计负荷（总加工时间）
        self.process_count = 0     # 已加工工序数
        
        # 历史记录
        self.schedule = []  # 加工记录 [(job_id, op_id, start, end), ...]
    
    def reset(self):
        """重置机器状态"""
        self.available_time = 0.0
        self.total_load = 0.0
        self.process_count = 0
        self.schedule = []
    
    def assign_operation(self, job_id, op_id, duration, start_time):
        """
        分配工序到该机器
        
        参数:
            job_id: int, 工件ID
            op_id: int, 工序ID
            duration: float, 加工时间
            start_time: float, 开始时间
        
        返回:
            end_time: float, 结束时间
        """
        # 计算实际开始时间（考虑机器可用时间）
        actual_start = max(start_time, self.available_time)
        end_time = actual_start + duration
        
        # 更新状态
        self.available_time = end_time
        self.total_load += duration
        self.process_count += 1
        self.schedule.append((job_id, op_id, actual_start, end_time))
        
        return end_time
    
    def get_available_time(self):
        """获取机器下次可用时间"""
        return self.available_time
    
    def get_load(self):
        """获取机器当前负荷"""
        return self.total_load
    
    def get_utilization(self, makespan):
        """
        计算机器利用率
        
        参数:
            makespan: float, 总完工时间
        
        返回:
            utilization: float, 利用率 (0~1)
        """
        if makespan <= 0:
            return 0.0
        return self.total_load / makespan
    
    def get_idle_time(self, makespan):
        """获取机器空闲时间"""
        return makespan - self.total_load
    
    def __repr__(self):
        return f"Machine({self.name}, load={self.total_load:.2f}, ops={self.process_count})"
    
    def get_stats(self):
        """获取机器统计信息"""
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "total_load": self.total_load,
            "process_count": self.process_count,
            "available_time": self.available_time,
            "schedule": self.schedule
        }
