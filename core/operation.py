# core/operation.py
"""
工序（Operation）数据模型
表示工件的一道工序，包含可选机器和加工时间信息
"""

class Operation:
    def __init__(self, job_id, op_id, machines, times):
        """
        初始化一个工序对象
        
        参数:
            job_id: int, 所属工件ID
            op_id: int, 工序ID（在该工件内的序号）
            machines: list, 可选机器ID列表（0-based）
            times: list, 对应机器的加工时间列表
        """
        self.job_id = job_id
        self.op_id = op_id
        self.machines = list(machines)
        self.times = list(times)
        
        # 可选机器数量
        self.num_machines = len(machines)
        
        # 最小/最大加工时间
        self.min_time = min(times) if times else 0
        self.max_time = max(times) if times else 0
        self.avg_time = sum(times) / len(times) if times else 0
    
    def get_time(self, machine_id):
        """获取在指定机器上的加工时间"""
        try:
            idx = self.machines.index(machine_id)
            return self.times[idx]
        except ValueError:
            raise ValueError(f"机器 {machine_id} 不在工序 ({self.job_id}, {self.op_id}) 的可选列表中")
    
    def can_process_on(self, machine_id):
        """检查该工序是否可以在指定机器上加工"""
        return machine_id in self.machines
    
    def get_machine_index(self, machine_id):
        """获取机器在可选列表中的索引"""
        return self.machines.index(machine_id)
    
    def __repr__(self):
        return f"Op(J{self.job_id},O{self.op_id},machines={self.machines},times={self.times})"
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "machines": self.machines,
            "times": self.times
        }
    
    @classmethod
    def from_dict(cls, job_id, op_id, data):
        """从字典创建工序对象"""
        return cls(job_id, op_id, data["machines"], data["times"])
