# core/job.py
"""
工件（Job）数据模型
表示一个需要加工的工件，包含多道工序
"""

from core.operation import Operation

class Job:
    def __init__(self, job_id, operations_data=None):
        """
        初始化一个工件对象
        
        参数:
            job_id: int, 工件ID
            operations_data: list, 工序数据列表，格式为:
                [{"machines": [...], "times": [...]}, ...]
        """
        self.job_id = job_id
        self.operations = []
        
        # 如果有初始数据，创建工序对象
        if operations_data:
            for op_id, op_data in enumerate(operations_data):
                op = Operation(job_id, op_id, op_data["machines"], op_data["times"])
                self.operations.append(op)
        
        # 工序总数
        self.num_operations = len(self.operations)
        
        # 进度跟踪
        self.current_op_index = 0  # 下一道待加工工序
        self.completion_time = 0.0  # 完工时间
    
    def add_operation(self, machines, times):
        """添加一道工序"""
        op_id = len(self.operations)
        op = Operation(self.job_id, op_id, machines, times)
        self.operations.append(op)
        self.num_operations += 1
        return op
    
    def get_operation(self, op_id):
        """获取指定ID的工序"""
        if 0 <= op_id < self.num_operations:
            return self.operations[op_id]
        raise IndexError(f"工序ID {op_id} 超出范围")
    
    def get_next_operation(self):
        """获取下一道待加工工序"""
        if self.current_op_index < self.num_operations:
            return self.operations[self.current_op_index]
        return None  # 所有工序已完成
    
    def advance(self):
        """推进到下一道工序"""
        if self.current_op_index < self.num_operations:
            self.current_op_index += 1
    
    def is_completed(self):
        """检查工件是否已完成所有工序"""
        return self.current_op_index >= self.num_operations
    
    def reset(self):
        """重置工件状态"""
        self.current_op_index = 0
        self.completion_time = 0.0
    
    def get_total_min_time(self):
        """获取所有工序的最小加工时间之和（理论下界）"""
        return sum(op.min_time for op in self.operations)
    
    def get_total_max_time(self):
        """获取所有工序的最大加工时间之和"""
        return sum(op.max_time for op in self.operations)
    
    def get_remaining_operations(self):
        """获取剩余未加工的工序列表"""
        return self.operations[self.current_op_index:]
    
    def __repr__(self):
        return f"Job(J{self.job_id}, ops={self.num_operations}, progress={self.current_op_index}/{self.num_operations})"
    
    def to_dict(self):
        """转换为字典格式"""
        return [op.to_dict() for op in self.operations]
    
    @classmethod
    def from_dict(cls, job_id, operations_list):
        """从字典列表创建工件对象"""
        return cls(job_id, operations_list)


class JobSet:
    """工件集合管理器"""
    
    def __init__(self, jobs_data=None):
        """
        初始化
        
        参数:
            jobs_data: list, 工件数据列表，格式为:
                [[{"machines": [...], "times": [...]}, ...], ...]
        """
        self.jobs = []
        if jobs_data:
            for job_id, job_data in enumerate(jobs_data):
                job = Job(job_id, job_data)
                self.jobs.append(job)
        
        self.num_jobs = len(self.jobs)
    
    def get_job(self, job_id):
        """获取指定ID的工件"""
        return self.jobs[job_id]
    
    def reset_all(self):
        """重置所有工件状态"""
        for job in self.jobs:
            job.reset()
    
    def all_completed(self):
        """检查所有工件是否都已完成"""
        return all(job.is_completed() for job in self.jobs)
    
    def get_remaining_jobs(self):
        """获取还有未加工工序的工件列表"""
        return [job for job in self.jobs if not job.is_completed()]
    
    def get_total_operations(self):
        """获取所有工件的总工序数"""
        return sum(job.num_operations for job in self.jobs)
    
    def get_completed_operations(self):
        """获取已完成的工序总数"""
        return sum(job.current_op_index for job in self.jobs)
    
    def __repr__(self):
        completed = sum(1 for job in self.jobs if job.is_completed())
        return f"JobSet({self.num_jobs} jobs, {completed} completed)"
