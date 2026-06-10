# core/instance_parser.py
"""
FJSP 数据集解析器
支持 Brandimarte 格式（MK系列）以及 Fisher-Thompson mt06 等标准格式

格式规则：
- 第一行：jobs machines [可选平均机器数]
- 每个工件一行：
    第一个数字：工序数 ops
    然后重复每个工序：
        可选机器数 k
        之后 k 对 (机器ID, 加工时间)
"""

def load_fjsp_from_file(filepath):
    """
    解析 FJSP 实例文件
    
    参数:
        filepath: str, 文件路径（如 "data/Brandimarte_Data/mk01.fjs"）
    
    返回:
        jobs: list of jobs
              每个 job 是一个 list of operations
              每个 operation 是一个 dict:
                  {
                      "machines": [m1, m2, ...],
                      "times": [t1, t2, ...]
                  }
        num_machines: int, 总机器数（索引从1开始）
        num_jobs: int, 工件数
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 跳过空行，取第一个非空行作为第一行
    idx = 0
    while idx < len(lines) and lines[idx].strip() == '':
        idx += 1
    first_line = lines[idx].strip()
    idx += 1
    
    parts = list(map(float, first_line.split()))
    num_jobs = int(parts[0])
    num_machines = int(parts[1])
    # 第三个数字（平均可选机器数）可忽略
    
    jobs = []
    
    for job_id in range(num_jobs):
        # 跳过空行
        while idx < len(lines) and lines[idx].strip() == '':
            idx += 1
        if idx >= len(lines):
            raise ValueError(f"文件行数不足，期望 {num_jobs} 个工件，实际只读到 {job_id}")
        
        data_line = lines[idx].strip()
        idx += 1
        



        data = list(map(float, data_line.split()))
        
        num_ops = int(data[0])
        job_operations = []
        pos = 1
        
        for op_id in range(num_ops):
            if pos >= len(data):
                # 如果数据不够，尝试读取下一行（极少情况）
                next_line = lines[idx].strip()
                idx += 1
                data.extend(map(int, next_line.split()))
            

            k = int(data[pos])
            pos += 1
            
            if pos + 2*k > len(data):
                next_line = lines[idx].strip()
                idx += 1
                data.extend(map(int, next_line.split()))
            
            machines = []
            times = []



            for _ in range(k):
                machine_id = int(data[pos])
                proc_time = float(data[pos+1])
                pos += 2
                machines.append(machine_id)
                times.append(proc_time)
            
            job_operations.append({
                "machines": machines,
                "times": times
            })
        
        jobs.append(job_operations)
    
    return jobs, num_machines, num_jobs


def convert_to_zero_index(jobs):
    """将机器ID从1-based转为0-based"""
    new_jobs = []
    for job in jobs:
        new_job = []
        for op in job:
            new_op = {
                "machines": [m-1 for m in op["machines"]],
                "times": op["times"][:]
            }
            new_job.append(new_op)
        new_jobs.append(new_job)
    return new_jobs


def get_global_operations_list(jobs):
    """提取全局所有工序的扁平列表"""
    ops_list = []
    for job_id, job in enumerate(jobs):
        for op_id, op in enumerate(job):
            ops_list.append({
                "job_id": job_id,
                "op_id": op_id,
                "machines": op["machines"],
                "times": op["times"]
            })
    return ops_list


if __name__ == "__main__":
    # 测试 mt06 示例
    sample = """6   6   1   
6   1   3   1   1   1   3   1   2   6   1   4   7   1   6   3   1   5   6   
6   1   2   8   1   3   5   1   5   10  1   6   10  1   1   10  1   4   4   
6   1   3   5   1   4   4   1   6   8   1   1   9   1   2   1   1   5   7   
6   1   2   5   1   1   5   1   3   5   1   4   3   1   5   8   1   6   9   
6   1   3   9   1   2   3   1   5   5   1   6   4   1   1   3   1   4   1   
6   1   2   3   1   4   3   1   6   9   1   1   10  1   5   4   1   3   1   """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fjs', delete=False) as f:
        f.write(sample)
        tmp = f.name
    try:
        jobs, n_mach, n_jobs = load_fjsp_from_file(tmp)
        print(f"工件数: {n_jobs}, 机器数: {n_mach}")
        print("第一个工件第一道工序可选机器:", jobs[0][0]["machines"])
        print("对应加工时间:", jobs[0][0]["times"])
    finally:
        import os
        os.unlink(tmp)