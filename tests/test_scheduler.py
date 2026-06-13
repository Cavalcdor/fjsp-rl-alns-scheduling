"""核心调度器与GA解码的基础单元测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.scheduler import evaluate


class TestEvaluate:
    """evaluate() 函数测试"""

    def test_simple_two_jobs(self):
        """2工件2机器简单排程"""
        jobs = [
            [{"machines": [0, 1], "times": [3, 4]}],
            [{"machines": [1], "times": [5]}],
        ]
        assignment = [(0, 0, 0, 3), (1, 0, 1, 5)]
        cmax, var = evaluate(jobs, assignment, 2)
        assert cmax == 5
        assert var >= 0

    def test_sequential_same_machine(self):
        """同机器串行工序"""
        jobs = [
            [{"machines": [0], "times": [3]}],
            [{"machines": [0], "times": [4]}],
        ]
        assignment = [(0, 0, 0, 3), (1, 0, 0, 4)]
        cmax, var = evaluate(jobs, assignment, 1)
        assert cmax == 7  # 3 + 4 = 7

    def test_parallel_different_machines(self):
        """不同机器并行工序"""
        jobs = [
            [{"machines": [0], "times": [5]}],
            [{"machines": [1], "times": [6]}],
        ]
        assignment = [(0, 0, 0, 5), (1, 0, 1, 6)]
        cmax, var = evaluate(jobs, assignment, 2)
        assert cmax == 6  # max(5, 6)
        assert var > 0

    def test_return_details_structure(self):
        """return_details=True 返回正确结构"""
        jobs = [
            [{"machines": [0], "times": [3]}],
        ]
        assignment = [(0, 0, 0, 3)]
        _, _, details = evaluate(jobs, assignment, 1, return_details=True)
        assert "machine_load" in details
        assert "schedule" in details
        assert "cmax" in details
        assert details["cmax"] == 3
        assert len(details["schedule"]) == 1
        s = details["schedule"][0]
        assert s[0] == 0  # job_id
        assert s[1] == 0  # op_id
        assert s[2] == 0  # machine_id
        assert s[3] == 0  # start
        assert s[4] == 3  # end

    def test_multi_operation_job(self):
        """多工序工件的顺序约束"""
        jobs = [
            [
                {"machines": [0], "times": [2]},
                {"machines": [0], "times": [3]},
            ],
        ]
        assignment = [(0, 0, 0, 2), (0, 1, 0, 3)]
        cmax, var = evaluate(jobs, assignment, 1)
        assert cmax == 5  # 2 + 3 = 5

    def test_empty_assignment(self):
        """空分配"""
        jobs = [[{"machines": [0], "times": [1]}]]
        cmax, var = evaluate(jobs, [], 1)
        assert cmax == 0
        assert var == 0.0
