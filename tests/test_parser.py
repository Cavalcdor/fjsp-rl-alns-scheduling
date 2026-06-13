"""实例解析器测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.instance_parser import load_fjsp_from_file, convert_to_zero_index


class TestInstanceParser:
    """FJSP 文件解析测试"""

    @pytest.fixture
    def mk01_path(self):
        return "data/Brandimarte_Data/Mk01.fjs"

    def test_load_mk01_basic(self, mk01_path):
        """解析 Mk01 基本结构"""
        jobs, num_machines, num_jobs = load_fjsp_from_file(mk01_path)
        assert num_jobs == 10
        assert num_machines == 6
        assert len(jobs) == 10

    def test_load_mk01_total_ops(self, mk01_path):
        """Mk01 总工序数 = 55"""
        jobs, _, _ = load_fjsp_from_file(mk01_path)
        total_ops = sum(len(job) for job in jobs)
        assert total_ops == 55

    def test_load_mk01_operation_structure(self, mk01_path):
        """每个工序有 machines 和 times 字段"""
        jobs, _, _ = load_fjsp_from_file(mk01_path)
        for job in jobs:
            for op in job:
                assert "machines" in op
                assert "times" in op
                assert len(op["machines"]) == len(op["times"])
                assert len(op["machines"]) > 0

    def test_machine_ids_positive(self, mk01_path):
        """原始机器 ID >= 1"""
        jobs, _, _ = load_fjsp_from_file(mk01_path)
        for job in jobs:
            for op in job:
                for m in op["machines"]:
                    assert m >= 1

    def test_convert_zero_index(self, mk01_path):
        """0-index 转换后机器 ID < num_machines"""
        jobs, num_machines, _ = load_fjsp_from_file(mk01_path)
        jobs0 = convert_to_zero_index(jobs)
        for job in jobs0:
            for op in job:
                for m in op["machines"]:
                    assert 0 <= m < num_machines
                for t in op["times"]:
                    assert t > 0

    def test_convert_preserves_length(self, mk01_path):
        """转换不改变工序数和加工时间"""
        jobs, _, _ = load_fjsp_from_file(mk01_path)
        jobs0 = convert_to_zero_index(jobs)
        assert len(jobs0) == len(jobs)
        for j_raw, j_zero in zip(jobs, jobs0):
            assert len(j_zero) == len(j_raw)
            for op_raw, op_zero in zip(j_raw, j_zero):
                assert op_zero["times"] == op_raw["times"]

    def test_load_mk02(self):
        """Mk02 10x6 58 ops"""
        jobs, nm, nj = load_fjsp_from_file("data/Brandimarte_Data/Mk02.fjs")
        assert nj == 10
        assert nm == 6
        assert sum(len(j) for j in jobs) == 58

    def test_load_mk09(self):
        """Mk09 20x10 ops"""
        jobs, nm, nj = load_fjsp_from_file("data/Brandimarte_Data/Mk09.fjs")
        assert nj == 20
        assert nm == 10
        total_ops = sum(len(j) for j in jobs)
        assert total_ops > 0

    def test_file_not_found(self):
        """文件不存在引发正确错误"""
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            load_fjsp_from_file("data/nonexistent.fjs")
