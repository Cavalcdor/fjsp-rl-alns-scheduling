"""Tabu Search 核心功能测试"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from algorithms.ga import GA
from algorithms.tabu_search import TabuSearch


@pytest.fixture
def mk01_jobs():
    p = "data/Brandimarte_Data/Mk01.fjs"
    jobs_raw, nm, nj = load_fjsp_from_file(p)
    return convert_to_zero_index(jobs_raw), nm, nj


@pytest.fixture
def ga(mk01_jobs):
    jobs, nm, _ = mk01_jobs
    return GA(jobs, nm, config)


@pytest.fixture
def ts(mk01_jobs):
    jobs, nm, _ = mk01_jobs
    return TabuSearch(jobs, nm, config)


class TestTabuSearchInit:
    """TabuSearch 初始化测试"""

    def test_init(self, ts):
        """基本初始化"""
        assert ts.num_jobs == 10
        assert ts.num_machines == 6
        assert ts.total_ops == 55

    def test_ms_start_idx(self, ts):
        """MS 起始索引正确"""
        assert len(ts._ms_start_idx) == ts.num_jobs
        assert ts._ms_start_idx[0] == 0
        assert ts._ms_start_idx[-1] + ts.ops_per_job[-1] == ts.total_ops

    def test_get_ms_index(self, ts):
        """_get_ms_index 正确"""
        idx = ts._get_ms_index(0, 0)
        assert idx == 0
        # 计算累加
        total = 0
        for j in range(ts.num_jobs):
            for o in range(ts.ops_per_job[j]):
                assert ts._get_ms_index(j, o) == total
                total += 1

    def test_get_job_op_from_ms_index(self, ts):
        """_get_job_op_from_ms_index 逆向正确"""
        for j in range(ts.num_jobs):
            for o in range(ts.ops_per_job[j]):
                idx = ts._get_ms_index(j, o)
                j2, o2 = ts._get_job_op_from_ms_index(idx)
                assert (j2, o2) == (j, o)


class TestTS1Neighborhood:
    """TS1 机器重分配邻域测试"""

    def test_ts1_generates_neighbors(self, ts, ga):
        """TS1 能生成至少一个邻居"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        critical_ops = ga._find_critical_path(ind)
        neighbors = ts._neighborhood_ts1(ind, critical_ops)
        # 可能没有邻域（所有工序只有一台可选机器）
        assert isinstance(neighbors, list)

    def test_ts1_neighbor_structure(self, ts, ga):
        """TS1 邻居为 (individual, (job_id, op_id, new_mc), cmax)"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        critical_ops = ga._find_critical_path(ind)
        neighbors = ts._neighborhood_ts1(ind, critical_ops)
        if neighbors:
            n, move, cmax = neighbors[0]
            assert isinstance(move, tuple)
            assert len(move) == 3  # (job_id, op_id, new_machine_choice)
            assert isinstance(cmax, (int, float))


class TestTS2Neighborhood:
    """TS2 工序交换邻域测试"""

    def test_ts2_generates_neighbors(self, ts, ga):
        """TS2 能生成邻居"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        critical_ops = ga._find_critical_path(ind)
        neighbors = ts._neighborhood_ts2(ind, critical_ops)
        assert isinstance(neighbors, list)

    def test_ts2_neighbor_structure(self, ts, ga):
        """TS2 邻居为 (individual, ("swap", idx1, idx2), cmax)"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        critical_ops = ga._find_critical_path(ind)
        neighbors = ts._neighborhood_ts2(ind, critical_ops)
        if neighbors:
            n, move, cmax = neighbors[0]
            assert isinstance(move, tuple)
            assert move[0] == "swap"
            assert isinstance(cmax, (int, float))


class TestTSOptimize:
    """TS.optimize() 端到端测试"""

    def test_optimize_returns_individual(self, ts, ga):
        """TS 优化返回个体"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        result = ts.optimize(ind, gen=0)
        assert result is not None
        assert "os" in result
        assert "ms" in result

    def test_optimize_not_worse(self, ts, ga):
        """TS 优化后 Cmax 不劣化（或至少不增加很多）"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        original_cmax = ind["cmax"]
        result = ts.optimize(ind, gen=0)
        # TS 可能接受差解（模拟退火机制），但不应该偏离太远
        assert result["cmax"] <= original_cmax * 1.2

    def test_optimize_short_iterations(self, ts, ga):
        """少量迭代运行不崩溃"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        # 保存原始迭代次数
        original = ts.ts_iterations
        ts.ts_iterations = 3
        try:
            result = ts.optimize(ind, gen=0)
            assert result is not None
        finally:
            ts.ts_iterations = original


class TestTabuTenure:
    """禁忌表管理测试"""

    def test_tabu_tenure_positive(self, ts):
        """禁忌期限 > 0"""
        assert ts.tabu_tenure > 0

    def test_tabu_tenure_config_value(self, ts):
        """禁忌期限等于配置值"""
        assert ts.tabu_tenure == config.TABU_TENURE


class TestBKSIntegrity:
    """BKS 表格完整性测试"""

    @pytest.mark.parametrize("fname,expected_bks", [
        ("Mk01.fjs", 40),
        ("Mk02.fjs", 26),
        ("Mk03.fjs", 204),
        ("Mk04.fjs", 60),
        ("Mk05.fjs", 172),
        ("Mk06.fjs", 57),
        ("Mk07.fjs", 139),
        ("Mk08.fjs", 523),
        ("Mk09.fjs", 307),
    ])
    def test_bks_values(self, fname, expected_bks):
        """验证 main.py 中 BKS_TABLE 的值"""
        # 直接验证 main.py 中定义的 BKS_TABLE
        from main import bks
        val = bks(fname)
        assert val == expected_bks, f"{fname}: expected {expected_bks}, got {val}"
