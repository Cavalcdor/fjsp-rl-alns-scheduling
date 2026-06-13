"""GA 核心功能测试：解码、关键路径、交叉、变异、适应度"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.instance_parser import load_fjsp_from_file, convert_to_zero_index
from algorithms.ga import GA


# ===== Fixtures =====

@pytest.fixture
def mk01_jobs():
    p = "data/Brandimarte_Data/Mk01.fjs"
    jobs_raw, nm, nj = load_fjsp_from_file(p)
    return convert_to_zero_index(jobs_raw), nm, nj


@pytest.fixture
def ga(mk01_jobs):
    jobs, nm, _ = mk01_jobs
    return GA(jobs, nm, config)


# ===== Decode 测试 =====

class TestDecode:
    """GA.decode() 确定性测试"""

    def test_decode_returns_list(self, ga):
        """返回 assignment list"""
        ind = ga.initialize_population()[0]
        assign = ga.decode(ind)
        assert isinstance(assign, list)
        assert len(assign) == ga.total_ops

    def test_decode_tuple_structure(self, ga):
        """每个元素是 (job_id, op_id, machine_id, duration)"""
        ind = ga.initialize_population()[0]
        assign = ga.decode(ind)
        for item in assign:
            assert len(item) == 4
            jid, oid, mid, dur = item
            assert isinstance(jid, int)
            assert isinstance(oid, int)
            assert isinstance(mid, int)
            assert isinstance(dur, (int, float))
            assert 0 <= jid < ga.num_jobs
            assert 0 <= mid < ga.num_machines

    def test_decode_deterministic(self, ga):
        """相同个体重复解码结果一致"""
        ind = ga.initialize_population()[0]
        a1 = ga.decode(ind)
        a2 = ga.decode(ind)
        assert a1 == a2

    def test_decode_maintains_sequence(self, ga):
        """同一工件工序按递增顺序出现"""
        ind = ga.initialize_population()[0]
        assign = ga.decode(ind)
        op_count = {}
        for jid, oid, _, _ in assign:
            prev = op_count.get(jid, -1)
            assert oid == prev + 1, f"Job {jid}: op {oid} after op {prev}"
            op_count[jid] = oid

    def test_decode_machine_exists(self, ga):
        """分配的机器在可选机器列表中"""
        ind = ga.initialize_population()[0]
        assign = ga.decode(ind)
        for jid, oid, mid, _ in assign:
            op = ga.jobs[jid][oid]
            assert mid in op["machines"]


# ===== Evaluate Fitness 测试 =====

class TestEvaluateFitness:
    """GA.evaluate_fitness() 测试"""

    def test_fitness_is_scalar(self, ga):
        """适应度为标量 cmax + load_penalty"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        assert isinstance(ind["fitness"], (int, float))

    def test_cmax_loaded(self, ga):
        """fitness 包含 cmax + load_penalty"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        assert ind["fitness"] >= ind["cmax"]

    def test_fitness_non_negative(self, ga):
        """Cmax > 0"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        assert ind["cmax"] > 0

    def test_all_individuals_evaluated(self, ga):
        """初始种群全部评估"""
        pop = ga.initialize_population()
        for ind in pop:
            ga.evaluate_fitness(ind)
            assert ind["fitness"] is not None
            assert ind["cmax"] is not None


# ===== Critical Path 测试 =====

class TestCriticalPath:
    """_find_critical_path() 测试"""

    def test_critical_path_not_empty(self, ga):
        """关键路径至少包含一道工序"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        cp = ga._find_critical_path(ind)
        assert len(cp) > 0

    def test_critical_path_end_time_equals_cmax(self, ga):
        """关键路径最后工序的结束时间 == cmax"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        cp = ga._find_critical_path(ind)
        assert max(op["end"] for op in cp) == ind["cmax"]

    def test_critical_path_structure(self, ga):
        """关键路径元素包含必要字段"""
        ind = ga.initialize_population()[0]
        ga.evaluate_fitness(ind)
        cp = ga._find_critical_path(ind)
        for op in cp:
            for key in ("job_id", "op_id", "machine_id", "start", "end", "duration"):
                assert key in op


# ===== Crossover 测试 =====

class TestCrossover:
    """GA.crossover() 测试"""

    def test_crossover_returns_two_children(self, ga):
        """交叉返回两个子代"""
        pop = ga.initialize_population()
        c1, c2 = ga.crossover(pop[0], pop[1], 0.8)
        assert c1 is not None
        assert c2 is not None

    def test_crossover_preserves_length(self, ga):
        """子代编码长度与父代一致"""
        pop = ga.initialize_population()
        os_len = len(pop[0]["os"])
        ms_len = len(pop[0]["ms"])
        c1, c2 = ga.crossover(pop[0], pop[1], 0.8)
        assert len(c1["os"]) == os_len
        assert len(c1["ms"]) == ms_len
        assert len(c2["os"]) == os_len
        assert len(c2["ms"]) == ms_len

    def test_crossover_valid_os(self, ga):
        """子代 OS 中每个工件出现次数正确"""
        pop = ga.initialize_population()
        c1, c2 = ga.crossover(pop[0], pop[1], 0.8)
        for child in (c1, c2):
            counts = [0] * ga.num_jobs
            for jid in child["os"]:
                counts[jid] += 1
            for jid in range(ga.num_jobs):
                assert counts[jid] == ga.ops_per_job[jid], (
                    f"Job {jid}: expected {ga.ops_per_job[jid]}, got {counts[jid]}"
                )


# ===== Mutate 测试 =====

class TestMutate:
    """GA.mutate() 测试"""

    def test_mutate_preserves_length(self, ga):
        """变异不改变编码长度"""
        ind = ga.initialize_population()[0]
        orig_os = ind["os"][:]
        orig_ms = ind["ms"][:]
        ga.mutate(ind, 0.3, 0.5)
        assert len(ind["os"]) == len(orig_os)
        assert len(ind["ms"]) == len(orig_ms)

    def test_mutate_valid_os(self, ga):
        """变异后 OS 计数正确"""
        ind = ga.initialize_population()[0]
        ga.mutate(ind, 0.3, 0.5)
        counts = [0] * ga.num_jobs
        for jid in ind["os"]:
            counts[jid] += 1
        for jid in range(ga.num_jobs):
            assert counts[jid] == ga.ops_per_job[jid]


# ===== Initialize 测试 =====

class TestInitialize:
    """GA.initialize_population() 测试"""

    def test_population_size(self, ga):
        """种群规模 == POP_SIZE"""
        pop = ga.initialize_population()
        assert len(pop) == config.POP_SIZE

    def test_all_individuals_have_fields(self, ga):
        """每个个体有所需字段"""
        pop = ga.initialize_population()
        for ind in pop:
            assert "os" in ind
            assert "ms" in ind
            assert len(ind["os"]) == ga.total_ops
            assert len(ind["ms"]) == ga.total_ops

    def test_os_valid_counts(self, ga):
        """每个个体的 OS 编码合法"""
        pop = ga.initialize_population()
        for ind in pop:
            counts = [0] * ga.num_jobs
            for jid in ind["os"]:
                counts[jid] += 1
            for jid in range(ga.num_jobs):
                assert counts[jid] == ga.ops_per_job[jid]


# ===== End-to-end 测试 =====

class TestGAEndToEnd:
    """GA 端到端运行测试（小规模快速验证）"""

    def test_ga_runs_mk01(self, ga):
        """GA 能在 Mk01 上运行 5 代不崩溃"""
        # 用 5 代快速验证
        original = config.MAX_GEN
        config.MAX_GEN = 5
        try:
            best = ga.run()
            assert best is not None
            assert best["cmax"] > 0
        finally:
            config.MAX_GEN = original

    def test_ga_convergence(self, ga):
        """GA 运行后最优 Cmax 不大于最差初始 Cmax"""
        pop = ga.initialize_population()
        for ind in pop:
            ga.evaluate_fitness(ind)
        worst_initial = max(ind["cmax"] for ind in pop)

        original = config.MAX_GEN
        config.MAX_GEN = 10
        config.VERBOSE = False
        try:
            best = ga.run()
            assert best["cmax"] <= worst_initial
        finally:
            config.MAX_GEN = original
            config.VERBOSE = True

    def test_seed_determinism(self):
        """相同种子产生相同结果"""
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        p = "data/Brandimarte_Data/Mk01.fjs"
        jobs_raw, nm, _ = load_fjsp_from_file(p)
        jobs = convert_to_zero_index(jobs_raw)

        g1 = GA(jobs, nm, config)
        g1.population = g1.initialize_population()
        s1 = [ind["os"][:5] for ind in g1.population[:3]]

        random.seed(42)
        np.random.seed(42)
        g2 = GA(jobs, nm, config)
        g2.population = g2.initialize_population()
        s2 = [ind["os"][:5] for ind in g2.population[:3]]

        assert s1 == s2
