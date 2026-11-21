"""优化验证：与旧版暴力枚举等价、大目录性能、作业内索引复用、报坏后重排。"""
import random
import time

import pytest

from app.planner import Copy, CopyIndex, build_plan
from app.state import CopyRecord, Job, JobStore


# ---------- 优化前的参考实现（逐文件逐子集的暴力枚举），用于等价性对照 ----------

def _reference_choose_volumes(targets, copies, current_volume, head, volumes):
    n = len(volumes)
    best_key = None
    best_mask = 0
    for mask in range(1, 1 << n):
        selected = {volumes[i] for i in range(n) if mask & (1 << i)}
        if not all(any(c.volume in selected for c in copies[f]) for f in targets):
            continue
        mounts = len(selected) - (1 if current_volume in selected else 0)
        rewinds = 0
        if current_volume is not None and current_volume in selected:
            for f in targets:
                in_sel = [c for c in copies[f] if c.volume in selected]
                if all(
                    c.volume == current_volume and c.position < head for c in in_sel
                ):
                    rewinds = 1
                    break
        key = (mounts + rewinds, mounts, len(selected), mask)
        if best_key is None or key < best_key:
            best_key = key
            best_mask = mask
    return best_mask


def _reference_plan(pending, copies, current_volume, head):
    """优化前 planner.build_plan 的逐字复刻（仅用于测试对照）。"""
    unretrievable = sorted(f for f in pending if not copies.get(f))
    targets = [f for f in pending if copies.get(f)]
    if not targets:
        return {"mounts": 0, "rewinds": 0, "reads": [],
                "volumes": [], "unretrievable": unretrievable}
    volumes = sorted({c.volume for f in targets for c in copies[f]})
    mask = _reference_choose_volumes(targets, copies, current_volume, head, volumes)
    selected = {volumes[i] for i in range(len(volumes)) if mask & (1 << i)}

    def assign(file):
        candidates = [c for c in copies[file] if c.volume in selected]
        off_current = [c for c in candidates if c.volume != current_volume]
        if off_current:
            return min(off_current, key=lambda c: (c.volume, c.position))
        after_head = [c for c in candidates if c.position >= head]
        if after_head:
            return min(after_head, key=lambda c: c.position)
        return min(candidates, key=lambda c: c.position)

    by_volume = {}
    for f in targets:
        chosen = assign(f)
        by_volume.setdefault(chosen.volume, []).append(chosen)
    ordered = sorted(by_volume, key=lambda v: (v != current_volume, v))
    reads = []
    for volume in ordered:
        items = sorted(by_volume[volume], key=lambda c: (c.position, c.file))
        if volume == current_volume:
            reads.extend((c.file, volume, c.position)
                         for c in items if c.position >= head)
            reads.extend((c.file, volume, c.position)
                         for c in items if c.position < head)
        else:
            reads.extend((c.file, volume, c.position) for c in items)
    return {
        "mounts": len(selected) - (1 if current_volume in selected else 0),
        "rewinds": 1 if any(
            current_volume is not None and current_volume in selected
            and all(c.volume == current_volume and c.position < head
                    for c in copies[f] if c.volume in selected)
            for f in targets
        ) else 0,
        "reads": reads,
        "volumes": ordered,
        "unretrievable": unretrievable,
    }


def _plan_signature(plan):
    return {
        "mounts": plan.mounts,
        "rewinds": plan.rewinds,
        "reads": [(s.file, s.volume, s.position)
                  for s in plan.steps if s.action == "read"],
        "volumes": plan.volumes_used,
        "unretrievable": plan.unretrievable,
    }


def _make_catalog(rng, n_files, n_volumes, copies_per_file, no_copy=()):
    copies = {}
    for i in range(n_files):
        name = f"f{i:04d}"
        if name in no_copy:
            continue
        k = min(copies_per_file, n_volumes)
        vs = rng.sample(range(1, n_volumes + 1), k)
        copies[name] = [Copy(name, v, rng.randint(1, 10_000)) for v in vs]
    return copies


# ---------- 1. 与暴力枚举的随机等价性 ----------

@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_on_random_catalogs(seed):
    rng = random.Random(seed)
    n_volumes = rng.randint(1, 12)
    n_files = rng.randint(1, 25)
    copies = _make_catalog(rng, n_files, n_volumes, rng.randint(1, 3))
    files = [f"f{i:04d}" for i in range(n_files)]
    if rng.random() < 0.4:  # 制造一些完全没有副本的文件
        for f in rng.sample(files, k=min(3, len(files))):
            copies.pop(f, None)
    current_volume = rng.choice([None] + list(range(1, n_volumes + 1)))
    head = rng.randint(0, 10_000)

    actual = _plan_signature(build_plan(files, copies, current_volume, head))
    expected = _reference_plan(files, copies, current_volume, head)
    assert actual == expected


def test_matches_brute_force_with_current_volume_edge_cases():
    rng = random.Random(2026)
    for _ in range(30):
        n_volumes = rng.randint(2, 8)
        copies = _make_catalog(rng, 15, n_volumes, rng.randint(1, 4))
        files = sorted(copies)
        # 当前卷固定为目录中的某一卷，且带头取一个能触发回卷判定的位置
        cv = rng.randint(1, n_volumes)
        head = rng.choice([0, 1, 5000, 9999, 10000])
        actual = _plan_signature(build_plan(files, copies, cv, head))
        expected = _reference_plan(files, copies, cv, head)
        assert actual == expected


# ---------- 2. 大目录性能：选卷汇总线性、枚举 O(V·2^V) ----------

@pytest.mark.parametrize("n_files", [1_000, 5_000])
def test_large_directory_planning_time(n_files):
    rng = random.Random(7)
    copies = _make_catalog(rng, n_files, 12, copies_per_file=3)
    files = [f"f{i:04d}" for i in range(n_files)]
    start = time.perf_counter()
    plan = build_plan(files, copies, current_volume=None, head=0)
    elapsed = time.perf_counter() - start
    assert plan.cost >= 1
    # 5000 文件 × 12 卷 × 3 副本的首版规划应在 1 秒内完成
    assert elapsed < 1.0, f"build_plan 耗时 {elapsed:.3f}s"


def test_repeated_replans_share_index_and_stay_fast():
    rng = random.Random(11)
    n_files = 2_000
    catalog = _make_catalog(rng, n_files, 10, copies_per_file=2)
    files = [f"f{i:04d}" for i in range(n_files)]
    index = CopyIndex(catalog, all_files=files)

    timings = []
    for cv, head in [(None, 0), (1, 5000), (3, 123), (7, 9999), (10, 1)]:
        start = time.perf_counter()
        plan = index.build_plan(cv, head)
        timings.append(time.perf_counter() - start)
        assert plan.steps, "每次重排都应有可执行步骤"
    # 同一索引复用 5 次，单次重排远低于全量重建的成本
    assert max(timings) < 0.25


# ---------- 3. 作业内索引复用：回执只动所涉文件 ----------

def _job_with(n_files, n_volumes, copies_per_file, seed=3):
    rng = random.Random(seed)
    catalog = _make_catalog(rng, n_files, n_volumes, copies_per_file)
    files = [f"f{i:04d}" for i in range(n_files)]
    records = [
        CopyRecord(file=c.file, volume=c.volume, position=c.position)
        for cs in catalog.values() for c in cs
    ]
    return Job(files=files, copies=records)


def test_index_object_is_reused_across_replans():
    job = _job_with(50, 6, 2)
    assert job.index is not None
    first_id = id(job.index)
    job.plan()
    job.plan()
    assert id(job.index) == first_id  # 重排不重建索引
    assert job.index.pending_mask.bit_count() == 50


def test_successful_receipt_only_moves_involved_file_bit():
    job = _job_with(20, 5, 2)
    index = job.index
    plan = job.plan()
    step = plan.next_read
    before_cover = dict(index.cover)
    before_present = index.present_mask

    job.current_volume, job.head = step.volume, step.position
    index.mark_done(step.file)

    assert index.pending_mask & (1 << index.bit_of_file[step.file]) == 0
    # 成功不改变任何卷的副本汇总
    assert index.cover == before_cover
    assert index.present_mask == before_present
    # 其余文件仍在待取掩码中
    assert index.pending_mask.bit_count() == 19


def test_failed_receipt_only_updates_involved_file_masks():
    job = _job_with(20, 5, 2)
    index = job.index
    plan = job.plan()
    step = plan.next_read
    file_bit = 1 << index.bit_of_file[step.file]

    # 记录该文件原本出现在哪些卷上
    expected_vols = {c.volume for c in index.copies_by_file[step.file]
                     if not (c.volume == step.volume and c.position == step.position)}
    untouched = {v: m for v, m in index.cover.items()}
    index.disable_copy(step.file, step.volume, step.position)

    # 停用副本所在卷摘掉该文件位；其余卷掩码完全不变
    assert not (index.cover[step.volume] & file_bit)
    for v, mask in untouched.items():
        if v != step.volume:
            assert index.cover[v] == mask
    # 该文件仍在持有其剩余副本的卷上
    still_present = {v for v, m in index.cover.items() if m & file_bit}
    assert still_present == expected_vols
    # 停用的副本已从目录移除
    assert all(not (c.volume == step.volume and c.position == step.position)
               for c in index.copies_by_file[step.file])


def test_failure_exhausting_copies_marks_file_unretrievable():
    job = _job_with(10, 4, copies_per_file=1)
    index = job.index
    plan = job.plan()
    step = plan.next_read  # 单副本文件
    assert len(index.copies_by_file[step.file]) == 1
    index.disable_copy(step.file, step.volume, step.position)
    assert job.unretrievable() == [step.file]
    replan = job.plan()
    assert step.file in replan.unretrievable
    assert step.file not in [s.file for s in replan.steps]


# ---------- 4. 报坏后重排：端到端流程与无副本清单 ----------

def test_replan_after_failures_on_large_directory():
    store = JobStore()
    rng = random.Random(99)
    n_files, n_volumes = 300, 12
    catalog = _make_catalog(rng, n_files, n_volumes, copies_per_file=2)
    files = [f"f{i:04d}" for i in range(n_files)]
    records = [
        CopyRecord(file=c.file, volume=c.volume, position=c.position)
        for cs in catalog.values() for c in cs
    ]
    job = store.create(files, records)
    index = job.index
    assert index.pending_mask.bit_count() == n_files

    failures = 0
    steps_seen = 0
    deadline = time.perf_counter() + 10  # 整条 300 文件流程 10 秒内跑完
    while True:
        plan = job.plan()
        nxt = plan.next_read
        if nxt is None:
            break
        steps_seen += 1
        # 前 20 步一律报坏，之后成功；坏副本被停用后计划必须立刻换副本
        result = "failed" if steps_seen <= 20 else "ok"
        job, _before, after = store.apply_receipt(
            nxt.file, nxt.volume, nxt.position, result
        )
        if result == "failed":
            failures += 1
            # 报坏后同一步骤不会再出现在计划首位
            again = after.next_read
            assert again is None or (
                again.file, again.volume, again.position
            ) != (nxt.file, nxt.volume, nxt.position)
        assert time.perf_counter() < deadline

    state_files = set(job.done) | set(job.unretrievable())
    assert state_files == set(files)
    assert job.reads_failed == failures
    assert len(job.history) == steps_seen
    # 停用副本数与失败次数一致
    assert sum(1 for c in job.copies if not c.enabled) == failures
    # 流程结束时所有文件要么已取（移出待取），要么无副本可读（移出 present）
    assert index.pending_mask & index.present_mask == 0
    assert set(job.done) | set(job.unretrievable()) == set(files)
