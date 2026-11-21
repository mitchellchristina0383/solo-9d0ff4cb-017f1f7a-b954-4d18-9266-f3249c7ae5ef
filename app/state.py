"""作业状态：文件清单、副本目录、带头位置，以及回执的应用。

约定：
- 只有“当前计划第一步（第一个 read 动作）”的回执才会被接受并推进进度；
  其余回执一律拒绝（MismatchError），不更新任何状态。
- 无论读带成功还是失败，带头都推进到该步骤的卷内序号；
  若该步骤在另一卷上，说明装带已发生，当前卷随之切换。
- 失败的副本立即停用，不再参与后续规划。

性能：作业持有一个 ``CopyIndex``，创建时一次性汇总副本目录并在各次
重排之间复用；每张有效回执只更新所涉单个文件的掩码（成功移出待取掩码，
失败摘掉该副本在对应卷上的贡献），不重建整份索引。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .planner import Copy, CopyIndex, Plan, Step


class NoJobError(Exception):
    """尚未创建作业。"""


class NoPendingStepError(Exception):
    """当前计划没有可回报的读带步骤（已全部完成或全部无副本可读）。"""


class MismatchError(Exception):
    """回执与当前计划第一步不符。"""

    def __init__(self, expected: Step):
        super().__init__("receipt does not match the first step of the current plan")
        self.expected = expected


@dataclass
class CopyRecord:
    file: str
    volume: int
    position: int
    enabled: bool = True


@dataclass
class ReceiptEvent:
    seq: int
    file: str
    volume: int
    position: int
    result: str  # "ok" | "failed"


@dataclass
class Job:
    files: list[str]
    copies: list[CopyRecord]
    done: set[str] = field(default_factory=set)
    current_volume: int | None = None  # 当前已挂载的卷（初始为空机）
    head: int = 0  # 带头所在的卷内序号
    mounts_done: int = 0
    rewinds_done: int = 0
    reads_ok: int = 0
    reads_failed: int = 0
    history: list[ReceiptEvent] = field(default_factory=list)
    index: CopyIndex | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            grouped: dict[str, list[Copy]] = {}
            for c in self.copies:
                if c.enabled:
                    grouped.setdefault(c.file, []).append(
                        Copy(c.file, c.volume, c.position)
                    )
            # 索引按文件清单顺序分配文件位号；目录中无副本的文件同样登记
            self.index = CopyIndex(grouped, all_files=self.files)

    def enabled_copies(self) -> dict[str, list[Copy]]:
        assert self.index is not None
        return self.index.copies_by_file

    def pending(self) -> list[str]:
        return [f for f in self.files if f not in self.done]

    def unretrievable(self) -> list[str]:
        assert self.index is not None
        return sorted(
            self.index.names_from_mask(
                self.index.pending_mask & ~self.index.present_mask
            )
        )

    def plan(self) -> Plan:
        assert self.index is not None
        return self.index.build_plan(self.current_volume, self.head)


class JobStore:
    """内存作业仓库：同一时刻只处理一个取回作业。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None

    def create(self, files: list[str], copies: list[CopyRecord]) -> Job:
        with self._lock:
            self._job = Job(files=files, copies=copies)
            return self._job

    def get(self) -> Job:
        with self._lock:
            if self._job is None:
                raise NoJobError
            return self._job

    def reset(self) -> bool:
        with self._lock:
            existed = self._job is not None
            self._job = None
            return existed

    def apply_receipt(self, file: str, volume: int, position: int, result: str) -> tuple[Job, Plan, Plan]:
        """应用一张回执，返回 (作业, 执行前计划, 重排后计划)。"""
        with self._lock:
            if self._job is None:
                raise NoJobError
            job = self._job
            assert job.index is not None
            before = job.plan()
            step = before.next_read
            if step is None:
                raise NoPendingStepError
            if (step.file, step.volume, step.position) != (file, volume, position):
                raise MismatchError(step)

            # 该读带步骤之前的装带/回卷动作随回执一并确认
            for s in before.steps:
                if s.action == "read":
                    break
                if s.action == "mount":
                    job.mounts_done += 1
                elif s.action == "rewind":
                    job.rewinds_done += 1

            # 成功或失败都推进带头；跨卷步骤意味着装带已发生
            job.current_volume = volume
            job.head = position

            if result == "ok":
                job.done.add(file)
                job.reads_ok += 1
                job.index.mark_done(file)  # 仅更新该文件的待取掩码
            else:
                for c in job.copies:
                    if c.file == file and c.volume == volume and c.position == position:
                        c.enabled = False  # 停用出错的副本
                        break
                job.reads_failed += 1
                # 仅重算所涉文件在各卷上的掩码贡献，其余索引原样复用
                job.index.disable_copy(file, volume, position)

            job.history.append(
                ReceiptEvent(
                    seq=len(job.history) + 1,
                    file=file,
                    volume=volume,
                    position=position,
                    result=result,
                )
            )
            return job, before, job.plan()


store = JobStore()
