"""读取顺序规划器：在“装带次数 + 回卷次数”最少的约束下给出读取顺序。

代价模型（与值班室约定一致）：
- 装带：挂载一卷计 1 次；当前已挂载的卷继续读不计装带。
- 装带后带头在卷首（序号 0 之前），只能向序号增大的方向读。
- 回卷：在当前卷上需要读取比带头位置更小的序号时，必须回卷 1 次，
  回卷后带头回到卷首，再按序号升序读。
- 同一卷内按序号升序读取不需要回卷；因此每卷至多回卷 1 次，
  只有“当前已挂载卷”可能被迫回卷（其余卷从卷首开始升序读即可）。

算法（卷数 V ≤ 12）：
- 副本索引 ``CopyIndex`` 在作业内复用：文件与卷均用位掩码表示，
  每卷线性汇总一次“持有副本的文件掩码”；一张回执只更新所涉文件的汇总。
- 选卷阶段枚举全部非空卷子集（≤ 4095 个），覆盖关系由子集并集的
  LSB 递推得到；回卷判定用“免回卷并集”的同构递推。
  汇总为 O(文件数 + 副本数) 的线性工作量，评完全部卷组至多 O(V·2^V)。
- 平局时依次按：装带更少、用卷更少、子集位掩码更小（保证确定性）。
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_VOLUMES = 12


@dataclass(frozen=True)
class Copy:
    """一个可用副本：文件 file 位于卷 volume 的卷内序号 position。"""

    file: str
    volume: int
    position: int


@dataclass(frozen=True)
class Step:
    """计划中的一个动作：mount（装带）/ rewind（回卷）/ read（读带）。"""

    seq: int
    action: str
    volume: int | None = None
    file: str | None = None
    position: int | None = None


@dataclass
class Plan:
    steps: list[Step]
    volumes_used: list[int]
    mounts: int
    rewinds: int
    unretrievable: list[str]  # 已无任何可用副本的待取文件

    @property
    def cost(self) -> int:
        return self.mounts + self.rewinds

    @property
    def next_read(self) -> Step | None:
        """计划中的第一个读带步骤（值班员下一步要执行并回报的步骤）。"""
        for step in self.steps:
            if step.action == "read":
                return step
        return None


class CopyIndex:
    """作业内复用的副本索引：文件/卷位掩码 + 每卷持有的文件掩码。

    - 创建时一次性汇总；后续规划只做位运算，不再逐副本扫描全目录。
    - ``disable_copy`` / ``mark_done`` 只改动回执所涉单个文件的掩码，
      其余文件的索引原样复用。
    """

    def __init__(
        self,
        copies_by_file: dict[str, list[Copy]],
        all_files: list[str] | tuple[str, ...],
    ) -> None:
        # 文件位号按 all_files 的顺序分配（作业中即文件清单顺序），保证确定性
        names: list[str] = list(all_files)
        seen = set(names)
        for f in copies_by_file:
            if f not in seen:
                names.append(f)
                seen.add(f)
        self.names: list[str] = names
        self.bit_of_file: dict[str, int] = {f: i for i, f in enumerate(names)}

        self.copies_by_file: dict[str, list[Copy]] = {
            f: list(cs) for f, cs in copies_by_file.items()
        }
        self.volumes: list[int] = sorted(
            {c.volume for cs in self.copies_by_file.values() for c in cs}
        )
        # cover[v]：卷 v 上仍有可用副本的文件位掩码
        self.cover: dict[int, int] = {v: 0 for v in self.volumes}
        for f, cs in self.copies_by_file.items():
            file_bit = 1 << self.bit_of_file[f]
            for v in {c.volume for c in cs}:
                self.cover[v] |= file_bit
        # present_mask：至少有一个可用副本的文件；pending_mask：仍待取的文件。
        # 注意没有任何副本的文件不在 present_mask 中，但仍是待取（无副本可读）文件。
        self.present_mask: int = 0
        for mask in self.cover.values():
            self.present_mask |= mask
        self.pending_mask = (1 << len(names)) - 1

    # ---- 作业内增量更新（只触及回执所涉文件）----

    def disable_copy(self, file: str, volume: int, position: int) -> None:
        """停用一个副本：仅重算该文件在各卷上的掩码贡献。"""
        cs = self.copies_by_file.get(file)
        if not cs:
            return
        remaining = [c for c in cs if not (c.volume == volume and c.position == position)]
        if len(remaining) == len(cs):
            return
        self.copies_by_file[file] = remaining
        file_bit = 1 << self.bit_of_file[file]
        for v in self.volumes:  # 先摘掉该文件的全部贡献，再按剩余副本放回
            self.cover[v] &= ~file_bit
        for v in {c.volume for c in remaining}:
            self.cover[v] |= file_bit
        if not remaining:
            self.present_mask &= ~file_bit

    def mark_done(self, file: str) -> None:
        """文件取回成功：从待取掩码移除（副本汇总保持不变，仍可被复用查询）。"""
        self.pending_mask &= ~(1 << self.bit_of_file[file])

    def names_from_mask(self, mask: int) -> list[str]:
        return [self.names[i] for i in range(len(self.names)) if mask & (1 << i)]

    # ---- 规划 ----

    def build_plan(self, current_volume: int | None, head: int) -> Plan:
        """按当前待取掩码与带头位置生成最优读取计划。"""
        present = self.pending_mask & self.present_mask
        unretrievable = sorted(
            self.names_from_mask(self.pending_mask & ~self.present_mask)
        )
        if present == 0:
            return Plan([], [], 0, 0, unretrievable)

        # 仅保留与待取文件有关的卷，按卷号升序分配子集位号（与历史行为一致）
        rel_volumes = [v for v in self.volumes if self.cover[v] & present]
        n = len(rel_volumes)
        bit_of_volume = {v: i for i, v in enumerate(rel_volumes)}
        rel_cover = [self.cover[v] & present for v in rel_volumes]

        # 子集覆盖并集：LSB 递推，全部子集 O(2^V)
        size = 1 << n
        union = [0] * size
        for mask in range(1, size):
            lsb = mask & -mask
            union[mask] = union[mask ^ lsb] | rel_cover[lsb.bit_length() - 1]

        # “免回卷并集”：当前卷只贡献带头之后（含当前位置）的副本，
        # 其余卷的贡献与普通并集相同。某文件被选中卷覆盖却不在免回卷并集中，
        # 当且仅当它在选中卷内的副本全部落在当前卷且全部低于带头——必须回卷。
        current_bit = (
            bit_of_volume[current_volume]
            if current_volume is not None and current_volume in bit_of_volume
            else None
        )
        safe_cover = list(rel_cover)
        if current_bit is not None:
            after_head = 0
            remaining = present
            while remaining:
                lsb = remaining & -remaining
                remaining ^= lsb
                f = self.names[lsb.bit_length() - 1]
                if any(
                    c.volume == current_volume and c.position >= head
                    for c in self.copies_by_file[f]
                ):
                    after_head |= lsb
            safe_cover[current_bit] = after_head
        safe_union = [0] * size
        for mask in range(1, size):
            lsb = mask & -mask
            safe_union[mask] = safe_union[mask ^ lsb] | safe_cover[lsb.bit_length() - 1]

        # 枚举全部非空子集并评分：覆盖检查与回卷判定均为 O(1) 位运算
        best_key: tuple[int, int, int, int] | None = None
        best_mask = 0
        for mask in range(1, size):
            if union[mask] != present:
                continue
            used = mask.bit_count()
            has_current = current_bit is not None and bool(mask & (1 << current_bit))
            mounts = used - (1 if has_current else 0)
            rewinds = 1 if has_current and safe_union[mask] != present else 0
            key = (mounts + rewinds, mounts, used, mask)
            if best_key is None or key < best_key:
                best_key = key
                best_mask = mask

        selected = {
            rel_volumes[i] for i in range(n) if best_mask & (1 << i)
        }

        # 待取且有副本的文件，按文件清单顺序指派副本
        targets = [
            self.names[i]
            for i in range(len(self.names))
            if present & (1 << i)
        ]
        by_volume: dict[int, list[Copy]] = {}
        for f in targets:
            chosen = _assign(
                f, self.copies_by_file, selected, current_volume, head
            )
            by_volume.setdefault(chosen.volume, []).append(chosen)

        # 当前卷最先读（避免卸下再装），其余卷按卷号升序
        ordered = sorted(by_volume, key=lambda v: (v != current_volume, v))

        steps: list[Step] = []
        seq = 1

        def emit(action: str, **kw) -> None:
            nonlocal seq
            steps.append(Step(seq=seq, action=action, **kw))
            seq += 1

        for volume in ordered:
            items = sorted(by_volume[volume], key=lambda c: (c.position, c.file))
            if volume == current_volume:
                after = [c for c in items if c.position >= head]
                before = [c for c in items if c.position < head]
                for c in after:
                    emit("read", volume=volume, file=c.file, position=c.position)
                if before:
                    emit("rewind", volume=volume)
                    for c in before:
                        emit("read", volume=volume, file=c.file, position=c.position)
            else:
                emit("mount", volume=volume)
                for c in items:
                    emit("read", volume=volume, file=c.file, position=c.position)

        mounts = sum(1 for s in steps if s.action == "mount")
        rewinds = sum(1 for s in steps if s.action == "rewind")
        return Plan(steps, ordered, mounts, rewinds, unretrievable)


def _assign(
    file: str,
    copies: dict[str, list[Copy]],
    selected: set[int],
    current_volume: int | None,
    head: int,
) -> Copy:
    """为文件在选定卷中挑一个副本（实现上面计算出的回卷数）。"""
    candidates = [c for c in copies[file] if c.volume in selected]
    # 优先放到非当前卷（从卷首升序读，绝无回卷）
    off_current = [c for c in candidates if c.volume != current_volume]
    if off_current:
        return min(off_current, key=lambda c: (c.volume, c.position))
    # 只能放当前卷：优先带头之后（含当前位置）的副本，避免回卷
    after_head = [c for c in candidates if c.position >= head]
    if after_head:
        return min(after_head, key=lambda c: c.position)
    return min(candidates, key=lambda c: c.position)


def build_plan(
    pending: list[str],
    copies: dict[str, list[Copy]],
    current_volume: int | None,
    head: int,
) -> Plan:
    """按当前状态为剩余待取文件生成最优读取计划（无状态入口，便于单测）。

    pending: 待取文件名列表。
    copies: 每个文件当前可用（未停用）的副本列表。
    current_volume: 当前已挂载的卷号（未挂载为 None）。
    head: 带头当前所在的卷内序号。
    """
    index = CopyIndex(copies, all_files=pending)
    return index.build_plan(current_volume, head)
