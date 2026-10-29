"""R5: ưu tiên đoạn của CLB được nhắc tới trong câu hỏi (boost mềm, không lọc cứng).

- Boost: mỗi đoạn thuộc CLB được nhắc tới được cộng `weight / (k + 1)` — tương đương được xếp hạng 1
  trong thêm một "kênh" RRF ảo, nên cùng thang với điểm RRF.
- Quota: câu nhắc ≥ 2 CLB (so sánh) thì mỗi CLB có ít nhất 1 đoạn trong top-5.
- Không lọc cứng: câu về CLB này vẫn có thể cần đoạn của cộng đồng S2B hoặc CLB khác.
"""

from __future__ import annotations

from neorag.data import Document
from s2b_rag.aliases import detect_clubs, enrich


class ClubBoostRetriever:
    def __init__(self, base, weight: float = 0.5, k: int = 60, quota_top: int = 5):
        self.base, self.weight, self.k, self.quota_top = base, weight, k, quota_top

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)
        self.club_of = {d.id: d.club for d in corpus}

    def search(self, query: str) -> list[tuple[int, float]]:
        results = self.base.search(query)
        clubs = set(detect_clubs(query))
        if not clubs:
            return results
        bonus = self.weight / (self.k + 1)
        boosted = [(i, s + bonus if self.club_of[i] in clubs else s) for i, s in results]
        boosted.sort(key=lambda r: -r[1])
        return self._apply_quota(boosted, clubs) if len(clubs) >= 2 else boosted

    def _apply_quota(self, ranked: list[tuple[int, float]], clubs: set[str]) -> list:
        """Đưa đoạn tốt nhất của CLB còn thiếu lên top-N (thay đoạn cuối top-N chưa cần giữ)."""
        ranked = list(ranked)
        for club in sorted(clubs):
            top = ranked[: self.quota_top]
            if any(self.club_of[i] == club for i, _ in top):
                continue
            best = next((r for r in ranked if self.club_of[r[0]] == club), None)
            if best is None:
                continue
            # chọn đoạn nhường chỗ: từ dưới lên, không làm CLB được hỏi nào mất suất duy nhất
            for pos in range(self.quota_top - 1, -1, -1):
                victim_club = self.club_of[top[pos][0]]
                same = sum(self.club_of[i] == victim_club for i, _ in top)
                if victim_club not in clubs or same > 1:
                    ranked.remove(best)
                    victim = ranked.pop(pos)
                    ranked.insert(pos, best)  # đoạn được đưa lên đứng đúng chỗ đoạn nhường
                    ranked.insert(self.quota_top, victim)  # đoạn nhường xuống ngay sau top-N
                    break
        return ranked


class EnrichedRetriever:
    """R4: index văn bản đã gắn tiền tố CLB + alias + chủ đề thay vì văn bản gốc."""

    def __init__(self, base):
        self.base = base

    def index(self, corpus: list[Document]) -> None:
        self.base.index(
            [Document(d.id, d.club, d.topic, enrich(d.club, d.topic, d.text)) for d in corpus]
        )

    def search(self, query: str) -> list[tuple[int, float]]:
        return self.base.search(query)
