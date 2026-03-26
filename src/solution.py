from __future__ import annotations

import heapq
import random
import time
from math import inf

from player import Player
from board import HexBoard


# ---------------------------------------------------------------------------
# Even-r offset direction tables (one per row parity)
# ---------------------------------------------------------------------------
_DIRS = (
    ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)),  # even row
    ((-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1, 1))   # odd  row
)


class SmartPlayer(Player):

    TIME_LIMIT: float = 4.5  # seconds per move (hard limit is 5 s)

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.opponent_id: int = 3 - player_id  # 1→2, 2→1

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def play(self, board: HexBoard) -> tuple[int, int]:
        """
        Return the best available move as (row, col).
        Phase 2: picks the empty cell that maximises _evaluate() after
        placing our piece there — a greedy 1-ply search with Dijkstra
        heuristic.
        """
        self._start_time = time.time()

        empty = self._get_empty_cells(board)
        if not empty:
            raise RuntimeError("No moves available — board is full.")

        best_move = empty[0]
        best_score = -inf

        for row, col in empty:
            # Simulate placing our piece
            board.board[row][col] = self.player_id
            score = self._evaluate(board)
            board.board[row][col] = 0  # undo

            if score > best_score:
                best_score = score
                best_move = (row, col)

        return best_move

    # -----------------------------------------------------------------------
    # Phase 1 — Board navigation (own helpers, no dependency on board.py extras)
    # -----------------------------------------------------------------------

    def _get_neighbors(self, row: int, col: int, size: int) -> list[tuple[int, int]]:
        """
        Return in-bounds hex neighbours of (row, col) using even-r layout.
        We keep this in SmartPlayer so solution.py is self-contained for
        all navigation logic beyond the base HexBoard API.
        """
        dirs = _DIRS[row % 2]
        return [
            (row + dr, col + dc)
            for dr, dc in dirs
            if 0 <= row + dr < size and 0 <= col + dc < size
        ]

    def _get_empty_cells(self, board: HexBoard) -> list[tuple[int, int]]:
        """All empty (row, col) positions — the legal move set."""
        n = board.size
        return [
            (r, c)
            for r in range(n)
            for c in range(n)
            if board.board[r][c] == 0
        ]

    # -----------------------------------------------------------------------
    # Phase 2 — Dijkstra heuristic
    # -----------------------------------------------------------------------

    def _dijkstra(self, board: HexBoard, player_id: int) -> int:
        """
        Return the *minimum number of empty cells* that player_id still
        needs to fill to connect their two sides.

        Cost model (resistance):
            - Own cell  → 0  (already placed, free to traverse)
            - Empty cell → 1  (must be played here in the future)
            - Opponent  → ∞  (impassable — never enqueued)

        The answer is the shortest-path distance from the player's
        source edge to their target edge under this cost model.

        Returns 0  if the player has already won.
        Returns inf (as a large int) if the target edge is unreachable.
        """
        n = board.size
        opp = 3 - player_id

        # --- initialise distances ---
        dist: list[list[float]] = [[inf] * n for _ in range(n)]
        heap: list[tuple[float, int, int]] = []

        # --- seed with the entire source edge ---
        if player_id == 1:          # left edge: col 0
            source_cells = [(r, 0) for r in range(n)]
            def is_goal(r: int, c: int) -> bool: return c == n - 1
        else:                        # top edge: row 0
            source_cells = [(0, c) for c in range(n)]
            def is_goal(r: int, c: int) -> bool: return r == n - 1

        for r, c in source_cells:
            if board.board[r][c] == opp:
                continue                  # blocked at source, skip
            cost = 0 if board.board[r][c] == player_id else 1
            if cost < dist[r][c]:
                dist[r][c] = cost
                heapq.heappush(heap, (cost, r, c))

        # --- Dijkstra relaxation ---
        while heap:
            d, r, c = heapq.heappop(heap)

            if d > dist[r][c]:
                continue              # stale entry

            if is_goal(r, c):
                return float(d)         # reached the target edge

            for nr, nc in self._get_neighbors(r, c, n):
                if board.board[nr][nc] == opp:
                    continue          # impassable
                step = 0 if board.board[nr][nc] == player_id else 1
                nd = d + step
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return inf               # target edge unreachable

    def _evaluate(self, board: HexBoard) -> float:
        """
        Heuristic board score from self.player_id's perspective.

        Formula:
            h(s) = dist_opponent - dist_self

        Interpretation:
            > 0  →  opponent needs more moves to win  →  we have an advantage
            = 0  →  balanced position
            < 0  →  we need more moves to win         →  opponent has advantage

        Terminal cases (handled first for correctness):
            Win  → +inf
            Loss → -inf
        """
        # Check terminal states first (most reliable, no approximation)
        if board.check_connection(self.player_id):
            return inf
        if board.check_connection(self.opponent_id):
            return -inf

        dist_self = self._dijkstra(board, self.player_id)
        dist_opp  = self._dijkstra(board, self.opponent_id)

        # Edge case: unreachable sides (should not occur in legal play)
        if dist_self == inf and dist_opp == inf:
            return 0.0
        if dist_self == inf:
            return -inf
        if dist_opp == inf:
            return inf

        return float(dist_opp - dist_self)

    # -----------------------------------------------------------------------
    # Phases 3-5 — stubs
    # -----------------------------------------------------------------------

    def _minimax(
        self,
        board: HexBoard,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        """Alpha-Beta Minimax — implemented in Phase 3."""
        raise NotImplementedError

    def _get_candidates(self, board: HexBoard) -> list[tuple[int, int]]:
        """CSP-informed candidate move list — implemented in Phase 4."""
        raise NotImplementedError