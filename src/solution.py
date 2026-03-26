from __future__ import annotations

import heapq
import time
from math import inf

from player import Player
from board import HexBoard


_DIRS = (
    ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)),  # even row
    ((-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1, 1)),  # odd  row
)


class SmartPlayer(Player):
    # Hard limit: 4.5 s.  Internal guard fires at 4.0 s so we always have
    # time to return a move before the harness kills us at 5 s.
    TIME_LIMIT:  float = 4.5
    _TIME_GUARD: float = 4.0   # internal early-stop threshold

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.opponent_id: int = 3 - player_id

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def play(self, board: HexBoard) -> tuple[int, int]:
        """
        Iterative Deepening Alpha-Beta (IDAB).
        Returns the best move from the deepest *fully completed* depth.
        Always has a fallback move — never returns None.
        """
        self._start_time = time.time()

        empty = self._get_empty_cells(board)
        if not empty:
            raise RuntimeError("No moves available — board is full.")

        # Guaranteed fallback: best greedy move at depth 1
        best_move = self._greedy_move(board, empty)

        for depth in range(1, board.size * board.size + 1):
            if self._time_up():
                break

            move, completed = self._best_move_at_depth(board, depth)

            if completed and move is not None:
                best_move = move
                # Found a forced win — no need to go deeper
                if self._last_score == inf:
                    break

            if not completed:
                break

        return best_move

    def _best_move_at_depth(
        self, board: HexBoard, depth: int
    ) -> tuple[tuple[int, int] | None, bool]:
        """
        One full Alpha-Beta pass at the given depth.

        Returns:
            (best_move, completed)
            completed=False → ran out of time; result may be unreliable.
        """
        best_move:  tuple[int, int] | None = None
        best_score: float = -inf
        self._last_score: float = -inf

        candidates = self._get_ordered_candidates(board, self.player_id)

        for row, col in candidates:
            if self._time_up():
                return best_move, (best_move is not None)

            board.board[row][col] = self.player_id
            score = self._minimax(
                board,
                depth=depth - 1,
                alpha=-inf,
                beta=inf,
                maximizing=False,
            )
            board.board[row][col] = 0

            if score > best_score:
                best_score = score
                best_move  = (row, col)

            if best_score == inf:   # forced win — stop early
                break

        self._last_score = best_score
        return best_move, True

    def _minimax(
        self,
        board:       HexBoard,
        depth:       int,
        alpha:       float,
        beta:        float,
        maximizing:  bool,
    ) -> float:
        """
        Alpha-Beta Minimax with in-place board mutation (no clone overhead).

        Cutoffs:
            β-cutoff (inside MAX): value ≥ beta  → MIN above would never pick this
            α-cutoff (inside MIN): value ≤ alpha → MAX above would never pick this
        """
        if self._time_up():
            return self._evaluate(board)

        # Terminal states
        if board.check_connection(self.player_id):
            return inf
        if board.check_connection(self.opponent_id):
            return -inf

        # Depth horizon
        if depth == 0:
            return self._evaluate(board)

        current_player = self.player_id if maximizing else self.opponent_id
        candidates     = self._get_ordered_candidates(board, current_player)

        if not candidates:
            return self._evaluate(board)

        if maximizing:
            value = -inf
            for r, c in candidates:
                board.board[r][c] = self.player_id
                value = max(value, self._minimax(board, depth-1, alpha, beta, False))
                board.board[r][c] = 0
                alpha = max(alpha, value)
                if value >= beta:
                    break   # β-cutoff
            return value
        else:
            value = inf
            for r, c in candidates:
                board.board[r][c] = self.opponent_id
                value = min(value, self._minimax(board, depth-1, alpha, beta, True))
                board.board[r][c] = 0
                beta = min(beta, value)
                if value <= alpha:
                    break   # α-cutoff
            return value

    # -----------------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------------

    def _get_ordered_candidates(
        self, board: HexBoard, current_player: int
    ) -> list[tuple[int, int]]:
        """
        Return empty cells sorted best-first for the given player.

        Ordering heuristic: prefer cells adjacent to existing pieces
        (own or opponent's) — they are almost always more relevant than
        isolated cells in the middle of nowhere.  Ties broken by a quick
        Dijkstra estimate only when the board is small enough.

        Phase 4 will add hard filtering (CSP) on top of this ordering.
        """
        empty = self._get_empty_cells(board)
        n = board.size

        # Fast adjacency score: count neighbouring occupied cells
        def adjacency(r: int, c: int) -> int:
            return sum(
                1 for nr, nc in self._get_neighbors(r, c, n)
                if board.board[nr][nc] != 0
            )

        # On small boards (N≤5) do full heuristic ordering — affordable
        # On larger boards use adjacency only to keep candidate gen fast
        if n <= 5:
            scored = []
            for r, c in empty:
                board.board[r][c] = current_player
                h = self._evaluate(board)
                board.board[r][c] = 0
                # Flip sign so higher-is-better sorts first
                key = -h if current_player == self.player_id else h
                scored.append((key, r, c))
            scored.sort()
            return [(r, c) for _, r, c in scored]
        else:
            # Primary: adjacency (descending), secondary: stable index
            scored = [(-adjacency(r, c), r, c) for r, c in empty]
            scored.sort()
            return [(r, c) for _, r, c in scored]

    def _greedy_move(
        self, board: HexBoard, empty: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """
        Pick the single best move by 1-ply heuristic evaluation.
        Used as the guaranteed fallback in play().
        """
        best_move  = empty[0]
        best_score = -inf
        for r, c in empty:
            board.board[r][c] = self.player_id
            score = self._evaluate(board)
            board.board[r][c] = 0
            if score > best_score:
                best_score = score
                best_move  = (r, c)
        return best_move

    def _dijkstra(self, board: HexBoard, player_id: int) -> float:
        """
        Minimum empty cells player_id still needs to fill to connect sides.
        Cost: own=0, empty=1, opponent=∞ (impassable).
        """
        n   = board.size
        opp = 3 - player_id

        dist: list[list[float]] = [[inf] * n for _ in range(n)]
        heap: list[tuple[float, int, int]] = []

        if player_id == 1:
            sources = [(r, 0) for r in range(n)]
            def is_goal(r: int, c: int) -> bool: return c == n - 1
        else:
            sources = [(0, c) for c in range(n)]
            def is_goal(r: int, c: int) -> bool: return r == n - 1

        for r, c in sources:
            if board.board[r][c] == opp:
                continue
            cost = 0.0 if board.board[r][c] == player_id else 1.0
            if cost < dist[r][c]:
                dist[r][c] = cost
                heapq.heappush(heap, (cost, r, c))

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue
            if is_goal(r, c):
                return d
            for nr, nc in self._get_neighbors(r, c, n):
                if board.board[nr][nc] == opp:
                    continue
                step = 0.0 if board.board[nr][nc] == player_id else 1.0
                nd = d + step
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return inf

    def _evaluate(self, board: HexBoard) -> float:
        """h(s) = dist_opponent - dist_self. ±inf for terminal states."""
        if board.check_connection(self.player_id):   return  inf
        if board.check_connection(self.opponent_id): return -inf

        d_self = self._dijkstra(board, self.player_id)
        d_opp  = self._dijkstra(board, self.opponent_id)

        if d_self == inf and d_opp == inf: return  0.0
        if d_self == inf:                  return -inf
        if d_opp  == inf:                  return  inf

        return d_opp - d_self

    def _get_neighbors(self, row: int, col: int, size: int) -> list[tuple[int, int]]:
        """In-bounds hex neighbours using even-r offset layout."""
        return [
            (row + dr, col + dc)
            for dr, dc in _DIRS[row % 2]
            if 0 <= row + dr < size and 0 <= col + dc < size
        ]

    def _get_empty_cells(self, board: HexBoard) -> list[tuple[int, int]]:
        """All empty (row, col) positions."""
        n = board.size
        return [
            (r, c)
            for r in range(n)
            for c in range(n)
            if board.board[r][c] == 0
        ]

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _time_up(self) -> bool:
        """True when we are within the internal safety margin."""
        return (time.time() - self._start_time) >= self._TIME_GUARD

    # -----------------------------------------------------------------------
    # Phase 4 stub
    # -----------------------------------------------------------------------

    def _get_candidates(self, board: HexBoard) -> list[tuple[int, int]]:
        """CSP-informed candidate filter — Phase 4."""
        raise NotImplementedError