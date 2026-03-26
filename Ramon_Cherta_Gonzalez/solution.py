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


def _max_depth_for_size(n: int) -> int:
    if n <= 5:  return 20   # tiny boards: search to near-terminal depth
    if n <= 7:  return 10
    if n <= 9:  return 7
    if n <= 11: return 5
    return 4 

_HARD_LIMIT    = 5.0   # tournament disqualification threshold (s)
_TIME_GUARD    = 4.3   # internal deadline — stop *starting* new work here
_DEPTH_BUDGET_FRAC = 0.45  # if last depth used > 45 % of budget, skip next


class SmartPlayer(Player):

    TIME_LIMIT:  float = 4.5   # public SLA (documented)

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.opponent_id: int = 3 - player_id

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def play(self, board: HexBoard) -> tuple[int, int]:
        
        self._start_time = time.time()
        n = board.size

        pieces_on_board = sum(
            board.board[r][c] != 0
            for r in range(n)
            for c in range(n)
        )
        if pieces_on_board <= 1:
            center = n // 2
            if board.board[center][center] == 0:
                return (center, center)
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)):
                nr, nc = center + dr, center + dc
                if 0 <= nr < n and 0 <= nc < n and board.board[nr][nc] == 0:
                    return (nr, nc)

        empty = self._get_empty_cells(board)
        if not empty:
            raise RuntimeError("No moves available — board is full.")

        best_move  = self._greedy_move(board, empty)
        max_depth  = _max_depth_for_size(n)

        prev_depth_elapsed = 0.0   # time spent on the last completed depth

        for depth in range(1, max_depth + 1):
            
            elapsed   = time.time() - self._start_time
            remaining = _TIME_GUARD - elapsed

            if remaining <= 0:
                break  # hard stop

            # If the previous depth took more than DEPTH_BUDGET_FRAC of what
            # is left, the next (deeper) search is very unlikely to complete —
            # skip it to avoid returning a half-baked result.
            if depth > 1 and prev_depth_elapsed > DEPTH_BUDGET_FRAC * remaining:
                break

            depth_start = time.time()
            move, completed = self._best_move_at_depth(board, depth)
            prev_depth_elapsed = time.time() - depth_start

            if completed and move is not None:
                best_move = move
                if self._last_score == inf:
                    break   # forced win found — no need to search deeper

            if not completed:
                # Timer fired mid-depth; the partial result may be unreliable.
                break

        return best_move

    def _best_move_at_depth(
        self, board: HexBoard, depth: int
    ) -> tuple[tuple[int, int] | None, bool]:
        
        best_move:  tuple[int, int] | None = None
        best_score: float = -inf
        self._last_score: float = -inf

        candidates = self._get_candidates(board, self.player_id)

        for row, col in candidates:
            if self._time_up():
                return best_move, (best_move is not None)

            board.board[row][col] = self.player_id
            score = self._minimax(board, depth - 1, -inf, inf, False)
            board.board[row][col] = 0

            if score > best_score:
                best_score = score
                best_move  = (row, col)

            if best_score == inf:
                break   # forced win — prune remaining root moves

        self._last_score = best_score
        return best_move, True

    def _minimax(
        self,
        board:      HexBoard,
        depth:      int,
        alpha:      float,
        beta:       float,
        maximizing: bool,
    ) -> float:
        """Alpha-Beta Minimax with CSP candidate filtering at every node."""
        if self._time_up():
            return self._evaluate(board)

        if board.check_connection(self.player_id):   return  inf
        if board.check_connection(self.opponent_id): return -inf
        if depth == 0:                               return self._evaluate(board)

        current_player = self.player_id if maximizing else self.opponent_id
        candidates     = self._get_candidates(board, current_player)

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
                    break
            return value
        else:
            value = inf
            for r, c in candidates:
                board.board[r][c] = self.opponent_id
                value = min(value, self._minimax(board, depth-1, alpha, beta, True))
                board.board[r][c] = 0
                beta = min(beta, value)
                if value <= alpha:
                    break
            return value


    def _get_candidates(
        self, board: HexBoard, current_player: int
    ) -> list[tuple[int, int]]:
    
        candidate_set: set[tuple[int, int]] = set()

        candidate_set |= self._constraint_adjacency(board)
        candidate_set |= self._constraint_critical_path(board, self.player_id)
        candidate_set |= self._constraint_critical_path(board, self.opponent_id)
        candidate_set |= self._constraint_virtual_bridge(board, self.player_id)
        candidate_set |= self._constraint_virtual_bridge(board, self.opponent_id)

        # Fallback: if NO piece has been placed yet, every cell is equally
        # valid — return the full empty set so the opening move is unrestricted.
        # Also guard against a truly empty candidate set (should not occur in
        # normal play, but defensive programming).
        all_empty = set(self._get_empty_cells(board))
        has_pieces = any(
            board.board[r][c] != 0
            for r in range(board.size)
            for c in range(board.size)
        )
        if not candidate_set or not has_pieces:
            candidate_set = all_empty

        # Sort: best moves first → maximises Alpha-Beta pruning
        return self._sort_candidates(board, list(candidate_set), current_player)

    def _constraint_adjacency(self, board: HexBoard) -> set[tuple[int, int]]:
        
        n = board.size
        candidates: set[tuple[int, int]] = set()

        for r in range(n):
            for c in range(n):
                if board.board[r][c] == 0:
                    continue
                for nr, nc in self._get_neighbors(r, c, n):
                    if board.board[nr][nc] == 0:
                        candidates.add((nr, nc))

        return candidates

    def _constraint_critical_path(
        self, board: HexBoard, player_id: int
    ) -> set[tuple[int, int]]:
        
        n   = board.size
        opp = 3 - player_id

        dist: list[list[float]] = [[inf] * n for _ in range(n)]
        prev: list[list[tuple[int,int] | None]] = [[None] * n for _ in range(n)]
        heap: list[tuple[float, int, int]] = []

        if player_id == 1:
            sources  = [(r, 0) for r in range(n)]
            def is_goal(r: int, c: int) -> bool: return c == n - 1
        else:
            sources  = [(0, c) for c in range(n)]
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
            for nr, nc in self._get_neighbors(r, c, n):
                if board.board[nr][nc] == opp:
                    continue
                step = 0.0 if board.board[nr][nc] == player_id else 1.0
                nd   = d + step
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    prev[nr][nc] = (r, c)
                    heapq.heappush(heap, (nd, nr, nc))

        # Find the cheapest goal cell
        goal_cells = (
            [(r, n-1) for r in range(n)] if player_id == 1
            else [(n-1, c) for c in range(n)]
        )
        best_cost = inf
        best_goal: tuple[int, int] | None = None
        for r, c in goal_cells:
            if dist[r][c] < best_cost:
                best_cost = dist[r][c]
                best_goal = (r, c)

        if best_goal is None:
            return set()

        # Trace path back from goal to source
        path_cells: set[tuple[int, int]] = set()
        cur: tuple[int, int] | None = best_goal
        while cur is not None:
            r, c = cur
            if board.board[r][c] == 0:
                path_cells.add(cur)
            cur = prev[r][c]

        return path_cells

    def _constraint_virtual_bridge(
        self, board: HexBoard, player_id: int
    ) -> set[tuple[int, int]]:
        
        n = board.size
        pieces = [
            (r, c)
            for r in range(n)
            for c in range(n)
            if board.board[r][c] == player_id
        ]
        pivots: set[tuple[int, int]] = set()

        for i in range(len(pieces)):
            r1, c1 = pieces[i]
            nb_a = set(self._get_neighbors(r1, c1, n))
            for j in range(i + 1, len(pieces)):
                r2, c2 = pieces[j]
                nb_b = set(self._get_neighbors(r2, c2, n))
                for mr, mc in nb_a & nb_b:
                    if board.board[mr][mc] == 0:
                        pivots.add((mr, mc))

        return pivots

    def _sort_candidates(
        self,
        board:          HexBoard,
        candidates:     list[tuple[int, int]],
        current_player: int,
    ) -> list[tuple[int, int]]:
        
        if not candidates:
            return candidates

        scored: list[tuple[float, int, int]] = []
        for r, c in candidates:
            board.board[r][c] = current_player
            h = self._evaluate(board)
            board.board[r][c] = 0
            # For self: higher h is better → negate for ascending sort
            # For opponent: lower h for us → opponent prefers lower h → use h directly
            key = -h if current_player == self.player_id else h
            scored.append((key, r, c))

        scored.sort()
        return [(r, c) for _, r, c in scored]


    def _dijkstra(self, board: HexBoard, player_id: int) -> float:
        
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
        """h(s) = dist_opponent − dist_self.  ±inf for terminal states."""
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

    def _greedy_move(
        self, board: HexBoard, empty: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """1-ply heuristic move — guaranteed non-None fallback."""
        best_move, best_score = empty[0], -inf
        for r, c in empty:
            board.board[r][c] = self.player_id
            score = self._evaluate(board)
            board.board[r][c] = 0
            if score > best_score:
                best_score, best_move = score, (r, c)
        return best_move

    def _time_up(self) -> bool:
        """True when internal deadline has been reached."""
        return (time.time() - self._start_time) >= _TIME_GUARD


# ---------------------------------------------------------------------------
# Module-level constant referenced inside play() — defined after the class
# so the docstring can reference it by name without a forward-reference issue.
# ---------------------------------------------------------------------------
DEPTH_BUDGET_FRAC = _DEPTH_BUDGET_FRAC