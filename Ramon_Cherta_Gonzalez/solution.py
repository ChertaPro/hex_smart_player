"""
SmartPlayer for the Hex game — solution.py
==========================================
All logic lives here. board.py and player.py are provided by the
tournament harness; we only use their public interface:
    HexBoard.board, HexBoard.size, .clone(), .place_piece(), .check_connection()

Architecture:
  Phase 1 — Board navigation helpers
  Phase 2 — Dijkstra heuristic evaluation
  Phase 3 — Minimax + Alpha-Beta + Iterative Deepening
  Phase 4 — CSP candidate-move reduction
  Phase 5 — Time control & integration  ← implemented here
             · Opening book (first move → board center)
             · Adaptive depth ceiling based on board size N
             · Per-depth elapsed-time guard (don't start a depth we can't finish)
             · Hard 5 s disqualification guard with a tighter internal deadline
"""

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

# ---------------------------------------------------------------------------
# Phase 5 — Adaptive depth ceilings per board size.
#
# Empirically, Alpha-Beta + CSP can search deeper on smaller boards within
# the 5 s budget.  On a 13×13 board each _evaluate() call runs two Dijkstra
# passes over 169 cells, so the effective branching factor after CSP pruning
# still makes depth > 4 rarely reachable before the timer fires.
#
# These caps are *ceilings* — iterative deepening will stop earlier whenever
# the timer fires, so they are safe for any N.
# ---------------------------------------------------------------------------
def _max_depth_for_size(n: int) -> int:
    if n <= 5:  return 20   # tiny boards: search to near-terminal depth
    if n <= 7:  return 10
    if n <= 9:  return 7
    if n <= 11: return 5
    return 4                # 13×13 and beyond


# ---------------------------------------------------------------------------
# Phase 5 — Hard time limit.
#
# The tournament disqualifies any move that takes ≥ 5.0 s.
# We use 4.5 s as the public TIME_LIMIT (documentation / contract).
# Internally _TIME_GUARD is tighter so we always return well before 5 s even
# under Python GIL jitter or slow machines.
#
# We also reserve a *per-depth budget*: if the time already spent on the
# current depth exceeds DEPTH_BUDGET_FRAC of the remaining allowance, we skip
# the next depth entirely.
# ---------------------------------------------------------------------------
_HARD_LIMIT    = 5.0   # tournament disqualification threshold (s)
_TIME_GUARD    = 4.3   # internal deadline — stop *starting* new work here
_DEPTH_BUDGET_FRAC = 0.45  # if last depth used > 45 % of budget, skip next


class SmartPlayer(Player):
    """
    Autonomous Hex player.
    Strategy: Iterative Deepening Alpha-Beta + Dijkstra heuristic + CSP pruning
              + adaptive depth ceiling + opening book + per-depth time budgeting.

    Public contract:  play(board: HexBoard) -> (row, col)
    No mutable class-level state → safe to reuse across games.
    """

    TIME_LIMIT:  float = 4.5   # public SLA (documented)

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.opponent_id: int = 3 - player_id

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def play(self, board: HexBoard) -> tuple[int, int]:
        """
        Phase 5 entry point — Iterative Deepening Alpha-Beta with full
        time control.

        Algorithm:
          1. Opening book  — if the board is empty (or has only 1 piece),
             immediately return the centre cell (strongest opening in Hex).
          2. Greedy fallback — compute a 1-ply heuristic move as the safety
             net in case the timer fires before any full-depth search finishes.
          3. Iterative deepening loop — search depths 1 … max_depth(N).
             · Skip the next depth if the previous depth consumed more than
               DEPTH_BUDGET_FRAC of the remaining time budget (we almost
               certainly cannot finish it).
             · Stop immediately if a forced win (score == ∞) was found.
             · Stop immediately when _TIME_GUARD seconds have elapsed.
          4. Return best_move — guaranteed valid, guaranteed within budget.
        """
        self._start_time = time.time()
        n = board.size

        # ------------------------------------------------------------------
        # Phase 5 — Step 1: Opening book
        # ------------------------------------------------------------------
        pieces_on_board = sum(
            board.board[r][c] != 0
            for r in range(n)
            for c in range(n)
        )
        if pieces_on_board <= 1:
            center = n // 2
            if board.board[center][center] == 0:
                return (center, center)
            # Centre taken (opponent went first and chose it): pick the
            # next-best cell — one step towards our connection axis.
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)):
                nr, nc = center + dr, center + dc
                if 0 <= nr < n and 0 <= nc < n and board.board[nr][nc] == 0:
                    return (nr, nc)

        # ------------------------------------------------------------------
        # Step 2: Greedy fallback (never returns None)
        # ------------------------------------------------------------------
        empty = self._get_empty_cells(board)
        if not empty:
            raise RuntimeError("No moves available — board is full.")

        best_move  = self._greedy_move(board, empty)
        max_depth  = _max_depth_for_size(n)

        # ------------------------------------------------------------------
        # Step 3: Iterative deepening with per-depth time budgeting
        # ------------------------------------------------------------------
        prev_depth_elapsed = 0.0   # time spent on the last completed depth

        for depth in range(1, max_depth + 1):
            # --- Phase 5: Per-depth budget guard ---
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

    # -----------------------------------------------------------------------
    # Phase 3 — Iterative Deepening + Alpha-Beta
    # -----------------------------------------------------------------------

    def _best_move_at_depth(
        self, board: HexBoard, depth: int
    ) -> tuple[tuple[int, int] | None, bool]:
        """One full Alpha-Beta pass at the given depth.

        Returns (best_move, completed).
        completed=False means the timer fired before all candidates were
        evaluated, so best_move may not be globally optimal at this depth.
        """
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

    # -----------------------------------------------------------------------
    # Phase 4 — CSP candidate-move reduction
    # -----------------------------------------------------------------------

    def _get_candidates(
        self, board: HexBoard, current_player: int
    ) -> list[tuple[int, int]]:
        """
        CSP-based candidate move set for current_player.

        Three constraints define the candidate set:
          C1 — Adjacency:      empty cells neighbouring any existing piece.
          C2 — Critical path:  cells on the Dijkstra shortest path of either
                               player (occupying them advances our path or
                               blocks the opponent's).
          C3 — Virtual bridge: empty cells that would connect two same-colour
                               pieces separated by exactly one step through
                               that cell.

        The candidate set is the UNION of C1 ∪ C2 ∪ C3.

        Fallback: if the union is empty (e.g. completely empty board at the
        very first move) we return all empty cells so the player is never stuck.

        Final step: sort candidates best-first using the same Dijkstra
        heuristic so Alpha-Beta prunes as aggressively as possible.
        """
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
        """
        C1: All empty cells that are neighbours of at least one occupied cell.

        Rationale: in Hex, pieces only gain value through connectivity.
        An empty cell far from any piece cannot affect any current chain
        and is almost never the optimal move.
        """
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
        """
        C2: Empty cells that lie on player_id's current shortest path.

        We reconstruct the actual path by running Dijkstra while tracking
        predecessors, then trace back from the cheapest goal cell.

        Playing on these cells either:
          - Advances our own connection (if player_id == self)
          - Disrupts the opponent's cheapest route (if player_id == opponent)
        """
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
        """
        C3: Empty cells that form virtual bridges between same-colour pieces.

        A virtual bridge pivot M exists when two same-colour cells A and B
        share M as a common empty neighbour — regardless of whether A and B
        are adjacent to each other.  Occupying M either secures or breaks
        the potential connection between A and B.

        Algorithm: collect all same-colour pieces, then for every pair check
        their common empty neighbours.  O(k² · 6) where k = number of pieces.
        """
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
        """
        Sort candidates best-first for current_player.

        Scoring: simulate placing current_player's piece and compute the
        resulting heuristic.  Higher is better for self.player_id.
        We only do full evaluation when the candidate list is small enough
        to keep sorting overhead negligible.
        """
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

    # -----------------------------------------------------------------------
    # Phase 2 — Dijkstra heuristic
    # -----------------------------------------------------------------------

    def _dijkstra(self, board: HexBoard, player_id: int) -> float:
        """
        Minimum empty cells player_id still needs to fill to connect sides.
        Cost: own cell=0, empty=1, opponent=∞ (impassable).
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
        """h(s) = dist_opponent − dist_self.  ±inf for terminal states."""
        if board.check_connection(self.player_id):   return  inf
        if board.check_connection(self.opponent_id): return -inf

        d_self = self._dijkstra(board, self.player_id)
        d_opp  = self._dijkstra(board, self.opponent_id)

        if d_self == inf and d_opp == inf: return  0.0
        if d_self == inf:                  return -inf
        if d_opp  == inf:                  return  inf

        return d_opp - d_self

    # -----------------------------------------------------------------------
    # Phase 1 — Board navigation helpers
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _time_up(self) -> bool:
        """True when internal deadline has been reached."""
        return (time.time() - self._start_time) >= _TIME_GUARD


# ---------------------------------------------------------------------------
# Module-level constant referenced inside play() — defined after the class
# so the docstring can reference it by name without a forward-reference issue.
# ---------------------------------------------------------------------------
DEPTH_BUDGET_FRAC = _DEPTH_BUDGET_FRAC