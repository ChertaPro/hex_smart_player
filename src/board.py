from __future__ import annotations
from collections import deque
from typing import Self

class HexBoard:
    """
    Represents an NxN Hex board using even-r offset coordinates.
 
    Cell values:
        0 = empty
        1 = Player 1  (connects left col-0 to right col N-1)
        2 = Player 2  (connects top row-0 to bottom row N-1)
    """

    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
    #Matriz NxN (0 = vacio, 1 = jugador 1, 2 = jugador 2)

    def clone(self) -> Self:
        """Return a deep copy of this board."""

        new = HexBoard(self.size)
        new.board = [row[:] for row in self.board]
        return new

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """Place a piece if the cell is empty. Returns True on success."""
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = player_id
        return True

    def check_connection(self, player_id: int) -> bool:
        """True if player_id has connected their two target sides (BFS)."""
        n = self.size
        if player_id == 1:
            sources = [(r, 0) for r in range(n) if self.board[r][0] == 1]
            goal = lambda r, c: c == n - 1
        else:
            sources = [(0, c) for c in range(n) if self.board[0][c] == 2]
            goal = lambda r, c: r == n - 1
 
        if not sources:
            return False
 
        visited = set(sources)
        queue = deque(sources)
        # even-r neighbor offsets
        dirs = (
            ((-1,-1),(-1,0),(0,-1),(0,1),(1,-1),(1,0)),
            ((-1,0),(-1,1),(0,-1),(0,1),(1,0),(1,1))
        )
        while queue:
            r, c = queue.popleft()
            if goal(r, c):
                return True
            for dr, dc in dirs[r % 2]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < n and 0 <= nc < n:
                    if self.board[nr][nc] == player_id:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False
 
    def __repr__(self) -> str:
        sym = {0: ".", 1: "R", 2: "B"}
        return "\n".join(
            " " * r + " ".join(sym[v] for v in row)
            for r, row in enumerate(self.board)
        )