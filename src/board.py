from typing import Self

class HexBoard:
    def __init__(self, size: int):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]
    #Matriz NxN (0 = vacio, 1 = jugador 1, 2 = jugador 2)

    def clone(self) -> Self:
        #Devuelve una copia del tablero actual
        pass

    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        #Coloca una pieza en la posición dada para el jugador especificado
        #Devuelve True si la pieza se colocó correctamente, False si la posición ya está ocupada
        pass

    def check_connection(self, player_id: int) -> bool:
        #Verifica si el jugador ha conectado sus dos lados 
        pass