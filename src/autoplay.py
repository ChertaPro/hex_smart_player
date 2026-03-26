"""
autoplay.py — Fase 6: Motor de autoplay para SmartPlayer
=========================================================
Ejecuta partidas automáticas entre SmartPlayer vs SmartPlayer
y SmartPlayer vs RandomPlayer para distintos tamaños de tablero.

Uso:
    python autoplay.py [--sizes 5 7 9 11] [--games 20] [--verbose]

Salida:
    Tabla con win-rate, nº de movimientos promedio y tiempo por movimiento.
    Exporta resultados a autoplay_results.json para el benchmark.
"""

import argparse
import json
import random
import time
import sys
import os

# Asegurarse de que src/ está en el path si se ejecuta desde la raíz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from board import HexBoard
from player import Player
from solution import SmartPlayer


# ---------------------------------------------------------------------------
# Jugador aleatorio de referencia
# ---------------------------------------------------------------------------

class RandomPlayer(Player):
    """Juega en una celda vacía al azar. Baseline de referencia."""

    def play(self, board: HexBoard) -> tuple[int, int]:
        empty = [
            (r, c)
            for r in range(board.size)
            for c in range(board.size)
            if board.board[r][c] == 0
        ]
        return random.choice(empty)


# ---------------------------------------------------------------------------
# Motor de partida
# ---------------------------------------------------------------------------

def play_game(
    board_size: int,
    player1: Player,
    player2: Player,
    verbose: bool = False,
) -> dict:
    """
    Juega una partida completa y devuelve estadísticas.

    Returns dict con:
        winner        : 1 o 2
        total_moves   : número de movimientos realizados
        move_times    : lista de tiempos por movimiento (segundos)
        move_history  : lista de (row, col, player_id)
        disqualified  : True si algún jugador superó 5 s
    """
    board    = HexBoard(board_size)
    players  = {1: player1, 2: player2}
    current  = 1
    move_times: list[float] = []
    move_history: list[tuple[int, int, int]] = []
    disqualified = False

    while True:
        p = players[current]
        t0 = time.time()
        row, col = p.play(board)
        elapsed = time.time() - t0

        move_times.append(elapsed)

        # Control de tiempo: descalificación si > 5 s
        if elapsed > 5.0:
            print(f"  ⚠  Jugador {current} DESCALIFICADO ({elapsed:.2f}s > 5.0s)")
            disqualified = True
            winner = 3 - current
            break

        ok = board.place_piece(row, col, current)
        if not ok:
            # Movimiento inválido — se considera descalificación
            print(f"  ⚠  Jugador {current} hizo movimiento inválido ({row},{col})")
            disqualified = True
            winner = 3 - current
            break

        move_history.append((row, col, current))

        if verbose:
            print(f"  P{current} → ({row},{col})  [{elapsed*1000:.1f} ms]")
            print(board)
            print()

        if board.check_connection(current):
            winner = current
            break

        current = 3 - current

    return {
        "winner":       winner,
        "total_moves":  len(move_history),
        "move_times":   move_times,
        "move_history": move_history,
        "disqualified": disqualified,
    }


# ---------------------------------------------------------------------------
# Suite de benchmarks
# ---------------------------------------------------------------------------

def run_suite(
    sizes:   list[int],
    n_games: int,
    verbose: bool,
) -> dict:
    """Ejecuta n_games partidas por tamaño y configuración."""

    all_results = {}

    configs = [
        ("smart_vs_smart",  lambda: SmartPlayer(1), lambda: SmartPlayer(2)),
        ("smart_vs_random", lambda: SmartPlayer(1), lambda: RandomPlayer(2)),
        ("random_vs_smart", lambda: RandomPlayer(1), lambda: SmartPlayer(2)),
    ]

    for size in sizes:
        all_results[size] = {}
        print(f"\n{'='*60}")
        print(f"  Tablero {size}×{size}")
        print(f"{'='*60}")

        for config_name, make_p1, make_p2 in configs:
            wins   = {1: 0, 2: 0}
            moves_list: list[int]   = []
            times_list: list[float] = []
            max_move_time: float    = 0.0
            disqs = 0

            print(f"\n  [{config_name}] — {n_games} partidas")

            for g in range(n_games):
                p1 = make_p1()
                p2 = make_p2()
                result = play_game(size, p1, p2, verbose=verbose)

                wins[result["winner"]] += 1
                moves_list.append(result["total_moves"])
                times_list.extend(result["move_times"])
                if result["move_times"]:
                    max_move_time = max(max_move_time, max(result["move_times"]))
                if result["disqualified"]:
                    disqs += 1

                bar = "█" * (g + 1) + "░" * (n_games - g - 1)
                print(f"    [{bar}] {g+1}/{n_games}  ganador=P{result['winner']}  "
                      f"movs={result['total_moves']}  "
                      f"t_max={max(result['move_times'])*1000:.0f}ms",
                      end="\r")

            print()  # newline tras la barra de progreso

            avg_moves = sum(moves_list) / len(moves_list) if moves_list else 0
            avg_time  = sum(times_list) / len(times_list) if times_list else 0

            stats = {
                "wins_p1":       wins[1],
                "wins_p2":       wins[2],
                "win_rate_p1":   wins[1] / n_games,
                "win_rate_p2":   wins[2] / n_games,
                "avg_moves":     round(avg_moves, 1),
                "avg_move_time": round(avg_time * 1000, 2),   # ms
                "max_move_time": round(max_move_time * 1000, 2),
                "disqualifications": disqs,
            }
            all_results[size][config_name] = stats

            print(f"    P1 wins: {wins[1]}/{n_games} ({wins[1]/n_games*100:.0f}%)  "
                  f"P2 wins: {wins[2]}/{n_games} ({wins[2]/n_games*100:.0f}%)")
            print(f"    Movs promedio: {avg_moves:.1f}  "
                  f"t_prom: {avg_time*1000:.1f}ms  "
                  f"t_max: {max_move_time*1000:.1f}ms  "
                  f"descalificaciones: {disqs}")

    return all_results


# ---------------------------------------------------------------------------
# Reporte final
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                  RESUMEN — SmartPlayer Fase 6               ║")
    print("╠══════════╦══════════╦═══════════╦══════════╦════════════════╣")
    print("║  Tamaño  ║  Config  ║  WR Smart ║ t_prom   ║  t_max (<5s?) ║")
    print("╠══════════╬══════════╬═══════════╬══════════╬════════════════╣")

    smart_configs = {
        "smart_vs_smart":  "SvsS P1",
        "smart_vs_random": "SvsR P1",
        "random_vs_smart": "RvsS P2",
    }
    wr_key = {
        "smart_vs_smart":  "win_rate_p1",
        "smart_vs_random": "win_rate_p1",
        "random_vs_smart": "win_rate_p2",
    }

    for size, configs in results.items():
        first = True
        for cfg, label in smart_configs.items():
            if cfg not in configs:
                continue
            s   = configs[cfg]
            wr  = s[wr_key[cfg]] * 100
            tav = s["avg_move_time"]
            tmx = s["max_move_time"]
            ok  = "✅" if tmx < 5000 else "❌"
            sz  = f"{size}×{size}" if first else ""
            print(f"║  {sz:<8}║  {label:<8}║  {wr:>6.0f}%   ║  {tav:>5.1f}ms  ║  {tmx:>7.1f}ms {ok}  ║")
            first = False
        print("╠══════════╬══════════╬═══════════╬══════════╬════════════════╣")

    print("╚══════════╩══════════╩═══════════╩══════════╩════════════════╝")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoplay benchmark — SmartPlayer Hex")
    parser.add_argument("--sizes",  nargs="+", type=int, default=[5, 7, 9, 11],
                        help="Tamaños de tablero a probar (default: 5 7 9 11)")
    parser.add_argument("--games",  type=int, default=10,
                        help="Partidas por configuración (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar cada movimiento y el tablero")
    parser.add_argument("--output", default="autoplay_results.json",
                        help="Fichero JSON de salida")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       SmartPlayer — Autoplay & Benchmark  (Fase 6)          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Tamaños : {args.sizes}")
    print(f"  Partidas: {args.games} por configuración")
    print(f"  Verbose : {args.verbose}")

    results = run_suite(args.sizes, args.games, args.verbose)

    print_summary(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
