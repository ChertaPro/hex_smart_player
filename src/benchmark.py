"""
benchmark.py — Fase 6: Benchmark de tiempos por tamaño de tablero
==================================================================
Mide el tiempo de respuesta de SmartPlayer.play() en condiciones
controladas: principio, mitad y final de partida.

Verifica el invariante de Fase 5: ningún movimiento supera 5 s.

Uso:
    python benchmark.py [--sizes 5 7 9 11 13] [--samples 5]

Salida por consola + benchmark_report.json
"""

import argparse
import json
import random
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from board import HexBoard
from solution import SmartPlayer


# ---------------------------------------------------------------------------
# Helpers para construir estados de tablero representativos
# ---------------------------------------------------------------------------

def _random_filled(size: int, fill_fraction: float, seed: int) -> HexBoard:
    """Devuelve un tablero con fill_fraction de sus celdas ocupadas."""
    rng   = random.Random(seed)
    board = HexBoard(size)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    n_fill = int(len(cells) * fill_fraction)
    for i, (r, c) in enumerate(cells[:n_fill]):
        board.board[r][c] = (i % 2) + 1  # alterna P1/P2
    return board


def _phases(size: int) -> list[tuple[str, float]]:
    """Fracciones de tablero lleno a medir."""
    return [
        ("inicio  (10%)", 0.10),
        ("medio   (50%)", 0.50),
        ("final   (80%)", 0.80),
    ]


# ---------------------------------------------------------------------------
# Benchmark por tamaño
# ---------------------------------------------------------------------------

def benchmark_size(size: int, n_samples: int) -> dict:
    """
    Para cada fase (inicio/medio/final) genera n_samples tableros distintos,
    mide el tiempo de play() de SmartPlayer y devuelve estadísticas.
    """
    results = {}

    for phase_name, fill in _phases(size):
        times = []
        valid = 0
        over_limit = 0

        for seed in range(n_samples):
            board = _random_filled(size, fill, seed=seed * 1000 + size)

            # Comprobar que el tablero no sea ya terminal
            if board.check_connection(1) or board.check_connection(2):
                continue
            empty = [(r, c) for r in range(size) for c in range(size)
                     if board.board[r][c] == 0]
            if not empty:
                continue

            player = SmartPlayer(player_id=1)
            t0 = time.perf_counter()
            move = player.play(board)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            valid += 1

            # Verificar que el movimiento es legal
            r, c = move
            assert 0 <= r < size and 0 <= c < size, f"Movimiento fuera de tablero: {move}"
            assert board.board[r][c] == 0, f"Celda no vacía: {move}"

            if elapsed > 5.0:
                over_limit += 1

        if not times:
            results[phase_name] = {"error": "No se pudo medir (tableros terminales)"}
            continue

        results[phase_name] = {
            "samples":      valid,
            "avg_ms":       round(statistics.mean(times) * 1000, 1),
            "median_ms":    round(statistics.median(times) * 1000, 1),
            "max_ms":       round(max(times) * 1000, 1),
            "min_ms":       round(min(times) * 1000, 1),
            "stdev_ms":     round((statistics.stdev(times) * 1000 if len(times) > 1 else 0), 1),
            "over_5s":      over_limit,
            "pass":         over_limit == 0,
        }

    return results


# ---------------------------------------------------------------------------
# Reporte en consola
# ---------------------------------------------------------------------------

def print_report(all_results: dict) -> None:
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║              BENCHMARK DE TIEMPOS — SmartPlayer (Fase 6)              ║")
    print("╠══════════╦═══════════════╦════════╦════════╦════════╦═════════════════╣")
    print("║  Tamaño  ║  Fase         ║ avg    ║ median ║ max    ║ ¿< 5s siempre?  ║")
    print("╠══════════╬═══════════════╬════════╬════════╬════════╬═════════════════╣")

    for size, phases in all_results.items():
        first = True
        for phase, stats in phases.items():
            if "error" in stats:
                continue
            ok  = "✅  SÍ" if stats["pass"] else f"❌  NO ({stats['over_5s']} fallos)"
            sz  = f"{size}×{size}" if first else ""
            print(f"║  {sz:<8}║  {phase}  ║ {stats['avg_ms']:>5.0f}ms║ "
                  f"{stats['median_ms']:>5.0f}ms║ {stats['max_ms']:>5.0f}ms║  {ok:<15} ║")
            first = False
        print("╠══════════╬═══════════════╬════════╬════════╬════════╬═════════════════╣")

    print("╚══════════╩═══════════════╩════════╩════════╩════════╩═════════════════╝")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de tiempos SmartPlayer")
    parser.add_argument("--sizes",   nargs="+", type=int, default=[5, 7, 9, 11, 13],
                        help="Tamaños de tablero (default: 5 7 9 11 13)")
    parser.add_argument("--samples", type=int, default=8,
                        help="Tableros de muestra por fase (default: 8)")
    parser.add_argument("--output",  default="benchmark_report.json",
                        help="Fichero JSON de salida")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║          SmartPlayer — Benchmark de Tiempos  (Fase 6)                 ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print(f"  Tamaños : {args.sizes}")
    print(f"  Muestras: {args.samples} por fase\n")

    all_results: dict = {}

    for size in args.sizes:
        print(f"  Midiendo {size}×{size}...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = benchmark_size(size, args.samples)
        elapsed = time.perf_counter() - t0
        all_results[size] = res
        print(f"listo ({elapsed:.1f}s)")

    print_report(all_results)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Informe guardado en: {args.output}")


if __name__ == "__main__":
    main()
