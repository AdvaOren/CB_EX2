#!/usr/bin/env python
"""
Genetic-algorithm solver for *ordinary* and *most-perfect* magic squares.

✱ 2025-05-25 — **all modes use random initial populations**.
   Any deterministic shortcut has been removed.

Running options (see `interactive_menu`):
  1) Ordinary magic square  – GA search               (perfect=False)
  2) Most-perfect square    – Classic GA              (mode="classic")
  3) Most-perfect square    – Darwinian GA            (mode="darwin")
  4) Most-perfect square    – Lamarckian GA           (mode="lamarck")
  5) Benchmark all modes

Author: rewritten for Adva, 2025-05-25
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from itertools import islice
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
#  Hyper-parameter bundle
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class GAConfig:
    order: int = 3                    # magic-square size N
    generations: int = 200            # maximal GA iterations
    pop_size: int = 100
    elite_frac: float = 0.20
    mut_rate: float = 0.45
    stagnation_patience: int = 6
    strong_mut_every: int = 50       # generations of stagnation before burst
    strong_mut_rate: float = 0.90
    strong_mut_count: int = 35
    mode: str = "classic"          # classic | darwin | lamarck
    perfect: bool = True           # True → most-perfect; False → ordinary
    plot: bool = True

# ────────────────────────────────────────────────────────────────────────────────
#  Main GA implementation
# ────────────────────────────────────────────────────────────────────────────────
class MagicGA:
    def __init__(self, cfg: GAConfig) -> None:
        self.cfg = cfg
        self.target = self.magic_constant(cfg.order)
        # Fully random initial population for *all* tasks
        self.population = [self._spawn_individual() for _ in range(cfg.pop_size)]
        self._fig = None

    # ---------- Utility ---------------------------------------------------- #
    @staticmethod
    def magic_constant(n: int) -> int:
        return n * (n**2 + 1) // 2

    # ---------- Genetic operations ---------------------------------------- #
    def _spawn_individual(self) -> list[int]:
        seq = list(range(1, self.cfg.order ** 2 + 1))
        random.shuffle(seq)
        return seq
    
    def _adjust_mut_rate(self, best_fit: int) -> None:
        hi, lo = 10, 2
        if best_fit >= hi:
            self.cfg.mut_rate = 0.85
        else:
            frac = max(0, (best_fit - lo) / (hi - lo))
            self.cfg.mut_rate = 0.95

    def evaluate(self, chrom: list[int]) -> int:
        """Lower is better; 0 → valid magic square (ordinary or most-perfect)."""
        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)
        tgt = self.target

        # Row / column / diagonal sums
        score = int(np.abs(board.sum(1) - tgt).sum() +
                     np.abs(board.sum(0) - tgt).sum() +
                     abs(board.trace() - tgt) +
                     abs(np.fliplr(board).trace() - tgt))

        # Extra constraints for *most-perfect* squares
        if self.cfg.perfect:
            half = n // 2
            r = np.arange(half)[:, None]
            c = np.arange(n)
            # Complementary pairs (horizontal + vertical)
            score += int(np.abs(board[r, c] + board[-1 - r, -1 - c] - (n**2 + 1)).sum())
            score += int(np.abs(board[r, -1 - c] + board[-1 - r,  c] - (n**2 + 1)).sum())
            # 2×2 subsquares
            if n % 2 == 0:
                s = (board[:-1:2, :-1:2] + board[1::2, :-1:2] +
                     board[:-1:2, 1::2] + board[1::2, 1::2])
                score += int(np.abs(s - tgt).sum())
        return score

    def _tournament(self, k: int = 3) -> list[int]:
        competitors = random.sample(self.population, k)
        competitors.sort(key=self.evaluate)
        return competitors[0]

    def _breed(self, mum: list[int], dad: list[int]) -> list[int]:
        n = self.cfg.order ** 2
        child = [-1] * n
        a, b = sorted(random.sample(range(n), 2))
        child[a:b] = mum[a:b]
        fill = (g for g in dad if g not in child)
        child = [g if g != -1 else next(fill) for g in child]
        return child

    def _jiggle(self, chrom: list[int], *, rate: float | None = None,
                force: bool = False) -> list[int]:
        p = self.cfg.mut_rate if rate is None else rate
        print(f"with p={p:.2f} ")
        if random.random() > p and not force:
            return chrom

        n_sq = self.cfg.order ** 2
        action = random.choice(["swap", "reverse", "rotate", "shuffle"])
        if action == "swap":
            i, j = random.sample(range(n_sq), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
        elif action == "reverse":
            i, j = sorted(random.sample(range(n_sq), 2))
            chrom[i:j] = reversed(chrom[i:j])
        elif action == "rotate":
            i, j = sorted(random.sample(range(n_sq), 2))
            k = random.randint(1, self.cfg.order // 2)
            seg = chrom[i:j]
            chrom[i:j] = seg[-k:] + seg[:-k]
        else:  # shuffle
            i = random.randint(0, n_sq - self.cfg.order)
            j = i + random.randint(self.cfg.order, n_sq // 2)
            random.shuffle(chrom[i:j])
        return chrom

    # ---------- Optional local search for Darwin/Lamarck ------------------ #
    def _hill_climb(self, chrom: list[int]) -> list[int]:
        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)
        best = board.copy()
        best_fit = self.evaluate(best.ravel().tolist())
        for _ in range(n):
            row_err = np.abs(best.sum(1) - self.target)
            col_err = np.abs(best.sum(0) - self.target)
            r1, r2 = row_err.argsort()[-2:]
            c1, c2 = col_err.argsort()[-2:]
            if (r1, c1) == (r2, c2):
                continue
            test = best.copy()
            test[r1, c1], test[r2, c2] = test[r2, c2], test[r1, c1]
            f = self.evaluate(test.ravel().tolist())
            if f < best_fit:
                best, best_fit = test, f
        return best.ravel().tolist()

    # ---------- Visual ---------------------------------------------------- #
    def _show(self, chrom: list[int], gen: int) -> None:
        if not self.cfg.plot:
            return
        if self._fig is None:
            self._fig = plt.figure(figsize=(6, 6))
        plt.clf()
        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)
        norm = board.astype(float) / (n * n)
        colours = plt.cm.YlGnBu(norm)
        tbl = plt.table(cellText=board, cellColours=colours, cellLoc="center",
                        loc="center", colWidths=[0.12] * n)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        plt.title(f"Generation {gen} | Fitness {self.evaluate(chrom)}")
        plt.axis("off")
        plt.pause(0.001)

    # ---------- Evolution loop ------------------------------------------- #
    def run(self) -> list[int]:
        cfg = self.cfg
        patience_ctr = 0
        strong_ctr = 0
        best_fit_hist: List[int] = []

        for g in range(cfg.generations):
            print(f"mutation rate: {cfg.mut_rate:.2f} | ")
            # Optional local search
            if cfg.mode in ("darwin", "lamarck"):
                refined_pop = []
                for indiv in self.population:
                    refined = self._hill_climb(indiv)
                    refined_pop.append(refined if cfg.mode == "lamarck" else indiv)
                self.population = refined_pop

            # Sort by fitness, record best
            self.population.sort(key=self.evaluate)
            best = self.population[0]
            best_fit = self.evaluate(best)
            self._adjust_mut_rate(best_fit)
            best_fit_hist.append(best_fit)
            self._show(best, g)

            if best_fit == 0:
                print(f"✨ Solution found in {g} generations (fitness 0)")
                break

            # Stagnation detection
            if len(best_fit_hist) > cfg.stagnation_patience and \
               all(b == best_fit for b in best_fit_hist[-cfg.stagnation_patience:]):
                patience_ctr += 1
                strong_ctr += 1
            else:
                patience_ctr = strong_ctr = 0

            # Elitism copy
            elite_cut = int(cfg.elite_frac * cfg.pop_size)
            next_gen = self.population[:elite_cut]

            # Strong mutation burst
            if strong_ctr >= cfg.strong_mut_every:
                for _ in range(cfg.strong_mut_count):
                    mutant = best.copy()
                    self._jiggle(mutant, rate=cfg.strong_mut_rate, force=True)
                    if cfg.mode == "lamarck":
                        mutant = self._hill_climb(mutant)
                    next_gen.append(mutant)
                strong_ctr = 0

            # Standard reproduction
            while len(next_gen) < cfg.pop_size:
                mum = self._tournament()
                dad = self._tournament()
                child = self._breed(mum, dad)
                self._jiggle(child)
                next_gen.append(child)

            self.population = next_gen

        if cfg.plot:
            plt.show()
        return best

    # ------------------------------------------------------------------- #
    # Static helpers: benchmarking & CSV logs
    # ------------------------------------------------------------------- #
    @staticmethod
    def benchmark(ns: list[int], cfg: GAConfig, runs: int = 5) -> pd.DataFrame:
        rows = []
        for n in ns:
            for mode in ("classic", "darwin", "lamarck"):
                times, finals = [], []
                cfg_n = cfg | GAConfig(order=n, mode=mode, plot=False)  # type: ignore
                for _ in range(runs):
                    start = time.perf_counter()
                    best = MagicGA(cfg_n).run()
                    times.append(time.perf_counter() - start)
                    finals.append(MagicGA(cfg_n).evaluate(best))
                rows.append(dict(N=n,
                                 Mode=mode,
                                 AvgTime=np.mean(times),
                                 StdTime=np.std(times),
                                 AvgFitness=np.mean(finals),
                                 StdFitness=np.std(finals)))
        df = pd.DataFrame(rows)
        df.to_csv("ga_magic_benchmark.csv", index=False)
        return df

# ────────────────────────────────────────────────────────────────────────────────
#  CLI wrapper
# ────────────────────────────────────────────────────────────────────────────────
def parse_args() -> GAConfig:
    p = argparse.ArgumentParser(description="Genetic magic-square solver (random init)")
    p.add_argument("--n", type=int, default=3, help="square size (≥3)")
    p.add_argument("--gens", type=int, default=200, help="number of generations")
    p.add_argument("--mode", choices=["classic", "darwin", "lamarck"], default="classic")
    p.add_argument("--pop", type=int, default=100, help="population size")
    p.add_argument("--perfect", action="store_true", help="solve most-perfect variant")
    p.add_argument("--no-plot", action="store_true", help="disable live plot")
    args = p.parse_args()
    return GAConfig(order=args.n, generations=args.gens, pop_size=args.pop,
                    mode=args.mode, perfect=args.perfect, plot=not args.no_plot)


def main() -> None:
    cfg = parse_args()
    result = MagicGA(cfg).run()
    print("\nBest board (fitness =", MagicGA(cfg).evaluate(result), "):")
    print(np.asarray(result).reshape(cfg.order, cfg.order))

# ────────────────────────────────────────────────────────────────────────────────
#  Interactive text-menu front-end
# ────────────────────────────────────────────────────────────────────────────────
def interactive_menu() -> None:
    MODE_LABELS = {
        "1": ("classic", False),   # ordinary magic square (GA)
        "2": ("classic", True),    # most-perfect, classic GA
        "3": ("darwin",  True),    # most-perfect, Darwinian
        "4": ("lamarck", True),    # most-perfect, Lamarckian
    }
    while True:
        print("""
╔════════════════════════════════════════╗
║           MAGIC-SQUARE  SOLVER         ║
╠════════════════════════════════════════╣
║ 1) Ordinary magic square  (GA)         ║
║ 2) Perfect magic square  (classic)     ║
║ 3) Perfect magic square  (darwinian)   ║
║ 4) Perfect magic square  (lamarckian)  ║
║ 5) Benchmark all modes                 ║
║ 6) Exit                                ║
╚════════════════════════════════════════╝""")
        choice = input("Select option → ").strip()

        # ------ run a GA variant ------------------------------------- #
        if choice in MODE_LABELS:
            try:
                n = int(input(" Square order N (≥3)        : "))
                gens = int(input(" Generations                : "))
                pop = int(input(" Population size [100]      : ") or "100")
                plot = input(" Show live plot? [y]/n      : ").lower().strip() != "n"
                algo, perfect = MODE_LABELS[choice]
                cfg = GAConfig(order=n, generations=gens, pop_size=pop,
                                mode=algo, perfect=perfect, plot=plot)
                start = time.perf_counter()
                result = MagicGA(cfg).run()
                elapsed = time.perf_counter() - start
                print(f"\nFinished in {elapsed:.2f}s – final fitness:",
                      MagicGA(cfg).evaluate(result))
                print(np.asarray(result).reshape(n, n))
            except ValueError:
                print("❌  Please enter integers only.")
            continue

        # ------ benchmark -------------------------------------------- #
        elif choice == "5":
            try:
                ns = [int(x) for x in input(" N list (e.g. 3,4,5)       : ")
                                         .replace(" ", "")
                                         .split(",") if x]
                gens = int(input(" Generations per run        : "))
                runs = int(input(" Runs per variant [5]       : ") or "5")
                cfg = GAConfig(generations=gens, plot=False)
                df = MagicGA.benchmark(ns, cfg, runs=runs)
                print("\nBenchmark complete – results saved to 'ga_magic_benchmark.csv'\n")
                print(df.to_string(index=False))
            except ValueError:
                print("❌  Invalid numeric input.")
            continue

        # ------ quit -------------------------------------------------- #
        elif choice == "6":
            print("👋  Goodbye!")
            break
        else:
            print("❓  Unknown option – please try again.")

if __name__ == "__main__":
    interactive_menu()
