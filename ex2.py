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
    order: int = 3                      # magic-square size N
    generations: int = 200              # number of GA iterations
    pop_size: int = 100
    elite_frac: float = 0.20
    mut_rate: float = 0.45
    stagnation_patience: int = 6
    strong_mut_every: int = 50       # generations of stagnation before burst
    strong_mut_every: int = 2           # generations of stagnation before strong mut.
    strong_mut_rate: float = 0.90
    strong_mut_count: int = 35
    mode: str = "classic"          # classic | darwin | lamarck
    perfect: bool = True           # True → most-perfect; False → ordinary
    strong_mut_count: int = 5
    mode: str = "classic"           # classic | darwin | lamarck
    perfect: bool = True            # True → most-perfect; False → ordinary
    plot: bool = True

# ────────────────────────────────────────────────────────────────────────────────
# --------------------------------------------------------------------------- #
#  Deterministic builders for *ordinary* magic squares
# --------------------------------------------------------------------------- #
def deterministic_magic(n: int) -> np.ndarray:
    """Return an *ordinary* magic square of order n (n ≥ 3)."""
    if n % 2 == 1:                                # --- odd -------------
        m = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for k in range(1, n * n + 1):
            m[i, j] = k
            i2, j2 = (i - 1) % n, (j + 1) % n
            if m[i2, j2]:
                i = (i + 1) % n
            else:
                i, j = i2, j2
        return m

    elif n % 4 == 0:                                # --- doubly even -----
        m = np.arange(1, n * n + 1).reshape(n, n)
        mask = np.logical_xor(
            np.indices((n, n))[0] % 4 // 2,
            np.indices((n, n))[1] % 4 // 2,
        )
        m[mask] = n * n + 1 - m[mask]
        return m

    else:                                           # --- singly even -----
        half = n // 2
        sub = deterministic_magic(half)
        m = np.block([[sub, sub + 2 * half**2],
                      [sub + 3 * half**2, sub + half**2]])

        k = (n - 2) // 4
        cols = list(range(k)) + list(range(n - k + 1, n))
        extra = [k]

        m[:, cols] = m[:, cols][[1, 0, 3, 2]]      # swap blocks A↔D, B↔C
        m[:, extra] = m[:, extra][[3, 2, 1, 0]]     # special central col
        return m


# --------------------------------------------------------------------------- #
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
        # Base score: row / col / main diag sums (required for *any* magic square)
        score  = np.abs(board.sum(1) - tgt).sum()
        score += np.abs(board.sum(0) - tgt).sum()
        score += abs(board.trace() - tgt)
        score += abs(np.fliplr(board).trace() - tgt)

        # Extra constraints for *most-perfect* squares
        if self.cfg.perfect:
        # Extra constraints only for *perfect* squares
        if self.cfg.perfect: # THIS IS THE KEY CHECK
            half = n // 2
            r = np.arange(half)[:, None]
            c = np.arange(n)
            # Complementary pairs (horizontal + vertical)
            score += int(np.abs(board[r, c] + board[-1 - r, -1 - c] - (n**2 + 1)).sum())
            score += int(np.abs(board[r, -1 - c] + board[-1 - r,  c] - (n**2 + 1)).sum())
            # 2×2 subsquares
            # Check for property M (complementary pairs summing to n^2 + 1)
            score += np.abs(board[r, c] + board[-1 - r, -1 - c] - (n**2 + 1)).sum()
            score += np.abs(board[r, -1 - c] + board[-1 - r, c] - (n**2 + 1)).sum()

            if n % 2 == 0:
                # Check for 2x2 sub-square sums for doubly-even perfect squares
                s = (board[:-1:2, :-1:2] + board[1::2, :-1:2] +
                     board[:-1:2, 1::2] + board[1::2, 1::2])
                score += int(np.abs(s - tgt).sum())
        return score
                     board[:-1:2, 1::2] + board[1::2, 1::2])
                score += np.abs(s - tgt).sum()

        return int(score)


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
    def _jiggle(self, chrom: list[int],
                 rate: float | None = None,
                 force: bool = False) -> list[int]:
        """Return mutated individual (in-place)."""
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

        # This hill climbing is specifically designed to swap elements
        # within rows/cols to fix sum errors. It might be too aggressive
        # or not general enough for ordinary squares if they don't have
        # certain properties that make these swaps effective.
        # For a general GA, simple random swaps or more complex local searches
        # might be needed, or ensure the GA's own mutation is strong enough.
        # For simplicity, we'll keep it as is, but be aware it's geared towards
        # improving sums by local swaps.
        for _ in range(n): # Iterate n times to try and improve
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

            # Find the two rows/cols with largest errors
            # We take the top two, ensuring we don't pick the same index twice if n=1
            r_indices = row_err.argsort()
            c_indices = col_err.argsort()

            # Attempt swaps between cells in rows/cols with high errors
            # This is a heuristic; more sophisticated hill climbing would explore more neighbors.
            improved_this_iteration = False
            for r1_idx in r_indices[::-1]: # Iterate from highest error rows
                for r2_idx in r_indices[::-1]:
                    if r1_idx == r2_idx: continue
                    for c1_idx in c_indices[::-1]: # Iterate from highest error cols
                        for c2_idx in c_indices[::-1]:
                            if c1_idx == c2_idx: continue

                            test = best.copy()
                            # Swap elements at (r1, c1) and (r2, c2)
                            test[r1_idx, c1_idx], test[r2_idx, c2_idx] = test[r2_idx, c2_idx], test[r1_idx, c1_idx]
                            f = self.evaluate(test.ravel().tolist())
                            if f < best_fit:
                                best, best_fit = test, f
                                improved_this_iteration = True
                                # If an improvement is found, restart search for this iteration
                                # or simply break and let the next iteration try again
                                break
                        if improved_this_iteration: break
                    if improved_this_iteration: break
                if improved_this_iteration: break
            if not improved_this_iteration and _ > 0: # If no improvement in an iteration, can break early
                break

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
        plt.title(f"Generation {gen} | Fitness {self.evaluate(chrom)}") # Evaluate with current cfg
        plt.axis("off")
        plt.pause(0.001)

    # ---------- Evolution loop ------------------------------------------- #
    def run(self) -> list[int]:
        cfg = self.cfg
        patience_ctr = 0
        strong_ctr = 0
        best_fit_hist: List[int] = []

        # Removed the deterministic early exit. The GA will now always run.

        for g in range(cfg.generations):
            print(f"mutation rate: {cfg.mut_rate:.2f} | ")
            # Optional local search
            if cfg.mode in ("darwin", "lamarck"):
                refined_pop = []
            # optional local search
            if cfg.mode in ("darwin", "lamarck"): # Only these modes use hill_climb
                new_pop = []
                for indiv in self.population:
                    refined = self._hill_climb(indiv)
                    refined_pop.append(refined if cfg.mode == "lamarck" else indiv)
                self.population = refined_pop
                    if cfg.mode == "lamarck":
                        new_pop.append(refined)
                    else:
                        new_pop.append(indiv)
                self.population = new_pop

            # Sort by fitness, record best
            self.population.sort(key=self.evaluate)
            best = self.population[0]
            best_fit = self.evaluate(best)
            self._adjust_mut_rate(best_fit)
            best_fit_hist.append(best_fit)
            self._show(best, g)

            if best_fit == 0:
                print(f"✨ Solution found in {g} generations (fitness 0)")
                # This message is now more general, as it applies to both
                # ordinary (if perfect=False) and perfect (if perfect=True) squares.
                square_type = "Most-perfect" if cfg.perfect else "Ordinary"
                print(f"✨ {square_type} square found in {g} generations")
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
        # Define the modes to benchmark, including a specific entry for ordinary GA
        benchmark_modes = [
            ("classic", False), # Ordinary magic square using GA
            ("classic", True),  # Perfect magic square using classic GA
            ("darwin", True),
            ("lamarck", True),
        ]

        for n in ns:
            for mode, perfect_setting in benchmark_modes:
                times, finals = [], []
                cfg_n = cfg | GAConfig(order=n, mode=mode, plot=False, perfect=perfect_setting) # type: ignore
                for _ in range(runs):
                    start = time.perf_counter()
                    best = MagicGA(cfg_n).run()
                    times.append(time.perf_counter() - start)
                    finals.append(MagicGA(cfg_n).evaluate(best)) # Evaluate with the correct perfect setting
                rows.append(dict(N=n,
                                  Mode=f"{mode} (Perfect={perfect_setting})", # Add perfect setting to mode label
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
    p.add_argument("--mode", choices=["classic", "darwin", "lamarck"],
                     default="classic",
                     help="evolution variant")
    p.add_argument("--pop", type=int, default=100, help="population size")
    p.add_argument("--perfect", action="store_true", help="solve most-perfect variant")
    p.add_argument("--no-plot", action="store_true", help="disable live plot")
    p.add_argument("--ordinary", action="store_true",
                     help="Solve for ordinary magic square (disables perfect constraints)")
    args = p.parse_args()
    return GAConfig(order=args.n, generations=args.gens, pop_size=args.pop,
                    mode=args.mode, perfect=args.perfect, plot=not args.no_plot)

    # Determine perfect setting based on --ordinary flag
    perfect_setting = not args.ordinary

    return GAConfig(order=args.n,
                     generations=args.gens,
                     pop_size=args.pop,
                     mode=args.mode,
                     perfect=perfect_setting,
                     plot=not args.no_plot)


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
        "1": ("classic", False), # Ordinary magic square (GA)
        "2": ("classic", True),  # Perfect magic square (classic GA)
        "3": ("darwin",  True),  # Darwinian GA (perfect)
        "4": ("lamarck", True),  # Lamarckian GA (perfect)
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
        print("\n╔════════════════════════════════════════╗")
        print("║  MAGIC-SQUARE  SOLVER                  ║")
        print("╠════════════════════════════════════════╣")
        print("║ 1) Ordinary magic square (using GA)    ║") # Updated label
        print("║ 2) Perfect magic square (classic GA)   ║")
        print("║ 3) Darwinian GA (perfect)              ║")
        print("║ 4) Lamarckian GA (perfect)             ║")
        print("║ 5) Benchmark all modes                 ║")
        print("║ 6) Exit                                ║")
        print("╚════════════════════════════════════════╝")
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
                n = int(input(" Square order N (≥3)          : "))
                gens = int(input(" Generations                    : "))
                pop = int(input(" Population size [100]        : ") or "100")
                plot = input(" Show live plot? [y]/n        : ").lower().strip() != "n"

                if n < 3:
                    print("❌ Order N must be ≥ 3 for magic squares.")
                    continue

                algo, perfect_setting = MODE_LABELS[choice]
                cfg = GAConfig(order=n,
                                generations=gens,
                                pop_size=pop,
                                mode=algo,
                                perfect=perfect_setting,
                                plot=plot)
                start = time.perf_counter()
                result = MagicGA(cfg).run()
                elapsed = time.perf_counter() - start
                print(f"\nFinished in {elapsed:.2f}s – final fitness:",
                      MagicGA(cfg).evaluate(result)) # Evaluate with the correct perfect setting
                print(np.asarray(result).reshape(n, n))
            except ValueError:
                print("❌  Please enter integers only.")
            continue

        # ------ benchmark -------------------------------------------- #
        elif choice == "5":
            try:
                ns = [int(x) for x in input(" N list (e.g. 3,4,5)        : ")
                                         .replace(" ", "")
                                         .split(",") if x]
                gens = int(input(" Generations per run          : "))
                runs = int(input(" Runs per variant [5]         : ") or "5")
                # For benchmarking, we now explicitly create configs for both
                # ordinary and perfect variants for each GA mode.
                # The benchmark function has been updated to reflect this.
                cfg = GAConfig(generations=gens, plot=False) # Base config for benchmark
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