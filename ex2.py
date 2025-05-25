#!/usr/bin/env python
# magic_ga.py
"""
Genetic-algorithm solver for classic, Darwinian, or Lamarckian magic squares.

Author: (rewritten for Adva, 2025-05-21)
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Hyper-parameter bundle
# --------------------------------------------------------------------------- #
@dataclass
class GAConfig:
    order: int = 3                    # magic-square size N
    generations: int = 200            # number of GA iterations
    pop_size: int = 100
    elite_frac: float = 0.20
    mut_rate: float = 0.85
    stagnation_patience: int = 6
    strong_mut_every: int = 2         # generations of stagnation before strong mut.
    strong_mut_rate: float = 0.90
    strong_mut_count: int = 5
    mode: str = "classic"          # classic | darwin | lamarck
    perfect: bool = True           # True → most-perfect; False → ordinary
    plot: bool = True

# --------------------------------------------------------------------------- #
#  Validation functions
# --------------------------------------------------------------------------- #
def validate_perfect_square_order(n: int) -> tuple[bool, str]:
    """
    Validate if order n is suitable for perfect magic squares.
    Returns (is_valid, error_message)
    """
    if n < 3:
        return False, "Order must be at least 3"
    
    # Perfect magic squares work best with multiples of 4
    if n % 4 != 0:
        return False, f"Perfect magic squares require order to be a multiple of 4 (got {n})"
    
    return True, ""

# --------------------------------------------------------------------------- #
#  Deterministic builders for *ordinary* magic squares
# --------------------------------------------------------------------------- #
def deterministic_magic(n: int) -> np.ndarray:
    """Return an *ordinary* magic square of order n (n ≥ 3)."""
    if n % 2 == 1:                                  # --- odd -------------
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

        m[:, cols] = m[:, cols][[1, 0, 3, 2]]        # swap blocks A↔D, B↔C
        m[:, extra] = m[:, extra][[3, 2, 1, 0]]       # special central col
        return m


# --------------------------------------------------------------------------- #
#  Main GA implementation
# --------------------------------------------------------------------------- #
class MagicGA:
    def __init__(self, cfg: GAConfig) -> None:
        # Validate configuration for perfect squares
        if cfg.perfect:
            is_valid, error_msg = validate_perfect_square_order(cfg.order)
            if not is_valid:
                raise ValueError(f"Invalid configuration for perfect magic square: {error_msg}")
        
        self.cfg = cfg
        self.target = self.magic_constant(cfg.order)
        self.population = [self._spawn_individual() for _ in range(cfg.pop_size)]
        self._fig = None

    # ---------- Utility ---------------------------------------------------- #
    @staticmethod
    def magic_constant(n: int) -> int:
        """Return the row/col/diag sum of an n×n magic square."""
        return n * (n**2 + 1) // 2

    # ---------- Genetic operations ---------------------------------------- #
    def _spawn_individual(self) -> list[int]:
        seq = list(range(1, self.cfg.order ** 2 + 1))
        random.shuffle(seq)
        return seq

    def evaluate(self, chrom: list[int]) -> int:
        """Lower is better; 0 → valid magic square."""
        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)
        tgt = self.target

        # Basic magic square constraints (for both ordinary and perfect)
        score  = np.abs(board.sum(1) - tgt).sum()  # rows
        score += np.abs(board.sum(0) - tgt).sum()  # columns  
        score += abs(board.trace() - tgt)          # main diagonal
        score += abs(np.fliplr(board).trace() - tgt)  # anti-diagonal

        # Additional constraints only for *perfect* squares
        if self.cfg.perfect:
            # Constraint 1: Opposite corners sum to n²+1
            half = n // 2
            r = np.arange(half)[:, None]
            c = np.arange(n)
            score += np.abs(board[r, c] + board[-1 - r, -1 - c] - (n**2 + 1)).sum()
            score += np.abs(board[r, -1 - c] + board[-1 - r, c] - (n**2 + 1)).sum()

            # Constraint 2: For even n, 2x2 blocks sum to magic constant
            if n % 2 == 0:
                for i in range(0, n, 2):
                    for j in range(0, n, 2):
                        block_sum = (board[i, j] + board[i, j+1] + 
                                   board[i+1, j] + board[i+1, j+1])
                        score += abs(block_sum - tgt)
            
            # Constraint 3: Bent diagonals (for odd n >= 5)
            if n % 2 == 1 and n >= 5:
                # Example bent diagonals - you can add more
                bent_sum1 = sum(board[i, (i + n//2) % n] for i in range(n))
                bent_sum2 = sum(board[i, (i - n//2) % n] for i in range(n))
                score += abs(bent_sum1 - tgt) + abs(bent_sum2 - tgt)
            
            # Constraint 4: Center symmetry for all pairs
            for i in range(n):
                for j in range(n):
                    opposite_i, opposite_j = n - 1 - i, n - 1 - j
                    if i < opposite_i or (i == opposite_i and j < opposite_j):
                        pair_sum = board[i, j] + board[opposite_i, opposite_j]
                        score += abs(pair_sum - (n**2 + 1))

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

    def _jiggle(self, chrom: list[int],
                rate: float | None = None,
                force: bool = False) -> list[int]:
        """Return mutated individual (in-place)."""
        p = self.cfg.mut_rate if rate is None else rate
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

        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)

        # Normalize values between 0 and 1 for colormap
        norm = board.astype(float) / (n * n)

        # Custom brighter colormap (avoiding dark colors)
        from matplotlib.colors import LinearSegmentedColormap
        bright_cmap = LinearSegmentedColormap.from_list("bright", ["#fffae5", "#ffd07a", "#ffad33", "#ff8800"])

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(n, n))
        self._ax.clear()

        # Draw cells with colored background
        for i in range(n):
            for j in range(n):
                color = bright_cmap(norm[i, j])
                self._ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=color, ec="black", lw=1.5))
                self._ax.text(j + 0.5, n - 1 - i + 0.5, str(board[i, j]),
                            va="center", ha="center", fontsize=16, weight="bold", color="black")

        self._ax.set_xlim(0, n)
        self._ax.set_ylim(0, n)
        self._ax.set_aspect("equal")
        self._ax.axis("off")

        self._ax.set_title(f"Generation {gen} | Fitness {self.evaluate(chrom)}", fontsize=14)
        plt.tight_layout()
        plt.pause(0.001)


    # ---------- Evolution loop ------------------------------------------- #
    def run(self) -> list[int]:
        cfg = self.cfg
        patience_ctr = 0
        strong_ctr = 0
        best_fit_hist: List[int] = []
        
        # ---------- Different approaches for ordinary vs perfect squares ---- #
        if not self.cfg.perfect:
            print("🎯 Solving ORDINARY magic square (basic constraints only)...")
            print("   - Rows, columns, and diagonals must sum to magic constant")
        else:
            print("🌟 Solving PERFECT magic square (extensive constraints)...")
            print("   - All ordinary constraints PLUS:")
            print("   - Opposite corners sum to n²+1")
            print("   - 2×2 blocks sum to magic constant (even n)")
            print("   - Bent diagonals sum to magic constant (odd n)")
            print("   - Perfect center symmetry for all pairs")
        
        # ---------- Main GA loop ------------------------------------------- #
        for g in range(cfg.generations):
            # optional local search
            if cfg.mode in ("darwin", "lamarck"):
                new_pop = []
                for indiv in self.population:
                    refined = self._hill_climb(indiv)
                    if cfg.mode == "lamarck":          # inherit improvement
                        new_pop.append(refined)
                    else:                              # Darwin – keep genotype
                        new_pop.append(indiv)
                self.population = new_pop

            # sort by score
            self.population.sort(key=self.evaluate)
            best = self.population[0]
            best_fit = self.evaluate(best)
            best_fit_hist.append(best_fit)
            self._show(best, g)

            if best_fit == 0:
                square_type = "PERFECT" if cfg.perfect else "ORDINARY"
                print(f"🎉 SUCCESS! Valid {square_type} magic square found in {g+1} generations!")
                
                # Verify the solution
                board = np.asarray(best).reshape(cfg.order, cfg.order)
                print(f"\n📊 Solution verification:")
                print(f"   Magic constant: {self.target}")
                print(f"   Row sums: {board.sum(1).tolist()}")
                print(f"   Col sums: {board.sum(0).tolist()}")
                print(f"   Main diagonal: {board.trace()}")
                print(f"   Anti-diagonal: {np.fliplr(board).trace()}")
                
                if cfg.perfect:
                    print(f"   Perfect square constraints also satisfied! ✨")
                
                break

            # stagnation?
            if len(best_fit_hist) > cfg.stagnation_patience and \
               all(b == best_fit for b in best_fit_hist[-cfg.stagnation_patience:]):
                patience_ctr += 1
                strong_ctr += 1
            else:
                patience_ctr = strong_ctr = 0

            # breeding pool
            elite_cut = int(cfg.elite_frac * cfg.pop_size)
            next_gen = self.population[:elite_cut]

            if strong_ctr >= cfg.strong_mut_every:
                for _ in range(cfg.strong_mut_count):
                    mutant = best.copy()
                    self._jiggle(mutant, rate=cfg.strong_mut_rate, force=True)
                    if cfg.mode == "lamarck":
                        mutant = self._hill_climb(mutant)
                    next_gen.append(mutant)
                strong_ctr = 0  # reset

            # standard reproduction
            while len(next_gen) < cfg.pop_size:
                mum = self._tournament()
                dad = self._tournament()
                child = self._breed(mum, dad)
                self._jiggle(child)
                next_gen.append(child)

            self.population = next_gen
        
        else:
            # If we didn't break out of the loop (didn't find perfect solution)
            square_type = "PERFECT" if cfg.perfect else "ORDINARY"
            print(f"\n⏱️  GA completed {cfg.generations} generations without finding perfect {square_type} square.")
            print(f"Best fitness achieved: {best_fit} (lower is better, 0 = perfect)")
            
            if best_fit > 0:
                print(f"💡 Tip: Try increasing generations, population size, or mutation rate")

        # final board
        if cfg.plot:
            plt.show()
        return best

    # ------------------------------------------------------------------- #
    # Static helpers for benchmarking & CSV logs
    # ------------------------------------------------------------------- #
    @staticmethod
    def benchmark(ns: list[int], cfg: GAConfig, runs: int = 5) -> pd.DataFrame:
        rows = []
        for n in ns:
            # Validate n for perfect squares
            if cfg.perfect:
                is_valid, error_msg = validate_perfect_square_order(n)
                if not is_valid:
                    print(f"⚠️  Skipping N={n} for perfect squares: {error_msg}")
                    continue
                    
            for mode in ("classic", "darwin", "lamarck"):
                times, finals = [], []
                cfg_n = GAConfig(
                    order=n, 
                    mode=mode, 
                    plot=False,
                    generations=cfg.generations,
                    pop_size=cfg.pop_size,
                    perfect=cfg.perfect
                )
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


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> GAConfig:
    p = argparse.ArgumentParser(description="Genetic magic-square solver")
    p.add_argument("--n", type=int, default=3, help="square size (≥3)")
    p.add_argument("--gens", type=int, default=200, help="number of generations")
    p.add_argument("--mode", choices=["classic", "darwin", "lamarck"],
                   default="classic",
                   help="evolution variant")
    p.add_argument("--pop", type=int, default=100, help="population size")
    p.add_argument("--no-plot", action="store_true", help="disable live plot")
    p.add_argument("--perfect", action="store_true", help="solve perfect magic square")
    args = p.parse_args()

    # Validate perfect square configuration
    if args.perfect:
        is_valid, error_msg = validate_perfect_square_order(args.n)
        if not is_valid:
            print(f"❌ Error: {error_msg}")
            print("💡 For perfect magic squares, please use multiples of 4 (e.g., 4, 8, 12, 16)")
            exit(1)

    return GAConfig(order=args.n,
                    generations=args.gens,
                    pop_size=args.pop,
                    mode=args.mode,
                    plot=not args.no_plot,
                    perfect=args.perfect)


def main() -> None:
    cfg = parse_args()
    result = MagicGA(cfg).run()
    print("\nBest board:\n", np.asarray(result).reshape(cfg.order, cfg.order))

# --------------------------------------------------------------------------- #
#  Simple text-menu front-end with validation
# --------------------------------------------------------------------------- #
def get_valid_order_input(perfect_mode: bool) -> int:
    """Get valid order input from user with appropriate validation."""
    while True:
        try:
            if perfect_mode:
                n = int(input(" Square order N (multiple of 4): "))
                is_valid, error_msg = validate_perfect_square_order(n)
                if not is_valid:
                    print(f" ❌ {error_msg}")
                    print(" 💡 Please enter a multiple of 4 (e.g., 4, 8, 12)")
                    continue
            else:
                n = int(input(" Square order N (≥3): "))
                if n < 3:
                    print(" ❌ Order must be at least 3")
                    continue
            
            return n
            
        except ValueError:
            print(" ❌ Please enter a valid integer")

def interactive_menu() -> None:
    """
    Console menu that wraps MagicGA.  Keeps asking until the user quits.
    """
    MODE_LABELS = {
        "2": ("classic", True),
        "3": ("darwin",  True),
        "4": ("lamarck", True),
        "1": ("classic", False)   # ordinary magic square
    }

    while True:
        print("\n╔════════════════════════════════════════╗")
        print("║  MAGIC-SQUARE  SOLVER                  ║")
        print("╠════════════════════════════════════════╣")
        print("║ 1) Standard magic square               ║")
        print("║ 2) Perfect magic square                ║")
        print("║ 3) Darwinian GA (perfect)              ║")
        print("║ 4) Lamarckian GA (perfect)             ║")
        print("║ 5) Benchmark all modes                 ║")
        print("║ 6) Exit                                ║")
        print("╚════════════════════════════════════════╝")
        choice = input("Select option → ").strip()

        # --------------- run one GA variant ---------------- #
        if choice in MODE_LABELS:
            try:
                algo, perfect = MODE_LABELS[choice]
                
                # Get valid order with appropriate validation
                n = get_valid_order_input(perfect)
                
                gens = int(input(" Generations                : "))
                pop = int(input(" Population size [100]      : ") or "100")
                plot = input(" Show live plot? [y]/n      : ").lower().strip() != "n"

                cfg = GAConfig(order=n,
                        generations=gens,
                        pop_size=pop,
                        mode=algo,
                        perfect=perfect,
                        plot=plot)
                start = time.perf_counter()
                result = MagicGA(cfg).run()
                elapsed = time.perf_counter() - start
                print(f"\nFinished in {elapsed:.2f}s – final fitness:",
                      MagicGA(cfg).evaluate(result))
                print(np.asarray(result).reshape(n, n))

            except ValueError:
                print("❌  Please enter integers only.")
            continue

        # --------------- benchmark ------------------------- #
        elif choice == "5":
            try:
                perfect = input(" Perfect squares? [y]/n     : ").lower().strip() != "n"
                
                if perfect:
                    print(" 🌟 For perfect squares, only multiples of 4 are supported")
                    print(" 💡 Example: 4,8,12 or just 4")
                
                ns_input = input(" N list (e.g. 3,4,5)       : ").replace(" ", "").split(",")
                ns = [int(x) for x in ns_input if x]
                
                # Validate all N values for perfect squares
                if perfect:
                    valid_ns = []
                    for n in ns:
                        is_valid, error_msg = validate_perfect_square_order(n)
                        if is_valid:
                            valid_ns.append(n)
                        else:
                            print(f" ⚠️  Skipping N={n}: {error_msg}")
                    
                    if not valid_ns:
                        print(" ❌ No valid orders provided for perfect squares")
                        continue
                    ns = valid_ns
                
                gens = int(input(" Generations per run        : "))
                runs = int(input(" Runs per variant [5]       : ") or "5")
                
                cfg = GAConfig(generations=gens, plot=False, perfect=perfect)
                df = MagicGA.benchmark(ns, cfg, runs=runs)
                print("\nBenchmark complete – results saved to "
                      "'ga_magic_benchmark.csv'\n")
                print(df.to_string(index=False))
            except ValueError:
                print("❌  Invalid numeric input.")
            continue

        # --------------- quit ------------------------------ #
        elif choice == "6":
            print("👋  Goodbye!")
            break
        else:
            print("❓  Unknown option – please try again.")

if __name__ == "__main__":
    interactive_menu()