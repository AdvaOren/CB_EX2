#!/usr/bin/env python
"""
Interactive genetic-algorithm solver for ordinary and most-perfect magic squares
with an enhanced benchmark wizard that spotlights the most interesting findings.

🗓️ 2025-05-26 – Completed wizard & interactive menu
    • Benchmark menu: 1-Normal (N=5) 2-Advanced (N=4,8) 3-Both
    • Validates numeric inputs via prompt_int
    • Reports success-rate, avg gens & eval calls to solve, best/avg fitness
    • Displays solved runs or best-fit row per N, sample solution
    • Plots average fitness curves, prints quick insights
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────
@dataclass
class GAConfig:
    order: int = 3
    generations: int = 200
    pop_size: int = 100
    elite_frac: float = 0.2
    mut_rate: float = 0.45
    stagnation_patience: int = 6
    strong_mut_every: int = 50
    strong_mut_rate: float = 0.9
    strong_mut_count: int = 35
    mode: str = "classic"  # classic | darwin | lamarck
    perfect: bool = False
    plot: bool = True


def validate_order(n: int, perfect: bool = False) -> bool:
    return n >= 3 and (n % 4 == 0 if perfect else True)

# ────────────────────────────────────────────────────────────────────────
# GA engine
# ────────────────────────────────────────────────────────────────────────
class MagicGA:
    def __init__(self, cfg: GAConfig):
        if not validate_order(cfg.order, cfg.perfect):
            raise ValueError("Invalid order for selected square-type")
        self.cfg = cfg
        self.target = cfg.order * (cfg.order ** 2 + 1) // 2
        self.population = [self._spawn() for _ in range(cfg.pop_size)]
        self.eval_calls = 0
        self.history: List[int] = []

    def _spawn(self) -> List[int]:
        seq = list(range(1, self.cfg.order ** 2 + 1))
        random.shuffle(seq)
        return seq

    def evaluate(self, chrom: List[int]) -> int:
        self.eval_calls += 1
        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)
        tgt = self.target
        err = int(np.abs(board.sum(1) - tgt).sum() +
                   np.abs(board.sum(0) - tgt).sum() +
                   abs(board.trace() - tgt) +
                   abs(np.fliplr(board).trace() - tgt))
        if self.cfg.perfect:
            half = n // 2
            r = np.arange(half)[:, None]
            c = np.arange(n)
            err += int(np.abs(board[r, c] + board[-1 - r, -1 - c] - (n**2 + 1)).sum())
            err += int(np.abs(board[r, -1 - c] + board[-1 - r,  c] - (n**2 + 1)).sum())
            if n % 2 == 0:
                s = (board[:-1:2, :-1:2] + board[1::2, :-1:2] +
                     board[:-1:2, 1::2] + board[1::2, 1::2])
                err += int(np.abs(s - tgt).sum())
        return err

    def _tournament(self):
        return min(random.sample(self.population, 3), key=self.evaluate)

    def _breed(self, mom: List[int], dad: List[int]) -> List[int]:
        n = len(mom)
        child = [-1] * n
        i, j = sorted(random.sample(range(n), 2))
        child[i:j] = mom[i:j]
        fill = (g for g in dad if g not in child)
        return [x if x != -1 else next(fill) for x in child]

    def _mutate(self, chrom: List[int], p: float | None = None, force: bool = False):
        if not force and random.random() > (p or self.cfg.mut_rate):
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
        else:
            i = random.randint(0, n_sq - self.cfg.order)
            j = i + random.randint(self.cfg.order, n_sq // 2)
            random.shuffle(chrom[i:j])
        return chrom

    def _hill(self, chrom: List[int]):
        n = self.cfg.order
        board = np.asarray(chrom).reshape(n, n)
        best = board.copy()
        best_f = self.evaluate(best.ravel().tolist())
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
            if f < best_f:
                best, best_f = test, f
        return best.ravel().tolist()

    def run(self) -> Tuple[List[int], int | None]:
        cfg = self.cfg
        burst = 0
        for gen in range(cfg.generations):
            if cfg.mode in ("darwin", "lamarck"):
                self.population = [self._hill(ind) if cfg.mode == "lamarck" else ind for ind in self.population]
            self.population.sort(key=self.evaluate)
            best = self.population[0]
            best_fit = self.evaluate(best)
            self.history.append(best_fit)
            if best_fit == 0:
                return best, gen
            if len(self.history) > cfg.stagnation_patience and all(f == best_fit for f in self.history[-cfg.stagnation_patience:]):
                burst += 1
            else:
                burst = 0
            next_gen = self.population[:int(cfg.elite_frac * cfg.pop_size)]
            if burst >= cfg.strong_mut_every:
                for _ in range(cfg.strong_mut_count):
                    m = best.copy()
                    self._mutate(m, p=cfg.strong_mut_rate, force=True)
                    if cfg.mode == "lamarck":
                        m = self._hill(m)
                    next_gen.append(m)
                burst = 0
            while len(next_gen) < cfg.pop_size:
                child = self._breed(self._tournament(), self._tournament())
                self._mutate(child)
                next_gen.append(child)
            self.population = next_gen
        return self.population[0], None

# ────────────────────────────────────────────────────────────────────────
# Experiment runner
# ────────────────────────────────────────────────────────────────────────

def experiment(order: int, mode: str, generations: int, runs: int, perfect: bool = False):
    cfg = GAConfig(order=order, generations=generations, mode=mode, perfect=perfect, plot=False)
    curves, calls, times, solved, solve_gen, boards = [], [], [], [], [], []
    for _ in range(runs):
        ga = MagicGA(cfg)
        t0 = time.perf_counter(); best, gen = ga.run(); elapsed = time.perf_counter() - t0
        curves.append(ga.history)
        calls.append(ga.eval_calls)
        times.append(elapsed)
        solved.append(ga.evaluate(best) == 0)
        solve_gen.append(gen)
        boards.append(np.asarray(best).reshape(order, order))
    return dict(curves=curves, calls=calls, times=times, solved=solved, solve_gen=solve_gen, boards=boards)

# ────────────────────────────────────────────────────────────────────────
# Prompt helper
# ────────────────────────────────────────────────────────────────────────

def prompt_int(msg: str, default: int | None = None, min_val: int = 1) -> int:
    while True:
        raw = input(msg).strip()
        if not raw and default is not None:
            return default
        try:
            val = int(raw)
            if val >= min_val:
                return val
        except ValueError:
            pass
        print(f"❌ Please enter an integer ≥ {min_val}.")

# ────────────────────────────────────────────────────────────────────────
# Benchmark wizard
# ────────────────────────────────────────────────────────────────────────

def benchmark_wizard() -> None:
    print("\n⚙️  Experiment wizard")
    print("1) Normal")
    print("2) Advanced")
    print("3) Normal & Advanced")
    choice = input("Select option (1/2/3) ⇒ ").strip()
    if choice not in ("1", "2", "3"):
        print("❌ Invalid selection.")
        return
    NsNotAdd = False
    Ns = []
    while not NsNotAdd:
        userNs = input("Select order N ⇒ ").strip()
        if userNs in ("0", "1", "2"):
            print("Invalid selection.")
            continue
        if choice == "2" or choice == "3":
            if int(userNs) % 4 != 0:
                print("Perfect squares require N % 4 == 0.")
                NsNotAdd = False
                continue
        Ns.append(int(userNs))
        continueNs = input("Add another order? [Y]/n ⇒ ").strip().lower()
        if continueNs in ("n", "no"):
            NsNotAdd = True
        elif continueNs in ("y", "yes", ""):
            NsNotAdd = False
        else:
            print("❌ Invalid input, exiting.")
            return
    perfect = choice in ("2", "3")
    generations = prompt_int("Generations [200] ⇒ ", default=200)
    runs        = prompt_int("Runs per case [3] ⇒ ", default=3)
    summary_rows: List[Dict] = []
    curve_store: Dict[Tuple[int, str], List[List[int]]] = {}
    for n in Ns:
        if not validate_order(n, perfect=perfect):
            print(f"🚫 Skipping invalid N={n}")
            continue
        for mode in ("classic", "darwin", "lamarck"):
            print(f"› Running N={n}, mode={mode}, ×{runs}")
            res = experiment(n, mode, generations, runs, perfect)
            curve_store[(n, mode)] = res["curves"]
            finals = [c[-1] for c in res["curves"]]
            solved_idx = [i for i, s in enumerate(res["solved"]) if s]
            success_rate = 100 * sum(res["solved"]) / runs
            avg_gen_solve = (sum(res["solve_gen"][i] for i in solved_idx) / len(solved_idx)
                             if solved_idx else None)
            avg_calls_solve = (sum(res["calls"][i] for i in solved_idx) / len(solved_idx)
                               if solved_idx else None)
            avg_calls_all = sum(res["calls"]) / len(res["calls"])
            best_fit = min(finals)
            avg_fit = sum(finals) / len(finals)
            summary_rows.append({
                "N": n, "Mode": mode, "SuccessRate": success_rate,
                "AvgGenSolve": avg_gen_solve, "AvgCallsSolve": avg_calls_solve,
                "AvgCalls": avg_calls_all, "BestFit": best_fit, "AvgFit": avg_fit
            })
            flag = "✔" if success_rate > 0 else "✖"
            print(f"  {mode:<8} | success {success_rate:.0f}% {flag} | best fit {best_fit:.1f}", end="")
            if solved_idx:
                idx = solved_idx[0]
                print(" + sample solution:")
                print(res["boards"][idx])
            else:
                print()
    df = pd.DataFrame(summary_rows)
    # filter interesting
    interesting = df[df.SuccessRate > 0].copy()
    for n in Ns:
        slice_n = df[df.N == n]
        if slice_n.SuccessRate.max() == 0:
            best_row = slice_n.loc[slice_n.BestFit.idxmin()]
            interesting = pd.concat([interesting, best_row.to_frame().T], ignore_index=True)
    interesting = interesting.drop_duplicates().reset_index(drop=True)
    print("\n===== Interesting Results =====")
    print(interesting.to_string(index=False, formatters={
        "SuccessRate": lambda x: f"{x:.0f}%",
        "AvgGenSolve": lambda x: f"{int(x)}" if pd.notna(x) else "–",
        "AvgCallsSolve": lambda x: f"{int(x)}" if pd.notna(x) else "–",
        "AvgCalls": lambda x: f"{int(x)}",
        "BestFit": lambda x: f"{x:.1f}",
        "AvgFit": lambda x: f"{x:.1f}"
    }))
    print("==============================")
    # plots
    for n in Ns:
        plt.figure(figsize=(6, 4))
        for mode in ("classic", "darwin", "lamarck"):
            curves = curve_store.get((n, mode))
            if not curves:
                continue
            L = max(len(c) for c in curves)
            aligned = [c + [c[-1]] * (L - len(c)) for c in curves]
            plt.plot(np.mean(aligned, axis=0), label=mode.capitalize())
        plt.title(f"Fitness vs Generation (N={n})")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (lower=better)")
        plt.grid(alpha=0.3)
        plt.legend()
    plt.show()
    # quick insights
    for n in Ns:
        sub = interesting[interesting.N == n]
        if sub.empty: continue
        best_call = sub.loc[sub.AvgCalls.idxmin()]
        print(f"For N={n}, {best_call.Mode} had lowest avg eval-calls ({int(best_call.AvgCalls)}).")

# ────────────────────────────────────────────────────────────────────────
# Interactive menu
# ────────────────────────────────────────────────────────────────────────
OPTIONS = {
    "1": ("classic", False),
    "2": ("darwin", False),
    "3": ("lamarck", False),
    "4": ("classic", True),
    "5": ("darwin", True),
    "6": ("lamarck", True)
}

def interactive_menu() -> None:
    while True:
        print("""
╔════════════════════════════════════════╗
║              MAGIC-SQUARE              ║
╠════════════════════════════════════════╣
║ 1) Ordinary (Classic)                  ║
║ 2) Ordinary (Darwinian)                ║
║ 3) Ordinary (Lamarckian)               ║
║ 4) Perfect  (Classic)                  ║
║ 5) Perfect  (Darwinian)                ║
║ 6) Perfect  (Lamarckian)               ║
║ 7) Benchmark wizard                    ║
║ 8) Exit                                ║
╚════════════════════════════════════════╝""")
        ch = input("Select option → ").strip()
        if ch in OPTIONS:
            mode, perfect = OPTIONS[ch]
            n = prompt_int("Order N (>=3) ⇒ ", min_val=3)
            if not validate_order(n, perfect):
                print("❌ Invalid order for this square type.")
                continue
            gens = prompt_int("Generations ⇒ ", default=200)
            pop_size = prompt_int("Population size ⇒ ", default=100)
            plot_live = input("Show live plot? [Y]/n ⇒ ").lower().strip() != "n"
            cfg = GAConfig(order=n, generations=gens, pop_size=pop_size,
                           mode=mode, perfect=perfect, plot=plot_live)
            ga = MagicGA(cfg)
            best, gen = ga.run()
            solved_msg = f"Solved in {gen+1} gens" if gen is not None else "No solution found"
            print(f"{solved_msg} | Fitness: {ga.evaluate(best)}")
            print(np.asarray(best).reshape(n, n))
        elif ch == "7":
            benchmark_wizard()
        elif ch == "8":
            print("👋 Goodbye!")
            break
        else:
            print("❓ Unknown option – please try again.")

if __name__ == "__main__":
    interactive_menu()
