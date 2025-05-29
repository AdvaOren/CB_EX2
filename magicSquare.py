"""
Interactive genetic-algorithm solver for ordinary and most-perfect magic squares
with an enhanced benchmark wizard that spotlights the most interesting findings.
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration
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
    optimization: str = "hill"  # hill | mc | tabu
    tabu_tenure: int = 7
    tabu_iters: int = 50


def validate_order(n: int, perfect: bool = False) -> bool:
    return n >= 3 and (n % 4 == 0 if perfect else True)

# GA engine
class MagicGA:
    def __init__(self, cfg: GAConfig):
        if not validate_order(cfg.order, cfg.perfect):
            raise ValueError("Invalid order for selected square-type")
        self.cfg = cfg
        self.target = cfg.order * (cfg.order ** 2 + 1) // 2
        self.population = [self._spawn() for _ in range(cfg.pop_size)]
        self.eval_calls = 0
        self.history: List[int] = []
        self.avg_history: List[float] = []

        # Initialize plotting variables
        self._fig = None
        self._ax = None
        self._fitness_fig = None 
        self._fitness_ax = None

    def _spawn(self) -> List[int]:
        seq = list(range(1, self.cfg.order ** 2 + 1))
        random.shuffle(seq)
        return seq

    def _adjust_mut_rate(self, best_fit: int) -> None:
        hi = 10
        if best_fit >= hi:
            self.cfg.mut_rate = 0.85
        else:
            self.cfg.mut_rate = 0.95
            
    def evaluate(self, chrom: List[int]) -> int:
        """
        Fitness for a *flat* magic-square chromosome.
        Returns total absolute error for rows, columns, diagonals,
        plus the extra “most-perfect” constraints if cfg.perfect is True.
        """
        self.eval_calls += 1
        n = self.cfg.order
        expected_len = n * n
        if len(chrom) != expected_len:
            raise ValueError(f"chrom length must be {expected_len}, got {len(chrom)}")

        board = np.asarray(chrom, dtype=int).reshape(n, n)
        tgt   = self.target

        # row, column, and diagonal errors
        err  = np.abs(board.sum(axis=1) - tgt).sum()
        err += np.abs(board.sum(axis=0) - tgt).sum()
        err += abs(board.trace()             - tgt)
        err += abs(np.fliplr(board).trace()  - tgt)
        err  = int(err)

        # “most-perfect” extra checks (optional)
        if getattr(self.cfg, "perfect", False):
            half = n // 2
            r = np.arange(half)[:, None]
            c = np.arange(n)

            # complementary pairs on both diagonals
            err += int(
                np.abs(board[r,  c] + board[-1 - r, -1 - c] - (n**2 + 1)).sum()
            + np.abs(board[r, -1 - c] + board[-1 - r,   c] - (n**2 + 1)).sum()
            )

            # 2×2 subsquare sums (only when n is even)
            if n % 2 == 0:
                s = (
                    board[:-1:2, :-1:2] + board[1::2, :-1:2]
                    + board[:-1:2,  1::2] + board[1::2,  1::2]
                )
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
    
    
    def _tabu_search(self, chrom: List[int]) -> List[int]:
        """
        Tabu search local optimization: explore swap neighbors,
        avoid reversing recent moves stored in a Tabu list.
        """
        best = chrom.copy()
        best_fit = self.evaluate(best)
        tabu_list: List[Tuple[int,int]] = []
        tenure = self.cfg.tabu_tenure
        iters = self.cfg.tabu_iters
        n_sq = self.cfg.order ** 2
        for _ in range(iters):
            neighbors: List[Tuple[int,List[int],int,Tuple[int,int]]] = []
            # generate all swap neighbors
            for i in range(n_sq-1):
                for j in range(i+1, n_sq):
                    if (i,j) in tabu_list:
                        continue
                    cand = best.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    f = self.evaluate(cand)
                    neighbors.append((f, cand, i, j))
            if not neighbors:
                break
            # pick best neighbor
            neighbors.sort(key=lambda x: x[0])
            f, cand, i, j = neighbors[0]
            # update Tabu list
            tabu_list.append((i,j))
            if len(tabu_list) > tenure:
                tabu_list.pop(0)
            if f < best_fit:
                best, best_fit = cand, f
        return best

    def _monte_carlo(self, chrom: List[int], iterations: Optional[int] = None) -> List[int]:
        """
        Monte Carlo random‐search on a flat magic‐square encoding.
        - Optionally takes `iterations` (default: self.cfg.mc_iterations or order^3).
        - Each step: pick two random positions, swap them, evaluate.
        - Keep the best solution found over all trials.
        """
        n = self.cfg.order
        total_cells = n * n

        # determine how many random swaps to try
        if iterations is None:
            iterations = getattr(self.cfg, "mc_iterations", total_cells)

        # initialize
        best = chrom.copy()
        best_f = self.evaluate(best)
        curr = best.copy()

        for _ in range(iterations):
            # pick two distinct random indices
            i, j = random.sample(range(total_cells), 2)
            # swap them in the current candidate
            curr[i], curr[j] = curr[j], curr[i]

            # evaluate
            f = self.evaluate(curr)
            if f < best_f:
                best, best_f = curr.copy(), f

            # undo swap (so each trial is independent)
            curr[i], curr[j] = curr[j], curr[i]

        return best

    def _hill(self, chrom: List[int]) -> List[int]:
        """
        Steepest-descent hill-climber on a flat list of length n^2.
        - Scores each cell by row, col, and (if on them) both diagonals.
        - Picks the top-error cells (configurable count) and tries every swap among them.
        - Accepts the best improving swap per round, repeats until converged.
        """
        n = self.cfg.order
        target = self.target
        max_cand = getattr(self.cfg, "hill_candidates", n * 2)

        best = chrom.copy()
        best_f = self.evaluate(best)

        while True:
            # --- compute row and col errors ---
            row_sums = [sum(best[i*n:(i+1)*n]) for i in range(n)]
            col_sums = [sum(best[j::n]) for j in range(n)]
            row_err = [abs(r - target) for r in row_sums]
            col_err = [abs(c - target) for c in col_sums]

            # --- compute diagonal errors ---
            diag_main = sum(best[i*n + i] for i in range(n))
            diag_anti = sum(best[i*n + (n-1-i)] for i in range(n))
            diag_err_main = abs(diag_main - target)
            diag_err_anti = abs(diag_anti - target)

            # --- score each position ---
            cell_err = []
            for pos in range(n*n):
                i, j = divmod(pos, n)
                score = row_err[i] + col_err[j]
                if i == j:
                    score += diag_err_main
                if i + j == n - 1:
                    score += diag_err_anti
                cell_err.append(score)

            # --- pick top-error positions ---
            sorted_idx = np.argsort(cell_err)
            candidates = sorted_idx[-max_cand:]

            # --- search best swap among candidates ---
            improved = False
            best_neighbor = best
            best_neighbor_f = best_f

            for a, b in combinations(candidates, 2):
                if best[a] == best[b]:
                    continue  # swapping identical values won't help
                test = best.copy()
                test[a], test[b] = test[b], test[a]
                f = self.evaluate(test)
                if f < best_neighbor_f:
                    best_neighbor_f = f
                    best_neighbor = test
                    improved = True

            if not improved:
                break

            best, best_f = best_neighbor, best_neighbor_f

        return best

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

        # Create figure with subplots if not exists
        if self._fig is None:
            self._fig, (self._ax, self._fitness_ax) = plt.subplots(1, 2, figsize=(15, 6))
            
        self._ax.clear()

        # Draw magic square (your existing code)
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

        # Plot fitness curves
        self._fitness_ax.clear()
        if len(self.history) > 0:
            self._fitness_ax.plot(range(len(self.history)), self.history, 
                                label='Best Fitness', color='blue', linewidth=2)
        if len(self.avg_history) > 0:
            self._fitness_ax.plot(range(len(self.avg_history)), self.avg_history, 
                                label='Average Fitness', color='red', linewidth=2, alpha=0.7)
        
        self._fitness_ax.set_xlabel('Generation')
        self._fitness_ax.set_ylabel('Fitness (lower is better)')
        self._fitness_ax.set_title('Fitness Evolution')
        self._fitness_ax.legend()
        self._fitness_ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 for better visualization
        if len(self.history) > 0 or len(self.avg_history) > 0:
            max_fit = max(max(self.history) if self.history else 0, 
                        max(self.avg_history) if self.avg_history else 0)
            self._fitness_ax.set_ylim(0, max_fit * 1.1)

        plt.tight_layout()
        plt.pause(0.001)

    def run(self) -> Tuple[List[int], int | None]:
        cfg = self.cfg
        burst = 0
        optim_func = self._hill if cfg.optimization == "hill" else (
            self._monte_carlo if cfg.optimization == "mc" else self._tabu_search)
        
        for gen in range(cfg.generations):
            if cfg.mode in ("darwin", "lamarck"):
                self.population = [optim_func(ind) if cfg.mode == "lamarck" else ind for ind in self.population]
            
            # Calculate fitness for all individuals ONCE
            fitness_scores = [self.evaluate(ind) for ind in self.population]
            
            # Sort population based on pre-calculated fitness
            sorted_pairs = sorted(zip(self.population, fitness_scores), key=lambda x: x[1])
            self.population = [ind for ind, _ in sorted_pairs]
            fitness_scores = [fit for _, fit in sorted_pairs]
            
            best = self.population[0]
            best_fit = fitness_scores[0]
            
            # Calculate average fitness from pre-calculated scores
            avg_fit = sum(fitness_scores) / len(fitness_scores)
            
            self._adjust_mut_rate(best_fit)
            self.history.append(best_fit)
            self.avg_history.append(avg_fit)  # Add this line
            
            if cfg.plot:
                self._show(best, gen)
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
                        m = optim_func(m)
                    next_gen.append(m)
                burst = 0
            while len(next_gen) < cfg.pop_size:
                child = self._breed(self._tournament(), self._tournament())
                self._mutate(child)
                next_gen.append(child)
            self.population = next_gen
        return self.population[0], None

# Experiment runner
def experiment(order: int, mode: str, generations: int, runs: int, perfect: bool = False, optimization: str = "hill") -> Dict[str, List]:
    cfg = GAConfig(order=order, generations=generations, mode=mode, perfect=perfect, plot=False, optimization=optimization)
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

# Prompt helper
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
        print(f"Please enter an integer ≥ {min_val}.")

# Benchmark wizard
def benchmark_wizard() -> None:
    print("\nExperiment wizard")
    print("1) Normal")
    print("2) Advanced")
    print("3) Normal & Advanced")
    choice = input("Select option (1/2/3) ⇒ ").strip()
    if choice not in ("1", "2", "3"):
        print("Invalid selection.")
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
            print("Invalid input, exiting.")
            return
    generations = prompt_int("Generations [200] ⇒ ", default=200)
    runs        = prompt_int("Runs per case [3] ⇒ ", default=3)
    optimization = input("Optimization method (hill/mc/tabu) [hill] ⇒ ").strip().lower()
    summary_rows: List[Dict] = []
    curve_store: Dict[Tuple[int, str], List[List[int]]] = {}
    # decide which variants to run
    perfect_flags = []
    if choice == "1":
        perfect_flags = [False]
    elif choice == "2":
        perfect_flags = [True]
    else:          # "3"  → run BOTH
        perfect_flags = [False, True]
    for n in Ns:
        for perfect in perfect_flags:
            if not validate_order(n, perfect=perfect):
                print(f"Skipping invalid N={n}")
                continue
            perfect_str = "Perfect" if perfect else "Ordinary"
            for mode in ("classic", "darwin", "lamarck"):
                print(f"› Running N={n}, mode={mode}, type: {perfect_str}, ×{runs}")
                res = experiment(n, mode, generations, runs, perfect, optimization)
                key = (n, perfect, mode)
                curve_store[key] = res["curves"]
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
                    "N": n, "Mode": mode, "SuccessRate": success_rate, "Algorithm": optimization.upper(),
                    "Perfect": perfect,
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
    print("\n====== Complete Results ======")
    print(df.to_string(index=False, formatters={
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
        for perfect in (False, True):
            curves_exist = any((n, perfect, m) in curve_store for m in ("classic", "darwin", "lamarck"))
            if not curves_exist:
                continue    
            plt.figure(figsize=(6, 4))
            for mode in ("classic", "darwin", "lamarck"):
                curves = curve_store.get((n, perfect, mode))
                if not curves:
                    print(f"No curves for N={n}, mode={mode}, perfect={perfect}.")
                    continue
                L = max(len(c) for c in curves)
                aligned = [c + [c[-1]] * (L - len(c)) for c in curves]
                plt.plot(np.mean(aligned, axis=0), label=mode.capitalize())
            sq_type = "Perfect" if perfect else "Ordinary"
            plt.title(f"Fitness vs Generation • N={n} • {sq_type}")
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness (lower is better)")
            plt.grid(alpha=0.3)
            plt.legend()
    plt.show()

    # quick insights
    for n in Ns:
        sub = interesting[interesting.N == n]
        if sub.empty: continue
        best_call = sub.loc[sub.AvgCalls.idxmin()]
        print(f"For N={n}, {best_call.Mode} had lowest avg eval-calls ({int(best_call.AvgCalls)}).")

# Interactive menu
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
                print("Invalid order for this square type.")
                continue
            gens = prompt_int("Generations ⇒ ", default=200)
            pop_size = prompt_int("Population size ⇒ ", default=100)
            optimization = input("Optimization method (hill/mc/tabu) [hill] ⇒ ").strip().lower()
            plot_live = input("Show live plot? [Y]/n ⇒ ").lower().strip() != "n"
            cfg = GAConfig(order=n, generations=gens, pop_size=pop_size,
                           mode=mode, perfect=perfect, plot=plot_live, optimization=optimization)
            ga = MagicGA(cfg)
            best, gen = ga.run()
            solved_msg = f"Solved in {gen+1} gens" if gen is not None else "No solution found"
            print(f"{solved_msg} | Fitness: {ga.evaluate(best)}")
            print(np.asarray(best).reshape(n, n))
        elif ch == "7":
            benchmark_wizard()
        elif ch == "8":
            print("Goodbye!")
            break
        else:
            print("Unknown option – please try again.")

if __name__ == "__main__":
    interactive_menu()