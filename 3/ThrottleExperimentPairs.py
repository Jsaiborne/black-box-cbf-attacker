import random
import time
import argparse
import statistics
import math
import matplotlib
matplotlib.use('Agg')  # Safe for servers/headless
import matplotlib.pyplot as plt

# =================================================
# ============== USER-EDITABLE CONSTANTS ==========
# Edit these for quick experiments (CLI will override)
# =================================================
# Filter / universe / workload
M = 5000               # filter size (number of counters)
K = None               # number of hash functions; if None, compute optimal k from M/N
N = 500                # number of true elements initially inserted
U_SIZE = 50000          # size of universe (scannable range)
RATE_LIMITS = [10000, 5000, 2000, 1000, 500, 200, 100]  # list of rate limits to test
CHURN_PER_CYCLE = 100    # number of elements changed per churn event

# Attack / algorithm choices
ENABLE_PAIRS = True    # set to False to disable pair-testing (edit in-code)
PAIR_BUDGET = 500      # max pair tests per experiment (edit in-code)
CHECK_BUDGET = None    # None => full-scan P; int => sample this many y in P when building Prem

# Trials / reproducibility / churn model
TRIALS = 5            # default number of independent trials per rate-limit
SEED_BASE = None       # if int, base seed; per-trial seed used = SEED_BASE + trial_index
CHURN_MODEL = 'query'  # 'query' (churn triggered by hitting query_limit) or 'time' (periodic churn)
TIME_INTERVAL = None   # when churn_model='time', churn every TIME_INTERVAL accesses (None -> use query_limit)

# Misc
VERBOSE = False        # set True to get more prints (per-trial messages)
OUTPUT_FILENAME = None # if None a filename will be auto-generated

# =================================================
# End constants
# =================================================


# Import your existing class
try:
    from CountingBloomFilter import CountingBloomFilter
except ImportError:
    print("Error: CountingBloomFilter.py not found. Please ensure the file is in the same directory.")
    exit()


# ==========================================
# 1. THE THROTTLED FILTER WRAPPER
# ==========================================
class ThrottledOracle:
    """
    Wraps the CBF to simulate an API that:
      - Counts queries (Rate Limiting)
      - Changes data automatically (Data Churn)

    churn_model:
      - 'query' : churn fires when query_limit is reached (original behavior)
      - 'time'  : churn fires every time_between_churns accesses regardless of query limit
    """
    def __init__(self, cbf, true_set, universe, query_limit, churn_amount,
                 churn_model='query', time_between_churns=None):
        self.cbf = cbf
        self.true_set = true_set
        self.universe = universe
        self.query_limit = query_limit
        self.churn_amount = churn_amount
        self.churn_model = churn_model
        self.time_between_churns = time_between_churns if time_between_churns is not None else query_limit

        self.queries_made = 0
        self.time_counter = 0
        self.total_days_passed = 0
        self.churn_events = 0

    def access(self, operation, item):
        # update counters
        self.queries_made += 1
        self.time_counter += 1

        # query-driven churn
        if self.churn_model == 'query':
            if self.queries_made >= self.query_limit:
                self._apply_data_churn()
                self.queries_made = 0
                self.total_days_passed += 1

        # time-driven churn
        elif self.churn_model == 'time':
            if self.time_counter >= self.time_between_churns:
                self._apply_data_churn()
                self.time_counter = 0
                self.total_days_passed += 1

        # actual CBF operation
        if operation == 'check':
            return self.cbf.check(item)
        elif operation == 'add':
            self.cbf.add(item)
        elif operation == 'remove':
            self.cbf.remove(item)

    def _apply_data_churn(self):
        self.churn_events += 1
        for _ in range(self.churn_amount):
            if not self.true_set:
                break
            idx_to_remove = random.randrange(len(self.true_set))
            old_item = self.true_set.pop(idx_to_remove)
            self.cbf.remove(old_item)

            # add a new item (rejection sample)
            new_item = random.randint(1, len(self.universe))
            while new_item in self.true_set:
                new_item = random.randint(1, len(self.universe))
            self.true_set.append(new_item)
            self.cbf.add(new_item)


# ==========================================
# 2. THE BLACK-BOX ATTACKER (WITH PAIR TOGGLE)
#    Accepts check_budget to control Prem-building
# ==========================================
def run_black_box_attack(oracle, universe, enable_pairs=False, pair_budget=1000, check_budget=None, verbose=False):
    """
    - check_budget: None => full scan of P when building Prem; otherwise random sample of up to check_budget elements.
    """
    # PHASE 1: DISCOVERY
    if verbose:
        print("Scanning universe to compute P (positives)...")
    P = []
    for x in universe:
        if oracle.access('check', x):
            P.append(x)
    if verbose:
        print(f"Initial positives found: {len(P)}")

    Srec = []

    # helper to build Prem using check_budget
    def build_Prem_excluding(exclude_set):
        """
        exclude_set: set of elements to exclude from checks (e.g., {x} or {x,y})
        Returns: Prem set (subset of P that remain positive)
        """
        candidates = [y for y in P if y not in exclude_set]
        # full scan
        if check_budget is None:
            subset = candidates
        else:
            if len(candidates) > check_budget:
                subset = random.sample(candidates, check_budget)
            else:
                subset = candidates

        Prem_local = set()
        for y in subset:
            if oracle.access('check', y):
                Prem_local.add(y)
        return Prem_local, subset

    # Single-element test
    def test_element(x, P):
        # remove x
        oracle.access('remove', x)
        # check x
        x_still_positive = oracle.access('check', x)
        if x_still_positive:
            oracle.access('add', x)
            return False

        Prem, _ = build_Prem_excluding({x})
        R = [y for y in P if y != x and y not in Prem]

        # Case A: only x disappeared
        if set(P) == Prem.union({x}):
            if x in P:
                P.remove(x)
            Srec.append(x)
            if verbose:
                print(f"test_element: confirmed singleton {x}")
            return True

        # Case B: some R disappeared -> insert R & re-check
        if R:
            for z in R:
                oracle.access('add', z)
            x_pos_after_inserting_R = oracle.access('check', x)

            if not x_pos_after_inserting_R:
                # x confirmed; R are false positives
                for z in R:
                    if z in P:
                        P.remove(z)
                    oracle.access('remove', z)
                if x in P:
                    P.remove(x)
                Srec.append(x)
                if verbose:
                    print(f"test_element: confirmed {x} after inserting R (|R|={len(R)})")
                return True
            else:
                for z in R:
                    oracle.access('remove', z)

        # restore x
        oracle.access('add', x)
        return False

    # Pair test (heuristic)
    def test_pair(x, y, P):
        oracle.access('remove', x)
        oracle.access('remove', y)

        x_pos = oracle.access('check', x)
        y_pos = oracle.access('check', y)
        if x_pos and y_pos:
            oracle.access('add', x)
            oracle.access('add', y)
            return False

        Prem, _ = build_Prem_excluding({x, y})
        R = [z for z in P if z not in Prem and z not in (x, y)]

        # Case A: both disappeared and no others
        if set(P) == Prem.union({x, y}):
            if x in P: P.remove(x)
            if y in P: P.remove(y)
            Srec.extend([x, y])
            if verbose:
                print(f"test_pair: confirmed pair ({x},{y})")
            return True

        if R:
            for z in R:
                oracle.access('add', z)
            x_pos_after_R = oracle.access('check', x)
            y_pos_after_R = oracle.access('check', y)

            found_any = False
            if not x_pos_after_R and y_pos_after_R:
                if x in P: P.remove(x)
                Srec.append(x)
                found_any = True
            elif not y_pos_after_R and x_pos_after_R:
                if y in P: P.remove(y)
                Srec.append(y)
                found_any = True
            elif not x_pos_after_R and not y_pos_after_R:
                if x in P: P.remove(x)
                if y in P: P.remove(y)
                Srec.extend([x, y])
                found_any = True

            for z in R:
                if z in P and found_any:
                    P.remove(z)
                oracle.access('remove', z)

            if found_any:
                if verbose:
                    print(f"test_pair: confirmed from pair ({x},{y}), found_any={found_any}, |R|={len(R)}")
                return True
            else:
                oracle.access('add', x)
                oracle.access('add', y)
                return False

        oracle.access('add', x)
        oracle.access('add', y)
        return False

    # Peeling loop
    run_process = True
    while run_process:
        run_process = False

        # single-element peeling
        run_test = True
        while run_test:
            run_test = False
            snapshot_P = list(P)
            for x in snapshot_P:
                if x not in P:
                    continue
                if test_element(x, P):
                    run_test = True
                    run_process = True

        # optional pair-testing
        if enable_pairs and len(P) >= 2:
            pair_checks_done = 0
            candidates = list(P)
            random.shuffle(candidates)
            n = len(candidates)
            for i in range(n):
                for j in range(i + 1, n):
                    if pair_checks_done >= pair_budget:
                        break
                    x = candidates[i]; y = candidates[j]
                    if x not in P or y not in P:
                        continue
                    pair_checks_done += 1
                    if test_pair(x, y, P):
                        run_process = True
                    if len(P) < 2:
                        break
                if pair_checks_done >= pair_budget or len(P) < 2:
                    break

    if verbose:
        print(f"Recovered {len(Srec)} elements via peeling (pairs={enable_pairs}).")
    return Srec


# ==========================================
# 3. EXPERIMENT EXECUTION (MULTI-TRIALS)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="CBF black-box peeling attack simulation.")
    parser.add_argument("--pairs", action="store_true", help="Enable pair-testing (Algorithm 2). Overrides ENABLE_PAIRS.")
    parser.add_argument("--pair-budget", type=int, default=None, help="Max pair tests per experiment (overrides PAIR_BUDGET).")
    parser.add_argument("--trials", type=int, default=None, help="Number of independent trials per rate-limit (overrides TRIALS).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed base. If provided, trial seed = seed + trial_index.")
    parser.add_argument("--churn-model", choices=['query', 'time'], default=None, help="Churn model override.")
    parser.add_argument("--time-interval", type=int, default=None, help="Used with --churn-model time: accesses between churn events.")
    parser.add_argument("--k", type=int, default=None, help="Override K (number of hash functions). If omitted and K constant is None, optimal k is computed.")
    parser.add_argument("--check-budget", type=int, default=None, help="Override CHECK_BUDGET (how many y in P to check when building Prem).")
    parser.add_argument("--verbose", action="store_true", help="Verbose output (overrides VERBOSE).")
    parser.add_argument("--output", type=str, default=None, help="Output filename for plot (overrides OUTPUT_FILENAME).")

    args = parser.parse_args()

    # Resolve settings: constants first, override with CLI where provided
    M_local = M
    K_local = K
    if args.k is not None:
        K_local = args.k
    N_local = N
    U_size_local = U_SIZE
    RATE_LIMITS_local = RATE_LIMITS.copy()
    CHURN_PER_CYCLE_local = CHURN_PER_CYCLE

    trials_local = args.trials if args.trials is not None else TRIALS
    enable_pairs_local = args.pairs if args.pairs else ENABLE_PAIRS
    pair_budget_local = args.pair_budget if args.pair_budget is not None else PAIR_BUDGET
    check_budget_local = args.check_budget if args.check_budget is not None else CHECK_BUDGET
    seed_base_local = args.seed if args.seed is not None else SEED_BASE
    churn_model_local = args.churn_model if args.churn_model is not None else CHURN_MODEL
    time_interval_local = args.time_interval if args.time_interval is not None else TIME_INTERVAL
    verbose_local = args.verbose if args.verbose else VERBOSE
    output_filename_local = args.output if args.output is not None else OUTPUT_FILENAME

    # compute K if needed
    if K_local is None:
        if N_local <= 0:
            K_local = 1
        else:
            k_float = (M_local / N_local) * math.log(2)
            K_local = max(1, int(round(k_float)))
            if verbose_local:
                print(f"Computed optimal k from formula: k_float={k_float:.3f} -> k_opt={K_local}")

    # Universe
    universe = list(range(1, U_size_local + 1))

    print(f"--- Starting Simulation [N={N_local}, M={M_local}, K={K_local}, trials={trials_local}, churn_model={churn_model_local}] ---")
    if seed_base_local is not None:
        print(f"Using seed base = {seed_base_local} (per-trial seeds = seed_base + trial_index).")
    if check_budget_local is None:
        print("CHECK_BUDGET = None (full scan of P when building Prem).")
    else:
        print(f"CHECK_BUDGET = {check_budget_local} (sample up to this many y from P when building Prem).")

    print(f"{'Rate Limit':<12} | {'Recall (mean±std)':<22} | {'Precision (mean±std)':<24} | {'Days Passed (mean)':<16}")
    print("-" * 90)

    mean_recalls = []
    std_recalls = []
    mean_precisions = []
    std_precisions = []
    mean_days = []

    for limit in RATE_LIMITS_local:
        recalls = []
        precisions = []
        days = []

        for t in range(trials_local):
            # per-trial seed for reproducibility (if provided)
            if seed_base_local is not None:
                trial_seed = seed_base_local + t
                random.seed(trial_seed)
            else:
                # if no seed specified leave RNG as-is (non-deterministic)
                pass

            # Setup CBF and server true set
            cbf = CountingBloomFilter(M_local, K_local)
            true_set = random.sample(universe, N_local)
            for x in true_set:
                cbf.add(x)

            # Setup oracle
            oracle = ThrottledOracle(cbf, true_set, universe,
                                     query_limit=limit,
                                     churn_amount=CHURN_PER_CYCLE_local,
                                     churn_model=churn_model_local,
                                     time_between_churns=(time_interval_local if time_interval_local is not None else limit))

            # Run attack
            extracted_elements = run_black_box_attack(oracle, universe,
                                                      enable_pairs=enable_pairs_local,
                                                      pair_budget=pair_budget_local,
                                                      check_budget=check_budget_local,
                                                      verbose=verbose_local)

            # Evaluate
            current_server_set = set(oracle.true_set)
            found_set = set(extracted_elements)
            true_positives_found = len(found_set.intersection(current_server_set))

            recall = (true_positives_found / N_local) * 100 if N_local > 0 else 0
            precision = (true_positives_found / len(found_set)) * 100 if len(found_set) > 0 else 0

            recalls.append(recall)
            precisions.append(precision)
            days.append(oracle.total_days_passed)

            if verbose_local:
                print(f"trial {t+1}/{trials_local}, limit={limit}: recall={recall:.1f}%, precision={precision:.1f}%, days={oracle.total_days_passed}")

        # aggregate
        mean_rec = statistics.mean(recalls)
        std_rec = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        mean_prec = statistics.mean(precisions)
        std_prec = statistics.stdev(precisions) if len(precisions) > 1 else 0.0
        mean_day = statistics.mean(days)

        mean_recalls.append(mean_rec)
        std_recalls.append(std_rec)
        mean_precisions.append(mean_prec)
        std_precisions.append(std_prec)
        mean_days.append(mean_day)

        print(f"{limit:<12} | {mean_rec:6.1f}% ±{std_rec:5.2f}     | {mean_prec:6.1f}% ±{std_prec:5.2f}     | {mean_day:>12.2f}")

    # PLOTTING
    plt.figure(figsize=(10, 6))
    plt.errorbar(RATE_LIMITS_local, mean_recalls, yerr=std_recalls, marker='o', label='Recall (mean ± std)')
    plt.errorbar(RATE_LIMITS_local, mean_precisions, yerr=std_precisions, marker='x', linestyle='--', label='Precision (mean ± std)')

    plt.title(f'Attack Success vs. Rate Limit (with Data Churn)\nChurn={CHURN_PER_CYCLE_local} items per churn | trials={trials_local} | pairs={enable_pairs_local} | churn_model={churn_model_local}')
    plt.xlabel('Allowed Queries per Cycle (Rate Limit)')
    plt.ylabel('Attack Success (%)')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()

    if output_filename_local is None:
        output_filename_local = f"churn_attack_results_trials{trials_local}_M{M_local}_N{N_local}_K{K_local}_pairs{enable_pairs_local}_churn-{churn_model_local}.png"
    plt.savefig(output_filename_local)
    print(f"\nGraph saved to {output_filename_local}")


if __name__ == "__main__":
    main()
