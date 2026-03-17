import random
import time
import argparse
import statistics
import math
import csv
import matplotlib
matplotlib.use('Agg')  # Safe for servers/headless
import matplotlib.pyplot as plt

# =================================================
# ============== USER-EDITABLE CONSTANTS ==========
# Edit these for quick experiments (CLI will override)
# =================================================
# Basic workload
N = 150                # number of true elements initially inserted
U_SIZE = 5000          # size of universe (scannable range)
RATE_LIMITS = [10000, 5000, 2000, 1000, 500, 200, 100]

# Sweep M/N ratios (list). The code computes M = round(ratio * N)
M_N_RATIOS = [8, 12, 16]   # example ratios to sweep; add/remove as needed

# Attack / algorithm choices
ENABLE_PAIRS = True    # set to False to disable pair-testing (edit in-code)
PAIR_BUDGET = 1000      # max pair tests per experiment (edit in-code)
CHECK_BUDGET = None    # None => full-scan P; integer => sample when building Prem

# Trials / reproducibility / churn model
TRIALS = 5        # default number of independent trials per rate-limit
SEED_BASE = 42       # if int, base seed; per-trial seed used = SEED_BASE + trial_index
CHURN_MODEL = 'poisson'  # 'query' (churn triggered by hitting query_limit) or 'time'/'poisson'/'bursty'
TIME_INTERVAL = None   # when churn_model='time', churn every TIME_INTERVAL accesses (None -> use query_limit)
LAMBDA_CHURN = 2.0     # NEW: Average churn events per cycle (e.g., 2 times per rate-limit window)

# FP measurement
FP_SAMPLE_SIZE =3000  # number of non-members to sample for empirical FP measurement per trial

# Misc
VERBOSE = False        # set True to see per-trial prints
OUTPUT_FILENAME = None # if None a filename will be auto-generated

# =================================================
# End constants
# =================================================


# Import your existing class (must be in same directory)
try:
    from CountingBloomFilter import CountingBloomFilter
except ImportError:
    print("Error: CountingBloomFilter.py not found. Please ensure the file is in the same directory.")
    exit()


# ---------- ThrottledOracle (instrumented for logging) ----------
# ---------- ThrottledOracle (instrumented for logging) ----------
class ThrottledOracle:
    def __init__(self, cbf, true_set, universe, query_limit, churn_amount,
                 churn_model='query', time_between_churns=None, lambda_churn=1.0):
        self.cbf = cbf
        self.true_set = true_set
        self.universe = universe
        self.query_limit = query_limit
        self.churn_amount = churn_amount
        self.churn_model = churn_model
        self.time_between_churns = time_between_churns if time_between_churns is not None else query_limit
        self.lambda_churn = lambda_churn

        # Counters for diagnostics / logging
        self.queries_made = 0         
        self.total_accesses = 0       
        self.time_counter = 0
        self.total_days_passed = 0
        self.churn_events = 0
        
        # --- NEW: Continuous Time Tracking ---
        self.current_time = 0.0
        # If Poisson, schedule the first independent churn event
        if self.churn_model == 'poisson':
            self.next_churn_time = random.expovariate(self.lambda_churn)
        else:
            self.next_churn_time = float('inf')

    def access(self, operation, item):
        self.total_accesses += 1
        self.queries_made += 1
        self.time_counter += 1

        # --- NEW: Advance the clock ---
        # 1 cycle = query_limit. Therefore, 1 query takes (1.0 / query_limit) time units.
        time_per_query = 1.0 / self.query_limit
        self.current_time += time_per_query

        if self.churn_model == 'query':
            if self.queries_made >= self.query_limit:
                self._apply_data_churn()
                self.queries_made = 0
                self.total_days_passed += 1
                
        elif self.churn_model == 'time':
            if self.time_counter >= self.time_between_churns:
                self._apply_data_churn()
                self.time_counter = 0
                self.total_days_passed += 1
                
        elif self.churn_model == 'poisson':
            # --- NEW: True Poisson Churn ---
            # While the attacker was querying, did we pass the scheduled churn time?
            # (Uses 'while' because a very slow attacker might miss multiple scheduled churns)
            while self.current_time >= self.next_churn_time:
                self._apply_data_churn()
                # Schedule the next churn event based on the exponential distribution
                self.next_churn_time += random.expovariate(self.lambda_churn)
                self.total_days_passed += 1 # Treating one churn event as a tracked "day/cycle" progression

        # perform the cbf operation
        if operation == 'check':
            return self.cbf.check(item)
        elif operation == 'add':
            return self.cbf.add(item)
        elif operation == 'remove':
            return self.cbf.remove(item)

    def _apply_data_churn(self):
        self.churn_events += 1
        for _ in range(self.churn_amount):
            if not self.true_set:
                break
            idx_to_remove = random.randrange(len(self.true_set))
            old_item = self.true_set.pop(idx_to_remove)
            try:
                self.cbf.remove(old_item)
            except Exception:
                pass

            # add new unique
            new_item = random.randint(1, len(self.universe))
            while new_item in self.true_set:
                new_item = random.randint(1, len(self.universe))
            self.true_set.append(new_item)
            try:
                self.cbf.add(new_item)
            except Exception:
                pass


# ---------- Attacker implementation (single + optional pair testing) ----------
def run_black_box_attack(oracle, universe, enable_pairs=False, pair_budget=1000, check_budget=None, verbose=False):
    # --- DISCOVERY ---
    P = []
    for x in universe:
        if oracle.access('check', x):
            P.append(x)

    Srec = []

    def build_Prem_excluding(exclude_set):
        candidates = [y for y in P if y not in exclude_set]
        if check_budget is None:
            subset = candidates
        else:
            subset = random.sample(candidates, min(len(candidates), check_budget))
        Prem_local = set()
        for y in subset:
            if oracle.access('check', y):
                Prem_local.add(y)
        return Prem_local, subset

    def test_element(x, P):
        oracle.access('remove', x)
        x_still_positive = oracle.access('check', x)
        if x_still_positive:
            oracle.access('add', x)
            return False

        Prem, _ = build_Prem_excluding({x})
        R = [y for y in P if y != x and y not in Prem]

        if set(P) == Prem.union({x}):
            if x in P: P.remove(x)
            Srec.append(x)
            if verbose:
                print(f"singleton confirm {x}")
            return True

        if R:
            for z in R:
                oracle.access('add', z)
            x_pos_after_inserting_R = oracle.access('check', x)
            if not x_pos_after_inserting_R:
                for z in R:
                    if z in P: P.remove(z)
                    oracle.access('remove', z)
                if x in P: P.remove(x)
                Srec.append(x)
                if verbose:
                    print(f"singleton confirm {x} after inserting R")
                return True
            else:
                for z in R:
                    oracle.access('remove', z)

        oracle.access('add', x)
        return False

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

        if set(P) == Prem.union({x, y}):
            if x in P: P.remove(x)
            if y in P: P.remove(y)
            Srec.extend([x, y])
            if verbose:
                print(f"pair confirm ({x},{y})")
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
                    print(f"pair deduced from ({x},{y}), found_any={found_any}")
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

    return Srec


# ---------- MAIN: sweep M/N ratios and rate_limits, measure empirical FP & CSV logging ----------
def main():
    parser = argparse.ArgumentParser(description="CBF black-box peeling attack simulation with m/n sweep and empirical FP and CSV logging.")
    parser.add_argument("--trials", type=int, default=None, help="override TRIALS constant")
    parser.add_argument("--pairs", action="store_true", help="enable pair-testing (overrides ENABLE_PAIRS)")
    parser.add_argument("--pair-budget", type=int, default=None, help="override PAIR_BUDGET")
    parser.add_argument("--seed", type=int, default=None, help="seed base (per-trial: seed+trial_index)")
    parser.add_argument("--check-budget", type=int, default=None, help="override CHECK_BUDGET")
    parser.add_argument("--fp-samples", type=int, default=None, help="override FP_SAMPLE_SIZE")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--output", type=str, default=None, help="plot filename override")
    args = parser.parse_args()

    trials = args.trials if args.trials is not None else TRIALS
    enable_pairs = args.pairs if args.pairs else ENABLE_PAIRS
    pair_budget = args.pair_budget if args.pair_budget is not None else PAIR_BUDGET
    seed_base = args.seed if args.seed is not None else SEED_BASE
    check_budget = args.check_budget if args.check_budget is not None else CHECK_BUDGET
    fp_samples = args.fp_samples if args.fp_samples is not None else FP_SAMPLE_SIZE
    verbose = args.verbose if args.verbose else VERBOSE
    output_filename = args.output if args.output is not None else OUTPUT_FILENAME

    N_local = N
    universe = list(range(1, U_SIZE + 1))
    churn_per_cycle = 5  # keep same as before; could be exposed as constant if desired

    # CSV diagnostics file (overwrite existing)
    csv_filename = "detailed_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "ratio", "M_passed", "K", "effective_m",
            "trial_idx", "seed",
            "rate_limit",
            "empirical_fp_pct", "theoretical_fp_pct",
            "occupancy_pct", "initial_P_size",
            "recovered_count", "true_positives_found",
            "recall_pct", "precision_pct",
            "total_accesses", "days_passed", "churn_events"
        ])

    # Prepare plotting containers: for each m/n ratio we'll collect mean recall/precision lists
    all_mean_recalls = {}
    all_std_recalls = {}
    all_mean_precisions = {}
    all_std_precisions = {}
    all_mean_days = {}
    all_mean_fp = {}
    all_std_fp = {}

    print(f"Running experiments: N={N_local}, trials={trials}, m/n ratios={M_N_RATIOS}, rate_limits={RATE_LIMITS}")
    for ratio in M_N_RATIOS:
        M_local = max(1, int(round(ratio * N_local)))
        # compute optimal K
        k_float = (M_local / N_local) * math.log(2)
        K_local = max(1, int(round(k_float)))
        if verbose:
            print(f"\n=== ratio={ratio} -> M_passed={M_local}, computed K_opt={K_local} ===\n")

        mean_recalls = []
        std_recalls = []
        mean_precisions = []
        std_precisions = []
        mean_days = []
        mean_fp_rates = []
        std_fp_rates = []

        # For each rate limit, run 'trials' experiments
        for limit in RATE_LIMITS:
            recalls = []
            precisions = []
            days = []
            fps = []  # empirical fp per trial (measured before attack)

            for t in range(trials):
                # reproducible seed per trial (if provided)
                trial_seed = None
                if seed_base is not None:
                    trial_seed = seed_base + t
                    random.seed(trial_seed)

                # Setup CBF and insert true_set
                cbf = CountingBloomFilter(M_local, K_local)
                true_set = random.sample(universe, N_local)
                for x in true_set:
                    cbf.add(x)

                # 1) Determine effective m from cbf internals (try several likely attributes)
                effective_m = None
                # first try attributes that hold arrays of counters/bits
                candidates = ['bloom_structure', 'counters', 'count_array', 'array', 'bits', 'counts', 'bitarray', 'bloom']
                for attr in candidates:
                    if hasattr(cbf, attr):
                        val = getattr(cbf, attr)
                        try:
                            effective_m = len(val)
                        except Exception:
                            # if it's a numeric attribute (not list), use directly if sensible
                            try:
                                effective_m = int(val)
                            except Exception:
                                effective_m = None
                        break
                # fallback: try common single-name fields
                if effective_m is None:
                    for key, val in cbf.__dict__.items():
                        # prefer list-like attributes
                        try:
                            if isinstance(val, (list, tuple)):
                                effective_m = len(val)
                                break
                        except Exception:
                            continue
                # final fallback: use passed M_local
                if effective_m is None:
                    effective_m = M_local

                # 2) theoretical FP using effective_m
                p_theory_eff = (1.0 - math.exp(- (K_local * N_local) / float(effective_m))) ** K_local
                p_theory_pct = p_theory_eff * 100.0

                # 3) empirical FP measurement: sample non-members and check via cbf.check (NOT oracle)
                non_members = [x for x in universe if x not in true_set]
                sample_size = min(fp_samples, len(non_members))
                sample_non_members = random.sample(non_members, sample_size)
                fp_count = 0
                for y in sample_non_members:
                    try:
                        if cbf.check(y):
                            fp_count += 1
                    except Exception:
                        # if check raises for unexpected reasons, treat as negative to avoid failing whole run
                        pass
                empirical_fp = (fp_count / sample_size) * 100.0 if sample_size > 0 else 0.0
                fps.append(empirical_fp)

                # 4) occupancy probe (try common internal names)
                occupancy = None
                counters = None
                candidates_occ = ['bloom_structure', 'counters', 'count_array', 'array', 'bits', 'counts', 'bitarray', 'bloom']
                for attr in candidates_occ:
                    if hasattr(cbf, attr):
                        counters = getattr(cbf, attr)
                        break
                if counters is not None:
                    try:
                        non_zero = sum(1 for v in counters if v > 0)
                        occupancy = (non_zero / float(len(counters))) * 100.0
                    except Exception:
                        occupancy = None

               # 5) Build oracle to compute attacker's observed P
                oracle_for_P = ThrottledOracle(cbf, list(true_set), universe,
                                               query_limit=limit, churn_amount=churn_per_cycle,
                                               churn_model=CHURN_MODEL,
                                               time_between_churns=(TIME_INTERVAL if TIME_INTERVAL is not None else limit),
                                               lambda_churn=LAMBDA_CHURN) # <-- ADD THIS
                P = []
                for x in universe:
                    if oracle_for_P.access('check', x):
                        P.append(x)
                initial_P_size = len(P)

                # 6) Run the actual attack on a fresh CBF + oracle
                cbf_attack = CountingBloomFilter(M_local, K_local)
                for x in true_set:
                    cbf_attack.add(x)
                oracle = ThrottledOracle(cbf_attack, list(true_set), universe,
                                         query_limit=limit, churn_amount=churn_per_cycle,
                                         churn_model=CHURN_MODEL,
                                         time_between_churns=(TIME_INTERVAL if TIME_INTERVAL is not None else limit),
                                         lambda_churn=LAMBDA_CHURN) # <-- ADD THIS

                extracted_elements = run_black_box_attack(oracle, universe,
                                                          enable_pairs=enable_pairs,
                                                          pair_budget=pair_budget,
                                                          check_budget=check_budget,
                                                          verbose=verbose)

                # Evaluate
                current_server_set = set(oracle.true_set)
                found_set = set(extracted_elements)
                true_positives_found = len(found_set.intersection(current_server_set))
                recovered_count = len(found_set)

                recall = (true_positives_found / N_local) * 100 if N_local > 0 else 0
                precision = (true_positives_found / recovered_count) * 100 if recovered_count > 0 else 0

                recalls.append(recall)
                precisions.append(precision)
                days.append(oracle.total_days_passed)

                # gather diagnostics for CSV
                total_accesses = getattr(oracle, "total_accesses", "")
                days_passed = getattr(oracle, "total_days_passed", "")
                churn_events = getattr(oracle, "churn_events", "")

                # write CSV row
                with open(csv_filename, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        ratio, M_local, K_local, effective_m,
                        t, (trial_seed if trial_seed is not None else ""),
                        limit,
                        f"{empirical_fp:.6f}", f"{p_theory_pct:.6f}",
                        (f"{occupancy:.6f}" if occupancy is not None else ""),
                        initial_P_size,
                        recovered_count, true_positives_found,
                        f"{recall:.6f}", f"{precision:.6f}",
                        (total_accesses if total_accesses is not None else ""), (days_passed if days_passed is not None else ""), (churn_events if churn_events is not None else "")
                    ])

                # compact debug line
                print(f"DEBUG trial ratio={ratio} M_passed={M_local} effective_m={effective_m} limit={limit} t={t} seed={trial_seed} theory_fp={p_theory_pct:.4f}% emp_fp={empirical_fp:.4f}% occ={('N/A' if occupancy is None else f'{occupancy:.2f}%')} |P|={initial_P_size} recov={recovered_count} tp={true_positives_found} recall={recall:.2f}% prec={precision:.2f}% accesses={total_accesses} days={days_passed} churns={churn_events}")

            # aggregate per rate limit
            mean_rec = statistics.mean(recalls)
            std_rec = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
            mean_prec = statistics.mean(precisions)
            std_prec = statistics.stdev(precisions) if len(precisions) > 1 else 0.0
            mean_day = statistics.mean(days)
            mean_fp = statistics.mean(fps)
            std_fp = statistics.stdev(fps) if len(fps) > 1 else 0.0

            mean_recalls.append(mean_rec)
            std_recalls.append(std_rec)
            mean_precisions.append(mean_prec)
            std_precisions.append(std_prec)
            mean_days.append(mean_day)
            mean_fp_rates.append(mean_fp)
            std_fp_rates.append(std_fp)

            print(f"ratio={ratio} limit={limit} => FP={mean_fp:.3f}% ±{std_fp:.3f}, recall={mean_rec:5.2f}% ±{std_rec:5.2f}, prec={mean_prec:5.2f}% ±{std_prec:5.2f}, days={mean_day:.2f}")

        # store results per ratio
        all_mean_recalls[ratio] = mean_recalls
        all_std_recalls[ratio] = std_recalls
        all_mean_precisions[ratio] = mean_precisions
        all_std_precisions[ratio] = std_precisions
        all_mean_days[ratio] = mean_days
        all_mean_fp[ratio] = mean_fp_rates
        all_std_fp[ratio] = std_fp_rates

    # Plot: one plot for recall (all ratios), one for precision
    plt.figure(figsize=(10, 6))
    for ratio in M_N_RATIOS:
        plt.errorbar(RATE_LIMITS, all_mean_recalls[ratio], yerr=all_std_recalls[ratio], marker='o', label=f"m/n={ratio}")
    plt.gca().invert_xaxis()
    plt.title(f"Recall vs Rate Limit (m/n sweep)")
    plt.xlabel("Allowed Queries per Cycle (Rate Limit)")
    plt.ylabel("Recall (%)")
    plt.grid(True)
    plt.legend()
    out_rec = output_filename if output_filename is not None else f"recall_mn_sweep.png"
    plt.savefig(out_rec)
    print(f"Saved recall plot to {out_rec}")

    plt.figure(figsize=(10, 6))
    for ratio in M_N_RATIOS:
        plt.errorbar(RATE_LIMITS, all_mean_precisions[ratio], yerr=all_std_precisions[ratio], marker='x', linestyle='--', label=f"m/n={ratio}")
    plt.gca().invert_xaxis()
    plt.title(f"Precision vs Rate Limit (m/n sweep)")
    plt.xlabel("Allowed Queries per Cycle (Rate Limit)")
    plt.ylabel("Precision (%)")
    plt.grid(True)
    plt.legend()
    out_prec = output_filename if output_filename is not None else f"precision_mn_sweep.png"
    plt.savefig(out_prec)
    print(f"Saved precision plot to {out_prec}")

    # Also print a compact summary table of empirical FP (means) per ratio (they are independent of rate limit)
    print("\nEmpirical FP summary (mean ± std) per m/n ratio (averaged across trials and rate limits):")
    for ratio in M_N_RATIOS:
        fps_list = all_mean_fp[ratio]
        avg_fp = statistics.mean(fps_list)
        stdev_fp = statistics.mean(all_std_fp[ratio]) if len(all_std_fp[ratio]) > 0 else 0.0
        print(f"m/n={ratio:>5} -> empirical FP (avg across rate_limits) = {avg_fp:.4f}%  (avg std {stdev_fp:.4f}%)")

    print("\nDone. CSV written to", csv_filename)

if __name__ == "__main__":
    main()
