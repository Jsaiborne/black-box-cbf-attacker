import random
import time
import argparse
import statistics
import matplotlib
matplotlib.use('Agg') # Safe for servers/headless
import matplotlib.pyplot as plt


# ----- user-editable constants -----
ENABLE_PAIRS = True    # set to False to disable pair-testing (edit in-code)
PAIR_BUDGET  = 500     # max number of pair tests per experiment (edit in-code)
TRIALS = 20             # default number of independent trials per rate-limit (edit in-code)
# -----------------------------------

# Import your existing class
# Ensure CountingBloomFilter.py is in the same directory
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
    1. Counts queries (Rate Limiting)
    2. Changes data automatically (Data Churn)
    """
    def __init__(self, cbf, true_set, universe, query_limit, churn_amount):
        self.cbf = cbf
        self.true_set = true_set   # The server's ground truth (list)
        self.universe = universe   # All possible elements (list)
        self.query_limit = query_limit # Queries allowed per "Day"
        self.churn_amount = churn_amount # Elements swapped per "Day"
        
        self.queries_made = 0
        self.total_days_passed = 0
        self.churn_events = 0

    def access(self, operation, item):
        """
        The only way the attacker interacts with the filter.
        """
        self.queries_made += 1
        
        # --- THE MECHANISM: Rate Limit triggers Churn ---
        if self.queries_made >= self.query_limit:
            self._apply_data_churn()
            self.queries_made = 0 # Reset daily budget
            self.total_days_passed += 1

        # Perform the actual CBF operation
        if operation == 'check':
            return self.cbf.check(item)
        elif operation == 'add':
            self.cbf.add(item)
        elif operation == 'remove':
            self.cbf.remove(item)

    def _apply_data_churn(self):
        """
        Simulates the set changing over time (Temporal Decay).
        Removes old items, adds new items.
        """
        self.churn_events += 1
        for _ in range(self.churn_amount):
            if not self.true_set: break
            
            # 1. Remove a random old element (Server updates its cache)
            idx_to_remove = random.randrange(len(self.true_set))
            old_item = self.true_set.pop(idx_to_remove)
            self.cbf.remove(old_item)

            # 2. Add a new random element
            # Simple rejection sampling to find a unique new item
            new_item = random.randint(1, len(self.universe)) 
            while new_item in self.true_set:
                new_item = random.randint(1, len(self.universe))
            
            self.true_set.append(new_item)
            self.cbf.add(new_item)

# ==========================================
# 2. THE BLACK-BOX ATTACKER (WITH PAIR TOGGLE)
# ==========================================
def run_black_box_attack(oracle, universe, enable_pairs=False, pair_budget=1000):
    """
    Implements the black-box peeling attack according to the paper's
    single-element test (Algorithm 1) and optional pair-testing
    (Algorithm 2) integrated into the peeling loop (Algorithm 3).
    - enable_pairs: if True, run pair-tests after single-element phase.
    - pair_budget: maximum number of pair-tests to attempt (to limit runtime).
    Returns the recovered elements (Srec).
    """

    # --- PHASE 1: DISCOVERY (Initial Scan) ---
    P = []
    for x in universe:
        if oracle.access('check', x):
            P.append(x)

    # Srec: recovered true positives
    Srec = []

    # ---------- Single-element test (Algorithm 1) ----------
    def test_element(x, P):
        # Remove element x from the filter
        oracle.access('remove', x)

        # Query x in the filter
        x_still_positive = oracle.access('check', x)
        if x_still_positive:
            # Undetermined: restore and exit
            oracle.access('add', x)
            return False

        # Build Prem by querying all elements in P \ {x}
        Prem = set()
        for y in P:
            if y == x: continue
            if oracle.access('check', y):
                Prem.add(y)

        # R = elements in P that disappeared after removing x (excluding x)
        R = [y for y in P if y != x and y not in Prem]

        # Case A: only x disappeared -> x is confirmed
        if set(P) == Prem.union({x}):
            if x in P:
                P.remove(x)
            Srec.append(x)
            return True

        # Case B: some R disappeared -> try inserting R and test again
        if R:
            for z in R:
                oracle.access('add', z)
            x_pos_after_inserting_R = oracle.access('check', x)

            if not x_pos_after_inserting_R:
                # x confirmed true positive; elements in R are false positives
                for z in R:
                    if z in P:
                        P.remove(z)
                    oracle.access('remove', z)
                if x in P:
                    P.remove(x)
                Srec.append(x)
                return True
            else:
                # undo insertions
                for z in R:
                    oracle.access('remove', z)

        # Not determined: restore x and return
        oracle.access('add', x)
        return False

    # ---------- Pair test (Algorithm 2, heuristic) ----------
    def test_pair(x, y, P):
        # Remove both x and y
        oracle.access('remove', x)
        oracle.access('remove', y)

        # Quick checks
        x_pos = oracle.access('check', x)
        y_pos = oracle.access('check', y)

        if x_pos and y_pos:
            oracle.access('add', x)
            oracle.access('add', y)
            return False

        # Build Prem
        Prem = set()
        for z in P:
            if z == x or z == y:
                continue
            if oracle.access('check', z):
                Prem.add(z)

        # R = elements in P that disappeared after removing x,y (excluding x,y)
        R = [z for z in P if z not in Prem and z not in (x, y)]

        # Case A: only x and/or y disappeared (no other losses)
        if set(P) == Prem.union({x, y}):
            if x in P: P.remove(x)
            if y in P: P.remove(y)
            Srec.extend([x, y])
            return True

        # Case B: some R disappeared; try inserting R and re-check x,y
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
                return True
            else:
                oracle.access('add', x)
                oracle.access('add', y)
                return False

        oracle.access('add', x)
        oracle.access('add', y)
        return False

    # ==========================================
    # Algorithm 3 peeling loop with optional pairs
    # ==========================================
    run_process = True
    while run_process:
        run_process = False

        # Single-element peeling (repeat until no more single-element discoveries)
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

        # If pair-testing enabled, try pair tests (bounded by budget)
        if enable_pairs and len(P) >= 2:
            pair_checks_done = 0
            candidates = list(P)
            random.shuffle(candidates)
            n = len(candidates)
            pair_found_in_pass = False
            for i in range(n):
                for j in range(i+1, n):
                    if pair_checks_done >= pair_budget:
                        break
                    x = candidates[i]; y = candidates[j]
                    if x not in P or y not in P:
                        continue
                    pair_checks_done += 1
                    if test_pair(x, y, P):
                        pair_found_in_pass = True
                        run_process = True
                    if len(P) < 2:
                        break
                if pair_checks_done >= pair_budget or len(P) < 2:
                    break

    return Srec

# ==========================================
# 3. EXPERIMENT EXECUTION (MULTI-TRIALS)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="CBF black-box peeling attack simulation.")
    parser.add_argument("--pairs", action="store_true", help="Enable pair-testing (Algorithm 2) as a toggle.")
    parser.add_argument("--pair-budget", type=int, default=None, help="Max pair tests per experiment (limits runtime).")
    parser.add_argument("--trials", type=int, default=None, help="Number of independent trials per rate-limit (overrides TRIALS constant).")
    args = parser.parse_args()

    # Parameters (can edit constants above or pass via CLI for trials/pairs)
    M = 2048   # Filter size
    K = 4      # Hash functions
    N = 150    # Number of elements in Set S
    U_size = 5000 # Size of Universe (scannable range)
    
    universe = list(range(1, U_size + 1))
    
    # Rate limits to test
    rate_limits = [10000, 5000, 2000, 1000, 500, 200, 100]
    churn_per_cycle = 5 # 5 elements change every cycle

    # Determine runtime params (CLI overrides constants where provided)
    trials = args.trials if args.trials is not None else TRIALS
    enable_pairs = args.pairs if args.pairs else ENABLE_PAIRS
    pair_budget = args.pair_budget if args.pair_budget is not None else PAIR_BUDGET

    print(f"--- Starting Simulation [N={N}, M={M}, trials={trials}] ---")
    print(f"{'Rate Limit':<12} | {'Recall (mean±std)':<22} | {'Precision (mean±std)':<24} | {'Days Passed (mean)':<16}")
    print("-" * 90)

    mean_recalls = []
    std_recalls = []
    mean_precisions = []
    std_precisions = []
    mean_days = []

    for limit in rate_limits:
        recalls = []
        precisions = []
        days = []

        for t in range(trials):
            # Setup Environment
            cbf = CountingBloomFilter(M, K)
            true_set = random.sample(universe, N)
            for x in true_set:
                cbf.add(x)
                
            # Setup Oracle (The Rate Limiter + Churner)
            oracle = ThrottledOracle(cbf, true_set, universe, 
                                     query_limit=limit, 
                                     churn_amount=churn_per_cycle)
            
            # Run Attack (single-element peeling + optional pairs)
            extracted_elements = run_black_box_attack(oracle, universe, enable_pairs=enable_pairs, pair_budget=pair_budget)
            
            # Evaluate Success
            current_server_set = set(oracle.true_set)
            found_set = set(extracted_elements)
            true_positives_found = len(found_set.intersection(current_server_set))
            
            # Recall & Precision
            recall = (true_positives_found / N) * 100 if N > 0 else 0
            precision = (true_positives_found / len(found_set)) * 100 if len(found_set) > 0 else 0

            recalls.append(recall)
            precisions.append(precision)
            days.append(oracle.total_days_passed)

        # aggregate statistics
        mean_rec = statistics.mean(recalls)
        std_rec = statistics.pstdev(recalls) if len(recalls) > 1 else 0.0
        mean_prec = statistics.mean(precisions)
        std_prec = statistics.pstdev(precisions) if len(precisions) > 1 else 0.0
        mean_day = statistics.mean(days)

        mean_recalls.append(mean_rec)
        std_recalls.append(std_rec)
        mean_precisions.append(mean_prec)
        std_precisions.append(std_prec)
        mean_days.append(mean_day)

        print(f"{limit:<12} | {mean_rec:6.1f}% ±{std_rec:5.2f}     | {mean_prec:6.1f}% ±{std_prec:5.2f}     | {mean_day:>12.2f}")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))
    # error bars show population stddev
    plt.errorbar(rate_limits, mean_recalls, yerr=std_recalls, marker='o', label='Recall (mean ± std)')
    plt.errorbar(rate_limits, mean_precisions, yerr=std_precisions, marker='x', linestyle='--', label='Precision (mean ± std)')
    
    plt.title(f'Attack Success vs. Rate Limit (with Data Churn)\nChurn={churn_per_cycle} items per limit cycle | trials={trials} | pairs={enable_pairs}')
    plt.xlabel('Allowed Queries per Cycle (Rate Limit)')
    plt.ylabel('Attack Success (%)')
    plt.gca().invert_xaxis() # High limit (left) -> Low limit (right)
    plt.grid(True)
    plt.legend()

    filename = f"churn_attack_results_trials{trials}_M{M}_N{N}_pairs{enable_pairs}.png"
    plt.savefig(filename)
    print(f"\nGraph saved to {filename}")

if __name__ == "__main__":
    main()
