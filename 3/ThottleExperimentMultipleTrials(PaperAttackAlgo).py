import random
import time
import argparse
import matplotlib
matplotlib.use('Agg') # Safe for servers/headless
import matplotlib.pyplot as plt

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
        self.true_set = true_set   # The server's ground truth
        self.universe = universe   # All possible elements
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
# 2. THE BLACK-BOX ATTACKER (UPDATED)
# ==========================================
def run_black_box_attack(oracle, universe):
    """
    Implements the black-box peeling attack according to the paper's
    Algorithm 1 (test(x)) and Algorithm 3 (peeling loop). Pairs/triplets
    are NOT implemented (as requested).
    Returns the recovered elements (Srec).
    """

    # --- PHASE 1: DISCOVERY (Initial Scan) ---
    print("Scanning universe to compute P (positives)...")
    P = []
    for x in universe:
        if oracle.access('check', x):
            P.append(x)
    print(f"Initial positives found: {len(P)}")

    # Srec: recovered true positives
    Srec = []

    # Helper: test single element x according to Algorithm 1.
    def test_element(x, P):
        """
        Perform Algorithm 1 test of element x.
        Returns:
          - True if x is determined a true positive (and performs necessary removals)
          - False if undetermined (and restores state)
        Side-effects:
          - Modifies oracle (removes/adds) and P (if we confirm true positive).
        """
        # 1: Remove element x from the filter
        oracle.access('remove', x)

        # 2: Query x in the filter
        x_still_positive = oracle.access('check', x)
        if x_still_positive:
            # 3-5: Positive => undetermined; restore x and return
            oracle.access('add', x)
            return False

        # 7-12: Build Prem by querying all elements in P
        Prem = set()
        for y in P:
            # skip x (we already removed it), but safe to check
            if y == x:
                continue
            if oracle.access('check', y):
                Prem.add(y)

        # R = elements that disappeared after removal of x (excluding x itself)
        R = [y for y in P if y != x and y not in Prem]

        # Case: P == Prem U {x}  -> only x disappeared -> confirmed true positive
        if set(P) == Prem.union({x}):
            # Remove x from P and do not restore x in filter (we want it removed)
            if x in P:
                P.remove(x)
            # x is recovered
            Srec.append(x)
            # Note: x already removed from filter
            return True

        # Else if P == Prem U R U {x}  -> some elements disappeared (R)
        # Try case 2: insert R and query x again
        if R:
            # Insert all z in R into the filter
            for z in R:
                oracle.access('add', z)
            # Query x again
            x_pos_after_inserting_R = oracle.access('check', x)

            if not x_pos_after_inserting_R:
                # x still negative => x must be a true positive, and all z in R are false positives
                # Remove z in R from P and remove z from filter (we added them above)
                for z in R:
                    if z in P:
                        P.remove(z)
                    # remove z from filter to restore state (they were false positives)
                    oracle.access('remove', z)
                # x was already removed earlier; remove x from P and add to Srec
                if x in P:
                    P.remove(x)
                Srec.append(x)
                return True
            else:
                # Not determined: undo insertions of R
                for z in R:
                    oracle.access('remove', z)

        # If not determined by above, restore x and return undetermined
        oracle.access('add', x)
        return False

    # ==========================================
    # Algorithm 3 peeling loop (single elements only)
    # ==========================================
    run_process = True
    iteration = 0
    while run_process:
        iteration += 1
        run_process = False
        # Single-element tests: keep running until no single-element extraction occurs
        run_test = True
        while run_test:
            run_test = False
            # iterate over a static snapshot since P can change during iteration
            snapshot_P = list(P)
            for x in snapshot_P:
                # if x was removed from P earlier in this pass, skip
                if x not in P:
                    continue
                result = test_element(x, P)
                if result:
                    # Found a true positive; that'll allow further peeling
                    run_test = True
                    run_process = True
                    # continue testing remaining elements (we restart outer single-element loop)
            # end for snapshot_P
        # end while run_test

        # Note: pairs testing is omitted (per user's instruction).
        # After single-element phase completes, loop exits if no changes.
    # end while run_process

    print(f"Recovered {len(Srec)} elements via single-element peeling.")
    return Srec

# ==========================================
# 3. EXPERIMENT EXECUTION
# ==========================================
def main():
    # Parameters optimized for a standard PC
    M = 2048   # Filter size
    K = 4      # Hash functions
    N = 150    # Number of elements in Set S
    U_size = 5000 # Size of Universe (scannable range)
    
    universe = list(range(1, U_size + 1))
    
    # Rate limits to test
    rate_limits = [10000, 5000, 2000, 1000, 500, 200, 100]
    churn_per_cycle = 5 # 5 elements change every cycle
    
    results_precision = []
    results_recall = []

    print(f"--- Starting Simulation [N={N}, M={M}] ---")
    print(f"{'Rate Limit':<12} | {'Recall':<10} | {'Precision':<10} | {'Days Passed':<12}")
    print("-" * 55)

    for limit in rate_limits:
        # 1. Setup Environment
        cbf = CountingBloomFilter(M, K)
        true_set = random.sample(universe, N)
        for x in true_set:
            cbf.add(x)
            
        # 2. Setup Oracle (The Rate Limiter + Churner)
        oracle = ThrottledOracle(cbf, true_set, universe, 
                                 query_limit=limit, 
                                 churn_amount=churn_per_cycle)
        
        # 3. Run Attack (single-element peeling as implemented above)
        extracted_elements = run_black_box_attack(oracle, universe)
        
        # 4. Evaluate Success
        current_server_set = set(oracle.true_set)
        found_set = set(extracted_elements)
        
        true_positives_found = len(found_set.intersection(current_server_set))
        
        # Recall: What % of the set did we steal?
        recall = (true_positives_found / N) * 100 if N > 0 else 0
        
        # Precision: How much of what we stole is actually valid?
        precision = (true_positives_found / len(found_set)) * 100 if len(found_set) > 0 else 0
        
        results_recall.append(recall)
        results_precision.append(precision)
        
        print(f"{limit:<12} | {recall:>6.1f}%    | {precision:>6.1f}%    | {oracle.total_days_passed:<12}")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(rate_limits, results_recall, marker='o', label='Recall (Set Recovery)')
    plt.plot(rate_limits, results_precision, marker='x', linestyle='--', label='Precision (Valid Info)')
    
    plt.title(f'Attack Success vs. Rate Limit (with Data Churn)\nChurn={churn_per_cycle} items per limit cycle')
    plt.xlabel('Allowed Queries per Cycle (Rate Limit)')
    plt.ylabel('Attack Success (%)')
    plt.gca().invert_xaxis() # High limit (left) -> Low limit (right)
    plt.grid(True)
    plt.legend()
    
    filename = f"churn_attack_results_{M}_{N}.png"
    plt.savefig(filename)
    print(f"\nGraph saved to {filename}")

if __name__ == "__main__":
    main()
