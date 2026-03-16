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
            # Use index to avoid slow value search, assume list for simulation speed
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
# 2. THE BLACK-BOX ATTACKER
# ==========================================
def run_black_box_attack(oracle, universe):
    """
    Implements the 'Peeling' logic WITHOUT looking at counters.
    Uses 'Pollution' (Insert/Remove) to infer membership.
    """
    # --- PHASE 1: DISCOVERY (Initial Scan) ---
    # The attacker scans the universe to find potential positives (P)
    potential_positives = []
    
    # We limit the universe scan for simulation speed on your PC
    # In reality, this takes months if Rate Limit is low.
    for x in universe:
        if oracle.access('check', x):
            potential_positives.append(x)

    # --- PHASE 2: REFINEMENT (The "Peeling" Attack) ---
    # Algorithm: Remove x, see if anyone else in P disappears.
    detected_true_positives = []
    
    # We work on a copy because we modify the filter
    candidates = list(potential_positives) 
    
    for x in candidates:
        # 1. Remove candidate X
        oracle.access('remove', x)
        
        # 2. Check for "Impact"
        # If X was a True Positive, removing it might drop a counter to 0,
        # causing OTHER elements (false positives) to disappear.
        impact_detected = False
        
        # Optimization: Attacker checks a subset of P to save queries
        # (Checking ALL of P is O(|P|^2) which is very slow)
        check_budget = min(len(candidates), 50) 
        
        for i in range(check_budget):
            y = candidates[i]
            if x == y: continue
            
            # If y is suddenly negative, X was likely a True Positive masking Y
            if not oracle.access('check', y):
                impact_detected = True
                break 
        
        # 3. Restore candidate X (Attacker cleans up)
        oracle.access('add', x)
        
        if impact_detected:
            detected_true_positives.append(x)

    return detected_true_positives

# ==========================================
# 3. EXPERIMENT EXECUTION (Modified for Trials)
# ==========================================
def main():
    # Parameters optimized for a standard PC
    # N=200, M=2000 runs instantly. 
    M = 2048   # Filter size
    K = 4      # Hash functions
    N = 150    # Number of elements in Set S
    U_size = 5000 # Size of Universe (scannable range)
    
    universe = list(range(1, U_size + 1))
    
    # Experiment Settings
    rate_limits = [10000, 5000, 2000, 1000, 500, 200, 100]
    churn_per_cycle = 5 # 5 elements change every cycle
    num_trials = 20     # Run each setting 20 times to average out noise
    
    results_precision = []
    results_recall = []

    print(f"--- Starting Simulation [N={N}, M={M}, Trials={num_trials}] ---")
    print(f"{'Rate Limit':<12} | {'Avg Recall':<12} | {'Avg Precision':<15} | {'Avg Days':<12}")
    print("-" * 60)

    for limit in rate_limits:
        total_recall = 0
        total_precision = 0
        total_days = 0
        
        # --- THE TRIALS LOOP ---
        for _ in range(num_trials):
            # 1. Setup Environment
            cbf = CountingBloomFilter(M, K)
            true_set = random.sample(universe, N)
            for x in true_set:
                cbf.add(x)
                
            # 2. Setup Oracle (The Rate Limiter + Churner)
            oracle = ThrottledOracle(cbf, true_set, universe, 
                                     query_limit=limit, 
                                     churn_amount=churn_per_cycle)
            
            # 3. Run Attack
            extracted_elements = run_black_box_attack(oracle, universe)
            
            # 4. Evaluate Success for this specific trial
            current_server_set = set(oracle.true_set)
            found_set = set(extracted_elements)
            
            true_positives_found = len(found_set.intersection(current_server_set))
            
            # Recall: What % of the set did we steal?
            recall = (true_positives_found / N) * 100 if N > 0 else 0
            
            # Precision: How much of what we stole is actually valid?
            precision = (true_positives_found / len(found_set)) * 100 if len(found_set) > 0 else 0
            
            # Accumulate totals
            total_recall += recall
            total_precision += precision
            total_days += oracle.total_days_passed

        # --- CALCULATE AVERAGES ---
        avg_recall = total_recall / num_trials
        avg_precision = total_precision / num_trials
        avg_days = total_days / num_trials
        
        results_recall.append(avg_recall)
        results_precision.append(avg_precision)
        
        print(f"{limit:<12} | {avg_recall:>9.1f}%   | {avg_precision:>12.1f}%   | {avg_days:<12.1f}")

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(rate_limits, results_recall, marker='o', label='Avg Recall (Set Recovery)', color='blue')
    plt.plot(rate_limits, results_precision, marker='x', linestyle='--', label='Avg Precision (Valid Info)', color='red')
    
    plt.title(f'Attack Success vs. Rate Limit (Averaged over {num_trials} trials)\nChurn={churn_per_cycle} items per limit cycle')
    plt.xlabel('Allowed Queries per Cycle (Rate Limit)')
    plt.ylabel('Average Attack Success (%)')
    plt.gca().invert_xaxis() # High limit (left) -> Low limit (right)
    plt.grid(True)
    plt.legend()
    
    filename = f"churn_attack_results_{M}_{N}_trials.png"
    plt.savefig(filename)
    print(f"\nGraph saved to {filename}")

if __name__ == "__main__":
    main()