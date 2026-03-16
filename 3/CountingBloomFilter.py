# CountingBloomFilter.py
"""
CountingBloomFilter
-------------------

m = number of counter slots (NOT bits, NOT bytes).

This implementation stores exactly m integer counters.
Each hash maps into [0, m-1].

Theoretical FP formula should use:
    p = (1 - exp(-kN/m))^k

where:
    m = number of counters
    k = number of hash functions
    N = number of inserted elements
"""

from GenericHashFunctionsMD5 import GenericHashFunctionsMD5


class CountingBloomFilter:

    def __init__(self, m=65536, nhash=5, hash_f=None):
        """
        Parameters
        ----------
        m : int
            Number of counter slots (size of the filter).
        nhash : int
            Number of hash functions (k).
        hash_f : optional custom hash generator
        """

        if m <= 0:
            raise ValueError("m must be positive")

        if nhash <= 0:
            raise ValueError("nhash must be positive")

        # Number of counters (this is THE m in Bloom theory)
        self.m = int(m)

        # Number of hash functions (this is k in Bloom theory)
        self.nhash = int(nhash)

        # Allocate counters
        self.counters = [0] * self.m

        # Initialize hash generator
        if hash_f is None:
            self.hash = GenericHashFunctionsMD5(self.m, self.nhash)
        else:
            self.hash = hash_f

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    def add(self, data):
        """Insert element into filter."""
        for i in range(self.nhash):
            idx = self.hash.getbit_idx(data, i)
            self.counters[idx] += 1

    def remove(self, data):
        """Remove element from filter (safe removal)."""
        for i in range(self.nhash):
            idx = self.hash.getbit_idx(data, i)
            if self.counters[idx] > 0:
                self.counters[idx] -= 1

    def check(self, data, threshold=1):
        """
        Check membership.
        Returns True if all k counters >= threshold.
        """
        for i in range(self.nhash):
            idx = self.hash.getbit_idx(data, i)
            if self.counters[idx] < threshold:
                return False
        return True

    # --------------------------------------------------
    # Diagnostics / Introspection
    # --------------------------------------------------

    def clear(self):
        """Reset all counters to zero."""
        self.counters = [0] * self.m

    def get_counters(self):
        """Return internal counter array."""
        return self.counters

    def get_counter(self, position):
        """Return value of counter at index."""
        if position < 0 or position >= self.m:
            raise IndexError("counter index out of bounds")
        return self.counters[position]

    def load_factor(self):
        """
        Fraction of non-zero counters.
        Useful for occupancy diagnostics.
        """
        non_zero = sum(1 for c in self.counters if c > 0)
        return non_zero / float(self.m)

    def __len__(self):
        """len(cbf) returns number of counters (m)."""
        return self.m

    def __repr__(self):
        return f"CountingBloomFilter(m={self.m}, k={self.nhash})"
