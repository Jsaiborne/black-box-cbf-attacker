# GenericHashFunctionsMD5.py
"""
GenericHashFunctionsMD5 (updated)
--------------------------------
Implements Kirsch–Mitzenmacher double hashing to produce an arbitrary number
of hash functions from two base hashes:

    h_i(x) = (h1(x) + i * h2(x)) mod k

This avoids slicing a single digest into fixed-size chunks and thus removes
the previous limitation that required nhash*bitidx_size <= 128.

API:
    gen = GenericHashFunctionsMD5(k=number_of_slots, nhash=number_of_hashes)
    idx = gen.getbit_idx(element_int, n)   # 0 <= n < nhash

Notes:
- h1 is derived from MD5(element)
- h2 is derived from SHA1(element)
- We cache the last element's h1 and h2 to speed repeated calls for same element.
- If h2 % k == 0 we add 1 to h2_mod to avoid returning the same index for all i.
"""

import hashlib
import math


class GenericHashFunctionsMD5:
    def __init__(self, k=1024, nhash=2):
        """
        k      : number of slots (range 0..k-1) - typically the Bloom filter size 'm'
        nhash  : number of hash functions requested
        """
        self.k = int(max(1, k))
        self.nhash = int(max(1, nhash))

        # cache for last element hashed -> (h1_int, h2_int)
        self.lastelement = None
        self._h1 = 0
        self._h2 = 0

    def _compute_base_hashes(self, element):
        """
        Compute and cache the two base hashes for 'element' as integers:
        - h1 from MD5 (128 bits)
        - h2 from SHA1 (160 bits)
        Store them on the instance for reuse.
        """
        # compute MD5-based h1
        md5_hex = hashlib.md5(element.encode()).hexdigest()
        h1 = int(md5_hex, 16)

        # compute SHA1-based h2
        sha1_hex = hashlib.sha1(element.encode()).hexdigest()
        h2 = int(sha1_hex, 16)

        # cache values
        self.lastelement = element
        self._h1 = h1
        self._h2 = h2

    def getbit_idx(self, element_int, n):
        """
        Return the n-th hash index for element_int, in range [0, k-1].
        Uses double hashing: (h1 + n*h2) mod k.

        Parameters:
            element_int : object convertible to str (we call str() on it)
            n           : index of hash (0 <= n < nhash)

        Returns:
            Integer index in 0..k-1
        """
        if n < 0 or n >= self.nhash:
            raise IndexError(f"hash index n={n} out of range (0..{self.nhash-1})")

        element = str(element_int)
        if self.lastelement != element:
            self._compute_base_hashes(element)

        # reduce base hashes modulo k to keep numbers small while preserving uniformity
        h1_mod = self._h1 % self.k
        h2_mod = self._h2 % self.k

        # Avoid degenerate case where h2_mod == 0 -> then h_i = h1_mod for all i
        if h2_mod == 0:
            h2_mod = 1

        idx = (h1_mod + (n * h2_mod)) % self.k
        return idx
