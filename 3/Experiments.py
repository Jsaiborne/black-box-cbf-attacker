import random
import sys
import getopt
from CountingBloomFilter import CountingBloomFilter
from CountingBloomFilterNoCol import CountingBloomFilterNoCol
from GenericHashFunctionsSHA512 import GenericHashFunctionsSHA512
from math import e
from math import log as ln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse

FULL = 0
IND = 1
PAIRS = 2

# Variables and functions to calculate the theoretical extraction values for all variants
WHITE_BOX = 0
BLACK_BOX_PAIRS = 1
BLACK_BOX_SINGLE = 2
VARIANTS = [WHITE_BOX,BLACK_BOX_SINGLE,BLACK_BOX_PAIRS]
variant = WHITE_BOX
COLORS = ['red','green','blue']

k = None
cs = None
def setK(kk):
    global k
    global cs
    k = kk
    cs = ln(2)/k

def round(ps,pf,cf):
    global cs
    λ1 = k*cf*(1-pf)**(k-1)
    λ2 = k*cs*(1-ps)**(k-1)
    if (variant == WHITE_BOX):
        ps = e**(-λ1)
    elif (variant == BLACK_BOX_SINGLE):
        ps = e**(-λ1)*e**(-λ2)
    elif (variant == BLACK_BOX_PAIRS):
        ps = e**(-λ1)*e**(-λ2)*(1+λ2)
    pf = e**(-λ2)
    return (ps,pf)

def getCoreDensity(cf):
    (ps,pf) = (0,0)
    for i in range(1000):
        (ps,pf) = round(ps,pf,cf)
    return (1-ps)**k

def find_threshold():
    (low,high) = (0,500)
    while(high - low > 0.0001):
        mid = (low + high)/2
        if (getCoreDensity(mid) < 0.000001):
            low = mid
        else:
            high = mid
    return low

# Testing parameters
# filter_size = 1024
# k = 4
# n = 175
# trials = 100
max_val = 1000000000

parser = argparse.ArgumentParser()
parser.add_argument("-m", dest="m", type=int, help="Filter size (default 1024)", default=1024)
parser.add_argument("-n", dest="n", type=int, help="Number of true positives (default 256)", default=256)
parser.add_argument("-t", dest="t", type=int, help="Number of iterations (default 100)", default=100)
parser.add_argument("-k", dest="k", type=int, help="Number of hashes (default 3)", default=3)
# parser.add_argument("-p", dest="pairs", type=int, help="Carry pair extraction or not (default 0 - False). Any other number means True", default=0)
args = parser.parse_args()
filter_size = args.m
n = args.n
trials = args.t
k = args.k

# Function to generate the random set of elements.
# Current version uses strings
# num is the number of elements to generate
# if a CountingBloomFilter bf is passed, elements are stored there
# if a list ds is passed, elements are stored there
# exclude set are a elements that should not be selected
# maxVal is the maximum int value
def generate_random_elements(num, bf=None, ds=None, max_val=1000000000, exclude=None):
    # Elements are added to the set to check for repetitions
    if exclude is None:
        exclude = set()
    s = set()
    s.clear()

    # Keeps data of stored elements
    stored = 0
    # Generate elements until the value "stored" is reached
    while stored < num:
        # Generate integers between 1 and max_val
        entry = random.randint(1, max_val)
        # if entry was already selected or is in the exclude set,
        # go to next iteration
        if entry in s or entry in exclude:
            continue
        # When an BloomFilter is received
        if bf is not None:
            # Add the entry to the filter
            bf.add(entry)
        # When a list is received
        if ds is not None:
            # Add the element to the list
            ds.append(entry)
        # Another element has been stored
        stored = stored + 1
        # Add it to the set so they are not repeated
        s.add(entry)
    return

# Function to generate a random set of false positives
# num is the number of false positives to generate
# bf is the CountingBloomFilter in which to test the elements to find false positives
# maxVal is the maximum int value
# exclude set are a elements that should not be selected (basically true positives)
# returns a list with the generated false positives
def generate_random_fp(num, bf, max_val=1000000000, exclude=None):
    # Elements are added to the set to check for repetitions
    if exclude is None:
        exclude = set()
    s = set()
    s.clear()

    # Keeps data of stored elements
    stored = 0
    false_positives = []
    # Generate elements until the value "stored" is reached
    while stored < num:
        # Generate integers between 1 and max_val
        entry = random.randint(1, max_val)
        # if entry was already selected or is in the exclude set,
        # go to next iteration
        if entry in s or entry in exclude or not bf.check(entry):
            continue
        # When an CountingBloomFilter is received
        false_positives.append(entry)
        # Another element has been stored
        stored = stored + 1
        # Add it to the set so they are not repeated
        s.add(entry)
    return false_positives

# Function to find all elements from the universe that returns a positive from CBF
# bf is the Counting Bloom Filter
# max_val is the maximum integer value. Universe will include elements from 1 to max_val
def find_p(bf, max_val):
    # Create the list P of (true and false) positive elements
    p = list()
    # Check all elements of the universe, from 1 to max_val
    for i in range(max_val+1):
        # If one of the positions is 0, then it is a negative
        # Otherwise, add it to P
        if bf.check(i, 1):
            p.append(i)

    return p

# Function to find all elements from a set that returns a positive from CBF
# bf is the Counting Bloom Filter
# set is the set of elements to be tested against the filter
def find_p_set(bf, set):
    # Create the list P of (true and false) positive elements
    p = list()
    # Check all elements of the set
    for element in set:
        # If one of the positions is 0, then it is a negative
        # Otherwise, add it to P
        if bf.check(element, 1):
            p.append(element)

    return p

# Function that clears the element from its positions in T and also clears all the related false positives
# m is the size of the filter
# elements is the T array
# positives is the list of elements to be removed
# count_cbf is the list of counters from the CBF
# count is the list of counters from T
# hashf is the hash function used in the CBF
# k is the number of positions
# is_positive indicates if it is a real positive (true) or a false positive (false)
# nocol is a boolean indicating whether no collision is activated or not
def clear_positions(m, elements, positives, count_cbf, count, hashf, k, is_positive, nocol):
    # Additional elements to be removed
    additional = list()
    # and iterate over them
    num = len(positives)

    # Traverse the positives list
    for i in range(num):
        # get next element to be removed
        next_positive = positives[i]
        # for the k hash functions
        hashes = []
        for j in range(k):
            # Get the position mapped for the element and the jth hash function
            jpos = hashf.getbit_idx(next_positive, j)
            if nocol:
                while jpos in hashes:
                    jpos = (jpos + 1) % m
                hashes.append(jpos)
            # Element might have been removed in a different level of recursion
            if elements[jpos].count(next_positive) == 0:
                break
            # Remove the element from the position
            elements[jpos].remove(next_positive)
            # Reduce the T counter for that position
            count[jpos] -= 1
            # Reduce the CBF counter only when it is a real positive
            if is_positive:
                count_cbf[jpos] -= 1
            # If no more elements are mapped to this position in the CBF
            # we can remove all the pending elements from T and they are false positives
            if count_cbf[jpos] == 0 and count[jpos] != 0:
                # Add those elements to additional list
                additional.extend(elements[jpos])

    # Recursive call to remove the false positive elements
    if len(additional) > 0:
        # Pass False as last parameter as they are false positives
        clear_positions(m, elements, additional, count_cbf, count, hashf, k, False, nocol)

    return

# Get the set of elements from the Universe (1 to maxVal) that may be included into the filter.
# m is the number of positions (counters) of the CBF
# k is the number of hash functions
# cbf is the Counting Bloom Filter
# p is the P array with all the elements from the universe that returned positive from CBF
# nocol is a boolean indicating whether no collision is activated or not
# top is the maximum counter the peeling will extract
# top = FULL means that the algorithm will extract everything it can, without constraints
# top = IND is equivalent to blackbox ind, top = PAIRS is equivalent to blackbox pairs
def peeling(m, k, cbf, p, nocol, top):
    # Retrieve the hash function used
    hashf = cbf.get_hash()
    # T array  with m positions to store the elements that are mapped to them
    elements = [None] * m
    # Count of elements mapped to each position
    count = [0] * m

    # For all the positions in p
    for i in range(len(p)):
        hashes = []
        # for the k hash functions
        for j in range(k):
            # Get the position mapped for the element p[i] and the jth hash function
            pos = hashf.getbit_idx(p[i], j)
            # If no collision is activated, we recalculate the value of the hash
            if nocol:
                while pos in hashes:
                    pos = (pos + 1) % m
                hashes.append(pos)
            # Retrieve the position pos of the T array
            list_pos = elements[pos]
            # If no elements are assigned to that position, create a list and assign it
            if list_pos is None:
                list_pos = list()
                elements[pos] = list_pos
            # Include the element into the list of elements mapped to the position
            list_pos.append(p[i])
            # Increase the count of elements mapped to the position
            count[pos] += 1

    # Set that will store the positives that were extracted from the filter
    positives = set()
    # Values for the CBF counters
    counters = cbf.get_counters()

    while True:
        # If we found an element that could be extracted in this iteration
        found = False
        # Go through the m positions
        # TODO: exclude positions that were 0 in the previous iterations to speed up the process
        for i in range(m):
            # when the ith position does not have values, go to next position
            if count[i] == 0:
                continue
            # the position has values => it is not empty
            # we only want those positions where we have the same number of elements in T and CBF
            # and where the counter is less or equal than the established top
            if count[i] != counters[i] or (top > 0 and count[i] > top):
                continue
            # we found at least one position
            found = True
            # add the elements of the ith position to the positives
            positives.update(elements[i])
            # all these elements must be removed as well, but not from elements
            removers = elements[i].copy()
            # call the function that clears the removers and related false positives
            # pass True as last parameter as they are real positives
            clear_positions(m, elements, removers, counters, count, hashf, k, True, nocol)
        # if no new positives were found in the iteration, we should finish the algorithm
        if not found:
            break

    # return elements that were retrieved from the CBF
    return positives

x_axis = []
y1_axis = []
y2_axis = []
y3_axis = []
y4_axis = []
y5_axis = []
y6_axis = []
y7_axis = []
y8_axis = []
y9_axis = []

dots = [(x*n)//10 for x in range(0,51)]

f = open(str(filter_size) + '_' + str(k) + '_' + str(n) + '.results', 'w')
f.write("Start: " + time.ctime(time.time()) + "\n")
for fals in dots:
    avg_blackbox_ind = 0
    worst_blackbox_ind = 100
    avg_blackbox_pairs = 0
    worst_blackbox_pairs = 100
    avg_whitebox = 0
    worst_whitebox = 100
    for _ in range(trials):

        # Generate a standard CBF with the testing parameters
        bf = CountingBloomFilterNoCol(filter_size, k)

        # Fill the filter with random elements
        true_positives = []
        generate_random_elements(n, bf, true_positives, max_val)

        # Generate a certain number of false positives
        false_positives = generate_random_fp(fals, bf, max_val, true_positives)

        # Run the algorithm
        all_positives = true_positives + false_positives
        all_positives_set = set(all_positives)

        # First, we carry out the unconstrained whitebox analysis
        found_tps = peeling(filter_size, k, bf, all_positives, 1, FULL)
        prct_obtained = (len(found_tps)/len(true_positives)) * 100
        avg_whitebox += prct_obtained/trials
        if prct_obtained < worst_whitebox:
            worst_whitebox = prct_obtained

        # Then, we carry out the whitebox analysis limited to counters with value 1 (equivalent to blackbox ind)
        bf = CountingBloomFilterNoCol(filter_size, k)
        for pos in true_positives:
            bf.add(pos)
        found_tps = peeling(filter_size, k, bf, all_positives, 1, IND)
        prct_obtained = (len(found_tps)/len(true_positives)) * 100
        avg_blackbox_ind += prct_obtained/trials
        if prct_obtained < worst_blackbox_ind:
            worst_blackbox_ind = prct_obtained

        # Finally, we carry out the whitebox analysis limited to counters with value 2 (equivalent to blackbox pairs)
        bf = CountingBloomFilterNoCol(filter_size, k)
        for pos in true_positives:
            bf.add(pos)
        found_tps = peeling(filter_size, k, bf, all_positives, 1, PAIRS)
        prct_obtained = (len(found_tps)/len(true_positives)) * 100
        avg_blackbox_pairs += prct_obtained/trials
        if prct_obtained < worst_blackbox_pairs:
            worst_blackbox_pairs = prct_obtained


    setK(k)
    for v in VARIANTS:
        variant = v
        if v is WHITE_BOX:
            th_whitebox = (1 - getCoreDensity((fals/n)*2**k*cs)) * 100
        if v is BLACK_BOX_SINGLE:
            th_blackbox_ind = (1 - getCoreDensity((fals/n)*2**k*cs)) * 100
        if v is BLACK_BOX_PAIRS:
            th_blackbox_pairs = (1 - getCoreDensity((fals/n)*2**k*cs)) * 100

    print("FPs:", fals, "Avg Whitebox:", avg_whitebox, "Worst Whitebox:", worst_whitebox, "Avg Blackbox Ind:", avg_blackbox_ind, "Worst Blackbox Ind:", worst_blackbox_ind, "Avg Blackbox Pairs:", avg_blackbox_pairs, "Worst Blackbox Pairs:", worst_blackbox_pairs, "Theoretical Whitebox:", th_whitebox, "Theoretical Blackbox Ind:", th_blackbox_ind, "Theoretical Blackbox Pairs:", th_blackbox_pairs)


    x_axis.append(fals/n)
    y1_axis.append(avg_whitebox)
    y2_axis.append(avg_blackbox_ind)
    y3_axis.append(avg_blackbox_pairs)
    y4_axis.append(worst_whitebox)
    y5_axis.append(worst_blackbox_ind)
    y6_axis.append(worst_blackbox_pairs)
    y7_axis.append(th_whitebox)
    y8_axis.append(th_blackbox_ind)
    y9_axis.append(th_blackbox_pairs)

f.write("End: " + time.ctime(time.time()) + "\n")
f.write("Proportion FP/TP\n")
f.write(str(x_axis) + "\n")
f.write("Avg Whitebox\n")
f.write(str(y1_axis) + "\n")
f.write("Avg Blackbox Ind\n")
f.write(str(y2_axis) + "\n")
f.write("Avg Blackbox Pairs\n")
f.write(str(y3_axis) + "\n")
f.write("Worst Whitebox\n")
f.write(str(y4_axis) + "\n")
f.write("Worst Blackbox Ind\n")
f.write(str(y5_axis) + "\n")
f.write("Worst Blackbox Pairs\n")
f.write(str(y6_axis) + "\n")
f.write("Theoretical Whitebox\n")
f.write(str(y7_axis) + "\n")
f.write("Theoretical Blackbox Ind\n")
f.write(str(y8_axis) + "\n")
f.write("Theoretical Blackbox Pairs\n")
f.write(str(y9_axis) + "\n")
f.close()

plt.xlabel('Ratio False positives/True positives')
plt.ylabel('% of successfully obtained elements')
plt.title('m='+str(filter_size)+', '+'k='+str(k)+', '+'n='+str(n))
plt.yticks([0,20,40,60,80,100])
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.xlim(right=5.25)
plt.grid()
lab1 = "Avg Whitebox"
lab2 = "Avg Blackbox Ind"
lab3 = "Avg Blackbox Pairs"
lab4 = "Worst Whitebox"
lab5 = "Worst Blackbox Ind"
lab6 = "Worst Blackbox Pairs"
lab7 = "Th. Whitebox"
lab8 = "Th. Blackbox Ind"
lab9 = "Th. Blackbox Pairs"
plt.plot(x_axis, y1_axis, label=lab1)
plt.plot(x_axis, y2_axis, label=lab2)
plt.plot(x_axis, y3_axis, label=lab3)
plt.plot(x_axis, y4_axis, label=lab4)
plt.plot(x_axis, y5_axis, label=lab5)
plt.plot(x_axis, y6_axis, label=lab6)
plt.plot(x_axis, y7_axis, label=lab7)
plt.plot(x_axis, y8_axis, label=lab8)
plt.plot(x_axis, y9_axis, label=lab9)
plt.legend()
plt.savefig(str(filter_size) + '_' + str(k) + '_' + str(n) + '.png')





