# Black-Box CBF Attacker vs. Temporal Defense

This repository explores the privacy vulnerabilities of **Counting Bloom Filters (CBFs)** and evaluates the effectiveness of **Rate Limiting** and **Temporal Data Churn** as defensive strategies.

## Overview
Based on the paper *"On the Privacy of Counting Bloom Filters Under a Black-Box Attacker"* (Galán et al., 2023), this project simulates a "Peeling" attack. Unlike a static environment, this simulation introduces:
1. **Rate Limiting:** Throttling the number of queries an attacker can make per time cycle.
2. **Temporal Decay (Churn):** Real-time updates to the set where old items are removed and new ones are added, simulating a dynamic cache.



## Key Findings
Our experiments demonstrate that when the query rate is sufficiently restricted, the "extraction time" exceeds the "data lifespan." This causes the attacker's **Recall** to drop significantly as the stolen information becomes stale before the attack completes.

## Project Structure
- `ExperimentMultipleTrials.py`: Main entry point for running Monte Carlo simulations.
- `CountingBloomFilter.py`: Implementation of the CBF logic and hash functions.
- `ThrottledOracle.py`: The "Server" implementation that enforces rate limits and applies data churn.
- `AttackLogic.py`: Implementation of the black-box "Peeling" algorithm.

## Getting Started
1. Clone the repo: `git clone https://github.com/Jsaiborne/black-box-cbf-attacker.git`
2. Run the simulation: `python ExperimentMultipleTrials.py`

## License
Distributed under the MIT License. See `LICENSE` for more information.