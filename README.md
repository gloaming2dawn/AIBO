# Acquisition function maximizer Initialization for Bayesian Optimization (AIBO)
This is the code-release for the AIBO method from _**Unleashing the Potential of Acquisition Functions in High-
Dimensional Bayesian Optimization: An empirical study to understand the role of acquisition function maximizer initialization**_ submitted to Transactions on Machine Learning Research (TMLR). 

Note that AIBO is is a minimization algorithm, so please make sure you reformulate potential maximization problems.

## Installation
First, make sure you have a Python 3 environment installed. We recommend miniconda3 or anaconda3.
>conda create -n test python=3.9.9

Then install the package via pip:
>pip install -r requirements.txt --use-deprecated=legacy-resolver

## Quick Test
Using AIBO with UCB1.96 as the acquisition function to optimize 100D Ackley function
```
cd AIBO
python run.py --func=Ackley --dim=100 --method=AIBO_mixed-grad-UCB1.96 --iters=5000 --batch-size=10
```
Using standard BO (BO-grad in the paper) for comparison
```
python run.py --func=Ackley --dim=100 --method=AIBO_random-grad-UCB1.96 --iters=5000 --batch-size=10
```
Change acquisition function from UCB1.96 to UCB4 or EI
```
python run.py --func=Ackley --dim=100 --method=AIBO_mixed-grad-UCB4 --iters=5000 --batch-size=10
python run.py --func=Ackley --dim=100 --method=AIBO_mixed-grad-EI --iters=5000 --batch-size=10
```
Use single heuristic initialization strategy (GA/CMA-ES) for gradient-based acquisition function maximizer
```
python run.py --func=Ackley --dim=100 --method=AIBO_ga-grad-UCB1.96 --iters=5000 --batch-size=10
python run.py --func=Ackley --dim=100 --method=AIBO_cmaes-grad-UCB1.96 --iters=5000 --batch-size=10
```
Test other functions
```
python run.py --func=Rosenbrock --dim=100 --method=AIBO_mixed-grad-UCB1.96 --iters=5000 --batch-size=10
python run.py --func=Rastrigin --dim=100 --method=AIBO_mixed-grad-UCB1.96 --iters=5000 --batch-size=10
python run.py --func=Griewank --dim=100 --method=AIBO_mixed-grad-UCB1.96 --iters=5000 --batch-size=10
```
For GPU acceleration, install cuda-based torch version and add --device=cuda to the command, for example
```
python run.py --func=Ackley --dim=100 --method=AIBO_mixed-grad-UCB1.96 --iters=5000 --batch-size=10 --device=cuda
```
