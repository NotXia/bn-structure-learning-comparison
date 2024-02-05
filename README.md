# Evaluation of structure learning algorithms for Bayesian networks

Project for the Fundamentals of Artificial Intelligence and Knowledge Representation (Module 3) course at the University of Bologna (A.Y. 2023-2024).

A comparison of structure learning algorithms available in open-source Python libraries.


## Installation
The experiments were done using Python `3.11.0`.
To install the libraries, run:
```
pip install -r requirements.txt
```

## Usage
`evaluate.py` runs the various algorithms and outputs a JSON with the evaluation results.
In the `src` directory, run the following:
```
python evaluate.py                                  \
    --dataset=<apple or heart>                      \
    --output-path=<path to the output JSON file>    \
    --metrics                                       \
    --time
```