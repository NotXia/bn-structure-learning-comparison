# Evaluation of Python libraries for Bayesian networks

An evaluation of structure learning algorithms available in open-source Python libraries.


## Installation
The experiments were done using Python `3.11.0`.
To install the libraries, move into `src` and run:
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