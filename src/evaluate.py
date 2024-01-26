import os
import argparse
from dataset import Dataset, AppleQualityDataset, HeartDiseaseDataset, RandomDataset
from bn_builder import MultiLibBayesianNetwork, Library, Category, Algorithm
from time import process_time_ns
import json 



class Results:
    def __init__(self):
        self.results = {}

    def add(self, accuracy:float, num_edges:int, library:Library, category:Category, algorithm:Algorithm, time:float, notes:str|None=None):
        category = category.value
        algorithm = algorithm.value
        if category not in self.results: self.results[category] = {}
        if algorithm not in self.results[category]: self.results[category][algorithm] = []

        self.results[category][algorithm].append({
            "accuracy": accuracy,
            "num_edges": num_edges,
            "library": self.__getLibraryName(library),
            "time": time,
            "notes": notes
        })

    def __getLibraryName(self, library:Library):
        if library == Library.PGMPY: return "pgmpy"
        elif library == Library.BNLEARN: return "bnlearn"
        elif library == Library.POMEGRANATE: return "pomegranate"
        elif library == Library.PYAGRUM: return "pyAgrum"


def evaluateAccuracy(data:Dataset, compute_accuracy:bool, measure_time:bool) -> dict:
    results = Results()

    to_test_configs = [
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "chi_square", {"ci_test": "chi_square", "significance_level": 0.01, "max_cond_vars": 5}),
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "g_sq", {"ci_test": "g_sq", "significance_level": 0.05, "max_cond_vars": 5}),
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "log_likelihood", {"ci_test": "log_likelihood", "significance_level": 0.05, "max_cond_vars": 5}),
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "freeman_tuckey", {"ci_test": "freeman_tuckey", "significance_level": 0.05, "max_cond_vars": 5}),
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "modified_log_likelihood", {"ci_test": "modified_log_likelihood", "significance_level": 0.05, "max_cond_vars": 5}),
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "neyman", {"ci_test": "neyman", "significance_level": 0.05, "max_cond_vars": 5}),
        (Library.PGMPY, Category.CONSTRAINT, Algorithm.PC, "cressie_read", {"ci_test": "cressie_read", "significance_level": 0.05, "max_cond_vars": 5}),
        (Library.PGMPY, Category.SCORE, Algorithm.HC, "k2score", {"scoring_method": "k2score"}),
        (Library.PGMPY, Category.SCORE, Algorithm.HC, "bdeuscore", {"scoring_method": "bdeuscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.HC, "bdsscore", {"scoring_method": "bdsscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.HC, "bicscore", {"scoring_method": "bicscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.HC, "aicscore", {"scoring_method": "aicscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.MMHC, "k2score", {"scoring_method": "k2score"}),
        (Library.PGMPY, Category.SCORE, Algorithm.MMHC, "bdeuscore", {"scoring_method": "bdeuscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.MMHC, "bdsscore", {"scoring_method": "bdsscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.MMHC, "bicscore", {"scoring_method": "bicscore"}),
        (Library.PGMPY, Category.SCORE, Algorithm.MMHC, "aicscore", {"scoring_method": "aicscore"}),
        (Library.PGMPY, Category.TREE, Algorithm.CL, None, {}),
        (Library.PGMPY, Category.TREE, Algorithm.NB, None, {}),
        (Library.PGMPY, Category.TREE, Algorithm.TAN, None, {"class_node": data.features[0]}),
        (Library.POMEGRANATE, Category.CONSTRAINT, Algorithm.ASTAR, None, {}),
        (Library.POMEGRANATE, Category.TREE, Algorithm.CL, None, {}),
        (Library.PYAGRUM, Category.CONSTRAINT, Algorithm.MIIC, "MDL correction", {"correction": "MDL"}),
        (Library.PYAGRUM, Category.CONSTRAINT, Algorithm.MIIC, "NML correction", {"correction": "NML"}),
        (Library.PYAGRUM, Category.CONSTRAINT, Algorithm.MIIC, "No correction", {"correction": None}),
        (Library.PYAGRUM, Category.SCORE, Algorithm.TABU, None, {}),
        (Library.PYAGRUM, Category.SCORE, Algorithm.HC, None, {}),
        (Library.PYAGRUM, Category.SCORE, Algorithm.K2, None, {"ordering": [*range(len(data.features)+1)]}),
    ]

    for library, category, algorithm, notes, kwargs in to_test_configs:
        print(f">>> Starting: {library} {category} {algorithm} {kwargs}")
        try:
            if measure_time:
                time_measures = []
                for _ in range(3):
                    start_time = process_time_ns()
                    model = MultiLibBayesianNetwork(library).fit(data, algorithm, **kwargs)
                    time_measures.append( process_time_ns() - start_time )
                elapsed_time = sum(time_measures) / len(time_measures)
            else:
                model = MultiLibBayesianNetwork(library).fit(data, algorithm, **kwargs)
                elapsed_time = None

            accuracy = None
            if compute_accuracy: accuracy = data.evaluateBN(model)

            results.add(
                accuracy = accuracy, 
                num_edges = model.countEdges(),
                library = library, 
                category = category, 
                algorithm = algorithm,
                time = elapsed_time,
                notes = notes
            )
        except Exception as e:
            print(f"ERROR: {library} {category} {algorithm} {kwargs} {e}")

    return results.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Structure learning evaluation")
    parser.add_argument("--dataset", type=str, choices=["apple", "heart", "random"], required=True, help="Evaluation dataset")
    parser.add_argument("--random-variables", type=int, default=10, help="Number of variables in the random dataset")
    parser.add_argument("--accuracy", action="store_true", help="Compute accuracy")
    parser.add_argument("--time", action="store_true", help="Compute training time")
    parser.add_argument("--output-path", type=str, required=True, help="Path where the results will be saved")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random generators")
    args = parser.parse_args()

    if args.dataset == "apple":
        data = AppleQualityDataset(data_path="./data/apple_quality.csv", seed=args.seed)
    elif args.dataset == "heart":
        data = HeartDiseaseDataset(data_path="./data/heart.csv", seed=args.seed)
    elif args.dataset == "random":
        data = RandomDataset(args.random_variables, seed=args.seed)

    results = evaluateAccuracy(data, compute_accuracy=args.accuracy, measure_time=args.time)

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=3)