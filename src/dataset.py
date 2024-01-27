from bn_builder import Library, MultiLibBayesianNetwork
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import KBinsDiscretizer
import networkx as nx
import matplotlib.pyplot as plt



class Dataset:
    def __init__(self, 
            df:pd.DataFrame, 
            features:list[str], 
            target:str, 
            plot_positioning:list[dict], 
            train_ratio:float = 0.75, 
            seed:int = 42
        ):
        self.train_df, self.test_df = train_test_split(df, train_size=train_ratio, random_state=seed)
        self.features = features
        self.target = target
        self.plot_positioning = plot_positioning


    def evaluateBN(self, model:MultiLibBayesianNetwork) -> float:
        """
            Computes some evaluation metrics of a Bayesian network.
        """
        preds = model.predict(self.test_df, self.features, self.target)
        return {
            "accuracy": accuracy_score(self.test_df[self.target], preds),
            "precision_macro": precision_score(self.test_df[self.target], preds, average="macro"),
            "accuracy_macro": recall_score(self.test_df[self.target], preds, average="macro"),
            "f1_macro": f1_score(self.test_df[self.target], preds, average="macro")
        }
    
    def evaluate(self, model, library:Library) -> float:
        return self.evaluateBN(MultiLibBayesianNetwork(library, model))

    def plot(self, model, library: Library):
        if library == Library.PGMPY:
            nx.draw(nx.DiGraph(model.edges()), pos=self.plot_positioning, with_labels=True)
            plt.title("pgmpy")
        elif library == Library.BNLEARN:
            nx.draw(nx.DiGraph(model["model_edges"]), pos=self.plot_positioning, with_labels=True)
            plt.title("pgmpy")
        elif library == Library.POMEGRANATE:
            edges = [(
                self.train_df.columns[ model._distribution_mapping[edge[0]] ], 
                self.train_df.columns[ model._distribution_mapping[edge[1]] ]
            ) for edge in model.edges]
            nx.draw(nx.DiGraph(edges), pos=self.plot_positioning, with_labels=True)
            plt.title("pomegranate")
        elif library == Library.PYAGRUM:
            edges = [(self.train_df.columns[edge[0]], self.train_df.columns[edge[1]]) for edge in model.arcs()]
            nx.draw(nx.DiGraph(edges), pos=self.plot_positioning, with_labels=True)
            plt.title("pyAgrum")



# https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
class AppleQualityDataset(Dataset):
    def __init__(self, data_path:str="./data/apple_quality.csv", n_bins:int=3, train_ratio:float=0.75, seed:int=42):
        df = self.__loadDataset(data_path, n_bins)
        super().__init__(
            df = df, 
            features = df.columns[:-1],
            target = df.columns[-1],
            plot_positioning = {
                "size": (0.3, 0),
                "weight": (0.5, 1.2),
                "crunchiness": (0.3, 2),
                "quality": (1, 0.8),
                "juiciness": (1, 1.8),
                "sweetness": (1.7, 0),
                "ripeness": (1.5, 1.2),
                "acidity": (1.7, 2),
            },
            train_ratio = train_ratio, 
            seed = seed
        )

    def __loadDataset(self, data_path:str, n_bins) -> pd.DataFrame:
        df = pd.read_csv(data_path)

        df.columns = [col.lower() for col in df.columns]
        df.drop(columns=["a_id"], inplace=True)
        df.drop(index=df.index[-1], inplace=True)
        df["quality"] = [1 if quality == "good" else 0 for quality in df["quality"]]
        df["quality"] = df["quality"].astype(np.int64)

        for col in ["size", "weight", "sweetness", "crunchiness", "juiciness", "ripeness", "acidity"]:
            kbins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
            discrete_enc = kbins.fit_transform(df[col].to_numpy().reshape(-1, 1))
            df[col] = [encode[0] for encode in discrete_enc]
            df[col] = df[col].astype(np.int64)

        return df
    


# https://www.kaggle.com/datasets/ineubytes/heart-disease-dataset
class HeartDiseaseDataset(Dataset):
    def __init__(self, data_path:str="./data/hearth.csv", train_ratio:float=0.75, seed:int=42):
        df = self.__loadDataset(data_path)
        super().__init__(
            df = df, 
            features = df.columns[:-1],
            target = df.columns[-1],
            plot_positioning = {
                "target": (1.5, 1.5),
                "age": (0, 0.2),
                "sex": (1, 0),
                "cp": (2, 0),
                "trestbps": (3, 0.2),
                "chol": (0, 1),
                "fbs": (1, 1),
                "restecg": (2, 1),
                "thalach": (2.8, 2),
                "exang": (0.2, 2),
                "oldpeak": (1, 2.5),
                "slope": (2, 2.5),
                "thal": (0.5, 3),
                "ca": (2.5, 3),
            },
            train_ratio = train_ratio, 
            seed = seed
        )

    def __loadDataset(self, data_path:str) -> pd.DataFrame:
        df = pd.read_csv(data_path)

        for col, n_bins in [("age", 5), ("trestbps", 3), ("chol", 3), ("thalach", 3), ("oldpeak", 3)]:
            kbins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
            discrete_enc = kbins.fit_transform(df[col].to_numpy().reshape(-1, 1))
            df[col] = [encode[0] for encode in discrete_enc]
            df[col] = df[col].astype(np.int64)

        return df
    


class RandomDataset(Dataset):
    def __init__(self, num_variables, train_ratio:float=0.75, seed:int=42):
        df = self.__loadDataset(num_variables, seed)
        super().__init__(
            df = df, 
            features = df.columns[:-1],
            target = df.columns[-1],
            plot_positioning = {},
            train_ratio = train_ratio, 
            seed = seed
        )

    def __loadDataset(self, num_variables:int, seed) -> pd.DataFrame:
        data = np.random.default_rng(seed).normal(size=(10000, num_variables))
        df = pd.DataFrame(data=data, columns=[f"col{i}" for i in range(num_variables)])

        for col in df.columns:
            kbins = KBinsDiscretizer(n_bins=5, encode="ordinal")
            discrete_enc = kbins.fit_transform(df[col].to_numpy().reshape(-1, 1))
            df[col] = [encode[0] for encode in discrete_enc]
            df[col] = df[col].astype(np.int64)

        return df