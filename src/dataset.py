from abc import ABC
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.inference import VariableElimination
import bnlearn
import torch
import pyAgrum
import networkx as nx
import matplotlib.pyplot as plt



class Library(Enum):
    PGMPY = 0
    BNLEARN = 1
    POMEGRANATE = 2
    PYAGRUM = 3



class Dataset(ABC):
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


    def evaluate(self, model, library:Library) -> float:
        """
            Computes the accuracy of a Bayesian network.
        """
        preds = np.zeros(len(self.test_df))

        if library == Library.POMEGRANATE:
            test_data = self.test_df.to_numpy()
            X = torch.tensor(test_data)
            mask = torch.tensor([[col in self.features for col in self.test_df.columns]]*test_data.shape[0])
            X_masked = torch.masked.MaskedTensor(X, mask=mask)

            preds = model.predict(X_masked)[:, self.test_df.columns.tolist().index(self.target)].numpy()
        else:
            for i, entry in enumerate(self.test_df.iloc):
                evidence = { f: int(entry[f]) for f in self.features }

                if library == Library.PGMPY:
                    phi = VariableElimination(model).query([self.target], evidence)
                    preds[i] = phi.state_names[self.target][np.argmax(phi.values)]
                elif library == Library.BNLEARN:
                    phi = bnlearn.inference.fit(model, variables=[self.target], evidence=evidence, verbose=0)
                    preds[i] = phi.state_names[self.target][np.argmax(phi.values)]
                elif library == Library.PYAGRUM:
                    preds[i] = np.argmax(pyAgrum.getPosterior(model, evs=evidence, target=self.target).tolist())


        return accuracy_score(self.test_df[self.target], preds)


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