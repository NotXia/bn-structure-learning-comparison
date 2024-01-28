from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset import Dataset
from enum import Enum, StrEnum
import pgmpy
import bnlearn
import pomegranate.bayesian_network
import pyAgrum
import numpy as np
import torch



class Library(Enum):
    PGMPY = 0
    BNLEARN = 1
    POMEGRANATE = 2
    PYAGRUM = 3


class Category(StrEnum):
    CONSTRAINT = "constraint"
    SCORE = "score"
    TREE = "tree"
    HYBRID = "hybrid"


class Algorithm(StrEnum):
    PC = "PC"
    ASTAR = "A*"
    MIIC = "MIIC"
    HC = "HC"
    MMHC = "MMHC"
    TABU = "TS"
    K2 = "K2"
    CL = "CL"
    NB = "NB"
    TAN = "TAN"


def _createPgmpyBNFromDag(dag: pgmpy.base.DAG, train_df) -> pgmpy.models.BayesianNetwork:
    model = pgmpy.models.BayesianNetwork()
    model.add_edges_from(dag.edges())
    model.fit(train_df)
    return model


def _pgmpyPC(train_df, ci_test, significance_level, max_cond_vars):
    dag = pgmpy.estimators.PC(data=train_df).estimate(variant="parallel", ci_test=ci_test, significance_level=significance_level, max_cond_vars=max_cond_vars, show_progress=False)
    return _createPgmpyBNFromDag(dag, train_df)

def _pgmpyHC(train_df, scoring_method):
    dag = pgmpy.estimators.HillClimbSearch(data=train_df).estimate(scoring_method=scoring_method, show_progress=False)
    return _createPgmpyBNFromDag(dag, train_df)

def _pgmpyMMHC(train_df):
    dag = pgmpy.estimators.MmhcEstimator(data=train_df).estimate()
    return _createPgmpyBNFromDag(dag, train_df)

def _pgmpyCL(train_df):
    dag = pgmpy.estimators.TreeSearch(data=train_df).estimate(estimator_type="chow-liu", show_progress=False)
    return _createPgmpyBNFromDag(dag, train_df)

def _pgmpyNB(train_df, features, target):
    dag = pgmpy.models.NaiveBayes(features, target)
    return _createPgmpyBNFromDag(dag, train_df)

def _pgmpyTAN(train_df, class_node):
    dag = pgmpy.estimators.TreeSearch(data=train_df).estimate(estimator_type="tan", class_node=class_node, show_progress=False)
    return _createPgmpyBNFromDag(dag, train_df)

def _bnlearnPC(train_df):
    dag = bnlearn.structure_learning.fit(train_df, methodtype="cs", verbose=0)
    return bnlearn.parameter_learning.fit(dag, train_df, methodtype="ml", verbose=0)

def _bnlearnHC(train_df, scoretype):
    dag = bnlearn.structure_learning.fit(train_df, methodtype="hc", scoretype=scoretype, verbose=0)
    return bnlearn.parameter_learning.fit(dag, train_df, methodtype="ml", verbose=0)

def _bnlearnCL(train_df):
    dag = bnlearn.structure_learning.fit(train_df, methodtype="cl", verbose=0)
    return bnlearn.parameter_learning.fit(dag, train_df, methodtype="ml", verbose=0)

def _bnlearnNB(train_df, root_node):
    dag = bnlearn.structure_learning.fit(train_df, methodtype="naivebayes", root_node=root_node, verbose=0)
    return bnlearn.parameter_learning.fit(dag, train_df, methodtype="ml", verbose=0)

def _bnlearnTAN(train_df, class_node):
    dag = bnlearn.structure_learning.fit(train_df, methodtype="tan", class_node=class_node, verbose=0)
    return bnlearn.parameter_learning.fit(dag, train_df, methodtype="ml", verbose=0)

def _pomegranateAstar(train_df):
    model = pomegranate.bayesian_network.BayesianNetwork(algorithm="exact")
    model.fit(train_df.to_numpy())
    return model

def _pomegranateCL(train_df):
    model = pomegranate.bayesian_network.BayesianNetwork(algorithm="chow-liu")
    model.fit(train_df.to_numpy())
    return model

def _pyAgrumMIIC(train_df, correction):
    learner = pyAgrum.BNLearner(train_df)
    if correction == "MDL":
        model = learner.useMIIC().useMDLCorrection().learnBN()
    elif correction == "NML":
        model = learner.useMIIC().useNMLCorrection().learnBN()
    else:
        model = learner.useMIIC().useNoCorrection().learnBN()
    return model

def _pyAgrumTabu(train_df):
    learner = pyAgrum.BNLearner(train_df)
    model = learner.useLocalSearchWithTabuList().learnBN()
    return model

def _pyAgrumHC(train_df):
    learner = pyAgrum.BNLearner(train_df)
    model = learner.useGreedyHillClimbing().learnBN()
    return model

def _pyAgrumK2(train_df, ordering):
    learner = pyAgrum.BNLearner(train_df)
    model = learner.useK2(ordering).learnBN()
    return model


class MultiLibBayesianNetwork:
    def __init__(self, library:Library, model=None):
        self.library = library
        self.model = model


    def fit(self, data:Dataset, algorithm:Algorithm, **kwargs):
        if self.library == Library.PGMPY:
            if algorithm == Algorithm.PC:           self.model = _pgmpyPC(data.train_df, **kwargs)
            elif algorithm == Algorithm.HC:         self.model = _pgmpyHC(data.train_df, **kwargs)
            elif algorithm == Algorithm.MMHC:       self.model = _pgmpyMMHC(data.train_df)
            elif algorithm == Algorithm.CL:         self.model = _pgmpyCL(data.train_df)
            elif algorithm == Algorithm.NB:         self.model = _pgmpyNB(data.train_df, data.features, data.target)
            elif algorithm == Algorithm.TAN:        self.model = _pgmpyTAN(data.train_df, **kwargs)
        elif self.library == Library.BNLEARN:
            if algorithm == Algorithm.PC:           self.model = _bnlearnPC(data.train_df)
            elif algorithm == Algorithm.HC:         self.model = _bnlearnHC(data.train_df, **kwargs)
            elif algorithm == Algorithm.CL:         self.model = _bnlearnCL(data.train_df)
            elif algorithm == Algorithm.NB:         self.model = _bnlearnNB(data.train_df, data.target)
            elif algorithm == Algorithm.TAN:        self.model = _bnlearnTAN(data.train_df, **kwargs)
        elif self.library == Library.POMEGRANATE:
            if algorithm == Algorithm.ASTAR:        self.model = _pomegranateAstar(data.train_df)
            elif algorithm == Algorithm.CL:         self.model = _pomegranateCL(data.train_df)
        elif self.library == Library.PYAGRUM:
            if algorithm == Algorithm.MIIC:         self.model = _pyAgrumMIIC(data.train_df, **kwargs)
            elif algorithm == Algorithm.TABU:       self.model = _pyAgrumTabu(data.train_df)
            elif algorithm == Algorithm.HC:         self.model = _pyAgrumHC(data.train_df)
            elif algorithm == Algorithm.K2:         self.model = _pyAgrumK2(data.train_df, [*range(len(data.features)+1)])
        return self
    

    def predict(self, df, evidence_vars, target_var):
        """
            Common interface to make predictions on a Dataframe.
        """
        if self.library == Library.POMEGRANATE:
            test_data = df.to_numpy()
            X = torch.tensor(test_data)
            mask = torch.tensor([ [(col in evidence_vars) for col in df.columns] ] * test_data.shape[0])
            X_masked = torch.masked.MaskedTensor(X, mask=mask)

            preds = self.model.predict(X_masked)[:, df.columns.tolist().index(target_var)].numpy()
        else:
            preds = np.zeros(len(df))

            for i, entry in enumerate(df.iloc):
                evidence = { f: int(entry[f]) for f in evidence_vars }

                if self.library == Library.PGMPY:
                    phi = pgmpy.inference.VariableElimination(self.model).query([target_var], evidence)
                    preds[i] = phi.state_names[target_var][np.argmax(phi.values)]
                elif self.library == Library.BNLEARN:
                    phi = bnlearn.inference.fit(self.model, variables=[target_var], evidence=evidence, verbose=0)
                    preds[i] = phi.state_names[target_var][np.argmax(phi.values)]
                elif self.library == Library.PYAGRUM:
                    preds[i] = np.argmax(pyAgrum.getPosterior(self.model, evs=evidence, target=target_var).tolist())
        
        return preds
    

    def countEdges(self):
        if self.library == Library.PGMPY:
            return len(self.model.edges())
        elif self.library == Library.BNLEARN:
            return len(self.model["model_edges"])
        elif self.library == Library.POMEGRANATE:
            return len(self.model.edges)
        elif self.library == Library.PYAGRUM:
            return len(self.model.arcs())