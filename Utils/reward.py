import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
#from rdkit.six.moves import cPickle
import pickle as cPickle
from rdkit.Chem import AllChem, QED, DataStructs, Descriptors
import subprocess
from boltz.data.parse.schema import compute_3d_conformer

from Utils.sascore import calculateScore

import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import hydra
import pandas as pd
from pathlib import Path


def getReward(name, seed_smiles="", th=0.6):
    if name == "QED":
        return QEDReward()
    elif name == "PLogP":
        return PenalizedLogPReward()
    elif name == "ConstPLogP":
        return ConstPLogPReward(seed_smiles=seed_smiles, th=th)
    elif name == "QSAR":
        return QSARReward()
    elif name =="Boltzina":
        return BoltziniaReward()

class QSARReward:
    def __init__(self):
        self.model: RandomForestRegressor = pickle.load(open(hydra.utils.get_original_cwd()+"/data/features/ver2/qsar.pickle","rb"))
    def reward(self, mol):
        #if mol is not None:
        ar = np.zeros(2048)
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        Chem.DataStructs.ConvertToNumpyArray(bitvect,ar)

        br = [ar.tolist()]
        #model: RandomForestRegressor = pickle.load(open("model.pickle","rb"))

        pred = self.model.predict(np.array(br))
        return pred[0]       
class PenalizedLogPReward:
    def __init__(self):
        self.vmin = -100
        return

    def reward(self, mol):
        """
            This code is obtained from https://github.com/DeepGraphLearning/GraphAF
            , which is a part of GraphAF(Chence Shi et al. ICLR2020) program.
            Reward that consists of log p penalized by SA and # long cycles,
            as described in (Kusner et al. 2017). Scores are normalized based on the
            statistics of 250k_rndm_zinc_drugs_clean.smi dataset
            :param mol: rdkit mol object
            :return: float
        """
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        if mol is not None:
            try:
                log_p = Descriptors.MolLogP(mol)
                SA = -calculateScore(mol)

                # cycle score
                cycle_list = nx.cycle_basis(nx.Graph(
                    Chem.rdmolops.GetAdjacencyMatrix(mol)))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([len(j) for j in cycle_list])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                cycle_score = -cycle_length

                normalized_log_p = (log_p - logP_mean) / logP_std
                normalized_SA = (SA - SA_mean) / SA_std
                normalized_cycle = (cycle_score - cycle_mean) / cycle_std
                score = normalized_log_p + normalized_SA + normalized_cycle
            except ValueError:
                score = self.vmin
        else:
            score = self.vmin

        return score


class QEDReward:
    def __init__(self):
        self.vmin = 0

    def reward(self, mol):
        try:
            if mol is not None:
                score = QED.qed(mol)
            else:
                score = -1
        except ValueError:
            score = -1

        return score


class ConstPLogPReward:
    def __init__(self, seed_smiles, th=0.6):
        self.vmin = -100
        mol = Chem.MolFromSmiles(seed_smiles)
        self.seed_fp = AllChem.GetMorganFingerprint(mol, 2)
        self.reward_module = PenalizedLogPReward()
        self.th = th

    def reward(self, mol):
        try:
            gent_fp = AllChem.GetMorganFingerprint(mol, 2)
            sim = DataStructs.TanimotoSimilarity(self.seed_fp, gent_fp)
        except RuntimeError:
            sim = 0

        score = self.reward_module.vmin
        if sim > self.th:
            score = self.reward_module.reward(mol)

        return score


class SimilarityReward:
    def __init__(self, seed_smiles):
        self.vmin = 0
        mol = Chem.MolFromSmiles(seed_smiles)
        self.seed_fp = AllChem.GetMorganFingerprint(mol, 2)

    def reward(self, mol):
        gent_fp = AllChem.GetMorganFingerprint(mol, 2)
        sim = DataStructs.TanimotoSimilarity(self.seed_fp, gent_fp)

        return sim

class BoltziniaReward:
    def __init__(self):
        self.vmin = 0.1
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"

    def reward(self, mol):
        Path(self.path_to_workspace+f"ligand.smi").write_text(Chem.MolToSmiles(mol)+"\n")
        subprocess.run([self.path_to_boltzinia+".venv/bin/python", self.path_to_boltzinia+"run.py", self.path_to_workspace+"3zos_ddr1_config.json","--work_dir",f"{self.path_to_workspace}"[:-1],"--output_dir",f"{self.path_to_workspace}"[:-1],"--output_file",f"{self.path_to_workspace}results.csv","--vina_override", "--ligand_files", f"{self.path_to_workspace}ligand.pdb"], cwd = hydra.utils.get_original_cwd())

        df = pd.read_csv(self.path_to_workspace+f'results.csv')
        score = df.affinity_pred_value[0]
        return score

        

