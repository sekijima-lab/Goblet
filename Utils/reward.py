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
import torch
import time
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit.Chem import AllChem
benzene_scores = {"akt1": 1.263, "ampc": 2.234 , "cxcr4": 0.9012, "cp3a4": 2.356, "gcr": 2.197, "hivpr": 1.464, "hivrt": 2.589, "kif11": 2.157}
constraints = {
    "akt1": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 87]],
            }
        }
    ],
    "ampc": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 62]],
            }
        }
    ],
    "cp3a4": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 274]],
            }
        }
    ],
    "cxcr4": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 131]],
            }
        }
    ],
    "gcr": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 126]],
            }
        }
    ],
    "hivpr": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 25]],
            }
        }
    ],
    "hivrt": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 103]],
            }
        }
    ],
    "kif11": [
        {
            "pocket": {
                "binder": "B",
                "contacts": [["A", 119]],
            }
        }
    ],
}

def getReward(name, seed_smiles="", th=0.6,tgt_file="../akt1_3cqw"):
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
    elif name == "Boltz":
        return BoltzReward()
    elif name == "vina":
        return VinaReward(tgt_file=tgt_file)
    elif name == "RBoltz":
        return RawBoltzReward()
    elif name == "ABoltz":
        return APIBoltzReward()
    elif name == "ASBoltz":
        return APISASBoltzReward()
    elif name == "Svina":
        return SASVinaReward(tgt_file=tgt_file)

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
from ruamel.yaml import YAML
import json
import shutil
from rdkit.Chem import Descriptors
class BoltzReward:
    def __init__(self):
        self.vmin = -3.0
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"
        self.benzene_score = 2.0


    def reward(self, mol):
        self.benzene_score = benzene_scores.get(self.path_to_workspace.split("/")[-1].split("_")[0])
        yaml = YAML()
        yaml.preserve_quotes = True
        
        yaml_path = Path(self.path_to_workspace+"input.yaml")
        data = yaml.load(yaml_path.read_text())

        for entry in data["sequences"]:
            if "ligand" in entry:
                entry["ligand"]["smiles"] = Chem.MolToSmiles(mol)
            else:
                continue
                    
        with yaml_path.open("w") as f:
            yaml.dump(data, f)
        #yaml_path.write_text(yaml.dump(data))

        #env = os.environ.copy()
        #boltz_cache= Path(env["T4TMPDIR"]+"/boltz_cache")
        # if boltz_cache.exists():
        #     shutil.rmtree(boltz_cache)
        #boltz_cache.mkdir(parents=True, exist_ok=True)
        print("boltz:start", flush=True)
        __start = time.perf_counter()
        subprocess.run([f"{self.path_to_boltzinia}.venv/bin/boltz", "predict",self.path_to_workspace+"input.yaml","--override", "--output_format","pdb", "--cache",os.environ.get("T4TMPDIR")+"/boltz_cache", "--out_dir",self.path_to_workspace], cwd = hydra.utils.get_original_cwd())
        # boltz predict yaml --use_msa_server --override --output_format pdb --cache ~ --out_dir ~
        __end = time.perf_counter()
        print(f"boltz:end, consumed: {__end - __start}", flush=True)
        json_path = Path(self.path_to_workspace+"boltz_results_input/predictions/input/affinity_input.json")

        with json_path.open("r") as f:
            results = json.load(f)

        affinity_pred_value = results.get("affinity_pred_value", 0)
        prob_binary = results.get("affinity_probability_binary", 0)

        #score = affinity_pred_value if prob_binary > 0.5 else 0
        #score = -1*affinity_pred_value
        molWt = Descriptors.MolWt(mol)
        #score = torch.sigmoid(-1*torch.tensor(affinity_pred_value +1.5)).item()  if molWt>=500 else torch.sigmoid(-1*torch.tensor(affinity_pred_value)).item()
        #score = -1*affinity_pred_value/3  if molWt>=300 else -1*affinity_pred_value
        benzene_score = self.benzene_score #2.157 # TODO: for KIF11
        score = -1*(affinity_pred_value - benzene_score)/3  if affinity_pred_value>=benzene_score else 0
        return score
import requests
class APISASBoltzReward:
    #TODO: batchedboltz
    def __init__(self):
        self.vmin = -3.0
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"
        self.data = ""
        self.msa_path = ""
        self.benzene_score= 2.0
    
    
    def set_data(self, path_to_workspace):
        yaml = YAML()
        yaml.preserve_quotes = True
    
        yaml_path = Path(path_to_workspace+"input.yaml")
        self.data = yaml.load(yaml_path.read_text())



    def reward(self, mol):
        self.benzene_score = benzene_scores.get(self.path_to_workspace.split("/")[-2].split("_")[0])
        print(self.path_to_workspace.split("/")[-2].split("_")[0])
        if calculateScore(mol) > 3.5:
            return self.vmin
        print(self.path_to_workspace)
        payload = {
            "ligand_smiles": Chem.MolToSmiles(mol),
            "msa_a3m_path": self.path_to_workspace+self.data["sequences"][0]["protein"]["msa"],
            "constraints":constraints.get(self.path_to_workspace.split("/")[-2].split("_")[0]), 
            "job_id": "temporary"+self.path_to_workspace.split("/")[-2]
        }
        print("boltz:start", flush=True)
        __start = time.perf_counter()
        my_res = requests.post(f'http://127.0.0.1:{os.environ.get("API_PORT")}/predict_affinity',json=payload)

        
        __end = time.perf_counter()
        print(f"boltz:end, consumed: {__end - __start}", flush=True)
        #json_path = Path(self.path_to_workspace+"boltz_results_input/predictions/input/affinity_input.json")

        #with json_path.open("r") as f:
        #    results = json.load(f)
        if my_res.status_code == 200:
            affinity_pred_value = my_res.json()["affinity_score"] #results.get("affinity_pred_value", 0)
        else:
            return self.vmin
        #prob_binary = results.get("affinity_probability_binary", 0)

        #score = affinity_pred_value if prob_binary > 0.5 else 0
        #score = -1*affinity_pred_value
        molWt = Descriptors.MolWt(mol)
        #score = torch.sigmoid(-1*torch.tensor(affinity_pred_value +1.5)).item()  if molWt>=500 else torch.sigmoid(-1*torch.tensor(affinity_pred_value)).item()
        #score = -1*affinity_pred_value/3  if molWt>=300 else -1*affinity_pred_value
        benzene_score = self.benzene_score#2.157 # TODO: for KIF11
        score = -1*(affinity_pred_value - benzene_score)/3  if affinity_pred_value<=benzene_score else 0
        return score

class APIBoltzReward:
    #TODO: batchedboltz
    def __init__(self):
        self.vmin = -3.0
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"
        self.data = ""
        self.msa_path = ""
        self.benzene_score = 2.0
    
    
    def set_data(self, path_to_workspace):
        yaml = YAML()
        yaml.preserve_quotes = True
    
        yaml_path = Path(path_to_workspace+"input.yaml")
        self.data = yaml.load(yaml_path.read_text())



    def reward(self, mol):
        self.benzene_score = benzene_scores.get(self.path_to_workspace.split("/")[-1].split("_")[0])
        payload = {
            "ligand_smiles": Chem.MolToSmiles(mol),
            "msa_a3m_path": self.path_to_workspace+self.data["sequences"][0]["protein"]["msa"],
            "job_id": "temporary"+self.path_to_workspace.split("/")[-1]
        }
        print("boltz:start", flush=True)
        __start = time.perf_counter()
        my_res = requests.post(f'http://127.0.0.1:{os.environ.get("API_PORT")}/predict_affinity',json=payload)

        
        __end = time.perf_counter()
        print(f"boltz:end, consumed: {__end - __start}", flush=True)
        #json_path = Path(self.path_to_workspace+"boltz_results_input/predictions/input/affinity_input.json")

        #with json_path.open("r") as f:
        #    results = json.load(f)
        if my_res.status_code == 200:
            affinity_pred_value = my_res.json()["affinity_score"] #results.get("affinity_pred_value", 0)
        else:
            return self.vmin
        #prob_binary = results.get("affinity_probability_binary", 0)

        #score = affinity_pred_value if prob_binary > 0.5 else 0
        #score = -1*affinity_pred_value
        molWt = Descriptors.MolWt(mol)
        #score = torch.sigmoid(-1*torch.tensor(affinity_pred_value +1.5)).item()  if molWt>=500 else torch.sigmoid(-1*torch.tensor(affinity_pred_value)).item()
        #score = -1*affinity_pred_value/3  if molWt>=300 else -1*affinity_pred_value
        benzene_score = self.benzene_score #2.157 # TODO: for KIF11
        score = -1*(affinity_pred_value - benzene_score)/3  if affinity_pred_value<=benzene_score else 0
        return score
class RawBoltzReward:
    def __init__(self):
        self.vmin = -3.0
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"

    def reward(self, mol):
        yaml = YAML()
        yaml.preserve_quotes = True
        
        yaml_path = Path(self.path_to_workspace+"input.yaml")
        data = yaml.load(yaml_path.read_text())

        for entry in data["sequences"]:
            if "ligand" in entry:
                entry["ligand"]["smiles"] = Chem.MolToSmiles(mol)
            else:
                continue
                    
        with yaml_path.open("w") as f:
            yaml.dump(data, f)
        #yaml_path.write_text(yaml.dump(data))

        env = os.environ.copy()
        boltz_cache= Path(env["T4TMPDIR"]+"/boltz_cache")
        # if boltz_cache.exists():
        #     shutil.rmtree(boltz_cache)
        boltz_cache.mkdir(parents=True, exist_ok=True)
        print("boltz:start", flush=True)
        __start = time.perf_counter()
        subprocess.run([f"{self.path_to_boltzinia}.venv/bin/boltz", "predict",self.path_to_workspace+"input.yaml","--use_msa_server","--override", "--output_format","pdb", "--cache",os.environ.get("T4TMPDIR")+"/boltz_cache", "--out_dir",self.path_to_workspace], cwd = hydra.utils.get_original_cwd())
        # boltz predict yaml --use_msa_server --override --output_format pdb --cache ~ --out_dir ~
        __end = time.perf_counter()
        print(f"boltz:end, consumed: {__end - __start}", flush=True)
        json_path = Path(self.path_to_workspace+"boltz_results_input/predictions/input/affinity_input.json")

        with json_path.open("r") as f:
            results = json.load(f)

        affinity_pred_value = results.get("affinity_pred_value", 0)
        prob_binary = results.get("affinity_probability_binary", 0)

        #score = affinity_pred_value if prob_binary > 0.5 else 0
        score = -1*affinity_pred_value
        #molWt = Descriptors.MolWt(mol)
        #score = torch.sigmoid(-1*torch.tensor(affinity_pred_value +1.5)).item()  if molWt>=500 else torch.sigmoid(-1*torch.tensor(affinity_pred_value)).item()
        return score

class VinaReward:
    def __init__(self,tgt_file):
        self.vmin = 0.0
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"
        self.prep = MoleculePreparation()
        self.tgt_file = tgt_file

    def get_all_vina_scores(self,pdbqt_path):
        """
        Extracts all Vina docking scores from a multi-model PDBQT file.
        Returns a list of floats.
        """
        scores = []
        with open(pdbqt_path, 'r') as file:
            for line in file:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try:
                        scores.append(float(parts[3]))
                    except (IndexError, ValueError):
                        continue # Skip malformed lines
        return scores
    def reward(self, mol):
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        
        use_meeko = True  # Meekoでの処理を行うかどうかのフラグ
        out_pdbqt_path = self.path_to_workspace + "tmp_ligand.pdbqt"

        # --- Meekoに渡す前の安全装置 ---
        if mol.GetNumConformers() == 0:
            print("Warning: 3D coordinates not found. Attempting to generate...")
            mol = Chem.AddHs(mol)
            
            # 1. 通常の3D構造生成を試みる
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
            
            # 2. もし失敗（戻り値が-1）したら、ランダム座標からの生成（フォールバック）を試みる
            if res == -1:
                print("Standard embedding failed. Retrying with random coordinates...")
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                
                # 3. それでもダメなら OpenBabel でのPDBQT生成に切り替える (最終フォールバック)
                if res == -1:
                    print("RDKit 3D generation failed. Falling back to OpenBabel...")
                    use_meeko = False  # OpenBabelが書き出しまで行うため、Meekoはスキップする
                    
                    # OpenBabelに渡すためのSMILESファイルを作成
                    tmp_smi_path = self.path_to_workspace + "tmp_ligand.smi"
                    with open(tmp_smi_path, "w") as f:
                        f.write(Chem.MolToSmiles(mol))
                    
                    # OpenBabelコマンドの実行
                    # obabel tmp_ligand.smi -opdbqt -O tmp_ligand.pdbqt -p --gen3D
                    cmd = [
                        "obabel", tmp_smi_path,
                        "-opdbqt",
                        "-O", out_pdbqt_path,
                        "-p",        # 水素付加 (pH指定なしなら~7.4相当)
                        "--gen3D"    # 3D座標生成
                    ]
                    
                    try:
                        # subprocessで実行
                        ob_res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print("OpenBabel successfully generated PDBQT.")
                    except subprocess.CalledProcessError as e:
                        # OpenBabelでも失敗した場合
                        print(f"OpenBabel Error: {e.stderr}")
                        return self.vmin
                    except FileNotFoundError:
                        print("Error: 'obabel' command not found. Is OpenBabel installed and in PATH?")
                        return self.vmin

            # RDKitでの3D生成に成功した場合のみMMFFOptを実行
            if use_meeko:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception as e:
                    print(f"MMFF Optimization skipped/failed: {e}")

        # --- Meekoによる準備と書き出し (RDKitで成功した場合のみ実行) ---
        if use_meeko:
            try:
                mol_list = self.prep.prepare(mol)
                mol_string = PDBQTWriterLegacy.write_string(mol_list[0])

                if mol_string[1] != True:
                    return self.vmin

                with open(out_pdbqt_path, "w") as f:
                    f.write(mol_string[0])
            except Exception as e:
                print(f"Meeko preparation failed: {e}")
                return self.vmin
        
        try:
            subprocess.run(["apptainer","run","--nv","--bind",".:/mnt","--bind","/gs:/gs","--pwd","/mnt",f"{os.environ.get('SHARED')}/sandbox/3vinacont" ,"--receptor",f"{self.path_to_workspace}{self.tgt_file}.pdbqt","--ligand",f"{self.path_to_workspace}tmp_ligand.pdbqt","--config",f"{self.path_to_workspace}{self.tgt_file}.box.txt","--out",f"{self.path_to_workspace}tmp_docked.pdbqt","--thread","8192"], cwd = hydra.utils.get_original_cwd())
        except Exception as e:
            print(Chem.MolToSmiles(mol))
            print(e)
            raise 
        scores = self.get_all_vina_scores(f"{self.path_to_workspace}tmp_docked.pdbqt")
        score = min(scores)
        return -1*score/10

class SASVinaReward:
    def __init__(self,tgt_file):
        self.vmin = 0.0
        self.path_to_workspace = hydra.utils.get_original_cwd()+"/wkdir/"
        self.path_to_boltzinia = os.environ.get("HOME")+"/big_shared/GARGOYLES_boltz/boltzina/"
        self.prep = MoleculePreparation()
        self.tgt_file = tgt_file

    def get_all_vina_scores(self,pdbqt_path):
        """
        Extracts all Vina docking scores from a multi-model PDBQT file.
        Returns a list of floats.
        """
        scores = []
        with open(pdbqt_path, 'r') as file:
            for line in file:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    try:
                        scores.append(float(parts[3]))
                    except (IndexError, ValueError):
                        continue # Skip malformed lines
        return scores
    def reward(self, mol):
        if calculateScore(mol) > 3.5:
            return self.vmin
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        
        use_meeko = True  # Meekoでの処理を行うかどうかのフラグ
        out_pdbqt_path = self.path_to_workspace + "tmp_ligand.pdbqt"

        # --- Meekoに渡す前の安全装置 ---
        if mol.GetNumConformers() == 0:
            print("Warning: 3D coordinates not found. Attempting to generate...")
            mol = Chem.AddHs(mol)
            
            # 1. 通常の3D構造生成を試みる
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
            
            # 2. もし失敗（戻り値が-1）したら、ランダム座標からの生成（フォールバック）を試みる
            if res == -1:
                print("Standard embedding failed. Retrying with random coordinates...")
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                
                # 3. それでもダメなら OpenBabel でのPDBQT生成に切り替える (最終フォールバック)
                if res == -1:
                    print("RDKit 3D generation failed. Falling back to OpenBabel...")
                    use_meeko = False  # OpenBabelが書き出しまで行うため、Meekoはスキップする
                    
                    # OpenBabelに渡すためのSMILESファイルを作成
                    tmp_smi_path = self.path_to_workspace + "tmp_ligand.smi"
                    with open(tmp_smi_path, "w") as f:
                        f.write(Chem.MolToSmiles(mol))
                    
                    # OpenBabelコマンドの実行
                    # obabel tmp_ligand.smi -opdbqt -O tmp_ligand.pdbqt -p --gen3D
                    cmd = [
                        "obabel", tmp_smi_path,
                        "-opdbqt",
                        "-O", out_pdbqt_path,
                        "-p",        # 水素付加 (pH指定なしなら~7.4相当)
                        "--gen3D"    # 3D座標生成
                    ]
                    
                    try:
                        # subprocessで実行
                        ob_res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print("OpenBabel successfully generated PDBQT.")
                    except subprocess.CalledProcessError as e:
                        # OpenBabelでも失敗した場合
                        print(f"OpenBabel Error: {e.stderr}")
                        return self.vmin
                    except FileNotFoundError:
                        print("Error: 'obabel' command not found. Is OpenBabel installed and in PATH?")
                        return self.vmin

            # RDKitでの3D生成に成功した場合のみMMFFOptを実行
            if use_meeko:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception as e:
                    print(f"MMFF Optimization skipped/failed: {e}")

        # --- Meekoによる準備と書き出し (RDKitで成功した場合のみ実行) ---
        if use_meeko:
            try:
                mol_list = self.prep.prepare(mol)
                mol_string = PDBQTWriterLegacy.write_string(mol_list[0])

                if mol_string[1] != True:
                    return self.vmin

                with open(out_pdbqt_path, "w") as f:
                    f.write(mol_string[0])
            except Exception as e:
                print(f"Meeko preparation failed: {e}")
                return self.vmin
        
        try:
            subprocess.run(["apptainer","run","--nv","--bind",".:/mnt","--bind","/gs:/gs","--pwd","/mnt",f"{os.environ.get('SHARED')}/sandbox/3vinacont" ,"--receptor",f"{self.path_to_workspace}{self.tgt_file}.pdbqt","--ligand",f"{self.path_to_workspace}tmp_ligand.pdbqt","--config",f"{self.path_to_workspace}{self.tgt_file}.box.txt","--out",f"{self.path_to_workspace}tmp_docked.pdbqt","--thread","8192"], cwd = hydra.utils.get_original_cwd())
        except Exception as e:
            print(Chem.MolToSmiles(mol))
            print(e)
            raise 
        scores = self.get_all_vina_scores(f"{self.path_to_workspace}tmp_docked.pdbqt")
        score = min(scores)
        return -1*score/10
