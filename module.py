import pandas as pd
import torch 
import torch.nn as nn
import math 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
import os, sys
from tqdm import tqdm
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect, GetHashedTopologicalTorsionFingerprintAsBitVect
import deepchem as dc 
from transformers import AutoTokenizer, AutoModelForMaskedLM

class molecularfingerprint:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles)
        if self.mol is not None:  
            self.fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(self.mol, radius=2, nBits=1024)
            self.fingerprint = self.fingerprint.ToBitString()
            self.fingerprint = [int(i) for i in self.fingerprint]
    def get_fingerprint(self):
        return self.fingerprint
    def get_image(self,idx):
        return Draw.MolToImage(self.mol[idx])        

class MolecularFingerprint:
    def __init__(self, smiles):
        if not isinstance(smiles, str):
            self.mols = None
        else:
            self.smiles = smiles
            self.mols = Chem.MolFromSmiles(smiles)
            if self.mols:
                self.natoms = self.mols.GetNumAtoms()
            else:
                self.natoms = 0
        # cdktypes = ['maccs', 'pubchem', 'klekota-roth', 'shortestpath', 'cdk-substructure', 'circular', 'cdk-atompairs']
        # rdktypes = ['rdkit', 'morgan', 'rdk-maccs', 'topological-torsion', 'avalon', 'atom-pair', 'rdk-descriptor']
        # babeltypes = ['fp2', 'fp3', 'fp4', 'spectrophore']
        # vectypes = ['mol2vec']
    def get_MACCS(self):
        self.maccs_fingerprint = MACCSkeys.GenMACCSKeys(self.mols)
        self.maccs_fingerprint = np.array([int(i) for i in self.maccs_fingerprint])
        return self.maccs_fingerprint
    def get_MorganFingerprint(self):
        self.morganfp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(self.mols, radius=2, nBits=1024)
        self.morganfp = self.morganfp.ToBitString()
        self.morganfp = np.array([int(i) for i in self.morganfp])
        return self.morganfp
    def get_RDKFingerprint(self):
        self.rdkfp = Chem.RDKFingerprint(self.mols, fpSize=64, nBitsPerHash=1)
        # self.numbits = self.rdkfp.GetNumOnBits()
        self.rdkfp = self.rdkfp.ToBitString()
        self.rdkfp = np.array([int(i) for i in self.rdkfp])
        return self.rdkfp
    def get_ECFP(self): 
        self.ecfp = get_fingerprint(self.smiles,'extended').to_numpy()
        return self.ecfp
    def get_PubchemFingerprint(self):
        self.pubchemfp = get_fingerprint(self.smiles, 'pubchem').to_numpy()
        return self.pubchemfp
    def get_KlekotaFingerprint(self):
        self.klekotafp = get_fingerprint(self.smiles, 'klekota-roth').to_numpy()
        return self.klekotafp
    def get_mol2vec(self): 
        self.mol2vec = get_fingerprint(self.smiles,'mol2vec').to_numpy()
        return self.mol2vec
    def get_RDKDesc(self): 
        self.getrdkdesc = get_fingerprint(self.smiles,'rdk-descriptor').to_numpy()
        return self.getrdkdesc
    # def get_PubchemFingerprint(self): #source: https://chem.libretexts.org/Courses/Intercollegiate_Courses/Cheminformatics/06%3A_Molecular_Similarity/6.04%3A_Python_Assignment
    #     padder = len(self.smiles) % 4
    #     if padder > 0:
    #       self.smiles += '=' * (4 - padder)
    #     self.pcfp = "".join(f"{decode:08b}" for decode in b64decode(self.smiles))
    #     self.pcfp = [int(i) for i in self.pcfp[32:913]]
    #     return self.pcfp #shape differ according to length --> maybe project it to conv layer? then baru mean gituu
    def get_CoulombMat(self):
        AllChem.EmbedMolecule(self.mols)
        # Chem.AddHs(self.mols) #this one better add Hs or not??
        self.coulombfeat = dc.feat.CoulombMatrixEig(max_atoms=100, remove_hydrogens=False) ## For now i think better to remove for faster computation
        self.coulombfeat = self.coulombfeat._featurize(self.mols)
        return self.coulombfeat

class GraphMolecule:
    def __init__(self, smiles):
        self.smiles = smiles
    def molgraphconv(self):
        # self.mols = Chem.MolFromSmiles(smiles)
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        self.molconv = featurizer.featurize(self.smiles)
        return self.molconv
    def goverfeat(self): 
        featurizer = GroverFeaturizer(features_generator = dc.feat.CircularFingerprint())
        self.molgover = featurizer.featurize(self.smiles)

class ChemBerta: 
    def __init__(self, smiles):
        self.smiles = smiles 
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    def getembed(self): 
        inputs = self.tokenizer(self.smiles, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, output_attentions=True)
        return outputs
            
class NumpyE2ELoader: 
    def __init__(self):
        self.rnae2e_m1, self.rnae2e_z = [],[]
        self.rnadict = pd.read_csv('/home/project/11003323/febrina/MirSmiPred/dataset/compiled_dataset/rnaidtosequence.csv')
    def forward(self):
        folder_path = '/home/project/11003323/febrina/MirSmiPred/dataset/endtoend/E2Esm2mir3'
        for i, row in self.rnadict.iterrows():
            rnae2e_m1 = np.load(os.path.join(folder_path, f'pred_{row.miRBase}_m1_pre_6.npy'))
            rnae2e_z = np.load(os.path.join(folder_path, f'pred_{row.miRBase}_z_pre_6.npy'))
        self.rnae2e_m1.append(rnae2e_m1)
        self.rnae2e_z.append(rnae2e_z)
        return self.rnae2e_m1, self.rnae2e_z

class CompletePosEncode:
    def __init__(self, max_len=1024):
        self.max_len = max_len

    def one_d(self, idx_, d):
        idx = idx_[None]
        K = torch.arange(d//2).to(idx.device)
        sin_e = torch.sin(idx[..., None] * math.pi / (self.max_len**(2*K[None]/d))).to(idx.device)
        cos_e = torch.cos(idx[..., None] * math.pi / (self.max_len**(2*K[None]/d))).to(idx.device)
        return torch.cat([sin_e, cos_e], axis=-1)[0]

    def pos_1d(self, length, s_dim, device):
        idx = torch.arange(length, device=device)
        idx = self.one_d(idx, s_dim)
        return idx

    def pos_2d(self, length, s_dim, device):
        seq_idx = torch.arange(length, device=device)[None]
        relative_pos = seq_idx[:, :, None] - seq_idx[:, None, :]
        relative_pos = relative_pos.reshape([1, length**2])
        relative_pos = self.one_d(relative_pos, s_dim)  
        return relative_pos.reshape([1, length, length, -1])[0]
        
