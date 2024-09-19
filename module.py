import pandas as pd
import torch 
import torch.nn as nn
import math 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import os, sys

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
        
