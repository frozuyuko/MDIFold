import pandas as pd 
import numpy as np
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from module import molecularfingerprint, NumpyE2ELoader, FeatureEncoders, PositionalEncoding_1D_Modified, PositionalEncoding_2D_Modified
from module import CompletePosEncode
from rdkit import Chem
from rdkit.Chem import AllChem 
from DeepE2EPotential.Evoformer import Evoformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class mirnadrugdata(Dataset): #modify to duplicate E2E and other features 
    def __init__(self, FilePath):
        self.data = pd.read_csv(FilePath)[['smiles','score','miRBase','sequence']]
        self.scores = []
        self.miRBase = []
        self.sequences = []
        self.smiles, self.mols, self.fps = [],[],[]
        self.rnae2e_m1, self.rnae2e_z = [],[]
        self.rnallm, self.rnass = [],[]
        for idx,i in enumerate(tqdm(self.data.itertuples())):
            if np.isnan(i.score): continue
            mol = molecularfingerprint(i.smiles)
            if mol.mol is None: continue
            self.fps.append([int(i) for i in mol.fingerprint])
            self.scores.append(i.score)
            self.sequences.append(i.sequence)
            self.miRBase.append(i.miRBase)
            folder_secondary = './MDIFold/CompiledRNA/SecondaryStructure/Hairpin/'
            rna_ss = np.load(os.path.join(folder_secondary,f'output_{idx}.npy'))
            self.rnass.append(rna_ss)
            # folder_path = '/home/project/11003846/febrina/MirSmiPred/dataset/compiled_dataset/RNA/EndtoEndHairpin/'
            folder_path = './MDIFold/CompiledRNA/EndtoEnd/Hairpin/'
            rnae2e_m1_data = np.load(os.path.join(folder_path, f'm1/pred_{idx}_m1_pre_6.npy'))
            rnae2e_z_data = np.load(os.path.join(folder_path, f'z/pred_{idx}_z_pre_6.npy'))
            self.rnae2e_m1.append(rnae2e_m1_data)
            self.rnae2e_z.append(rnae2e_z_data) 
            folder_rnallm = './MDIFold/RNA-FM/HairpinEmbed/' 
            rna_llm = np.load(os.path.join(folder_rnallm,f'miRNALLM_{idx}.npy'))
            self.rnallm.append(rna_llm) 
            self.vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(1,3),max_features=84)
            self.vectorizer.fit(self.sequences)                             
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, index):
        return {
                'fp':self.fps[index],
                'rna_tfidf':self.vectorizer.transform([self.sequences[index]]).toarray()[0],
                'rnae2e_m1':self.rnae2e_m1[index],
                'rnae2e_z':self.rnae2e_z[index],
                'rna_llm':self.rnallm[index],
                'rna_ss':self.rnass[index],
                'score':self.scores[index]}

class mirnadrugdatanostructure(Dataset): #modify to duplicate E2E and other features 
    def __init__(self, FilePath):
        self.data = pd.read_csv(FilePath)[['smiles','score','miRBase','sequence']]
        self.scores = []
        self.miRBase = []
        self.sequences = []
        self.smiles, self.mols, self.fps = [],[],[]
        self.rnae2e_m1, self.rnae2e_z = [],[]
        for idx,i in enumerate(tqdm(self.data.itertuples())):
            if np.isnan(i.score): continue
            mol = molecularfingerprint(i.smiles)
            if mol.mol is None: continue
            self.fps.append([int(i) for i in mol.fingerprint])
            self.scores.append(i.score)
            self.sequences.append(i.sequence)
            self.miRBase.append(i.miRBase)  
            self.vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(1,3),max_features=84)
            self.vectorizer.fit(self.sequences)                             
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, index):
        return {
                'fp':self.fps[index],
                'rna_tfidf':self.vectorizer.transform([self.sequences[index]]).toarray()[0],
                'score':self.scores[index]}

class DataCollater:
    def __init__(self, rnaMaxLen=100):
        self.rnaMaxLen = rnaMaxLen
    def __call__(self, data):
        fp = [i['fp'] for i in data]
        score = [i['score'] for i in data]
        rna_tfidf = [i['rna_tfidf'] for i in data]
        rnae2e_m1 = [i['rnae2e_m1'] for i in data]
        rnae2e_z = [i['rnae2e_z'] for i in data]
        rna_llm = [i['rna_llm'] for i in data]
        rna_ss = [i['rna_ss'] for i in data]
        return {
            'fp': torch.tensor(fp, dtype=torch.float32),
            'rnae2e_m1':torch.tensor(rnae2e_m1, dtype=torch.float32),
            'rna_tfidf':torch.tensor(rna_tfidf, dtype=torch.float32),
            'rnae2e_z':torch.tensor(rnae2e_z, dtype=torch.float32),
            'rna_llm':torch.tensor(rna_llm, dtype=torch.float32),
            'rna_ss':torch.tensor(rna_ss, dtype=torch.float32),
            'score':torch.tensor(score, dtype=torch.float32)}
    
class Baseline(nn.Module):
    def __init__(self, embedding_size):
        super(Baseline, self).__init__()
        self.ln = (nn.LayerNorm(embedding_size),nn.ReLU())
        self.e2e = nn.Sequential(nn.LayerNorm(64),nn.Linear(64,embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.EvoFormer = Evoformer(embedding_size,embedding_size,docheck=False)
        # ss = L x 4
        # m1 = L x 64 --> m1 = L x 4
        # z = L x L x  64 --> z = L x L x 4
        # llm = L x 640 --> llm = L x 4
        self.fingerprint = nn.Sequential(
            nn.Linear(1024,embedding_size),nn.ReLU(),nn.LayerNorm(embedding_size),nn.Linear(embedding_size,embedding_size),nn.ReLU(),
            nn.Dropout(0.3),nn.ReLU())
        self.olinear = nn.Linear(embedding_size, 1)
        self.rna_tfidf = nn.Sequential(nn.LayerNorm(84),nn.Linear(84,embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU(),
            nn.Dropout(0.3),nn.ReLU())
        self.rna_llm = nn.Sequential(nn.LayerNorm(640),nn.Linear(640,embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU(),
            nn.Dropout(0.3),nn.ReLU())
        self.rna_ss = nn.Sequential(nn.LayerNorm(embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU(),nn.Dropout(0.3),nn.ReLU())
        self.output = nn.Sequential(nn.Linear(embedding_size*2,1))
    def forward(self, data):
        # x_rnakmer = self.rna_tfidf(data['rna_tfidf'])
        x_molfp = self.fingerprint(data['fp'])
        x_rnass = self.rna_ss(data['rna_ss'])
        rnae2e_m1 = self.e2e(data['rnae2e_m1'])
        rnae2e_z = self.e2e(data['rnae2e_z'])
        x_rnallm = self.rna_llm(data['rna_llm'])
        L1 = rnae2e_z.shape[1]
        max_len = 200
        pos = CompletePosEncode(max_len=max_len)
        pos_1d = pos.pos_1d(length=L1, s_dim=4, device=torch.device('cpu'))
        pos_2d = pos.pos_2d(length=L1, s_dim=4, device=torch.device('cpu'))
        rnae2e_m1pos = rnae2e_m1 + pos_1d.to('cuda:0')
        rnae2e_zpos = rnae2e_z[0] + pos_2d.to('cuda:0')
        x_rnae2e_m1, x_rnae2e_z = self.EvoFormer(rnae2e_m1pos, rnae2e_zpos)
        x_rnass = torch.mean(x_rnass, dim=1)
        # x_rna = torch.cat((x_rnae2e_m1[0].mean(dim=0),x_rnallm.mean(dim=0),x_rnass[0].mean(dim=0)),dim=0)
        # x_rna = torch.mean((x_rna[0]),dim=0)
        x_rna = x_rnae2e_m1[0].mean(dim=0) + x_rnallm[0].mean(dim=0) + x_rnass[0].mean(dim=0)
        x = torch.cat([x_molfp[0],x_rna], dim=0)
        x = self.output(x)
        return {'y_logit':x}


class BaselineEval(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineEval, self).__init__()
        self.ln = (nn.LayerNorm(embedding_size),nn.ReLU())
        self.e2e = nn.Sequential(nn.LayerNorm(64),nn.Linear(64,embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.EvoFormer = Evoformer(embedding_size,embedding_size,docheck=False)
        # ss = L x 4
        # m1 = L x 64 --> m1 = L x 4
        # z = L x L x  64 --> z = L x L x 4
        # llm = L x 640 --> llm = L x 4
        self.fingerprint = nn.Sequential(
            nn.Linear(1024,embedding_size),nn.ReLU(),nn.LayerNorm(embedding_size),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.olinear = nn.Linear(embedding_size, 1)
        self.rna_tfidf = nn.Sequential(nn.LayerNorm(84),nn.Linear(84,embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.rna_llm = nn.Sequential(nn.LayerNorm(640),nn.Linear(640,embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.rna_ss = nn.Sequential(nn.LayerNorm(embedding_size),nn.ReLU(),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.output = nn.Sequential(nn.Linear(embedding_size*2,1))
    def forward(self, data):
        # x_rnakmer = self.rna_tfidf(data['rna_tfidf'])
        x_molfp = self.fingerprint(data['fp'])
        x_rnass = self.rna_ss(data['rna_ss'])
        rnae2e_m1 = self.e2e(data['rnae2e_m1'])
        rnae2e_z = self.e2e(data['rnae2e_z'])
        x_rnallm = self.rna_llm(data['rna_llm'])
        L1 = rnae2e_z.shape[1]
        max_len = 200
        pos = CompletePosEncode(max_len=max_len)
        pos_1d = pos.pos_1d(length=L1, s_dim=4, device=torch.device('cpu'))
        pos_2d = pos.pos_2d(length=L1, s_dim=4, device=torch.device('cpu'))
        rnae2e_m1pos = rnae2e_m1 + pos_1d.to('cuda:0')
        rnae2e_zpos = rnae2e_z[0] + pos_2d.to('cuda:0')
        x_rnae2e_m1, x_rnae2e_z = self.EvoFormer(rnae2e_m1pos, rnae2e_zpos)
        x_rnass = torch.mean(x_rnass, dim=1)
        x_rna = x_rnae2e_m1[0].mean(dim=0) + x_rnallm[0].mean(dim=0) + x_rnass[0].mean(dim=0)
        x = torch.cat([x_molfp[0],x_rna], dim=0)
        x = self.output(x)
        return {'y_logit':x} 

class BaselineNoStructure(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineNoStructure, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.fingerprint = nn.Sequential(
            nn.Linear(1024,embedding_size),nn.ReLU(),nn.LayerNorm(embedding_size),nn.Linear(embedding_size,embedding_size),nn.ReLU(),
            nn.Dropout(0.3),nn.ReLU())
        self.olinear = nn.Linear(embedding_size, 1)
        self.rna_tfidf = nn.Sequential(
            nn.LayerNorm(84),nn.Linear(84,embedding_size),nn.ReLU(),
            nn.Linear(embedding_size,embedding_size),nn.ReLU(),
            nn.Dropout(0.3),nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(embedding_size*2,1))
    def forward(self, data):
        x_molfp = self.fingerprint(data['fp'])
        x_rnakmer = self.rna_tfidf(data['rna_tfidf'])
        x = torch.cat([x_molfp[0],x_rnakmer[0]], dim=0)
        x = self.output(x)
        return {'y_logit':x} 

class BaselineNoStructureEval(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineNoStructureEval, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.fingerprint = nn.Sequential(
            nn.Linear(1024,embedding_size),nn.ReLU(),nn.LayerNorm(embedding_size),nn.Linear(embedding_size,embedding_size),nn.ReLU())
        self.olinear = nn.Linear(embedding_size, 1)
        self.rna_tfidf = nn.Sequential(
            nn.LayerNorm(84),
            nn.Linear(84,embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size,embedding_size),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_size*2,1))
    def forward(self, data):
        x_molfp = self.fingerprint(data['fp'])
        x_rnakmer = self.rna_tfidf(data['rna_tfidf'])
        x = torch.cat([x_molfp[0],x_rnakmer[0]], dim=0)
        x = self.output(x)
        return {'y_logit':x} 
    
