import torch
from torch import nn
from torch.nn import functional as F
from . import EvoPair,EvoMSA


class EvoBlock(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(EvoBlock,self).__init__()
        self.msa_row=EvoMSA.MSARow(m_dim,z_dim)
        self.msa_col=EvoMSA.MSACol(m_dim)
        self.msa_trans=EvoMSA.MSATrans(m_dim)

        self.msa_opm=EvoMSA.MSAOPM(m_dim,z_dim)

        self.pair_triout=EvoPair.TriOut(z_dim)
        self.pair_triin =EvoPair.TriIn(z_dim)
        self.pair_tristart=EvoPair.TriAttStart(z_dim)
        self.pair_triend  =EvoPair.TriAttEnd(z_dim)
        self.pair_trans = EvoPair.PairTrans(z_dim)
        self.docheck=docheck


    def layerfunc_msa_row(self,m,z):
        return self.msa_row(m,z) + m
    def layerfunc_msa_col(self,m):
        return self.msa_col(m) + m
    def layerfunc_msa_trans(self,m):
        return self.msa_trans(m) + m
    def layerfunc_msa_opm(self,m,z):
        return self.msa_opm(m) + z

    def layerfunc_pair_triout(self,z):
        return self.pair_triout(z) + z
    def layerfunc_pair_triin(self,z):
        return self.pair_triin(z) + z
    def layerfunc_pair_tristart(self,z):
        return self.pair_tristart(z) + z
    def layerfunc_pair_triend(self,z):
        return self.pair_triend(z) + z      
    def layerfunc_pair_trans(self,z):
        return self.pair_trans(z) + z  
    def forward(self,m,z):

        m = m + self.msa_row(m,z)
        m = m + self.msa_col(m)
        m = m + self.msa_trans(m)
        z = z + self.msa_opm(m)
        z = z + self.pair_triout(z)
        z = z + self.pair_triin(z)
        z = z + self.pair_tristart(z)
        z = z + self.pair_triend(z)
        z = z + self.pair_trans(z)
        return m,z



class Evoformer(nn.Module):
    def __init__(self,m_dim,z_dim,docheck=False):
        super(Evoformer,self).__init__()
        # self.layers=[48]
        self.layers=[2]
        self.docheck=docheck
        if docheck:
            pass
            #print('will do checkpoint')
        self.evos=nn.ModuleList([EvoBlock(m_dim,z_dim,True) for i in range(self.layers[0])])

    def layerfunc(self,layermodule,m,z):
        m_,z_=layermodule(m,z)
        return m_,z_


    def forward(self,m,z):
        for i in range(self.layers[0]):
            m,z=self.evos[i](m,z)

        return m,z









