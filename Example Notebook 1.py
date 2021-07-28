
# coding: utf-8

# In[3]:


import fisher
        
from numpy.linalg.linalg import _fastCopyAndTranspose
import math
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb

#Noise curves for 2019-2020 servay from Srini
d = np.load('spt3g_winter_2020_ilc_cmb_90-150-220_TT-EE.npy', allow_pickle=True)
dd = d.item()

        


# In[2]:


#def __init__(self, Gmu, CLTT_String_File, CLTE_String_File,CLBB_String_File,CLEE_String_File, TT,TE,BB, EE )
fisher12= FisherCalculator(10**(-7),'cl_tt120.d','cl_te120.d','cl_bb120.d','cl_ee120.d',True, False, False, False)


# In[5]:


fisher12.Gmu_variation()


# In[6]:


#Fisher_matrix(self, noise_coefficient, noiseTT, noiseEE,noiseBB, fsky,fsky_coefficient)
fisher12.Fisher_matrix(1,dd['cl_residual']['TT'], dd['cl_residual']['EE'],dd['cl_residual']['EE'],1500 / 41253,1)

