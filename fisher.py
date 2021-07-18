
# coding: utf-8

# In[ ]:


#The libraries needed to run the code and camb. 
from numpy.linalg.linalg import _fastCopyAndTranspose
import math
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb







class FisherCalculator:
    def __init__(self, Gmu, CLTT_String_File, CLTE_String_File,CLBB_String_File,CLEE_String_File ):
        '''
        Constructor. This function calculates the cl_matrices for fiducial values and for each varied cosmological parameter
        excluding Gmu. 
        
        Parameters: 
        Gmu(int): the value of Gmu assumed. 
        CLTT_String_File(file): TT string power spectrum with some nexp value.
        CLTE_String_File(file): TE string power spectrum with some nexp value.
        CLBB_String_File(file): BB string power spectrum with some nexp value.
        CLEE_String_File(file): EE string power spectrum with some nexp value.
         
        
        '''
        #String Data
        self.l, self.yt, self.zt, self.tt = np.loadtxt(CLTT_String_File, unpack = True)
        self.l, self.yte, self.zte, self.tte = np.loadtxt(CLTE_String_File, unpack = True)
        self.l, self.yb, self.zb = np.loadtxt(CLBB_String_File, unpack = True)
        self.l, self.ye, self.ze, self.te = np.loadtxt(CLEE_String_File, unpack = True)


        #Set up a new set of parameters for CAMB
        cos_dict =   {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122,'mnu': 0.06,'omk': 0 , 'tau': 0.06,'As':2.10058296e-9, 'ns':0.96605}
        H0_dict =    {'H0': 67.5*1.01, 'ombh2':0.022, 'omch2':0.122, 'mnu': 0.06,'omk': 0, 'tau': 0.06, 'As':2.10058296e-9, 'ns':0.96605}
        ombh2_dict = {'H0': 67.5, 'ombh2':0.022*1.01, 'omch2':0.122,'mnu': 0.06,'omk': 0, 'tau': 0.06, 'As':2.10058296e-9, 'ns':0.96605}
        omch2_dict=  {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122*1.01, 'mnu': 0.06,'omk': 0, 'tau': 0.06,'As':2.10058296e-9, 'ns':0.96605}
        As_dict =    {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122,'mnu': 0.06,'omk': 0 , 'tau': 0.06,'As':2.10058296e-9*1.01, 'ns':0.96605}
        ns_dict =    {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122,'mnu': 0.06,'omk': 0 , 'tau': 0.06,'As':2.10058296e-9, 'ns':0.96605*1.01}
        tau_dict =   {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122, 'mnu': 0.06,'omk': 0, 'tau': 0.06*1.01,'As':2.10058296e-9, 'ns':0.96605}
        parameters_dict= {'H0_value':H0_dict,'ombh2_value':ombh2_dict,'omch2_value':omch2_dict, 'As_value':As_dict, 'ns_value':ns_dict, 'tau_value':tau_dict,'fiducial_value': cos_dict}
        key_list = list(parameters_dict.keys())
        val_list = list(parameters_dict.values())
        self.Parameters_CLmatrix = {}
        self.Gmu = Gmu
        for dictionary in parameters_dict.values():
            cl_matrix = []
            cp = camb.set_params(**dictionary)
            cp.set_for_lmax(10000, lens_potential_accuracy=0)
            results = camb.get_results(cp)
            powers = results.get_cmb_power_spectra(cp, CMB_unit='muK')
            totCL=powers['total']
            self.ls = np.arange(totCL.shape[0])
            cl_t = totCL[:,0]
            cl_e =totCL[:,1]
            cl_b = totCL[:,2]
            cl_te = totCL[:,3]
     #calculating A in terms of Gmu
            SigmaTot_tt2 = 0
            SigmaStr_tt2 = 0            
            for i in range(2,len(self.yt)):
                SigmaTot_tt2 += ((2*i+1)/(4*np.pi))* ((cl_t[i] *(2*np.pi))/(i*(i+1)))
                SigmaStr_tt2 += ((2*i+1)/(4*np.pi))* ( (self.yt[i]+ self.zt[i]+self.tt[i])*((2*np.pi))/(i*(i+1)))
            A = (self.Gmu)**2 * SigmaTot_tt2 *(1.27*10**(-6))**(-2) *(SigmaStr_tt2)**(-1)
    #Calculating Cl_matrices for varied cosmological parameters and multipole values
            for i in self.ls[ (self.ls < len(cl_t))& (self.ls <len(self.yt))& (self.ls > 1)]:
                Cl_11 = A*(self.yt[i]+ self.zt[i]+self.tt[i])*(2*np.pi)/(i*(i+1)) + ((cl_t[i]*(2*np.pi))/(i*(i+1)))
                Cl_12 = A*(self.yte[i]+ self.zte[i]+self.te[i])*(2*np.pi)/(i*(i+1))+((cl_te[i]*(2*np.pi))/(i*(i+1)))
                Cl_22 = A*(self.ye[i]+ self.ze[i]+self.te[i])*(2*np.pi)/(i*(i+1)) + ((cl_e[i]*(2*np.pi))/(i*(i+1)))
                Cl_33 = ((cl_b[i]*(2*np.pi))/(i*(i+1))) + A*(self.yb[i]+ self.zb[i])*(2*np.pi)/(i*(i+1))
                c_l   = np.matrix([[Cl_11,  Cl_12, 0],[ Cl_12, Cl_22, 0], [0,0,Cl_33 ]])
                cl_matrix.insert(len(cl_matrix)-1,c_l)
            self.Parameters_CLmatrix[key_list[val_list.index(dictionary)]] = cl_matrix
    
    
    def Gmu_variation(self):
        ''' The function calculates Cl Matrices for different multipole values and varied Gmu.
        
        Adds the resulting CL matrices to the dictionary of all cosmological parameters and their CL matrices.
        '''
        cl_matrix = []
        cos_dict = {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122,'mnu': 0.06,'omk': 0 , 'tau': 0.06,'As':2.10058296e-9, 'ns':0.96605}
        cp = camb.set_params(**cos_dict)
        cp.set_for_lmax(10000, lens_potential_accuracy=0)
        results = camb.get_results(cp)
        powers = results.get_cmb_power_spectra(cp, CMB_unit='muK')
        totCL=powers['total']
        self.ls = np.arange(totCL.shape[0])
        cl_t = totCL[:,0]
        cl_e =totCL[:,1]
        cl_b = totCL[:,2]
        cl_te = totCL[:,3]
        SigmaTot_tt2 = 0
        SigmaStr_tt2 = 0 
        for i in range(2,len(self.yt)):
            SigmaTot_tt2 += ((2*i+1)/(4*np.pi))* ((cl_t[i] *(2*np.pi))/(i*(i+1)))
            SigmaStr_tt2 += ((2*i+1)/(4*np.pi))* ( (self.yt[i]+ self.zt[i]+self.tt[i])*((2*np.pi))/(i*(i+1)))
        A = (self.Gmu*1.01)**2 * SigmaTot_tt2 *(1.27*10**(-6))**(-2) *(SigmaStr_tt2)**(-1)
        for i in self.ls[ (self.ls < len(cl_t))& (self.ls <len(self.yt)) & self.ls > 1]:
            Cl_11 = A*(self.yt[i]+ self.zt[i]+self.tt[i])*(2*np.pi)/(i*(i+1)) + ((cl_t[i]*(2*np.pi))/(i*(i+1)))
            Cl_12 = A*(self.yte[i]+ self.zte[i]+self.te[i])*(2*np.pi)/(i*(i+1))+((cl_te[i]*(2*np.pi))/(i*(i+1)))
            Cl_22 = A*(self.ye[i]+ self.ze[i]+self.te[i])*(2*np.pi)/(i*(i+1)) + ((cl_e[i]*(2*np.pi))/(i*(i+1)))
            Cl_33 = ((cl_b[i]*(2*np.pi))/(i*(i+1))) + A*(self.yb[i]+ self.zb[i])*(2*np.pi)/(i*(i+1))
            c_l   = np.matrix([[Cl_11,  Cl_12, 0],[ Cl_12, Cl_22, 0], [0,0,Cl_33 ]])
            cl_matrix.insert(len(cl_matrix)-1,c_l)
        self.Parameters_CLmatrix["Gmu_value"] = cl_matrix
         

        
    
    def Fisher_matrix(self, noise_coefficient, noiseTT, noiseEE, noiseBB, fsky,fsky_coefficient):
        '''
           The function addes noise to each CL_matrix for each variable. 
           Calculates the fisher matrix and the uncertainties on each parameter. 
           
           Parameters: 
           
           noise_coefficient(int): the coefficient of the noise curves.
           noiseTT(list): Noise curves for the TT power spectrum. 
           noiseEE(list): Noise curves for the EE power spectrum. 
           fsky(float): the fraction of the sky we are examining. 
           fsky_coefficient(int): the coefficient of fsky.
           
           returns: 
           dict: a dictionary that contains all the uncertainties on cosmological parameters. 
           
        '''
       
        #Using SPT 2020 data, noiseEEE = noiseBBB is a relatively good approximation. 
        noiseTTT = noise_coefficient*noiseTT
        noiseEEE = noise_coefficient*noiseEE
        noiseBBB = noise_coefficient*noiseBB
        for matrix in self.Parameters_CLmatrix.values(): 
            for j in range(len(matrix)): 
                matrix[j][0,0]+= noiseTTT[j]
                matrix[j][1,1]+= noiseEEE[j]
                matrix[j][2,2]+= noiseBBB[j]
        
        cl_dict_values = {"H0_value": 67.5, "ombh2_value": 0.022  ,"omch2_value": 0.122, "As_value":2.10058296e-9, "ns_value":0.96605, "tau_value":0.06, "Gmu_value":1*10**(-7)} 
        variables = ["H0_value","ombh2_value","omch2_value","As_value","ns_value","tau_value","Gmu_value", "fiducial_value"]
        self.f_matrix1 = np.zeros((7,7))
        fsky1 = fsky_coefficient*fsky
        for row in range(7):
            for column in range(7):
                for i in range(100,2998):
                        Clinv = np.linalg.inv((self.Parameters_CLmatrix[variables[7]])[i])
                        dCldtheta_row = ((((self.Parameters_CLmatrix[variables[row]])[i]) - ((self.Parameters_CLmatrix[variables[7]])[i])) / ((0.01)*cl_dict_values[variables[row]]))
                        dCldtheta_column = ((((self.Parameters_CLmatrix[variables[column]])[i]) - ((self.Parameters_CLmatrix[variables[7]])[i])) / ((0.01)*cl_dict_values[variables[column]]))
                        self.f_matrix1[row,column] += ((2*i+1)/2) * fsky1* np.trace(Clinv * dCldtheta_row * Clinv * dCldtheta_column)
        self.f_matrix1[5,5] +=   (1/(0.0070)**2)
        
        a = np.linalg.inv(self.f_matrix1)
        a.diagonal()
        self.errors = {}
        for i in range(len(a.diagonal())): 
            self.errors[variables[i]] = (a.diagonal()[i])**(1/2)
        return self.errors

