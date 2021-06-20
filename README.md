# Fisher-Forecast
test repository
import math
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb


cl_matrix = []
cl_matrix1 = []
cos_dict = {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122,'mnu': 0.06,'omk': 0 , 'tau': 0.06}
H0_dict ={'H0': 67.5*1.01, 'ombh2':0.022, 'omch2':0.122, 'mnu': 0.06,'omk': 0, 'tau': 0.06}
ombh2_dict = {'H0': 67.5, 'ombh2':0.022*1.01, 'omch2':0.122,'mnu': 0.06,'omk': 0, 'tau': 0.06}
omch2_dict={'H0': 67.5, 'ombh2':0.022, 'omch2':0.122*1.01, 'mnu': 0.06,'omk': 0, 'tau': 0.06}
tau_dict = {'H0': 67.5, 'ombh2':0.022, 'omch2':0.122, 'mnu': 0.06,'omk': 0, 'tau': 0.06*1.01}

def  noise(i, s, theta):
        N_l = (s**2)* math.exp(i*(i+1) * (theta**2/ 8* math.log(2)))
        return N_l
    
def get_cl(dict, b, c):
        pars = camb.CAMBparams()
        cl_matrix = []
        pars.set_cosmology(**dict)
        pars.InitPower.set_params(As=b*2.10058296e-9, ns=c*0.96605, r =0)
        pars.set_for_lmax(10000, lens_potential_accuracy=0);
        results = camb.get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
        totCL=powers['total']
        unlensedCL=powers['unlensed_scalar']
        ls = np.arange(totCL.shape[0])
        unlensedCL=powers['unlensed_scalar']
        cl_t = totCL[:,0]
        cl_e =totCL[:,1]
        cl_te = totCL[:,3]
        noiseTT = dd['cl_residual']['TT']
        noiseEE = dd['cl_residual']['EE']
        for i in ls[(ls < len(noiseTT)) & (ls < len(cl_t))& (ls > 100)]:
            Cl_11 = noiseTT[i]+ (cl_t[i]*(((2*np.pi))/(i*(i+1))))
            Cl_12 = (cl_te[i]*(2*np.pi))/(i*(i+1))
            Cl_22 = noiseEE[i]+ (cl_e[i]*(((2*np.pi))/(i*(i+1))))
            c_l   = np.matrix([[Cl_11,  Cl_12],[ Cl_12, Cl_22]])
            cl_matrix.insert(len(cl_matrix)-1,c_l)
        return cl_matrix
def get_fisher():
    H0_cls = get_cl(H0_dict, 1, 1)
    cos_cls = get_cl(cos_dict, 1,1)
    #A_cls = get_cl(cos_dict, 1)
    ombh2_cls = get_cl(ombh2_dict, 1, 1)
    omch2_cls = get_cl(omch2_dict, 1,1)
    As_cls = get_cl(cos_dict,1.01,1)
    ns_cls = get_cl(cos_dict, 1,1.01)
    tau_cls = get_cl(tau_dict,1,1)
    cl_dict = {"H0": H0_cls, "ombh2": ombh2_cls ,"omch2": omch2_cls, "As":As_cls, "ns":ns_cls, "tau":tau_cls,  "cos": cos_cls}  
    cl_dict_values = {"H0": 67.5, "ombh2": 0.022  ,"omch2": 0.122, "As":2.10058296e-9, "ns":0.96605, "tau":0.06} 
    var = list(cl_dict.keys())
    f_matrix1 = np.zeros((6,6))
    for row in range(6):
        for column in range(6):
            for i in range(1,2898):
                Clinv = np.linalg.inv((cl_dict[var[6]])[i])
                dCldtheta_row = ((((cl_dict[var[row]])[i]) - ((cl_dict[var[6]])[i])) / ((0.01)*cl_dict_values[var[row]]))
                dCldtheta_column = ((((cl_dict[var[column]])[i]) - ((cl_dict[var[6]])[i])) / ((0.01)*cl_dict_values[var[column]]))
                fsky = 1500 / 41253
                f_matrix1[row,column] += ((2*i+1)/2) * fsky * np.trace(Clinv * dCldtheta_row * Clinv * dCldtheta_column)


    print(f_matrix1)
