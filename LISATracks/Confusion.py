
# This provides a fit for the confusion noise, originally taken from 1703.09722
# Updated to 2103.14598 + 2108.01167


import numpy as np


# LISA mean armlength
L_mean = 2500000000.0

clight = 299792458.0

# High frequency sensitivity factor
highFreqFactor = 0.6

# Confusion noise parameter from 2103.14598
Sconf_a1_thresh5_runningMean = -0.16
Sconf_ak_thresh5_runningMean = -0.34
Sconf_b1_thresh5_runningMean = -2.78
Sconf_bk_thresh5_runningMean = -2.53
Sconf_A_thresh5_runningMean = 1.15e-44
Sconf_f2_thresh5_runningMean = 0.59e-3
Sconf_alpha_thresh5_runningMean = 1.66

Sconf_a1_thresh7_runningMean = -0.25
Sconf_ak_thresh7_runningMean = -0.27
Sconf_b1_thresh7_runningMean = -2.70
Sconf_bk_thresh7_runningMean = -2.47
Sconf_A_thresh7_runningMean = 1.14e-44
Sconf_f2_thresh7_runningMean = 0.31e-3
Sconf_alpha_thresh7_runningMean = 1.80

Sconf_a1_thresh5_runningMedian = -0.15
Sconf_ak_thresh5_runningMedian = -0.34
Sconf_b1_thresh5_runningMedian = -2.78
Sconf_bk_thresh5_runningMedian = -2.55
Sconf_A_thresh5_runningMedian = 1.14e-44
Sconf_f2_thresh5_runningMedian = 0.59e-3
Sconf_alpha_thresh5_runningMedian = 1.66

Sconf_a1_thresh7_runningMedian = -0.15
Sconf_ak_thresh7_runningMedian = -0.37
Sconf_b1_thresh7_runningMedian = -2.72
Sconf_bk_thresh7_runningMedian = -2.49
Sconf_A_thresh7_runningMedian = 1.15e-44
Sconf_f2_thresh7_runningMedian = 0.67e-3
Sconf_alpha_thresh7_runningMedian = 1.56


Sconf_a1_default = Sconf_a1_thresh5_runningMedian
Sconf_ak_default = Sconf_ak_thresh5_runningMedian
Sconf_b1_default = Sconf_b1_thresh5_runningMedian
Sconf_bk_default = Sconf_bk_thresh5_runningMedian
Sconf_A_default = Sconf_A_thresh5_runningMedian
Sconf_f2_default = Sconf_f2_thresh5_runningMedian
Sconf_alpha_default = Sconf_alpha_thresh5_runningMedian


# Not found in 2103.14598, assumed to be 1 year
Tref = 31558149.7635456



# OMS noise from 2108.01167
def SOMS_VarLength(f, L):
    freqs = np.copy(f)
    freqs[freqs <= 1.e-8] = 1.e-8
    return (15.e-12/L)**2*(1. + (2e-3/freqs)**4)


def SOMS(f):
    return SOMS_VarLength(f, L_mean)
    
    
    
# acc noise from 2108.01167
def Sacc_VarLength(f, L):
    freqs = np.copy(f)
    freqs[freqs <= 1.e-8] = 1.e-8
    return (3.e-15/(L * (2.*np.pi*freqs)**2))**2 * (1. + (0.4e-3/freqs)**2) * (1. + (freqs/8.e-3)**4)


def Sacc(f):
    return Sacc_VarLength(f, L_mean)


def Sinst_VarLength(f, L):
    freqs = np.copy(f)
    freqs[freqs <= 1.e-8] = 1.e-8
    return (4.*Sacc_VarLength(f, L) + SOMS_VarLength(f, L)) * (1. + highFreqFactor*(2.*np.pi*freqs*L/clight)**2)

def Sinst(f):
    return Sinst_VarLength(f, L_mean)
    
    

# confusion noise from 2103.14598
def Sconf_f1(a1, b1, Tobs):
    return 10.**b1 * (Tobs/Tref)**a1
    
def Sconf_fknee(ak, bk, Tobs):
    return 10.**bk * (Tobs/Tref)**ak


def Sconf_VarParams(f, A, f1, alpha, fknee, f2):
    freqs = np.copy(f)
    freqs[freqs <= 1.e-8] = 1.e-8
    return (0.5*A) * freqs**(-7./3.) * np.exp(-(freqs/f1)**alpha) * (1. + np.tanh((fknee - freqs) / f2))



def Sconf_VarFit(f, A, a1, b1, alpha, ak, bk, f2, Tobs):
    return Sconf_VarParams(f, A, Sconf_f1(a1, b1, Tobs), alpha, Sconf_fknee(ak, bk, Tobs), f2)
    
def Sconf(f, Tobs):
    return Sconf_VarFit(f, Sconf_A_default, Sconf_a1_default, Sconf_b1_default, Sconf_alpha_default, Sconf_ak_default, Sconf_bk_default, Sconf_f2_default, Tobs)






# Ratio of the confusion noise to the instrumental noise from 1703.09722
def Sconf_ratio(f, Tobs):
    return Sconf(f, Tobs)/Sinst(f)


# Add confusion noise to the psd in Sn with frequencies in freqs,
# assuming Tobs observation time
def Add_confusion(f, Sn, Tobs):
    Sc = Sconf_ratio(f, Tobs)
    Sout = Sn*(1. + Sc)
    return Sout


# Simple PSD for LISA
def psd_SCIRD(f):
    s_1 = 5.86e-48*(1+((0.4e-3)/f)**2)
    s_2 = 3.6e-41
    r = 1 + (f/(25e-3))**2
    s_SciRD = 1/2*20/3*(s_1/(2*np.pi*f)**4 + s_2)*r
    return(s_SciRD)