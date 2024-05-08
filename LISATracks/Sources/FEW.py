from few.utils.utility import *
from few.utils.fdutils import *
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *


from ..Source_model import Model
import numpy as np 


class FEW_FD(Model):

    name='EMRI'
    # num_harmonics = 6
    param_names = ['M',# Central mass [solar masses]
                   'mu',# Secondary mass [solar masses]
                   'p0', # Initial semi-latus rectum [m]
                   'e0', # Initial eccentricity
                   'qk', # Polar spin angle
                   'phik',# azimuthal viewing angle
                   'qS',# polar sky angle
                   'phiS',# azimuthal viewing angle
                   'dist',# Distance
                   'modes'] # modes to include in the model
    def __init__(self,parameters,freqs,T):
        '''
        Args:
            parameters: dictionary of parameters for this source
            freqs: array of frequencies over which to evaluate the waveform (build the interpolant) over
            T: Observation time
        '''
        self.parameters = parameters
        self.freqs = freqs
        self.observation_time = T

        self.M = parameters['M']
        self.mu = parameters['mu']
        self.p0 = parameters['p0']
        self.e0 = parameters['e0']
        self.qk = parameters['qk']
        self.phik = parameters['phik']
        self.qS = parameters['qS']
        self.phiS = parameters['phiS']
        self.dist = parameters['dist']
        self.modes = parameters['modes']

        # Both these parameters are ignored in the Schwarzschild waveform model, but are required for the model to be created
        a = 0.1
        x0 = 1.0
        
        # initial phases (Dont matter for our animation)
        Phi_phi0 = np.pi/3
        Phi_theta0 = 0.0
        Phi_r0 = np.pi/3

        traj_module = EMRIInspiral(func="SchwarzEccFlux")

        # Rescaled to years for FEW
        T_rescaled = self.observation_time/(365.25*24*60*60)

        # get the initial p0 
        p0 = get_p_at_t(
            traj_module,
            T_rescaled*0.2,
            [self.M, self.mu, 0.0, self.e0, 1.0],
            index_of_p=3,
            index_of_a=2,
            index_of_e=4,
            index_of_x=5,
            traj_kwargs={},
            xtol=2e-12,
            rtol=8.881784197001252e-16,
            bounds=None,
        )


        emri_injection_params = [
            self.M,  
            self.mu,
            a, 
            p0, 
            self.e0, 
            x0,
            self.dist, 
            self.qS,
            self.phiS,
            self.qk, 
            self.phik, 
            Phi_phi0, 
            Phi_theta0, 
            Phi_r0
        ]

        waveform_kwargs = {
        "T": T_rescaled,
        "dt": 10.0,
        "include_minus_m": True,
        'mask_positive': True,
        }

        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj_module(self.M, self.mu, 0.0, self.p0, self.e0, 1.0, T=T_rescaled)
        OmegaPhi, OmegaTheta, OmegaR = get_fundamental_frequencies(0.0, p, e, 0.0)


        # frequency domain
        few_gen = GenerateEMRIWaveform(
            "FastSchwarzschildEccentricFlux", 
            sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
            return_list=True
        )

        self.f_high = []

        self.amps = []
        self.tf = []
        self.f0s = []
        for mode in self.modes:

            fd_kwargs = waveform_kwargs.copy()
            fd_kwargs['mode_selection']  = [mode]
            print(mode)
            hf = few_gen(*emri_injection_params,**fd_kwargs)

            # Amps
            freq_fd = few_gen.waveform_generator.create_waveform.frequency
            positive_frequency_mask = (freq_fd>=0.0)
            positive_frequencies = freq_fd[positive_frequency_mask]

            min_f = positive_frequencies[np.abs(hf[0]>0)][0]
            max_f = positive_frequencies[np.abs(hf[0]>0)][-1]


            self.f0s.append(min_f)
            self.f_high.append(max_f)

            amp_spline = CubicSpline(positive_frequencies, np.abs(hf[0]))
            self.amps.append(amp_spline(self.freqs[(self.freqs>=min_f) & (self.freqs<=max_f)]))
            print(np.max(amp_spline(self.freqs[(self.freqs>=min_f) & (self.freqs<=max_f)])))
            # Time-frequency map
            m_sel = mode[0]
            n_sel = mode[2]
            theo_f = (m_sel*OmegaPhi + n_sel*OmegaR)/(2*np.pi*self.M*MTSUN_SI)

            # spline to get time frequency map
            time_f_spline = CubicSpline(theo_f, t)

            self.tf.append(time_f_spline(self.freqs[(self.freqs>=min_f) & (self.freqs<=max_f)]))



    def Amplitudes(self):
        '''
        Returns the amplitudes for each harmonic for the given parameters.
        '''
        return(self.amps)
    
    def Phases(self):
        '''
        Returns the phases for each harmonic for the given parameters.
        '''
        return(np.zeros(len(self.amps)))
    
    def Time_frequency(self):
        '''
        Returns the time-frequency map array for each harmonic for the given parameters.
        '''
        return(self.tf)