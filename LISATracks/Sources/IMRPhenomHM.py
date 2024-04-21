from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.utils.constants import *
from ..Source_model import Model
import numpy as np 

class IMRPhenomHM(Model):

    name='IMRPhenomHM'
    num_harmonics = 6
    param_names = ['m1','m2','a1','a2','D','tc','phi0']

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

        self.m1 = self.parameters['m1']
        self.m2 = self.parameters['m2']
        self.a1 = self.parameters['a1']
        self.a2 = self.parameters['a2']
        self.D  = self.parameters['D']
        self.tc = self.parameters['tc']
        self.reference_phase = self.parameters['phi0']
        
        self.f0s = [0]*self.num_harmonics  

        # Unit conversions 
        self.D = (self.D)*3.086e+16 # Pc to metres

        phenomhm = PhenomHMAmpPhase(use_gpu=False, run_phenomd=False)

        f_ref = 0 # Reference frequency (allow code to set f_ref internally, to I think roughly the merger frequency)

        # Try and set the reference time at f_ref, ie near the merger to be 0
        t_ref = 0

        
        phenomhm(
            self.m1,
            self.m2,
            self.a1,
            self.a2,
            self.D,
            self.reference_phase,
            f_ref,
            t_ref,
            freqs = self.freqs,
            length = self.freqs.size,
        )

        amps = phenomhm.amp[0]  # shape (num_modes, length)
        phase = phenomhm.phase[0]  # shape (num_modes, length)
        tf = phenomhm.tf[0]  # shape (num_modes, length)
        self.modes = phenomhm.modes


        self.f_high = []

        self.amps = []
        self.phase = []
        self.tf = []
    
        # Find cutoff frequency close to merger past which the tf map is no longer monotonic, ie the SPA doesnt apply: 
        for i, mode in enumerate(self.modes):
            
            # Find the frequency where the tf map is no longer monotonic
            if np.any(np.diff(tf[i])<0):
                freq_index_of_violation = np.where(np.diff(tf[i])<0)[0][0]
            else:
                freq_index_of_violation = len(self.freqs)-1
            # find f here this will be our upper cutoff for freq for this mode 
            self.f_high.append(self.freqs[freq_index_of_violation])

            # Cut off the amps, phases and tf maps at the cutoff frequency for each mode
            self.amps.append(amps[i][self.freqs<=self.freqs[freq_index_of_violation]])
            self.phase.append(phase[i][self.freqs<=self.freqs[freq_index_of_violation]])
            # tf is roughly set up to be 0 at the merger, so if we += self.tc to this, everything "before" LISa merger will be negative, and since the mission 
            #      timer is set to 0 initially and only goes up, wont matter.
            self.tf.append(tf[i][self.freqs<=self.freqs[freq_index_of_violation]]+self.tc)

        # # tf is roughly set up to be 0 at the merger, so if we += self.tc to this, everything "before" LISa merger will be negative, and since the mission 
        # #      timer is set to 0 initially and only goes up, wont matter.
        # self.tf += self.tc

    def Amplitudes(self):
        '''
        Returns the amplitudes for each harmonic for the given parameters.
        '''
        return(self.amps)
    
    def Phases(self):
        '''
        Returns the phases for each harmonic for the given parameters.
        '''
        return(self.phase)
    
    def Time_frequency(self):
        '''
        Returns the time-frequency map array for each harmonic for the given parameters.
        '''
        return(self.tf)