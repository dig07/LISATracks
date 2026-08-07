import numpy as np
from . import Constants as const
from ..Source_model import Model


c = const.clight
MTsun = const.MTsun
pc = const.pc


class DWD(Model):
    '''
    Galactic double white dwarf binary, treated as monochromatic over the LISA mission.

    Unlike the chirping sources, a DWD's track is naturally parametrised by time rather than by
    frequency. The frequency is held fixed at f0 for the whole mission (the entire track lives
    inside one bin of the master frequency grid), and what actually evolves is the characteristic
    strain, which grows as the square root of the number of cycles observed,

        h_c(t) = h_0 sqrt(N(t)),    N(t) = f0 t.

    The frequency-parametrised default in Model.track_samples cannot resolve that, so this class
    overrides track_samples and samples the track in time instead.
    '''

    name = 'DWD'
    param_names = ['m1', 'm2', 'f0', 'D']
    num_harmonics = 1

    def __init__(self, parameters, freqs, T, t_min=24*3600., num_samples=512):
        '''
        Args:
            parameters: dictionary of parameters for this source (m1, m2 in solar masses, f0 the GW
                frequency in Hz, D in pc; matching the conventions of the other sources)
            freqs: array of frequencies over which to evaluate the waveform. Unused here (the track
                is sampled in time), kept so that every source shares a common interface.
            T: Observation time (s)
            t_min: Mission time at which the track starts (s). h_c ~ sqrt(t), so the source sits far
                below the lower strain limit of a typical plot before this (Defaults to 1 day).
            num_samples: Number of (log spaced) time samples used to build the splines (Defaults to 512).
        '''
        self.parameters = parameters
        self.freqs = np.asarray(freqs)
        self.observation_time = T
        self.t_min = t_min
        self.num_samples = num_samples

        self.m1 = self.parameters['m1']
        self.m2 = self.parameters['m2']
        self.f0 = self.parameters['f0']
        self.D = self.parameters['D']

        # Unit conversions into geometrized units (masses and distances in seconds), as TaylorF2Ecc
        # Absorbed 3 factors of c
        self.m1 = self.m1*MTsun
        self.m2 = self.m2*MTsun

        # Absorbs an extra factor of c. 
        self.D = (self.D)*pc/c

        self.Mc = (self.m1*self.m2)**(3/5)/(self.m1+self.m2)**(1/5)

        # Log spaced so that the fast early rise of h_c ~ sqrt(N_cycles) ~ sqrt(t) is well resolved by the spline
        self.times = np.logspace(np.log10(self.t_min),np.log10(self.observation_time),self.num_samples)

        # Monochromatic, so the source sits at f0 for the whole mission
        self.track_frequencies = np.full_like(self.times,self.f0)

        self.f0s = [self.f0]
        self.f_high = [self.f0]

        print('Cycles observed over the mission: ',self.cycles(self.observation_time))

    def cycles(self,times):
        '''
        Number of GW cycles observed since the start of the mission, N(t) = f0 t.

        Args:
            times (array of floats): Mission times (s).

        Returns:
            N (array of floats): Cycles accumulated by each time.
        '''
        return self.f0*times

    def Amplitudes(self):
        '''
        Returns the amplitudes for each harmonic for the given parameters.

        The Fourier domain amplitude of a monochromatic source is a delta function, so what is
        returned here is the (dimensionless) strain amplitude h_0 at f0, rather than |h~(f)| as for
        the chirping sources.
        '''
        return [4*self.Mc**(5/3)*(np.pi*self.f0)**(2/3)/self.D]

    def Time_frequency(self):
        '''
        Returns the time-frequency map array for each harmonic for the given parameters.

        These are the mission times corresponding to self.track_frequencies (all of which are f0).
        '''
        return [self.times]

    def track_samples(self,freqs):
        '''
        Sample the track in time rather than in frequency (see the class docstring).

        Args:
            freqs: array of frequencies over which to evaluate the waveform. Unused here.

        Returns:
            list of (times, frequencies, characteristic_strain) tuples of arrays, one per harmonic.
        '''
        # h_c = h_0 sqrt(number of cycles observed)
        characteristic_strain = self.Amplitudes()[0]*np.sqrt(self.cycles(self.times))

        return [(self.times, self.track_frequencies, characteristic_strain)]
