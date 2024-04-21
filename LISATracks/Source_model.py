from abc import ABCMeta,abstractmethod
import scipy
import scipy.interpolate 
import numpy as np 
class Model(object):
    """
    Base class for a source model. Sources should subclass this
    and implement the following properties:
        - name: name of sourcetype (e.g. 'TaylorF2Ecc')
        - param_names: list of parameters (e.g. ['p1','p2'])
        - num_harmonics: number of harmonics to use in the model
        - f0s: list of initial frequencies for each harmonic

    And the following methods: 
        - Amplitudes: Returns the amplitudes as a function of frequency for each harmonic. 
        - Phases:  Returns the Phases as a function of frequency for each harmonic. 
        - Time_frequency:  Returns the time-frequency map as a function of frequency for each harmonic. 

    """
    __metaclass__ = ABCMeta
    names =None # Name of the sourcetypes
    param_names =None # list of parameter_names
    num_harmonics = 1
    f0s = []

    @abstractmethod
    def Amplitudes(self):
        """
        Returns the amplitudes for each harmonic for the given parameters.
        """
        pass


    @abstractmethod
    def Phases(self):
        """
        Returns the phases for each harmonic for the given parameters.
        """
        pass


    @abstractmethod
    def Time_frequency(self):
        """
        Returns the time-frequency map array for each harmonic for the given parameters.
        """
        pass    


    def generate_track_splines(self,freqs):
        """
        Generate the splines for each harmonic over the frequency range for the given parameters.

        Args:
            parameters: list of parameters
            freqs: array of frequencies over which to evaluate the waveform. 
        Returns:
            t_f_container: list of splines for the time-frequency map for each harmonic
            amplitude_time_container: list of splines for the amplitude (in characheristic strain) as a function of time for each harmonic
        """
        Amps, Phases, Time_freqs = self.Amplitudes(), self.Phases(), self.Time_frequency()

        # For each harmonic
        t_f_container = []
        amplitude_time_container = []

        for i in range(self.num_harmonics):

            # Extract Amplitude and Time-frequency map for this harmonic
            Amplitude = Amps[i]
            t_f_map = Time_freqs[i]

            # Extract the initial frequency for this harmonic
            f0 = self.f0s[i]

            # Interpolate the function of t->f map
            t_f_spline = scipy.interpolate.CubicSpline(t_f_map,freqs[(freqs>=f0) & (freqs<=self.f_high[i])])
            amplitude_time_spline = scipy.interpolate.CubicSpline(t_f_map,2*t_f_spline(t_f_map)*np.abs(Amplitude))


            t_f_container.append(t_f_spline)
            amplitude_time_container.append(amplitude_time_spline)

        return(t_f_container,amplitude_time_container)
