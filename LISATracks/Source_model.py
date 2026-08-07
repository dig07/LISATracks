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


    def track_samples(self,freqs):
        """
        Sample each harmonic's track as (times, frequencies, characteristic strain) arrays.

        This is the hook to override for a source whose track is not naturally parametrised by
        frequency (e.g. a monochromatic DWD, whose whole track lives inside a single bin of the
        master frequency grid and so must be sampled in time instead).

        The default implementation is the frequency-parametrised one used by chirping sources: the
        master frequency grid is restricted to each harmonic's band, Time_frequency() supplies the
        mission time at each of those frequencies, and the characteristic strain is 2 f |h~(f)|.

        Note the DWD setup overloads this with a time-parametrised version, since the track is naturally parametrised 
        by mission time rather than frequency.

        Args:
            freqs: array of frequencies over which to evaluate the waveform.
        Returns:
            list of (times, frequencies, characteristic_strain) tuples of arrays, one per harmonic.
        """

        Amps, Time_freqs = self.Amplitudes(),  self.Time_frequency()

        samples = []

        for i in range(self.num_harmonics):

            # Frequencies of the master grid that lie inside this harmonic's band
            freqs_in_band = freqs[(freqs>=self.f0s[i]) & (freqs<=self.f_high[i])]

            samples.append((Time_freqs[i], freqs_in_band, 2*freqs_in_band*np.abs(Amps[i])))

        return samples


    def generate_track_splines(self,freqs):
        """
        Generate the splines for each harmonic over the frequency range for the given parameters.

        Args:
            parameters: list of parameters
            freqs: array of frequencies over which to evaluate the waveform.
        Returns:
            t_f_container: list of splines for the time-frequency map for each harmonic
            amplitude_time_container: list of splines for the amplitude (in characheristic strain) as a function of time for each harmonic
            time_window_container: list of (t_start,t_end) tuples giving the mission-time interval over which each harmonic's track is well defined (band entry to the end of the t-f map)
        """

        # For each harmonic
        t_f_container = []
        amplitude_time_container = []
        time_window_container = []

        for times, track_freqs, characteristic_strain in self.track_samples(freqs):

            t_f_container.append(scipy.interpolate.CubicSpline(times,track_freqs))
            amplitude_time_container.append(scipy.interpolate.CubicSpline(times,characteristic_strain))
            time_window_container.append((times[0], times[-1]))

        return(t_f_container,amplitude_time_container,time_window_container)
