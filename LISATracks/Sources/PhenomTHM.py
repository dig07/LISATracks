import numpy as np
from . import Constants as const
from ..Source_model import Model
import scipy
import scipy.interpolate


# Euler Mascheroni constant
c = const.clight
MTsun = const.MTsun
pc = const.pc


def instantaneous_frequency(times, phase):
    '''
    Calculates the instantaneous GW frequency of a mode from its (unwrapped) phase.

    Parameters:
      times (array of floats): Time samples (seconds).
      phase (array of floats): Unwrapped phase of the mode at each time sample (radians).

    Returns:
      f (array of floats): Instantaneous GW frequency f = (1/2pi)|dphase/dt| (Hz).
    '''
    return np.abs(np.gradient(phase, times)) / (2 * np.pi)


class PhenTHM(Model):

    name = 'PhenomTHM'
    param_names = ['m1', 'm2', 'chi1z', 'chi2z', 'D', 'phi_ref', 'inclination', 'psi', 'f_low', 't_c']

    def __init__(self, parameters, freqs, T, higher_modes='all', delta_t=10.0):
        '''
        Args:
            parameters: dictionary of parameters for this source
            freqs: array of frequencies over which to evaluate the waveform (build the interpolant) over
            T: Observation time
            higher_modes: Higher modes to include beyond (2,2), passed to phentax (Defaults to 'all').
            delta_t: Time step (seconds) used to generate the time domain waveform (Defaults to 10s).
        '''
        # phentax (and JAX) are only imported when this source is actually used
        import jax
        jax.config.update('jax_enable_x64', True)
        from phentax.waveform import IMRPhenomTHM

        self.parameters = parameters
        self.freqs = np.asarray(freqs)
        self.observation_time = T

        self.m1 = self.parameters['m1']
        self.m2 = self.parameters['m2']
        self.chi1z = self.parameters['chi1z']
        self.chi2z = self.parameters['chi2z']
        self.D = self.parameters['D']
        self.phi_ref = self.parameters['phi0']
        self.inclination = self.parameters['inclination']
        self.psi = self.parameters['psi']
        self.f_low = self.parameters['f_low']
        # Optional: mission time (seconds) at which coalescence occurs. If not given, the source
        #       instead enters the band at mission time t=0 (matching the TaylorF2Ecc convention).
        self.t_c = self.parameters.get('t_c', None)

        # phentax takes the distance in Mpc, but the source dictionaries use pc (as TaylorF2Ecc)
        D_Mpc = self.D / 1.e6

        # Generate the multi-modal time domain waveform. Each mode is treated as a 'harmonic'.
        #       The positive m modes are projected onto the spin-weighted spherical harmonics so that
        #       the per-mode amplitude is the mode's contribution to the observed strain.
        waveform = IMRPhenomTHM(higher_modes=higher_modes, include_negative_modes=False, T=self.observation_time)

        times, mask, amplitudes, phases = waveform.compute_strain_components_amp_phase(
            self.m1, self.m2, self.chi1z, self.chi2z, D_Mpc,
            self.phi_ref, self.inclination, self.psi,
            delta_t=delta_t, f_min=self.f_low, f_ref=self.f_low)

        # Single binary -> drop the batch dimension and select the valid time samples (merger at t=0)
        mask = np.asarray(mask[0])
        times = np.asarray(times[0])[mask]
        amplitudes = np.asarray(amplitudes[0])[:, mask]
        phases = np.asarray(phases[0])[:, mask]

        self.modes = [tuple(int(x) for x in mode) for mode in np.asarray(waveform.modes_list)]
        self.num_harmonics = len(self.modes)

        # The time-frequency map is built from the clean (2,2) phase: every mode (l,m) chirps at
        #       f_lm = (m/2) f_22. This avoids the spurious frequency spikes that a per-mode phase
        #       gradient produces wherever a higher mode amplitude passes through a minimum.
        index_22 = self.modes.index((2, 2))
        frequency_22 = instantaneous_frequency(times, phases[index_22])
        frequency_22_dot = np.gradient(frequency_22, times)

        # Restrict to the monotonically chirping inspiral-merger portion (up to the (2,2) peak
        #       frequency) and keep only points that set a new running maximum, so the frequency
        #       is strictly increasing.
        f_peak_index = np.argmax(frequency_22)
        frequency_22 = frequency_22[:f_peak_index + 1]
        increasing = np.empty(frequency_22.shape, dtype=bool)
        increasing[0] = True
        increasing[1:] = frequency_22[1:] > np.maximum.accumulate(frequency_22)[:-1]

        frequency_22 = frequency_22[increasing]
        frequency_22_dot = np.clip(frequency_22_dot[:f_peak_index + 1][increasing], a_min=1e-30, a_max=None)

        # Map the waveform time (coalescence at t=0) onto mission time. If t_c is given, coalescence
        #       happens at mission time t_c; otherwise the source enters the band at mission time t=0.
        times = times[:f_peak_index + 1][increasing]
        if self.t_c is None:
            time_offset = -times[0]
        else:
            time_offset = self.t_c
        mission_times = times + time_offset
        # Mission time of coalescence (waveform t=0)
        self.t_merger = time_offset

        # Per-mode interpolants in frequency: f->t (mission time), f->|h~(f)|, f->phase
        self._t_of_f = []
        self._amplitude_of_f = []
        self._phase_of_f = []
        self.f0s = []
        self.f_high = []

        for i, (ell, emm) in enumerate(self.modes):

            # Frequency, frequency derivative and SPA amplitude of this mode
            frequency = (emm / 2) * frequency_22
            frequency_dot = (emm / 2) * frequency_22_dot
            h_tilde = amplitudes[i][:f_peak_index + 1][increasing] / np.sqrt(frequency_dot)
            phase = phases[i][:f_peak_index + 1][increasing]

            # Monotonicity-preserving interpolants: the time grid is dense at low frequency and sparse
            #       near merger, so a cubic spline would overshoot and make the resampled t(f) map
            #       non-monotonic. PCHIP avoids this and keeps t(f) strictly increasing.
            self._t_of_f.append(scipy.interpolate.PchipInterpolator(frequency, mission_times))
            self._amplitude_of_f.append(scipy.interpolate.PchipInterpolator(frequency, np.abs(h_tilde)))
            self._phase_of_f.append(scipy.interpolate.PchipInterpolator(frequency, phase))

            # Clip the per-mode frequency band to the master frequency grid so we never extrapolate
            self.f0s.append(max(frequency[0], self.freqs[0]))
            self.f_high.append(min(frequency[-1], self.freqs[-1]))

        print('Time to merger is: ', (times[-1] - times[0]) / (const.YRSID_SI), ' years')
        print('Coalescence at mission time: ', self.t_merger / (const.YRSID_SI), ' years')
        print('Modes included: ', self.modes)

    def _freqs_in_band(self, i):
        '''
        Master frequency grid restricted to the frequency band of harmonic i (matches the slicing
        performed in Model.generate_track_splines).
        '''
        return self.freqs[(self.freqs >= self.f0s[i]) & (self.freqs <= self.f_high[i])]

    def Amplitudes(self):
        '''
        Returns the frequency domain amplitudes for each mode for the given parameters.
        '''
        return [self._amplitude_of_f[i](self._freqs_in_band(i)) for i in range(self.num_harmonics)]

    def Phases(self):
        '''
        Returns the phases for each mode for the given parameters.
        '''
        return [self._phase_of_f[i](self._freqs_in_band(i)) for i in range(self.num_harmonics)]

    def Time_frequency(self):
        '''
        Returns the time-frequency map array for each mode for the given parameters.
        '''
        return [self._t_of_f[i](self._freqs_in_band(i)) for i in range(self.num_harmonics)]
