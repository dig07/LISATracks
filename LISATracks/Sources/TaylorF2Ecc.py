import numpy as np 
from . import Constants as const
from ..Source_model import Model
import scipy 


# Euler Mascheroni constant
gamma_e = const.gamma_e 
c = const.clight
G = const.G
MTsun = const.MTsun
pc = const.pc

# Armlength of LISA in seconds
Armlength = 2.5e+9/c

# BBHx prefactor to get strains into same units as Balrog (See PSD comparison, compare the PSDs from LISAtools, equivalent to frequency_mode == True from Balrog)
bbhx_pre_factor = 1/(2j*np.pi*Armlength)


def calculate_T(v, v0, e0, eta):
    '''
    Calculates the value of T using the (TaylorT2) Equation 6.7b from arXiv:1605.00304v2.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.

    Returns:
      T (float or array of floats): The calculated value(s) of T at fs provided into v.

    '''

    term1 = (1 + ((743 / 252) + (11 / 3) * eta) * v**2 - (32 / 5) * np.pi * v**3 + 
        ((3058673 / 508032) + (5429 / 504) * eta + (617 / 72) * eta**2) * v**4 + 
        (-(7729 / 252) + (13 / 3) * eta) * np.pi * v**5)
    
    term2 = ((-10052469856691 / 23471078400) + 
         (6848 / 105) * gamma_e + 
         (128 / 3) * np.pi**2 + 
         ((3147553127 / 3048192) - 
          (451 / 12) * np.pi**2) * eta - 
         (15211 / 1728) * eta**2 + 
         (25565 / 1296) * eta**3 + 
         (3424 / 105) * np.log(16 * v**2)) * v**6
    
    term3 = ((-15419335 / 127008) - (75703 / 756) * eta + (14809 / 378) * eta**2) * np.pi * v**7

    term4 = -(157 / 43) * e0**2 * (v0 / v)**(19/3) * (1 + ((17592719 / 5855472) + (1103939 / 209124) * eta) * v**2 + \
        ((2833 / 1008) - (197 / 36) * eta) * v0**2 - (2819123 / 384336) * np.pi * v**3 + \
        (377 / 72) * np.pi * v0**3 + ((955157839 / 302766336) + (1419591809 / 88306848) * eta + \
        (91918133 / 6307632) * eta**2) * v**4 + ((49840172927 / 5902315776) - (42288307 / 26349624) * eta - \
        (217475983 / 7528464) * eta**2) * v**2 * v0**2 + (-(1193251 / 3048192) - (66317 / 9072) * eta + \
        (18155 / 1296) * eta**2)*v0**4 -((166558393 / 12462660) + (679533343 / 28486080) * eta) * np.pi * v**5 + \
        (-(7986575459 / 387410688) + (555367231 / 13836096) * eta) * np.pi * v**3 * v0**2 + \
        ((6632455063 / 421593984) + (416185003 / 15056928) * eta) * np.pi * v**2 * v0**3 + \
        ((764881 / 90720) - (949457 / 22680) * eta) * np.pi * v0**5 +(-(2604595243207055311 / 16582316889600000) + (31576663 / 2472750) * gamma_e + \
        (924853159 / 40694400) * np.pi**2 + ((17598403624381 / 86141905920) - (886789 / 180864) * np.pi**2) * eta + \
        (203247603823 / 5127494400) * eta**2 + (2977215983 / 109874880) * eta**3 + \
        (226088539 / 7418250) * np.log(2) - (65964537 / 2198000) * np.log(3) + \
        (31576663 / 4945500) * np.log(16 * v**2)) * v**6 + ((2705962157887 / 305188466688) + (14910082949515 / 534079816704) * eta - \
        (99638367319 / 2119364352) * eta**2 - (18107872201 / 227074752) * eta**3) * v**4 * v0**2 -(1062809371 / 27672192) * np.pi**2 * v**3 * v0**3 + \
        (-(20992529539469 / 17848602906624) - (15317632466765 / 637450103808) * eta + \
        (8852040931 / 2529563904) * eta**2 + (20042012545 / 271024704) * eta**3) * v**2 * v0**4 + \
        ((26531900578691 / 168991764480) - (3317 / 126) * gamma_e + (122833 / 10368) * np.pi**2 + \
         ((9155185261 / 548674560) - (3977 / 1152) * np.pi**2) * eta - (5732473 / 1306368) * eta**2 - \
         (3090307 / 139968) * eta**3 + (87419 / 1890) * np.log(2) - (26001 / 560) * np.log(3) - \
         (3317 / 252) * np.log(16 * v0**2)) * v0**6)

    
    T = term1+term2+term3+term4
    return T

def Lambda_f(v, v0, e0, eta):
    '''
    Calculates the value of Lambda_f using the (TaylorT2) Equation 6.6a from arXiv:1605.00304v2.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.

    Returns:
      lambda_ (float or array of floats): The calculated value(s) of lambda_f at fs provided into v.

    '''
    term1 = 1 + ((3715/1008) + (55/12) * eta) * v**2 - 10 * np.pi * v**3    
    term2 = ((15293365/1016064) + (27145/1008) * eta + (3085/144) * eta**2) * v**4
    term3 = ((38645/2016) - (65/24) * eta) * np.pi * np.log(v**3) * v**5
    term4 = (
        (12348611926451/18776862720)
        - (1712/21) * gamma_e
        - (160/3) * np.pi**2
        + ((-15737765635/12192768) + (2255/48) * np.pi**2) * eta
        + (76055/6912) * eta**2
        - (127825/5184) * eta**3
        - (856/21) * np.log(16 * v**2)
    ) * v**6
    term5 = ((77096675/2032128) + (378515/12096) * eta - (74045/6048) * eta**2) * np.pi * v**7
    term6 = -(785/272) * e0**2 * (v0/v)**(19/3) * (
        1 + ((6955261/2215584) + (436441/79128) * eta) * v**2
        + ((2833/1008) - (197/36) * eta) * v0**2
        - (1114537/141300) * np.pi * v**3
        + (377/72) * np.pi * v0**3
        + ((377620541/107433216) + (561233971/31334688) * eta + (36339727/2238192) * eta**2) * v**4
        + ((19704254413/2233308672) - (16718633/9970128) * eta - (85978877/2848608) * eta**2) * v**2 * v0**2
        + (-(1193251/3048192) - (66317/9072) * eta + (18155/1296) * eta**2) * v0**4
        - ((131697334/8456805) + (268652717/9664920) * eta) * np.pi * v**5
        + ((-3157483321/142430400) + (219563789/5086800) * eta) * np.pi * v**3 * v0**2
        + ((2622133397/159522048) + (164538257/5697216) * eta) * np.pi * v**2 * v0**3
        + ((764881/90720) - (949457/22680) * eta) * np.pi * v0**5
        + (
            (-204814565759250649/1061268280934400)
            + (12483797/791280) * gamma_e
            + (365639621/13022208) * np.pi**2
            + ((34787542048195/137827049472) - (8764775/1446912) * np.pi**2) * eta
            + (80353703837/1640798208) * eta**2
            + (5885194385/175799808) * eta**3
            + (89383841/2373840) * np.log(2)
            - (26079003/703360) * np.log(3)
            + (12483797/1582560) * np.log(16 * v**2)
        ) * v**6
        + ((1069798992653/108292681728) + (5894683956785/189512193024) * eta
           - (39391912661/752032512) * eta**2
           - (7158926219/80574912) * eta**3) * v**4 * v0**2
        - (420180449/10173600) * np.pi**2 * v**3 * v0**3
        + (
            (-8299372143511/6753525424128)
            - (6055808184535/241197336576) * eta
            + (3499644089/957132288) * eta**2
            + (7923586355/102549888) * eta**3
        ) * v**2 * v0**4
        + (
            (26531900578691/168991764480)
            - (3317/126) * gamma_e
            + (122833/10368) * np.pi**2
            + ((9155185261/548674560) - (3977/1152) * np.pi**2) * eta
            - (5732473/1306368) * eta**2
            - (3090307/139968) * eta**3
            + (87419/1890) * np.log(2)
            - (26001/560) * np.log(3)
            - (3317/252) * np.log(16 * v0**2)
        ) * v0**6
    )

    lamba_ = term1 + term2 + term3 + term4 + term5 + term6
    
    return lamba_

def F2EccPhase(freqs,eta,v,e0,v0,coallesence_phase,time_to_merger):
    '''
    Calculates the value of Lambda_f using the (TaylorT2) Equation 6.26 from arXiv:1605.00304v2. Waveform Fourier phase for TaylorF2Ecc.

    Parameters:
      v (float or array of floats): (pi*M*f)^{1/3} with f as frequencies of the binary system.
      v0 (float): (pi*M*f_0)^{1/3}, with f_0 as the initial GW frequency of the binary
      e0 (float): The initial eccentricity of the binary system.
      eta (float): The symmetric mass ratio of the binary system.

    Returns:
      psi (float or array of floats): The calculated value(s) of psi at fs provided into v.

    '''
    term1 = 3 / (128 * eta * v**5)
    
    term2 = (1 + (3715 / 756 + 55 / 9 * eta) * v**2 - 16 * np.pi * v**3 +
             (15293365 / 508032 + 27145 / 504 * eta + 3085 / 72 * eta**2) * v**4)
    
    term3 = (1 + np.log(v**3)) * (38645 / 756 - 65 / 9 * eta) * np.pi * v**5
    
    term4 = (11583231236531 / 4694215680 - 6848 / 21 * gamma_e - 640 / 3 * np.pi**2 +
             (-15737765635 / 3048192 + 2255 / 12 * np.pi**2) * eta +
             76055 / 1728 * eta**2 - 127825 / 1296 * eta**3 - 3424 / 21 * np.log(16 * v**2)) * v**6
    
    term5 = (77096675 / 254016 + 378515 / 1512 * eta - 74045 / 756 * eta**2) * np.pi * v**7
    
    term6 = -(2355 / 1462) * e0**2 * (v0 / v)**(19/3) * (
        1 + (299076223 / 81976608 + 18766963 / 2927736 * eta) * v**2 +
        (2833 / 1008 - 197 / 36 * eta) * v0**2 - 2819123 / 282600 * np.pi * v**3 +
        377 / 72 * np.pi * v0**3 +
        (16237683263 / 3330429696 + 24133060753 / 971375328 * eta + 1562608261 / 69383952 * eta**2) * v**4 +
        (847282939759 / 82632420864 - 718901219 / 368894736 * eta - 3697091711 / 105398496 * eta**2) * v**2 * v0**2 +
        (-1193251 / 3048192 - 66317 / 9072 * eta + 18155 / 1296 * eta**2) * v0**4 -
        (2831492681 / 118395270 + 11552066831 / 270617760 * eta) * np.pi * v**5 +
        (-7986575459 / 284860800 + 555367231 / 10173600 * eta) * np.pi * v**3 * v0**2 +
        (112751736071 / 5902315776 + 7075145051 / 210796992 * eta) * np.pi * v**2 * v0**3 +
        (764881 / 90720 - 949457 / 22680 * eta) * np.pi * v0**5 +
        (-43603153867072577087 / 132658535116800000 + 536803271 / 19782000 * gamma_e +
         15722503703 / 325555200 * np.pi**2 +
         (299172861614477 / 689135247360 - 15075413 / 1446912 * np.pi**2) * eta +
         3455209264991 / 41019955200 * eta**2 + 50612671711 / 878999040 * eta**3 +
         3843505163 / 59346000 * np.log(2) - 1121397129 / 17584000 * np.log(3) +
         536803271 / 39564000 * np.log(16 * v**2)) * v**6 + 
        (46001356684079 / 3357073133568 + 253471410141755 / 5874877983744 * eta -
        1693852244423 / 23313007872 * eta**2 - 307833827417 / 2497822272 * eta**3) * v**4 * v0**2 -
        (1062809371 / 20347200) * np.pi**2 * v**3 * v0**3 + 
        (-356873002170973/249880440692736 - 260399751935005/8924301453312 * eta + 
         150484695827/35413894656 * eta**2 + 340714213265/3794345856* eta**3) * v**2*v0**4 + 
        (26531900578691 / 168991764480 - 3317 / 126 * gamma_e + 122833 / 10368 * np.pi**2 +
        (9155185261 / 548674560 - 3977 / 1152 * np.pi**2) * eta -
        5732473 / 1306368 * eta**2 - 3090307 / 139968 * eta**3 +
        87419 / 1890 * np.log(2) - 26001 / 560 * np.log(3) -
        3317 / 252 * np.log(16 * v0**2)) * v0**6)

    psi_0 = -2*coallesence_phase + 2*np.pi*freqs*time_to_merger - np.pi/4
        
    psi = psi_0 + term1 * (term2 + term3 + term4 + term5 + term6)
    
    return(psi)

def time_to_merger(m1,m2,e0,f_0):

    '''
    Calculates time to merger using TaylorT2 Eqn 6.6a from arXiv:1605.00304v2

    Args:
        m1 (float): Mass of the first object in solar masses.
        m2 (float): Mass of the second object in solar masses.
        e0 (float): Initial eccentricity.
        f_0 (float): Initial GW frequency in Hz.

    Returns:
        float: Time to merger in seconds.
    '''
    m1 = m1*MTsun 
    m2 = m2*MTsun 
    
    M = m1+m2 
    eta = (m1*m2)/(M**2)# # Reduced mass ratio
    
    v_init = (np.pi*M*f_0)**(1/3)
    
    tm = (5/256* M/eta * 1/(v_init)**8*calculate_T(v_init, v_init, e0,eta))
    
    return(tm)

def estimate_initial_gw_frequency(m1, m2, e0, tc, logging=False):
    """
    Estimates the initial gravitational wave (GW) frequency given the parameters.

    This function uses simple rootfinding to estimate the initial GW frequency.
    The logging parameter allows for checking that the optimizer has converged.

    Args:
        m1 (float): The mass of the first object (solar masses).
        m2 (float): The mass of the second object (solar masses).
        e0 (float): The eccentricity of the orbit.
        tc (float): The time to coalescence (s).
        logging (bool, optional): Whether to enable logging. Defaults to False.

    Returns:
        float: The estimated initial GW frequency.
    """
    
    m1 = m1 * MTsun 
    m2 = m2 * MTsun

    M = m1 + m2 
    eta = (m1 * m2) / (M ** 2)  # Reduced mass ratio: m1m2/M**2
    
    time_to_merger_root_finding = lambda f_initial: (5 / 256 * M / eta * 1 / ((np.pi * M * f_initial) ** (1 / 3)) ** 8 * calculate_T((np.pi * M * f_initial) ** (1 / 3), (np.pi * M * f_initial) ** (1 / 3), e0, eta)) - tc

    # xtol is the difference in frequency between 1/year and 1/(year+1 second)
    f_low_rootfinder = scipy.optimize.root_scalar(time_to_merger_root_finding, bracket=[1e-4, 1e-1], x0=1.e-2, xtol=1.0055108727742241e-15)

    if logging:
        print('Root finding converged for f_low?: ', f_low_rootfinder.converged)
    
    return f_low_rootfinder.root

class TF2Ecc(Model):

    name='TaylorF2Ecc'
    num_harmonics = 1
    param_names = ['m1','m2','e0','D','initial_orbital_phase','f_low']

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
        self.e0 = self.parameters['e0']
        self.D  = self.parameters['D']
        self.initial_orbital_phase = self.parameters['phi0']
        self.f_low = self.parameters['f_low']

        self.f_high = [freqs[-1]]
        self.f0s = [self.f_low]

        # Unit conversions 
        self.m1 = self.m1*MTsun 
        self.m2 = self.m2*MTsun
        self.D = (self.D)*pc/c

        self.M = self.m1+self.m2 
        self.eta = (self.m1*self.m2)/(self.M**2)# # Reduced mass ratio #m1m2/M**2

        # Only compute the waveform for frequencies > the initial GW frequency
        freq_mask = freqs>=self.f_low
        self.freqs = np.asarray(self.freqs[freq_mask])

        # initial v 
        self.v0 = (np.pi*self.M*self.f_low)**(1/3) #(pi M f_0)**(1/3)
        # final v 
        self.v1 = (np.pi*self.M*self.f_high[0])**(1/3)
        # All vs
        self.v = (np.pi*self.M*self.freqs)**(1/3)

        self.time_to_merger = 5/256* self.M/self.eta * 1/(self.v0)**8*calculate_T(self.v0, self.v0, self.e0, self.eta)

        
        # Time to merger from the time the source exits the frequency band specified, no eccentricity evolution assumed
        time_to_merger_from_f_high = 5/256* self.M/self.eta * 1/(self.v1)**8*calculate_T(self.v1, self.v1, self.e0, self.eta)    
        print('Time to merger is: ',(self.time_to_merger)/(const.YRSID_SI),' years')
        print('Upper bound on time in band: ',(self.time_to_merger-time_to_merger_from_f_high)/(const.YRSID_SI),' years (no eccentricity evolution assumed)')


    def Amplitudes(self):
        '''
        Returns the amplitudes for each harmonic for the given parameters.
        '''

        Amp= (self.M*np.sqrt(5*np.pi/96)*(self.M/self.D)*np.sqrt(self.eta)*(np.pi*(self.M)*self.freqs)**(-7/6)) / np.sqrt(5/(64*np.pi))
        return([Amp])
    
    def Phases(self):
        '''
        Returns the phases for each harmonic for the given parameters.
        '''

        # Calculate coallesence phase using equn 6.6a from arXiv:1605.00304v2
        coallesence_phase = self.initial_orbital_phase-(-1/(32*self.v0**5*self.eta)*Lambda_f(self.v0, self.v0, self.e0, self.eta))

        # Waveform phase 
        Phi = F2EccPhase(self.freqs,self.eta,self.v,self.e0,self.v0,coallesence_phase,self.time_to_merger)+self.initial_orbital_phase
        return([Phi])
    
    def Time_frequency(self):
        '''
        Returns the time-frequency map array for each harmonic for the given parameters.
        '''
        # Calculate t-f map using 6.7a from arXiv:1605.00304v2
        time_to_merger_minus_time = 5/256* self.M/self.eta * 1/(self.v)**8*calculate_T(self.v, self.v0, self.e0, self.eta)

        #t-f map
        times = self.time_to_merger - time_to_merger_minus_time

        return([times])