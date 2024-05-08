import numpy as np 
from manim import *
from manim import WHITE, BLACK
from manim import config as global_config

from .Sources import TaylorF2Ecc, IMRPhenomHM, FEW
from functools import partial
import copy
from .Confusion import Add_confusion, psd_SCIRD

class Tracks(Scene):
    '''
    Class for creating the final animation tracks
    '''

    def __init__(self,T_obs,frequency_limits,freq_resolution,sources,
                 y_min=1.e-21,
                 y_max=1.e-19,
                 t_min = 1.e-10,
                 run_time=25,
                 psd_color='red',
                 light_or_dark_mode = 'dark',
                 render_axes = True,
                 render_mission_timer = True) -> None:
        '''
        Args:
            T_obs (float): Observation time.
            frequency_limits (list/tuple): Frequency limits for the plot.
            freq_resolution (int): Frequency resolution (Used logspaced).
            sources (list): list of sources, each source is a tuple of (source_type,stylistic_params,parameters), parameters is a dictionary of source parameters.
            y_min (float): Minimum strain for the plot (Defaults to 1.e-21).
            y_max (float): Maximum strain for the plot (Defaults to 1.e-19).
            t_min (float): Minimum time for the animation (Defauls to 1.e-10 to avoid T=0 errors).
            run_time (float): Total run time for the animation (Defaults to 25s).
            psd_color (str): Color of the ASD plot (Defaults to 'red').
            light_or_dark_mode (str): Light or dark mode for the plot (Defaults to 'dark').
            render_axes (bool): Render the axes (Defaults to True).
            render_mission_timer (bool): Render the mission time (Defaults to True).
      '''
        
        self.f_low = frequency_limits[0]
        self.f_high = frequency_limits[1]
        # Mission lifetime
        self.T_obs = T_obs
        self.freq_resolution = freq_resolution

        self.y_min = y_min
        self.y_max = y_max

        self.psd_color = psd_color

        # Master frequency array for the plot
        self.freqs = np.logspace(np.log10(self.f_low),np.log10(self.f_high),self.freq_resolution)

        # Used as the master time for the animation
        self.mission_time_tracker = ValueTracker(t_min)

        self.run_time = run_time

        self.sources = sources

        # Create the sources/Generate the splines
        self.generate_splines()

        self.light_or_dark_mode = light_or_dark_mode

        # Set color-scheme based on light or dark mode
        if self.light_or_dark_mode == 'light':
            self.background_color = WHITE
            self.axes_color = BLACK
            self.text_color = BLACK 
        elif self.light_or_dark_mode == 'dark':
            self.background_color = BLACK
            self.axes_color = WHITE
            self.text_color = WHITE
        
        self.render_axes = render_axes
        self.render_mission_timer = render_mission_timer

        config.background_color = self.background_color

        
        super().__init__()
        # Create the axes we will be animating on
        self.ax = Axes(x_range=[np.log10(self.f_low),np.log10(self.f_high),1],
            y_range=[np.log10(self.y_min),np.log10(self.y_max),1],
            y_axis_config={"scaling": LogBase(custom_labels=True),"font_size":25,"include_tip":False},
            x_axis_config={"scaling": LogBase(custom_labels=True),"font_size":25,"include_tip":False},
            axis_config={"include_numbers": True,"color":self.axes_color},)
        
        # Force axes colours
        for tick in self.ax.get_x_axis():
            tick.set_color(self.axes_color)
 
        for tick in self.ax.get_y_axis():
            tick.set_color(self.axes_color)


        # Axes labels
        self.axes_labels = self.ax.get_axis_labels(MathTex('f [Hz]').scale(0.6), 
                            Tex(r'Strain $\rm{1/\sqrt{Hz}}$' ).scale(0.6)).set_color(self.axes_color)




    def generate_splines(self):
        '''
        Generate the waveform splines for every source.

        The datastructure it ends up creating is the following

        source_splines = [[t_f_spline_1,amplitude_time_spline_1],[t_f_spline_2,amplitude_time_spline_2],...]   

        Where:
        
        t_f_spline_i : [[t_f_source_i_harmonic_1],[t_f__source_i_harmonic_2_],...,[t_f_source_i_harmonic_n]]
        amplitude_time_spline_i : [[amplitude_time_source_i_harmonic_1],[amplitude_time_source_i_harmonic_2],...,[amplitude_time_source_i_harmonic_n]]

        '''
        # Each source is a tuple of (source_type,parameters)
        self.source_names = []
        self.source_splines = []
        self.source_colors = []

        for source in self.sources:

            if source[0] == 'TaylorF2Ecc':
                # Initialise source
                smbbh = TaylorF2Ecc.TF2Ecc(source[2],self.freqs,self.T_obs)
                # Generate splines 
                t_f_spline, amplitude_time_spline = smbbh.generate_track_splines(self.freqs)

            elif source[0] == 'IMRPhenomHM':
                # Initialise source
                imrphenomhm = IMRPhenomHM.IMRPhenomHM(source[2],self.freqs,self.T_obs)
                # Generate splines 
                t_f_spline, amplitude_time_spline = imrphenomhm.generate_track_splines(self.freqs)

            elif source[0] == 'EMRI':
                # Initialise source
                emri = FEW.FEW_FD(source[2],self.freqs,self.T_obs)
                # Generate splines
                t_f_spline, amplitude_time_spline = emri.generate_track_splines(self.freqs)


            # Add other sources as else statements 
            else:
                assert False, "Source type not implemented"

            self.source_names.append(source[1]['Name'])

            self.source_colors.append(source[1]['Color'])

            self.source_splines.append([t_f_spline,amplitude_time_spline])
    def ASD_with_confusion(self,freqs):
        '''
        Calculate the ASD with confusion noise for the current mission time.
        '''

        # Given some mission time, calcute the PSD and add the bump to it 
        psd = psd_SCIRD(freqs)
        psd_with_confusion = Add_confusion(freqs, psd, self.mission_time_tracker.get_value())
        # Return ASD not PSD 
        return(np.sqrt(psd_with_confusion))



    def position_at_time(self,dot,spline_t_f=None,spline_t_A=None):
        '''
        Used to move the dot along the spline at the current time. 

        Args:
            dot (Dot): Dot to move along the spline.
            spline_t_f (CubicSpline (scipy)): Spline for the frequency (as a function of time).
            spline_t_A (CubicSpline (scipy)): Spline for the amplitude (as a function of time).

        '''

        position = self.ax.c2p(spline_t_f(self.mission_time_tracker.get_value()),
                                                                        spline_t_A(self.mission_time_tracker.get_value()))
        
        dot.move_to(position)

    def move_label_to_dot(self,label,tracer=None):
        '''
        Move the source label to the dot.
        '''
        label.next_to(tracer,UP)

    

    def construct(self,):
        '''
        Main function to construct the animation.
        This function is called by manim internally to create the animation.
        '''
        # Holding lists for tracing dot each harmonic of each sources
        tracers = []
        # Holding lists for the traces of each harmonic of each source
        traces = []
        # Labels for each source
        labels = []

        for source_index,source_spline_container in enumerate(self.source_splines):
            # Adds a dot to the plot at the initial position (f,A) for the source
            num_harmonics = len(source_spline_container[0])

            for i in range(num_harmonics):
                tracer = Dot(point=np.array(self.ax.c2p(source_spline_container[0][i](0),source_spline_container[1][i](0))),
                                color=self.source_colors[source_index],radius=0.06)
                
                # Traced path for the source
                trace = TracedPath(tracer.get_center,stroke_width=5,stroke_color=self.source_colors[source_index])


                # Partial function to lock in the arguments for the spline at the current source and harmonic and so it wont change as in 
                #       https://stackoverflow.com/questions/66131048/python-lambda-function-is-not-being-called-correctly-from-within-a-for-loop
                trace_func = partial(self.position_at_time,spline_t_f=copy.deepcopy(source_spline_container[0][i]),spline_t_A=copy.deepcopy(source_spline_container[1][i]))

                tracer.add_updater(trace_func)
                
                tracers.append(tracer)
                traces.append(trace)

            label = Tex(self.source_names[source_index],font_size=20,color = self.text_color)
            position_func  = partial(self.move_label_to_dot,tracer=tracers[-1])
            # label.add_updater(lambda d: d.next_to(tracers[-1],UP))
            label.add_updater(position_func)


            labels.append(label)


        # Initial state of the PSD
        ASD_plot = self.ax.plot(lambda freqs: np.sqrt(freqs)*self.ASD_with_confusion(freqs),color='red')

        # Set the ASD to update with the confusion noise as the mission timer updates. 
        ASD_plot.add_updater(lambda m: m.become(self.ax.plot(lambda freqs: np.sqrt(freqs)*self.ASD_with_confusion(freqs),color=self.psd_color)))


        # #Setting up label
        T_label = DecimalNumber(0,num_decimal_places=2,font_size=27,color=self.text_color)
        # UR being upper right
        T_label.to_edge(UR)
        T_label.add_updater(lambda d: d.set_value(self.mission_time_tracker.get_value()/(365.25*24*60*60)))

        T_label_ = Tex(r'Time from beginning of LISA mission (years) :',font_size=27,color=self.text_color)
        T_label_.add_updater(lambda d: d.next_to(T_label,LEFT))

        if self.render_axes: 
            self.add(self.ax,self.axes_labels)
        if self.render_mission_timer:
            self.add(T_label,T_label_)
            
        self.add(*tracers,*traces,*labels,ASD_plot)
        self.wait()
        
        # Increment the T_years all the way slowly to 4 years, this will automatically update our plot due to the updater function above
        self.play(ApplyMethod(self.mission_time_tracker.increment_value,self.T_obs),run_time=self.run_time,rate_func=linear)
        
        # self.play(FadeOut(T_label),FadeOut(T_label_),run_time=1)
        self.wait(3)
    