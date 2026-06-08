import numpy as np 
from manim import *
from manim import WHITE, BLACK
from manim import config as global_config

from .Sources import TaylorF2Ecc
from .Sources import PhenomTHM
from functools import partial
import copy
from .Confusion import Add_confusion, psd_SCIRD

class Tracks(Scene):
    '''
    Class for creating the final animation tracks
    '''

    def __init__(self,
                 T_obs,
                 frequency_limits,
                 freq_resolution,
                 sources,
                 y_min=1.e-21,
                 y_max=1.e-19,
                 t_min = 1.e-10,
                 run_time=25,
                 psd_color='red',
                 light_or_dark_mode = 'dark',
                 render_axes = True,
                 render_mission_timer = True,
                 axis_label_fontsize = 25,
                 source_label_fontsize = 20,
                 general_text_fontsize=27) -> None:
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
            axis_label_fontsize (int): Fontsize for the axis labels (Defaults to 25).
            source_label_fontsize (int): Fontsize for the source labels (Defaults to 20).
            general_text_fontsize (int): Fontsize for the general text (Defaults to 27).
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

        self.source_label_fontsize = source_label_fontsize
        self.general_text_fontsize = general_text_fontsize

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
            y_axis_config={"scaling": LogBase(custom_labels=True),"font_size":axis_label_fontsize,"include_tip":False},
            x_axis_config={"scaling": LogBase(custom_labels=True),"font_size":axis_label_fontsize,"include_tip":False},
            axis_config={"include_numbers": True,"color":self.axes_color},)
        
        # Force axes colours
        for tick in self.ax.get_x_axis():
            tick.set_color(self.axes_color)
 
        for tick in self.ax.get_y_axis():
            tick.set_color(self.axes_color)


        # Axes labels
        self.axes_labels = self.ax.get_axis_labels(MathTex('f \mathrm{[Hz]}',font_size = axis_label_fontsize), 
                            Tex(r'Characteristic strain',font_size = axis_label_fontsize)).set_color(self.axes_color)




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
        # Per-harmonic mode labels (e.g. '(2,2)') for each source, or None if the source has no
        #       meaningful mode decomposition (e.g. TaylorF2Ecc).
        self.source_mode_labels = []
        # Whether to only show a source while it is within its valid time-frequency window (used for
        #       merging sources like PhenomTHM). Inspiral-only sources (TaylorF2Ecc) stay visible.
        self.source_gated = []

        for source in self.sources:

            if source[0] == 'TaylorF2Ecc':
                # Initialise source
                smbbh = TaylorF2Ecc.TF2Ecc(source[2],self.freqs,self.T_obs)
                # Generate splines
                t_f_spline, amplitude_time_spline, time_windows = smbbh.generate_track_splines(self.freqs)
                mode_labels = [None]*len(t_f_spline)
                gated = False

            elif source[0] == 'PhenomTHM':
                # Initialise source (one track per mode, treated as harmonics)
                mbbh = PhenomTHM.PhenTHM(source[2],self.freqs,self.T_obs)
                # Generate splines
                t_f_spline, amplitude_time_spline, time_windows = mbbh.generate_track_splines(self.freqs)
                mode_labels = [f'({l},{m})' for (l,m) in mbbh.modes]
                gated = True

            # Add other sources as else statements
            else:
                assert False, "Source type not implemented"

            self.source_names.append(source[1]['Name'])

            self.source_colors.append(source[1]['Color'])

            self.source_splines.append([t_f_spline,amplitude_time_spline,time_windows])

            self.source_mode_labels.append(mode_labels)

            self.source_gated.append(gated)
    def ASD_with_confusion(self,freqs):
        '''
        Calculate the ASD with confusion noise for the current mission time.
        '''

        # Given some mission time, calcute the PSD and add the bump to it 
        psd = psd_SCIRD(freqs)
        psd_with_confusion = Add_confusion(freqs, psd, self.mission_time_tracker.get_value())
        # Return ASD not PSD 
        return(np.sqrt(psd_with_confusion))



    def position_at_time(self,dot,spline_t_f=None,spline_t_A=None,t_start=None,t_end=None,gate=True):
        '''
        Used to move the dot along the spline at the current time, and (if gated) to hide it outside
        of the interval over which its track is well defined.

        Args:
            dot (Dot): Dot to move along the spline.
            spline_t_f (CubicSpline (scipy)): Spline for the frequency (as a function of time).
            spline_t_A (CubicSpline (scipy)): Spline for the amplitude (as a function of time).
            t_start (float): Mission time at which the source enters the band (track becomes visible).
            t_end (float): Mission time at which the track ends (e.g. coalescence); the source is removed after this.
            gate (bool): If True, hide the dot outside [t_start,t_end]. If False, the dot stays visible (used for inspiral-only sources).
        '''

        time = self.mission_time_tracker.get_value()

        # Hide the dot before the source enters the band and after its track ends (gated sources only)
        if gate:
            dot.set_opacity(0 if (time < t_start or time > t_end) else 1)

        # Clamp the evaluation to the valid interval so the spline is never extrapolated
        time = min(max(time,t_start),t_end)

        position = self.ax.c2p(spline_t_f(time),spline_t_A(time))

        dot.move_to(position)

    def update_visibility(self,mobject,t_start=None,t_end=None):
        '''
        Show a mobject only while the current mission time lies within [t_start,t_end].
        '''
        time = self.mission_time_tracker.get_value()
        mobject.set_opacity(0 if (time < t_start or time > t_end) else 1)

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
        # Per-mode labels (e.g. '(2,2)') for each harmonic of each source
        mode_labels = []

        for source_index,source_spline_container in enumerate(self.source_splines):
            # Adds a dot to the plot at the initial position (f,A) for the source
            num_harmonics = len(source_spline_container[0])
            # Whether to hide this source outside of its valid time-frequency window
            gated = self.source_gated[source_index]

            for i in range(num_harmonics):
                spline_t_f = copy.deepcopy(source_spline_container[0][i])
                spline_t_A = copy.deepcopy(source_spline_container[1][i])
                # Mission-time interval over which this harmonic's track is well defined
                t_start, t_end = source_spline_container[2][i]

                # Place the dot at the position where the source enters the band
                tracer = Dot(point=np.array(self.ax.c2p(spline_t_f(t_start),spline_t_A(t_start))),
                                color=self.source_colors[source_index],radius=0.06)

                # Traced path for the source
                trace = TracedPath(tracer.get_center,stroke_width=5,stroke_color=self.source_colors[source_index])


                # Partial function to lock in the arguments for the spline at the current source and harmonic and so it wont change as in
                #       https://stackoverflow.com/questions/66131048/python-lambda-function-is-not-being-called-correctly-from-within-a-for-loop
                trace_func = partial(self.position_at_time,spline_t_f=spline_t_f,spline_t_A=spline_t_A,t_start=t_start,t_end=t_end,gate=gated)

                tracer.add_updater(trace_func)
                # Hide the trace before the source enters the band and after its track ends (gated sources only)
                if gated:
                    trace.add_updater(partial(self.update_visibility,t_start=t_start,t_end=t_end))

                tracers.append(tracer)
                traces.append(trace)

                # Add a small mode label (e.g. '(2,2)') above the dot, in the source colour
                mode_label_text = self.source_mode_labels[source_index][i]
                if mode_label_text is not None:
                    mode_label = MathTex(mode_label_text,font_size=self.source_label_fontsize*0.6,color=self.source_colors[source_index])
                    mode_label.add_updater(partial(self.move_label_to_dot,tracer=tracer))
                    mode_label.add_updater(partial(self.update_visibility,t_start=t_start,t_end=t_end))
                    mode_labels.append(mode_label)

            label = Tex(self.source_names[source_index],font_size=self.source_label_fontsize,color = self.text_color)
            position_func  = partial(self.move_label_to_dot,tracer=tracers[-1])
            # label.add_updater(lambda d: d.next_to(tracers[-1],UP))
            label.add_updater(position_func)
            # Show the source name only while a gated source is visible
            if gated:
                source_window = source_spline_container[2]
                source_start = min(window[0] for window in source_window)
                source_end = max(window[1] for window in source_window)
                label.add_updater(partial(self.update_visibility,t_start=source_start,t_end=source_end))


            labels.append(label)


        # Initial state of the PSD
        ASD_plot = self.ax.plot(lambda freqs: np.sqrt(freqs)*self.ASD_with_confusion(freqs),color='red')

        # Set the ASD to update with the confusion noise as the mission timer updates. 
        ASD_plot.add_updater(lambda m: m.become(self.ax.plot(lambda freqs: np.sqrt(freqs)*self.ASD_with_confusion(freqs),color=self.psd_color)))


        # #Setting up label
        T_label = DecimalNumber(0,num_decimal_places=2,font_size=self.general_text_fontsize,color=self.text_color)
        # UR being upper right
        T_label.to_edge(UR)
        T_label.add_updater(lambda d: d.set_value(self.mission_time_tracker.get_value()/(365.25*24*60*60)))

        T_label_ = Tex(r'Time from beginning of LISA mission (years) :',font_size=self.general_text_fontsize,color=self.text_color)
        T_label_.add_updater(lambda d: d.next_to(T_label,LEFT))

        if self.render_axes: 
            self.add(self.ax,self.axes_labels)
        if self.render_mission_timer:
            self.add(T_label,T_label_)
            
        self.add(*tracers,*traces,*labels,*mode_labels,ASD_plot)
        self.wait()
        
        # Increment the T_years all the way slowly to 4 years, this will automatically update our plot due to the updater function above
        self.play(ApplyMethod(self.mission_time_tracker.increment_value,self.T_obs),run_time=self.run_time,rate_func=linear)
        
        # self.play(FadeOut(T_label),FadeOut(T_label_),run_time=1)
        self.wait(3)
    
