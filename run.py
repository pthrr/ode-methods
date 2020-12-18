#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Simulate a first-order RC low-pass filter of the form:
           _____
    o-----|_____|----o-------o
             R       |
                   -----
                   -----
                     |  C
    o----------------o-------o

    with its Differential Equation:

    vin - vout = RC * d/dt * vout, which is equivalent to:
    d/dt * vout = (vin - vout) / RC.

    We use the cut-off frequency f_c = 1/(2*pi*159e-6) = 1e3 Hz.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
RC = 159.15e-6 # time constant (tau)
f_c = 1e3 # cut-off frequency
f_s = 100*f_c # sample rate
h = 1.0/f_s # step width

class Animation():
    """
        Class implementing an animated figure
        xlabel: label of the x-axis
        ylabel: label of the y-axis
        title: title of the figure
        xlim: limits of the x-axis
        ylim: limits of the y-axis
    """
    def __init__(self, \
            xlabel='Simulation time / s', \
            ylabel='Output voltage / V', \
            title='Response of first-order low pass filter with cut-off ' + \
              'frequency and excitation at 1 kHz and input amplitude of 1 V', \
            xlim=(0, 100), \
            ylim=(-1.1, 1.1)):
        """
            Constructor
        """
        # create plot figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def __del__(self):
        """
            Destructor
        """
        # destroy plot figure
        plt.close(self.fig)

    @property
    def pane(self):
        """
            Provides animated figure as class property
        """
        return self.fig

    def attach_functions(self, *functions):
        """
            Attaches anonymous functions to class
            functions: tuple of functions,their names and fmt string as
                       separate parameters
            returns: nothing
        """
        # save functions, their respective names and fmt strings
        self.styles = [style for _,_,style in functions]
        self.names = [name for _,name,_ in functions]
        self.functions = [f for f,_,_ in functions]
        n_fun = len(self.functions)

        # initialize state for each function
        self.states = np.zeros(n_fun)

        # initialize values for each function
        self.x = [np.zeros(1) for _ in range(n_fun)]
        self.y = [np.zeros(1) for _ in range(n_fun)]

        # initialize line for each function
        self.lines = [object() for _ in range(n_fun)]

    def update(self, frame):
        """
            Gets periodically called by animation routine
            frame: running number passed by animation routine for each frame
            returns: list with line2D object for each function
        """
        # generate test signal at cut-off frequency
        t = frame/f_s
        vin1 = 1.0 * np.sin(2*np.pi*f_c*t)
        vin12 = 1.0 * np.sin(2*np.pi*f_c*(t + h/2))
        vin2 = 1.0 * np.sin(2*np.pi*f_c*(t + h))

        # do calculation for each function
        for idx,fun in enumerate(self.functions):
            vout = fun(vin1, vin12, vin2, self.states[idx])

            # collect data
            self.x[idx] = np.append(self.x[idx], t)
            self.y[idx] = np.append(self.y[idx], self.states[idx])

            if len(self.x[idx]) > ((1.0/f_c)*f_s):
                self.x[idx] = self.x[idx][1:]

            if len(self.y[idx]) > ((1.0/f_c)*f_s):
                self.y[idx] = self.y[idx][1:]

            # save data for next step
            self.states[idx] = vout

        # update plot
        for idx,fun in enumerate(self.functions):
            self.lines[idx].set_data(self.x[idx], self.y[idx])

        self.ax.set_xlim([min([x.min() for x in self.x]), \
                          max([x.max() for x in self.x])])

        return self.lines,

    def start_animation(self):
        """
            Creates empty plot and starts animation with update as callback
            returns: nothing
        """
        # plot initial values for each function
        for idx,fun in enumerate(self.functions):
            line, = self.ax.plot(self.x[idx], self.y[idx], self.styles[idx], \
                    label=self.names[idx])
            self.lines[idx] = line

        # now that names are plotted, toggle legend
        self.ax.legend()

        # start animation
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=1)

def f(tau, vin, vout):
    """
        Explicit ODE to solve
    """
    return (vin - vout) / tau

if __name__ == "__main__":
    # functions for different calculation methods
    euler_f = lambda vin1, vin12, vin2, vout: \
            vout + h * \
            f(RC, vin1, vout) # Eulers method
    mid_f = lambda vin1, vin12, vin2, vout: \
            vout + h * \
            f(RC, vin12, vout+0.5*h* \
            f(RC, vin1, vout)) # Midpoint method
    heun_f = lambda vin1, vin12, vin2, vout: \
            vout + h/2 * ( \
            f(RC, vin1, vout) + \
            f(RC, vin2, vout + h * \
            f(RC, vin1, vout))) # Heuns method

    # start animation
    a = Animation()
    a.attach_functions((euler_f, 'Eulers method', 'b-'), \
                       (mid_f, 'Midpoint method', 'r--'), \
                       (heun_f, 'Heuns method', 'g-.'))
    a.start_animation()
    a.pane.show()

    input() # block here
