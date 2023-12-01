# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:06:00 2023

@author: Borislav.Polovnikov
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize 
import scipy.interpolate as interpolate
from matplotlib.widgets import Button


class LineDrawer:
    '''
    Class to draw functions in matplotlib figures
    Parameters:
        line: an empty matplotlib line object which will be filled with x- and y-data upon drawing.
        A minimal example looks like line = axes.plot([], [])[0] 
        
        offclick_method: callable function that will be called once drawing is complete and the mouse-button is released
    '''
    def __init__(self, line, offclick_call=None):
        self.line = line
        self.fig = line.figure
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        if offclick_call is not None:
            self.offclick_call = offclick_call
        
        #open a connection to the gui that when the mouse is clicked inside the axis
        self.conn_mouse_click=self.fig.canvas.mpl_connect('button_press_event', self.mouse_clicked_on) 
    
            
    def mouse_clicked_on(self,event):
        ### when the mouse is clicked, open connections to the gui-events of mouse-motion (actual drawing) and mouse-release (call self.offclick_call)
        if  event.inaxes !=self.line.axes:
            return
        self.conn_mouse_release=self.fig.canvas.mpl_connect('button_release_event', self.mouse_released_on)
        self.conn_mouse_clicked=self.fig.canvas.mpl_connect('motion_notify_event', self.move_mouse_while_clicked)
        

        
    def mouse_released_on(self,event):
        ### when the mouse is released, disconnect all gui connections and call self.offclick_call
        
        if  event.inaxes !=self.line.axes:
            return
        
        self.fig.canvas.mpl_disconnect(self.conn_mouse_clicked)
        self.fig.canvas.mpl_disconnect(self.conn_mouse_release)
        self.offclick_call()
        
    
    def move_mouse_while_clicked(self,event):
        ### while the mouse is pressed and moved, append new values to the x- and y-data of self.line
        
        if  event.inaxes !=self.line.axes: # go sure the mouse is inside the active axis
            return
        if len(self.xs) == 0: # if you just start drawing and there is no data yet in the line, just add the coordinates of the mouse as the first elements
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        elif event.xdata > self.xs[-1]: # go sure, that you only add new x-points to the right of the previous ones
            self.xs.append(event.xdata)
            self.ys.append(max(event.ydata, self.ys[-1])) # this is to make the drawn function non-decreasing. If you want to draw decreasing functions as well, just add event.ydata
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


class charging:
    '''
    Class to compute and fit dual-gate charging behavior of gated semiconductor heterostructures. 
    '''
    def __init__(self, epsilon_hBN, epsilon_sample, E_offset_cb, dW=56, dMo=55, d_sample=0.6, \
               moire_density=1.9, W_G_voltages=np.linspace(-22,22,200), Mo_G_voltages=np.linspace(-22,22,200), E_0=0, cmap='terrain_r'):
        
        ################################
        ### constants and parameters ###
        ################################
        self.epsilon_hBN = epsilon_hBN
        self.epsilon_sample = epsilon_sample
        self.offset = E_offset_cb
        self.dW = dW
        self.dMo = dMo
        self.d_sample = d_sample
        self.moire_density = moire_density
        self.W_G_voltages = W_G_voltages
        self.Mo_G_voltages = Mo_G_voltages
        self.E_0 = E_0 # define an offset in the applied voltage / the intrinsic Fermi energy
        self.last_solution = [4., self.offset + 2.]  ## keep the solution to the last pixel as a better guess for the starting point of the calculation for the next pixel
        
        ###### (# factors such that E/e is counted is meV, n in 10^12/cm^2, d in nm) 
        ######  --> divide E by 1000 in the right_hand_side of the capacitor equations!
        self.C_W = self.epsilon_hBN/self.dW*5.53 #because of possible confusion, always plot the W-gate vertically and avoid top/bottom nomenclature
        self.C_Mo = self.epsilon_hBN/self.dMo*5.53
        self.Cs = self.epsilon_sample/self.d_sample*5.53
        
        ### Functions to evaluate the charging density n from the Fermi Energy E
        ### There are prebuild functions with discrete moire bands (discretized DOS with 1 electron per moire cell each),
        ### and custom functional forms can be drawn in the figure. n_Mo/n_W are the fixed functions, and func_Mo/func_W are the containers for
        ### custom functions
        self.func_Mo = self.n_Mo
        self.func_W = self.n_W
        self.energies = self.compute_energies(self.func_Mo, self.func_W)
        
        
        ########################
        # Plotting environment #
        ########################
        self.fig, self.axes = plt.subplots(figsize=(13,10), nrows=2, ncols=2)

        titles=['MoSe2 charging', 'WS2 charging', 'Draw MoSe2 n(E)', 'Draw WS2 n(E)']
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i])
        
        
        self.axes[0,0].set_xlabel(r'$U_{MoSe2-Gate}$')
        self.axes[0,0].set_ylabel(r'$U_{WS2-Gate}$')
        
        self.axes[1,0].set_xlabel(r'$E_F$' + ' (meV)')
        self.axes[1,0].set_ylabel(r'$n_{Mo} / n_0$')
        
        self.axes[1,1].set_xlabel(r'$E_F$' + ' (meV)')
        self.axes[1,1].set_ylabel(r'$n_{W}/n_0$')
        
        self.axes[1,0].set_xlim(-25,130)
        self.axes[1,0].set_ylim(-2,5.5)
        self.axes[1,1].set_xlim(-25,130)
        self.axes[1,1].set_ylim(-2,5.5)
        
        self.axes[0,1].sharex(self.axes[0,0])
        self.axes[0,1].sharey(self.axes[0,0])
        
        self.im_Mo = self.axes[0,0].imshow(self.func_Mo(self.energies[:,:,0])/self.moire_density , cmap=cmap, origin='lower',
                                    extent=[self.Mo_G_voltages[0], self.Mo_G_voltages[-1], self.W_G_voltages[0], self.W_G_voltages[-1] ], vmin = -0.1, vmax = 3.15)
        self.im_W = self.axes[0,1].imshow(self.func_W(self.energies[:,:,1])/self.moire_density, cmap=cmap, origin='lower',
                                    extent=[self.Mo_G_voltages[0], self.Mo_G_voltages[-1], self.W_G_voltages[0], self.W_G_voltages[-1] ], vmin = -0.1, vmax = 3.15)
        
        Es = np.linspace(-100, 250, 1000) #### plot the prebuild charging functions n_Mo/n_W to keep them as a reference
        self.axes[1,0].plot(Es,self.n_Mo(Es)/self.moire_density, ls='',marker='o',markersize=1, color='grey', alpha=0.5)
        self.axes[1,1].plot(Es,self.n_W(Es)/self.moire_density, ls='',marker='o',markersize=1, color='grey', alpha=0.5)
        
        self.cb_Mo=self.fig.colorbar(self.im_Mo, ax=self.axes[0,0], orientation = 'vertical',extend='max')
        self.cb_W=self.fig.colorbar(self.im_W, ax=self.axes[0,1], orientation = 'vertical',extend='max')
        cbar_title = r'$\nu$'
        #cbar_title = r'$n[\frac{10^{12}}{cm^{2}}]$'
        self.cb_Mo.ax.set_ylabel(cbar_title)
        self.cb_W.ax.set_ylabel(cbar_title)
        
        
        ####################################################
        ### Interactive classes to draw custom functions ###
        ####################################################
        self.line_Mo = self.axes[1,0].plot([], [], linestyle="-.", marker="o", color="b", markersize = 3)[0]
        self.linebuilder_Mo = LineDrawer(self.line_Mo, offclick_call = self.update_Mo)
        self.line_W = self.axes[1,1].plot([], [], linestyle="-.", marker="o", color="b", markersize = 3)[0]
        self.linebuilder_W = LineDrawer(self.line_W, offclick_call = self.update_W)
        
        
        ###################################################
        ### Reset buttons to reset the drawing function ###
        ###################################################
        posMo = self.axes[1,0].get_position()
        posW = self.axes[1,1].get_position()
        self.clear_button_axis_Mo = self.fig.add_axes([posMo.x0+posMo.width*0.4, posMo.y0 - 0.08, 0.08, 0.02])
        self.clear_button_Mo = Button(self.clear_button_axis_Mo, 'Reset')
        self.clear_button_Mo.on_clicked(self.clear_Mo)
        
        self.clear_button_axis_W = self.fig.add_axes([posW.x0+posW.width*0.4, posMo.y0 - 0.08, 0.08, 0.02])
        self.clear_button_W = Button(self.clear_button_axis_W, 'Reset')
        self.clear_button_W.on_clicked(self.clear_W)
        
        
        
    
    def update_Mo(self):
        #############################################################
        ### Function called when a custom n_Mo(E) curve was drawn ###
        #############################################################

        self.func_Mo = interpolate.interp1d( np.array(self.linebuilder_Mo.xs), np.array(self.linebuilder_Mo.ys)*self.moire_density,\
                                            bounds_error=False, fill_value='extrapolate') 
        
        self.energies = self.compute_energies(self.func_Mo,self.func_W)
        self.im_Mo.set_data(self.func_Mo(self.energies[:,:,0])/self.moire_density)
        self.im_W.set_data(self.func_W(self.energies[:,:,1])/self.moire_density)
        plt.draw()
    
    def update_W(self):
        ############################################################
        ### Function called when a custom n_W(E) curve was drawn ###
        ############################################################

        self.func_W = interpolate.interp1d( np.array(self.linebuilder_W.xs), np.array(self.linebuilder_W.ys)*self.moire_density, \
                                           bounds_error=False, fill_value='extrapolate') 
            
        self.energies = self.compute_energies(self.func_Mo,self.func_W)
        self.im_Mo.set_data(self.func_Mo(self.energies[:,:,0])/self.moire_density)
        self.im_W.set_data(self.func_W(self.energies[:,:,1])/self.moire_density)
        plt.draw()
    
    def clear_Mo(self, event):
        ############################################################
        ### Reset the function n_Mo to the prebuild one and reset ## 
        ### the drawing possibility in the Mo axis #################
        ############################################################
        self.linebuilder_Mo.xs = []
        self.linebuilder_Mo.ys = []
        self.linebuilder_Mo.line.set_data([], [])
        self.linebuilder_Mo.line.figure.canvas.draw()
        
        self.func_Mo = self.n_Mo
        self.energies = self.compute_energies(self.func_Mo, self.func_W)
        self.im_Mo.set_data(self.func_Mo(self.energies[:,:,0])/self.moire_density)
        self.im_W.set_data(self.func_W(self.energies[:,:,1])/self.moire_density)
        plt.draw()
        
    def clear_W(self, event):
        ############################################################
        ### Reset the function n_W to the prebuild one and reset ###
        ### the drawing possibility in the Mo axis #################
        ############################################################
        self.linebuilder_W.xs = []
        self.linebuilder_W.ys = []
        self.linebuilder_W.line.set_data([], [])
        self.linebuilder_W.line.figure.canvas.draw()
        
        self.func_W = self.n_W
        self.energies = self.compute_energies(self.func_Mo, self.func_W)
        self.im_Mo.set_data(self.func_Mo(self.energies[:,:,0])/self.moire_density)
        self.im_W.set_data(self.func_W(self.energies[:,:,1])/self.moire_density)
        plt.draw()
        
    def unit_step(self, x, a, b):
        '''
        Function to apply a characteristic function on a numpy-array.
        Parameters
        ----------
        x : argument of the step_function, float
        a : left_boundary, float
        b : right_boundary, float
    
        Returns
        -------
        1 if a < x < b,
        0 if x=a or x=b,
        0 otherwise.
    
        '''
        return np.heaviside(x-a,0) - np.heaviside(x-b,0)
    
    def linear_step(self, x, a, b):
        '''
        Function to apply a linear step function between a and b on a numpy-array.
        Parameters
        ----------
        x : argument of the step_function, float
        a : left_boundary, float
        b : right_boundary, float
    
        Returns
        -------
        1 if b < x ,
        0 if x < a ,
        linear interpolation between a and b.
    
        '''
        assert a < b
        return np.heaviside(x-b, 0) + (x-a)/(b-a)*(np.heaviside(x-a,0)-np.heaviside(x-b,0))
        
    def n_Mo(self, E):
        ##############################################
        ### Custom n_Mo function with discrete DOS ###
        ##############################################

        return self.moire_density*(self.linear_step(E,0,5) + self.linear_step(E,57,62) + self.linear_step(E,110,114)+ np.heaviside(E-117,0)*(E-117)/4.8)
    
    def n_W(self, E):
        #############################################
        ### Custom n_W function with discrete DOS ###
        #############################################
        
        return self.moire_density*(self.linear_step(E,self.offset,self.offset+5) + self.linear_step(E,self.offset + 40,self.offset + 45) + self.linear_step(E,self.offset + 72,self.offset + 75)+ np.heaviside(E-self.offset - 80,0)*(E-self.offset - 80)/3.0) 
    
    ###############################################################
    ### Wrap the capacitor equations and solve them by the ########
    ### scipy.root method with the Levenberg-Marquardt minimizer ##
    ###############################################################
    def wrapper_res(self, U_W, U_Mo, nMo, nW):
        def residuals(Ef):
            out1 = nMo(Ef[0]) - (self.C_Mo*U_Mo - (self.C_Mo + self.Cs)*(Ef[0]+self.E_0)/1000 + self.Cs*(Ef[1]+self.E_0)/1000)
            out2 = nW(Ef[1]) - (self.Cs*(Ef[0]+self.E_0)/1000 - (self.C_W + self.Cs)*(Ef[1]+self.E_0)/1000 + self.C_W*U_W)
            return [out1, out2]
        return residuals
    
    def fermi_energies(self, U_W, U_Mo, nMo, nW):
        return scipy.optimize.root(self.wrapper_res(U_W, U_Mo, nMo, nW), x0=self.last_solution).x
    
    def compute_energies(self, n_Mo, n_W):
        energies = np.zeros((len(self.W_G_voltages), len(self.Mo_G_voltages),2))
        for i, Uw in enumerate(self.W_G_voltages):
            for j,Um in enumerate(self.Mo_G_voltages):
                if Uw+Um > -1:
                    energies[i,j]=self.fermi_energies(Uw, Um, n_Mo, n_W)
                    self.last_solution = energies[i,j]
        
        return energies

if __name__=='__main__':
    #
    ####################################################################
    ### Create a colormap to adjust the colors to the experimental data
    ####################################################################
    import matplotlib as mpl
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        '''
        https://stackoverflow.com/a/18926541
        '''
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    cmap_base = 'terrain_r'
    vmin, vmax = 0.5, 1.0
    cmap = truncate_colormap(cmap_base, vmin, vmax)
    
    
    lower = np.array([(0.5215686274509804, 0.38760784313725494, 0.3589019607843138, 0.15),(plt.get_cmap('terrain_r'))(0.45)])
    W_Yl = mpl.colors.LinearSegmentedColormap.from_list('white_to_yellow', lower, N=100)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', np.vstack((W_Yl(np.linspace(0, 1, 50)), (plt.get_cmap('terrain_r'))(np.linspace(0.45, 1.0, 100)))))
    ###################################################################
    
    #########################################
    #### call the class with given parameters
    #########################################
    out=charging(epsilon_hBN=4.0, epsilon_sample=8, E_offset_cb=30, moire_density=2.0, d_sample=0.6,E_0 = 300, cmap = cmap)
    
    #########################################
    
    ####################################################################
    # Get the computed densities and plot the filling up of the charge #
    ####################################################################
    # Us = out.W_G_voltages
    # NMo = np.diag(out.func_Mo(out.energies[:,:,0]))
    # NW = np.diag(out.func_W(out.energies[:,:,1]))
   
    # from matplotlib.ticker import MaxNLocator
    # cm=1/2.54
    # fig, ax = plt.subplots(figsize=(5*cm,4*cm))
    # fig.subplots_adjust(top=0.99,left=0.16,right=0.95,bottom=0.2)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.tick_params(axis='both', which='major', direction = 'in', width = 0.5, length = 2.1, top=True, bottom=True, right=True, left=True, labelsize=7)
    
    # clw = (0, 176/255, 240/255, 0.8)
    # clmo = (1, 83/255, 83/255, 0.8)
    # ax.plot(Us,NMo/out.moire_density,lw=1.5, color=clmo, label=r'$n_{MoSe_2}$')
    # ax.plot(Us, NW/out.moire_density, lw=1.5, color=clw, label=r'$n_{WS_2}$')
    # ax.plot(Us, (NMo+NW)/out.moire_density, lw=2, color = 'black', ls='-.', label=r'$n_{total}$')
    # ax.set_xlabel(r'$V_G$' + '(V)', fontsize=8)
    # ax.set_ylabel(r'$n/n_0$',fontsize=8)
    # ax.set_xlim(-2,11)
    # ax.set_ylim(-0.2,2.7)
    # ax.legend(frameon=False, fontsize=7.85)
        
        
    
    