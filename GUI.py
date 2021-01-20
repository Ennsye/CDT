#!/usr/bin/env python3

import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import matplotlib.animation as animation

import copy
import importlib
import inspect
import math
import multiprocessing
import networkx as nx
import numpy as np
import os
from pathlib import Path
import pickle
import sys
import time
import traceback


import CDTtools # a package that contains, among other things, dyn_core_tools.py
import custom_functions # a package that handles the various custom functions used for torque specification

toplevel_path = os.path.dirname(os.path.realpath('__file__'))

class CDT_GUI:
    def __init__(self, master):
        open('logfile', 'w').close() # wipe log file
        self.cpr, self.cps = multiprocessing.Pipe(duplex=False) # one-way pipe from solver process to parent
        # cpr is the receiving end
        self.rp = 4 # number of digits after decimal in rounded values
        self.cspcalls = 0
        self.solver_runningB = tk.BooleanVar() # is a solver currently running? 
        self.solver_runningB.set(False) 
        self.solver_runningB.trace_add("write", self._solver_state_change)
        # needs to be set before adding the trace, else the state change callback tries to disable
        # a button that doesn't yet exist
        self.prev_solve_type = None # needed by postproc and solver diagnostics to tell what kind
        # of run the self.sol object came from, as the user may have changed simTypeVar since the 
        # previous run
        
        self.master = master
        master.title("Catapult Design Tool")
        tk.Tk.report_callback_exception = self.show_error # turns off the default "silent failure mode"...
        self.back = tk.Frame(master=self.master)
#         self.back.pack_propagate(0) #Don't allow the widgets inside to determine the frame's width / height
        self.back.pack(fill=tk.BOTH, expand=1) #Expand the frame to fill the root window
        
        # PLOT WINDOW 1
        self.fig1 = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        self.ax1 = self.fig1.add_subplot(111)
        
        self.plotFrame1 = tk.Frame(master=self.back)
        self.plotFrame1.grid(row=2, column=0, rowspan=2, columnspan=2)
        self.plot1SubFrame = tk.Frame(master=self.plotFrame1)
        self.plot1SubFrame.grid(row=0, column=0, columnspan=2)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.plot1SubFrame)  # A tk.DrawingArea.
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack()
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.plot1SubFrame)
        self.toolbar1.update()
        
        self.p1_conditional_controls = tk.Frame(master=self.plotFrame1)
        self.p1_conditional_controls.grid(row=2, column=0, columnspan=2) # stuff like the play button for the animation
        # navigation toolbars use pack() internally, so you can't place them with grid
        # so you create a frame that's a child of the main frame, position the *child frame* using grid, 
        # and the toolbar packs itself within that frame...
        
        tk.Label(master=self.plotFrame1, text="Plot Type").grid(row=1, column=0)
        self.p1TypeVar = tk.StringVar()
        self.p1TypeVar.trace_add("write", self._change_p1)
        self.p1types = ['animation', 'arm load', 'projectile path', 
                       'efficiency', 'puller potential energy', 'axle reaction load']
        self.prevP1Type = None # necessary because destroy doesn't properly destroy all widgets in p1 cond frame
        self.p1TypeVar.set(self.p1types[0])
        self.p1TypeMenu = tk.OptionMenu(self.plotFrame1, self.p1TypeVar, *self.p1types)
        self.p1TypeMenu.grid(row=1, column=1)
        # PLOT WINDOW 1
        
        # PLOT WINDOW 2
        self.fig2 = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        self.ax2 = self.fig2.add_subplot(111)
        
        self.plotFrame2 = tk.Frame(master=self.back)
        self.plotFrame2.grid(row=2, column=3, rowspan=2)
        self.plot2SubFrame = tk.Frame(master=self.plotFrame2)
        self.plot2SubFrame.grid(row=0, column=0)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.plot2SubFrame)  # A tk.DrawingArea.
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack()
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.plot2SubFrame)
        self.toolbar2.update()
        
        self.p2ControlF = tk.Frame(master=self.plotFrame2) # child frame for plot 2 controls
        self.p2ControlF.grid(row=1, column=0)
        tk.Label(master=self.p2ControlF, text="x axis").grid(row=0, column=0)
        self.p2xVar = tk.StringVar()
        self.p2xOpts = ['time', 'theta', 'psi', 'projectile velocity']
        self.p2xVar.set(self.p2xOpts[0])
        self.p2xVar.trace_add("write", self._change_p2)
        self.p2xMenu = tk.OptionMenu(self.p2ControlF, self.p2xVar, *self.p2xOpts)
        self.p2xMenu.grid(row=0, column=1)
        tk.Label(master=self.p2ControlF, text="y axis").grid(row=0, column=2)
        self.p2yVar = tk.StringVar()
        self.p2yOpts = ['sling tension', 'projectile velocity', 'theta', 'psi', 
                        'arm rotational speed', 'puller speed']
        self.p2yVar.set(self.p2yOpts[0])
        self.p2yVar.trace_add("write", self._change_p2)
        self.p2yMenu = tk.OptionMenu(self.p2ControlF, self.p2yVar, *self.p2yOpts)
        self.p2yMenu.grid(row=0, column=3)        
        # PLOT WINDOW 2
        
        # MAIN CONTROLS
        self.mcFrame = tk.Frame(master=self.back)
        self.mcFrame.grid(row=3, column=2)
        self.simMsgM = tk.Message(master=self.mcFrame, text="", width=256, bg='white')
        self.simMsgM.grid(row=0, column=0, rowspan=2)
        self.msg_text = tk.StringVar()
        self.msg_text.trace_add("write", self._update_simMsg)
        self.msg_text.set("") # contents of simMsgM
        
        self.run_button = tk.Button(master=self.mcFrame, text="Start Solver", command=self._simulate)
        self.run_button.grid(row=0, column=1)
        
        self.stop_button = tk.Button(master=self.mcFrame, text="Stop Solver", command=self._stopsolve)
        self.stop_button.grid(row=1, column=1)
        self.stop_button.config(state=tk.DISABLED)
        # MAIN CONTROLS
        
        # SAVE AND LOAD BUTTONS
        self.saveFrame = tk.Frame(master=self.back)
        self.saveFrame.grid(row=0, column=0, columnspan=2, sticky=tk.NW)
        self.saveBtn = tk.Button(master=self.saveFrame, text="Save Design", command=self._save_design)
        self.saveBtn.grid(row=0, column=0)
        
        self.loadBtn = tk.Button(master=self.saveFrame, text="Load Design", command=self._load_design)
        self.loadBtn.grid(row=0, column=1)
        
        self.quit_button = tk.Button(master=self.saveFrame, text="Quit", command=self._quit)
        self.quit_button.grid(row=0, column=2)
        # SAVE AND LOAD BUTTONS
        
        # DYNAMICS PARAMETERS
        self.dpFrame = tk.LabelFrame(master=self.back, text="Dynamics Parameters")
        self.dpFrame.grid(row=1, column=0)
        
        DPF = self.dpFrame #shorthand name
        # correct order: (La, Ls, ds, mb, rc, Ia, mp, g)
        self.DPEL = ("Arm Length", "Sling Length", "Sling Linear Density", "Arm Mass", 
                "Distance from Pivot to Arm COM", "Arm Rotational Inertia", "Projectile + Pouch Mass", 
               "Gravitational Acceleration") # ORDER IS IMPORTANT!
        DPF.CW = self._gen_entries2(DPF, self.DPEL, (0,0), dict(), maxcol=len(self.DPEL))
        DPF.CW['Gravitational AccelerationE'].insert(0, "9.8")
        # DYNAMICS PARAMETERS
        
        # INITIAL CONDITIONS
        self.icFrame = tk.LabelFrame(master=self.back, text="Initial Conditions")
        self.icFrame.grid(row=1, column=1) # between the plot windows

        tk.Label(master=self.icFrame, text="\u03B8").grid(row=0, column=0)
        self.t0SV = tk.StringVar()
        self.theta0E = tk.Entry(master=self.icFrame, textvariable=self.t0SV)
        self.theta0E.grid(row=0, column=1)
        self.theta0E.insert(0, "3.14")

        self.ICEL = ("d\u03B8/dt", "\u03C8", "d\u03C8/dt")
        self.icFrame.CW = self._gen_entries2(self.icFrame, self.ICEL, (1,0), dict(), maxcol=len(self.ICEL))
        self.icFrame.CW['d\u03B8/dtE'].insert(0, "0")
        self.icFrame.CW['d\u03C8/dtE'].insert(0, "0")
        # INITIAL CONDITIONS
        
        # SIMULATION CONTROLS
        self.scFrame = tk.LabelFrame(master=self.back, text="Solver Controls")
        self.scFrame.grid(row=1, column=3)

        self.scCondFrame = tk.Frame(master=self.scFrame) 
        # frame containing the conditional entries, which depend on simtype
        self.scCondFrame.grid(row=4, column=0, columnspan=2)
        
        # --- widget definition for scCondFrame ---
        self.simtypes = ['psi', 'launch_angle', 'sling_len_opt', 'arm_inertia_opt', 'max_speed']
        self.sccw = {k: dict() for k in self.simtypes}
        self.sccw['psi']['psi_label'] = tk.Label(master=self.scCondFrame, text="psi at release")
        self.sccw['psi']['psi_label'].grid(row=0, column=0)
        self.sccw['psi']['psif'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['psi']['psif'].grid(row=0, column=1)
        
        self.sccw['launch_angle']['pb_label'] = tk.Label(master=self.scCondFrame, 
                                                                   text="psi final bounds", wraplength=80)
        self.sccw['launch_angle']['pb_label'].grid(row=0, column=0)
        self.sccw['launch_angle']['psifmin'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['launch_angle']['psifmin'].grid(row=0, column=1)
        self.sccw['launch_angle']['psifmax'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['launch_angle']['psifmax'].grid(row=0, column=2)
        self.sccw['launch_angle']['la_label'] = tk.Label(master=self.scCondFrame, text="launch angle")
        self.sccw['launch_angle']['la_label'].grid(row=2, column=0)
        self.sccw['launch_angle']['vpa'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['launch_angle']['vpa'].grid(row=2, column=1)
        
        self.sccw['sling_len_opt']['pb_label'] = self.sccw['launch_angle']['pb_label']
        self.sccw['sling_len_opt']['psifmin'] = self.sccw['launch_angle']['psifmin']
        self.sccw['sling_len_opt']['psifmax'] = self.sccw['launch_angle']['psifmax']
        self.sccw['sling_len_opt']['slb_label'] = tk.Label(master=self.scCondFrame, 
                                                           text="sling length bounds", wraplength=80)
        self.sccw['sling_len_opt']['slb_label'].grid(row=1, column=0)
        self.sccw['sling_len_opt']['lsmin'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['sling_len_opt']['lsmin'].grid(row=1, column=1)
        self.sccw['sling_len_opt']['lsmax'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['sling_len_opt']['lsmax'].grid(row=1, column=2)
        self.sccw['sling_len_opt']['la_label'] = self.sccw['launch_angle']['la_label']
        self.sccw['sling_len_opt']['vpa'] = self.sccw['launch_angle']['vpa']
        self.sccw['sling_len_opt']['tf_label'] = tk.Label(master=self.scCondFrame, text="\u03B8 final")
        self.sccw['sling_len_opt']['tf_label'].grid(row=3, column=0)
        self.sccw['sling_len_opt']['thetaf'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['sling_len_opt']['thetaf'].grid(row=3, column=1)

        self.sccw['arm_inertia_opt']['pb_label'] = self.sccw['launch_angle']['pb_label']
        self.sccw['arm_inertia_opt']['psifmin'] = self.sccw['launch_angle']['psifmin']
        self.sccw['arm_inertia_opt']['psifmax'] = self.sccw['launch_angle']['psifmax']
        self.sccw['arm_inertia_opt']['aib_label'] = tk.Label(master=self.scCondFrame, 
                                                           text="arm inertia bounds", wraplength=80)
        self.sccw['arm_inertia_opt']['aib_label'].grid(row=1, column=0)
        self.sccw['arm_inertia_opt']['aimin'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['arm_inertia_opt']['aimin'].grid(row=1, column=1)
        self.sccw['arm_inertia_opt']['aimax'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['arm_inertia_opt']['aimax'].grid(row=1, column=2)
        self.sccw['arm_inertia_opt']['la_label'] = self.sccw['launch_angle']['la_label']
        self.sccw['arm_inertia_opt']['vpa'] = self.sccw['launch_angle']['vpa']
        self.sccw['arm_inertia_opt']['tf_label'] = self.sccw['sling_len_opt']['tf_label']
        self.sccw['arm_inertia_opt']['thetaf'] = self.sccw['sling_len_opt']['thetaf']
        
        self.sccw['max_speed']['pb_label'] = self.sccw['launch_angle']['pb_label']
        self.sccw['max_speed']['psifmin'] = self.sccw['launch_angle']['psifmin']
        self.sccw['max_speed']['psifmax'] = self.sccw['launch_angle']['psifmax']
        self.sccw['max_speed']['slb_label'] = self.sccw['sling_len_opt']['slb_label']
        self.sccw['max_speed']['lsmin'] = self.sccw['sling_len_opt']['lsmin']
        self.sccw['max_speed']['lsmax'] = self.sccw['sling_len_opt']['lsmax']
        self.sccw['max_speed']['aib_label'] = tk.Label(master=self.scCondFrame, 
                                                           text="arm inertia bounds", wraplength=80)
        self.sccw['max_speed']['aib_label'].grid(row=2, column=0)
        self.sccw['max_speed']['aimin'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['max_speed']['aimin'].grid(row=2, column=1)
        self.sccw['max_speed']['aimax'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['max_speed']['aimax'].grid(row=2, column=2)
        
        self.sccw['max_speed']['la_label'] = tk.Label(master=self.scCondFrame, text="launch angle")
        self.sccw['max_speed']['la_label'].grid(row=4, column=0)
        self.sccw['max_speed']['vpa'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['max_speed']['vpa'].grid(row=4, column=1)
        self.sccw['max_speed']['tf_label'] = tk.Label(master=self.scCondFrame, text="\u03B8 final")
        self.sccw['max_speed']['tf_label'].grid(row=5, column=0)
        self.sccw['max_speed']['thetaf'] = tk.Entry(master=self.scCondFrame, width=8)
        self.sccw['max_speed']['thetaf'].grid(row=5, column=1)
        
        # -------------------------------

        tk.Label(master=self.scFrame, text="Time Step").grid(row=0, column=0)
        self.dtE = tk.Entry(master=self.scFrame)
        self.dtE.grid(row=0, column=1)
        self.dtE.insert(0, "0.001")

        self.platformVar = tk.BooleanVar()
        tk.Label(master=self.scFrame, text="Platform").grid(row=1, column=0)
        tk.Checkbutton(master=self.scFrame, variable=self.platformVar).grid(row=1, column=1)
        
        tk.Label(master=self.scFrame, text="Simulation Type").grid(row=2, column=0)
        self.simTypeVar = tk.StringVar(root)
        self.simTypeVar.trace_add("write", self._stchange) # calls _stchange when simTypeVar is written to
        self.simTypeVar.set(self.simtypes[1])
        self.simTypeMenu = tk.OptionMenu(self.scFrame, self.simTypeVar, *self.simtypes)
        self.simTypeMenu.grid(row=2, column=1)
        
        tk.Label(master=self.scFrame, text="Solver Timeout (s)").grid(row=5, column=0)
        self.timeoutE = tk.Entry(master=self.scFrame)
        self.timeoutE.grid(row=5, column=1)
        self.timeoutE.insert(0, "10")
        # SIMULATION CONTROLS
        
        # TORQUE CONTROLS
        self.tcFrame = tk.LabelFrame(master=self.back, text="Torque Source")
        self.tcFrame.grid(row=0, column=2, rowspan=3)
        
        tk.Label(master=self.tcFrame, text="Torque Specification Type").grid(row=0, column=0)
        self.tsTypeVar = tk.StringVar() # constructor argument just specifies the Tk instance, and we only have 1
        self.tsTypes = ['Configuration A', 'Configuration B', 'r(\u03B8) and F(\u03B8)', 
                        '\u03C4(\u03B8) and r(\u03B8)']
        self.tsTypeVar.trace_add("write", self._tcchange)
        self.tsTypeMenu = tk.OptionMenu(self.tcFrame, self.tsTypeVar, *self.tsTypes)
        self.tsTypeMenu.grid(row=0, column=1)
        
        self.tcCondFrame = tk.Frame(master=self.tcFrame)
        self.tcCondFrame.grid(row=1, column=0, columnspan=2)
        
        self.GPFpos = {'row': 1, 'column': 0, 'columnspan': 2, 'rowspan': 2} # shared b/w A, B
        
        tk.Label(master=self.tcFrame, text="Hysteresis Parameter (k_w)").grid(row=2, column=0)
        self.tc_kwE = tk.Entry(master=self.tcFrame)
        self.tc_kwE.grid(row=2, column=1)
        
        self.torqueMsgM = tk.Message(master=self.tcFrame, text="", width=430)
        self.torqueMsgM.grid(row=4, column=0, columnspan=2)
        
        self.tfig = Figure(figsize=(4, 3), dpi=100, tight_layout=True)
        self.tax = self.tfig.add_subplot(111)
        self.tplotFrame = tk.Frame(master=self.tcFrame)
        self.tplotFrame.grid(row=3, column=0, columnspan=2)
        self.tCanvas = FigureCanvasTkAgg(self.tfig, master=self.tplotFrame)
        self.tCanvas.draw()
        self.tCanvas.get_tk_widget().pack()
        
        # DICTIONARIES OF custom_func: callable FOR ALL CUSTOM FUNCTION MODULES
        self.RThetaDict = self.funcDict(custom_functions.RofTheta)
        self.FLDict = self.funcDict(custom_functions.FofL)
        self.TauThetaDict = self.funcDict(custom_functions.TauofTheta)
        self.FThetaDict = self.funcDict(custom_functions.FofTheta)       
        # ******************************************************
        
        self.tcFVar = tk.StringVar()
        self.tcTauVar = tk.StringVar()
        self.tcRVar = tk.StringVar()       
        self.tcF2Var = tk.StringVar()
        
        IRlabels = ['\u03B8 min', '\u03B8 max']
        IRstartpos = (4, 0)
        tpf_maxcol = 4
        
        self.tccw = {k: dict() for k in self.tsTypes}
        self.tccw['A'] = dict()
        self.tccw['A']['gpl'] = tk.Label(master=self.tcCondFrame, text="Geometry Parameters")
        self.tccw['A']['gpl'].grid(row=0, column=0, columnspan=2)
        self.tccw['A']['GPFrame'] = tk.Frame(master=self.tcCondFrame)
        self.tccw['A']['GPFrame'].grid(**self.GPFpos)
        GPFA = self.tccw['A']['GPFrame'] #shorthand name
        GPFED = {'A': ('r_s', 'd', '\u03B2'), 'B': ('r_w', 'r_c', 'd', '\u03B2')}
        GPFsp = {'A': (1, 0), 'B': (1, 0)}
        GPFA.CW = self._gen_entries2(GPFA, GPFED['A'], GPFsp['A'], dict()) # adds all relevant entries & labels
        
        self.tccw['A']['fpl'] = tk.Label(master=self.tcCondFrame, text='Force Parameters')
        self.tccw['A']['fpl'].grid(row=0, column=2, columnspan=2)
        self.tccw['A']['fll'] = tk.Label(master=self.tcCondFrame, text='F(L)')
        self.tccw['A']['fll'].grid(row=1, column=2)
        self.tccw['A']['tsTypeMenu'] = tk.OptionMenu(self.tcCondFrame, self.tcFVar, *self.FLDict.keys())
        self.tccw['A']['tsTypeMenu'].grid(row=1, column=3)
        self.tccw['A']['irl'] = tk.Label(master=self.tcCondFrame, text="Interpolation Range")
        self.tccw['A']['irl'].grid(row=3, column=0, columnspan=2)       
        self.tccw['common'] = dict()
        self.tccw['common']['irl'] = tk.Label(master=self.tcCondFrame, text="Interpolation Range")
        self.tccw['common']['irl'].grid(row=3, column=0, columnspan=2)
        self.tccw['common'] = self._gen_entries2(self.tcCondFrame, IRlabels, IRstartpos, self.tccw['common'])
        self.t0SV.trace_add("write", self._t0update) # NOW we can add the trace to the theta0 tracker
        # interpolation range stuff
        self.tccw['A']['cfBtn'] = tk.Button(master=self.tcCondFrame, text="Calculate Fit", command=self._calcfit)
        self.tccw['A']['cfBtn'].grid(row=5, column=2, columnspan=2)
        self.tccw['A']['FLFrame'] = tk.Frame(master=self.tcCondFrame)
        self.tccw['A']['FLFrame'].grid(row=2, column=2, columnspan=2) 
        # now we need to create all possible variations of FLFrame contents
        FLFA = self.tccw['A']['FLFrame'] # shorthand
        FLFA.CW = self._parameter_entry_initializer(dict(), FLFA, self.FLDict, (0,0), maxcol=tpf_maxcol) # child widgets   
        for w in FLFA.winfo_children():
            w.grid_remove() # otherwise all parameters for F(L) are present despite no func being selected
            
        def FLchange(*args):
            self._entry_change(self.tccw['A']['FLFrame'], self.tccw['A']['FLFrame'].CW, self.tcFVar.get())
            
        self.tcFVar.trace_add("write", FLchange) # when the F(L) specification type changes
        
        # CONFIGURATION B *****************
        self.tccw['B'] = dict()
        self.tccw['B']['gpl'] = tk.Label(master=self.tcCondFrame, text="Geometry Parameters")
        self.tccw['B']['gpl'].grid(row=0, column=0, columnspan=2)
        self.tccw['B']['GPFrame'] = tk.Frame(master=self.tcCondFrame)
        self.tccw['B']['GPFrame'].grid(**self.GPFpos)
        GPFB = self.tccw['B']['GPFrame'] #shorthand name
        GPFB.CW = self._gen_entries2(GPFB, GPFED['B'], GPFsp['B'], dict()) # adds all relevant entries & labels
        
        self.tccw['B']['fpl'] = tk.Label(master=self.tcCondFrame, text='Force Parameters')
        self.tccw['B']['fpl'].grid(row=0, column=2, columnspan=2)
        self.tccw['B']['fll'] = tk.Label(master=self.tcCondFrame, text='F(L)')
        self.tccw['B']['fll'].grid(row=1, column=2)
        self.tccw['B']['tsTypeMenu'] = tk.OptionMenu(self.tcCondFrame, self.tcFVar, *self.FLDict.keys())
        self.tccw['B']['tsTypeMenu'].grid(row=1, column=3)
        
        self.tccw['B']['cfBtn'] = tk.Button(master=self.tcCondFrame, text="Calculate Fit", command=self._calcfit)
        self.tccw['B']['cfBtn'].grid(row=5, column=2, columnspan=2)
        self.tccw['B']['FLFrame'] = self.tccw['A']['FLFrame'] # shared b/w A & B
                
        # R(theta) and F(theta) *****************
        self.tccw['C'] = dict() # R & F
        self.tccw['C']['fpl'] = tk.Label(master=self.tcCondFrame, text="Force Parameters")
        self.tccw['C']['fpl'].grid(row=0, column=0, columnspan=2)
        self.tccw['C']['ftl'] = tk.Label(master=self.tcCondFrame, text="F(\u03B8)")
        self.tccw['C']['ftl'].grid(row=1, column=0)
        self.tccw['C']['ftTypeMenu'] = tk.OptionMenu(self.tcCondFrame, self.tcF2Var, *self.FThetaDict.keys())
        self.tccw['C']['ftTypeMenu'].grid(row=1, column=1)
        self.tccw['C']['FPFrame'] = tk.Frame(master=self.tcCondFrame)
        self.tccw['C']['FPFrame'].grid(row=2, column=0, columnspan=2)
        FTFC = self.tccw['C']['FPFrame']
        FTFC.CW = self._parameter_entry_initializer(dict(), FTFC, self.FThetaDict, (0,0), maxcol=tpf_maxcol)
        for w in FTFC.winfo_children():
            w.grid_remove()
            
        def FTchange(*args):
            self._entry_change(self.tccw['C']['FPFrame'], self.tccw['C']['FPFrame'].CW, self.tcF2Var.get())
            
        self.tcF2Var.trace_add("write", FTchange)
        
        self.tccw['C']['rpl'] = tk.Label(master=self.tcCondFrame, text="Radius Parameters")
        self.tccw['C']['rpl'].grid(row=0, column=2, columnspan=2)
        self.tccw['C']['rtl'] = tk.Label(master=self.tcCondFrame, text="R(\u03B8)")
        self.tccw['C']['rtl'].grid(row=1, column=2)
        self.tccw['C']['rtTypeMenu'] = tk.OptionMenu(self.tcCondFrame, self.tcRVar, *self.RThetaDict.keys())
        self.tccw['C']['rtTypeMenu'].grid(row=1, column=3)
        self.tccw['C']['RPFrame'] = tk.Frame(master=self.tcCondFrame)
        self.tccw['C']['RPFrame'].grid(row=2, column=2, columnspan=2)
        RTFC = self.tccw['C']['RPFrame']
        RTFC.CW = self._parameter_entry_initializer(dict(), RTFC, self.RThetaDict, (0,0), maxcol=tpf_maxcol)
        for w in RTFC.winfo_children():
            w.grid_remove()
            
        def RTchange(*args):
            self._entry_change(self.tccw['C']['RPFrame'], self.tccw['C']['RPFrame'].CW, self.tcRVar.get())
            
        self.tcRVar.trace_add("write", RTchange)
        
        self.tccw['C']['cfBtn'] = tk.Button(master=self.tcCondFrame, text="Calculate Fit", command=self._calcfit)
        self.tccw['C']['cfBtn'].grid(row=5, column=2, columnspan=2)      
        
        # tau(theta) and r(theta) ********************
        self.tccw['D'] = dict() # tau and R
        self.tccw['D']['tpl'] = tk.Label(master=self.tcCondFrame, text="Torque Parameters")
        self.tccw['D']['tpl'].grid(row=0, column=0, columnspan=2)
        self.tccw['D']['ttl'] = tk.Label(master=self.tcCondFrame, text="\u03C4(\u03B8)")
        self.tccw['D']['ttl'].grid(row=1, column=0)
        self.tccw['D']['ttTypeMenu'] = tk.OptionMenu(self.tcCondFrame, self.tcTauVar, *self.TauThetaDict.keys())
        self.tccw['D']['ttTypeMenu'].grid(row=1, column=1)
        self.tccw['D']['TPFrame'] = tk.Frame(master=self.tcCondFrame)
        self.tccw['D']['TPFrame'].grid(row=2, column=0, columnspan=2)
        TTFD = self.tccw['D']['TPFrame']
        TTFD.CW = self._parameter_entry_initializer(dict(), TTFD, self.TauThetaDict, (0, 0), maxcol=tpf_maxcol)
        for w in TTFD.winfo_children():
            w.grid_remove()
        def TTchange(*args):
            self._entry_change(self.tccw['D']['TPFrame'], self.tccw['D']['TPFrame'].CW, self.tcTauVar.get())
            
        self.tcTauVar.trace_add("write", TTchange)
        
        self.tccw['D']['rpl'] = self.tccw['C']['rpl']
        self.tccw['D']['rtl'] = self.tccw['C']['rtl']
        self.tccw['D']['rtTypeMenu'] = self.tccw['C']['rtTypeMenu']
        self.tccw['D']['RPFrame'] = self.tccw['C']['RPFrame']
        self.tccw['D']['cfBtn'] = self.tccw['C']['cfBtn']


        self.tsTypeVar.set('Configuration A')
        # TORQUE CONTROLS
    
        # miscellaneous Vars
        # for loading to go smoothly, these should all be defined at initialization
        self.effxVar = tk.StringVar(root)
        self.effxvals = ['time', 'theta', 'psi', 'projectile speed']
        self.effxVar.set(self.effxvals[0])
        self.effxVar.trace_add("write", self._change_p1)
        
        
    def _save_design(self):
        classes_to_save = ['Frame', 'Labelframe', 'Entry', 'Checkbutton', 'Menubutton']
        blacklist = [self.plotFrame1, self.plotFrame2]
        D = dict()
        G = nx.DiGraph()
        G.add_node(0)
        G.nodes[0]['level'] = 0
        G.nodes[0]['wname'] = self.back.winfo_name()
        G.nodes[0]['widget'] = self.back # this gets wiped before saving, and regenerated on loading
        G.nodes[0]['class'] = self.back.winfo_class()
            
        def map_GUI(G, node):
            current_widget = G.nodes[node]['widget']
            if current_widget not in blacklist:
                child_widgets = current_widget.winfo_children()
                for cw in child_widgets:
                    add_widget(G, cw, node)
            # ignore any blacklisted widgets, like the frames that control plots
            return G
        
        def add_widget(G, widget, parent_node):
            C = widget.winfo_class()
            if C in classes_to_save:
                nnn = max(list(G.nodes)) + 1
                G.add_node(nnn)
                G.add_edge(parent_node, nnn)
                G.nodes[nnn]['level'] = G.nodes[parent_node]['level'] + 1
                G.nodes[nnn]['wname'] = widget.winfo_name()
                G.nodes[nnn]['widget'] = widget

                G.nodes[nnn]['class'] = C
                if (C == 'Frame') or (C == 'Labelframe'):
                    G = map_GUI(G, nnn) # the map_GUI->add_widget->map_GUI cycle is what does the recursive
                    # mapping of the tree structure of the GUI
                elif C == 'Entry':
                    G.nodes[nnn]['value'] = widget.get()
                elif C == 'Checkbutton':
                    varname = widget.cget('variable')
                    G.nodes[nnn]['value'] = widget.getvar(varname)
                    # to set in Load, widget.setvar(varname, newvalue)
                elif C == 'Menubutton':
    #                 for item in widget.keys():
    #                     print("item = ", item)
    #                     print("widget.cget(item) = ", widget.cget(item))
                    # above code allows inspecting all the valid cget options
                    varname = widget.cget('textvariable')
                    G.nodes[nnn]['value'] = widget.getvar(varname)
            # does nothing if the widget isn't a class to be saved
            return G       
        # these two together should recursively map the entire family tree of self.back
        G = map_GUI(G, 0)
        for n in G.nodes:
            G.nodes[n]['widget'] = False # can't pickle the widgets themselves, and they
            # need to be reconstituted later anyway
        default_dir = os.path.join(toplevel_path, "saved_designs")
        f = tk.filedialog.asksaveasfile(mode='wb', defaultextension=".pkl", initialdir=default_dir)
        if f is None:
            return
        else:
            pickle.dump(G, f)
        f.close()
        
                                
    def _load_design(self):
        default_dir = os.path.join(toplevel_path, "saved_designs")
        f = tk.filedialog.askopenfile(mode='rb', initialdir=default_dir)
        if f is None:
            return
        G = pickle.load(f)
        G.nodes[0]['widget'] = self.back
        
        def repair_node(G, node, parent_node):
            # reassign widgets to nodes, set entry values, and set variables associated with menus and checkboxes
            parent_widget = G.nodes[parent_node]['widget']
            try: 
                W = parent_widget.nametowidget(G.nodes[node]['wname'])
                G.nodes[node]['widget'] = W
                C = G.nodes[node]['class']
                if C == 'Entry':
    #                 varname = W.cget("variable")
                    W.delete(0, 'end')
                    W.insert(0, G.nodes[node]['value'])
    #                 W.setvar(varname, G.nodes[node]['value'])
                elif C == 'Checkbutton':
                    varname = W.cget('variable')
                    W.setvar(varname, G.nodes[node]['value'])
                elif C == 'Menubutton':
                    varname = W.cget('textvariable')
                    W.setvar(varname, G.nodes[node]['value'])
            except:
                print('\n\nERROR ON LEVEL: ' + str(G.nodes[node]['level']))
                pw_children = parent_widget.winfo_children()
                pw_children_names = [pwc.winfo_name() for pwc in pw_children]
                target_name = G.nodes[node]['wname']
                print("target name: ", target_name)
                print("available names: ", pw_children_names)
                print('--- END ERROR ---\n')
                
                
        def repair_G(G):
            # have to build the GUI state level by level
            # reassign widgets from top down, completing level n entirely before moving to level n+1
            # while doing this, set entries and associated variable values
            # sort nodes into list of lists. Inner lists are all nodes on a level, outer list is levels
            # outer_list[0] contains level 0 nodes, and so on
            max_level = max([G.nodes[n]['level'] for n in G.nodes])
            for level in range(1, max_level+1):
                # don't want to include the root node, as it has no predecessors
                Lnodes = [n for n in G.nodes if (G.nodes[n]['level'] == level)]
                for n in Lnodes:
                    parent_node = list(G.predecessors(n))[0]
                    repair_node(G, n, parent_node)
            
        repair_G(G)
        try:
            self._calcfit()
        except:
            pass
        
    def show_error(self, *args):
        err = traceback.format_exception(*args)
        tk.messagebox.showerror('Exception',err)
        
    def _update_simMsg(self, *args):
        self.simMsgM.configure(text=self.msg_text.get())        
        
    def _quit(self):
        self.master.quit()     # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
            
    def _t0update(self, *args):
        self.tccw['common']['\u03B8 maxE'].delete(0, "end")
        self.tccw['common']['\u03B8 maxE'].insert(0, self.theta0E.get())
        try:
            self._calcfit()
        except:
            pass # might not currently have all the other necessary parameters specified
            
    def _stopsolve(self):
        # user got sick of waiting for the solver to finish
        self.solve_proc.terminate()
        self.solver_runningB.set(False)
        
    def _solver_state_change(self, *args):
        # callback for what it says on the lid
        # *args is because tkinter passes some stuff by default
        # no docs on what it is, but safely ignored
        if self.solver_runningB.get():
            # solver is running
            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
    def _p1destroyer(self):
        self.p1_conditional_controls.destroy()
        self.p1_conditional_controls = tk.Frame(master=self.plotFrame1)
        self.p1_conditional_controls.grid(row=2, column=0, columnspan=2)
            
    def _change_p1(self, *args):
        # all the "weird" plots that draw multiple lines or mess with the axes / figures
        # as there is no need to save plots, the whole frame is destroyed whenever the plot type is changed
        p1T = self.p1TypeVar.get()          
        if p1T == 'animation':
            if self.prevP1Type == 'animation':
                self._run_animation()
            else:
                self._conf_animation()
                self._run_animation()
                
        elif p1T == 'arm load':
            self._p1destroyer()
            self.fig1.clear()
            axF = self.fig1.add_subplot(311)
            axO = self.fig1.add_subplot(312)
            axA = self.fig1.add_subplot(313)
            if hasattr(self, "sol"):
                if not self.sol is None:
                    group = (self.fig1, axF, axO, axA)
                    h = self.sol['H']
                    Y, Yd, t = (np.array(h['Y']), np.array(h['Yd']), np.array(h['t']))
                    CDTtools.dyn_core_tools.load_info(Y, Yd, t, self.dp, plot=True, group=group)
                    self.canvas1.draw()
            
            def sas():
                # def y_hist(y0, dt, dp, tp, psif, N=0, platform=True):
                Npoints = int(self.p1_snpE.get())
                D = CDTtools.dyn_core_tools.y_hist(self.y0, self.dt, self.dp, self.tp, self.sol['yf'][2],
                                                          N=Npoints, platform=self.p)
                data = CDTtools.dyn_core_tools.load_info(D['ys'], D['yds'], D['ts'], self.dp, plot=False)
                # {'Fx_tip': Ftx, 'Fy_tip': Fty, 'omega': omega, 'alpha': alpha}
                data_arr = np.stack((D['ts'], data['Fx_tip'], data['Fy_tip'], data['omega'], data['alpha']), 1)
                default_dir = os.path.join(toplevel_path, "saved_data")
                fname = tk.filedialog.asksaveasfilename(initialdir=default_dir)
                if fname is None:
                    return
                np.savetxt(fname, data_arr)
            # def load_info(Y, Yd, t, dp, plot=False, group=None):
            
            tk.Label(master=self.p1_conditional_controls, text="Number of Points").grid(row=0, column=0)
            self.p1_snpE = tk.Entry(master=self.p1_conditional_controls)
            self.p1_snpE.grid(row=0, column=1)
            tk.Button(master=self.p1_conditional_controls, text="Save Path", command=sas).grid(row=0, column=3)
                
        elif p1T == 'projectile path':
            def sp():
                if hasattr(self, "sol"):
                    if not self.sol is None:
                        Y = np.array(self.sol['H']['Y'])
                        default_dir = os.path.join(toplevel_path, "saved_data")
                        fname = tk.filedialog.asksaveasfilename(initialdir=default_dir)
                        scale = float(pathSaveScaleE.get())
                        CDTtools.dyn_core_tools.projectile_path(Y, self.dp, plot=False, fname=fname, 
                                                                scale_factor=scale, axPP=None)
            self._p1destroyer()            
            tk.Button(master=self.p1_conditional_controls, text="Save Path", command=sp).grid(row=1, column=2)
            tk.Label(master=self.p1_conditional_controls, text="scale factor").grid(row=1, column=0)
            pathSaveScaleE = tk.Entry(master=self.p1_conditional_controls)
            pathSaveScaleE.grid(row=1, column=1)
            pathSaveScaleE.insert(0, "1")
            self.fig1.clear()
            # def projectile_path(Y, dp, plot=False, fname=None, scale_factor=1, axPP=None):
            self.ax1 = self.fig1.add_subplot(111)
            if hasattr(self, "sol"):
                if not self.sol is None:
                    Y = np.array(self.sol['H']['Y'])
                    CDTtools.dyn_core_tools.projectile_path(Y, self.dp, plot=True, axPP=self.ax1)
                    self.canvas1.draw()
                    
        elif p1T == 'efficiency':
            if self.prevP1Type != 'efficiency':
                self._p1destroyer()
                tk.Label(master=self.p1_conditional_controls, text="x axis").grid(row=0, column=0)
                self.effxMenu = tk.OptionMenu(self.p1_conditional_controls, self.effxVar, *self.effxvals)
                self.effxMenu.grid(row=0, column=1)
            self.fig1.clear()
            self.ax1 = self.fig1.add_subplot(111)
            if hasattr(self, "sol"):
                if not self.sol is None:
                    Y = np.array(self.sol['H']['Y'])
                    Yd = np.array(self.sol['H']['Yd'])
                    t = np.array(self.sol['H']['t'])
                    # def energy_plot(Y, Yd, t, dp, x='time', axE=None):
                    CDTtools.dyn_core_tools.energy_plot(Y, Yd, t, self.dp, x=self.effxVar.get(), axE=self.ax1)
                    self.canvas1.draw()
                    
        elif p1T == 'puller potential energy':
            if self.prevP1Type != 'puller potential energy':
                self._p1destroyer()
                tk.Label(master=self.p1_conditional_controls, text="plot domain (theta)").grid(row=0, column=0)
                self.ppe_tminE = tk.Entry(master=self.p1_conditional_controls)
                self.ppe_tminE.grid(row=0, column=1)
                self.ppe_tmaxE = tk.Entry(master=self.p1_conditional_controls)
                self.ppe_tmaxE.grid(row=0, column=2)
                self.ppe_plotB = tk.Button(master=self.p1_conditional_controls, text="Replot", 
                                           command=self._change_p1)
                self.ppe_plotB.grid(row=0, column=3)
            self.fig1.clear()
            self.ax1 = self.fig1.add_subplot(111)
            if hasattr(self, "tp"):
                # def puller_potential_energy(tp, theta0, plot=False, plotdomain=None, axPPE=None):
                tmin = self.ppe_tminE.get()
                tmax = self.ppe_tmaxE.get()
                theta0 = float(self.theta0E.get())
                if (tmin != "") and (tmax != ""):
                    ppe = CDTtools.dyn_core_tools.puller_potential_energy(self.tp, theta0, plot=True, 
                                                                          plotdomain=(float(tmin), float(tmax)),
                                                                         axPPE = self.ax1)
                else:
                    ppe = CDTtools.dyn_core_tools.puller_potential_energy(self.tp, theta0, plot=True, 
                                                                          axPPE=self.ax1)
                msg1 = "Puller potential energy: " + str(round(ppe, self.rp))
                if hasattr(self, "vpf"):
                    Ekf = 0.5*float(self.dpFrame.CW['Projectile + Pouch MassE'].get())*(self.vpf)**2
                    msg2 = "\nOverall system efficiency: " + str(round(Ekf/ppe, self.rp))
                else:
                    msg2 = ""
                if self.prevP1Type == 'puller potential energy':
                    self.ppeMsg.configure(text=msg1+msg2)
                else:
                    self.ppeMsg = tk.Message(master=self.p1_conditional_controls, text=msg1+msg2, 
                                             bg='white', width=256)
                    self.ppeMsg.grid(row=1, column=0, columnspan=3)
                self.canvas1.draw()
            
        elif p1T == 'axle reaction load':
            # more or less the same as the projectile path - needs an option to save the load data
            def sarl():
                if hasattr(self, "sol"):
                    if not self.sol is None:
                        # def y_hist(y0, dt, dp, tp, psif, N=0, platform=True):
                        # return {'ts': sample_times, 'ys': ys, 'yds': yds}
                        Npoints = int(arlSaveNE.get())
                        D = CDTtools.dyn_core_tools.y_hist(self.y0, self.dt, self.dp, self.tp, self.sol['yf'][2],
                                                          N=Npoints, platform=self.p)
                        default_dir = os.path.join(toplevel_path, "saved_data")
                        fname = tk.filedialog.asksaveasfilename(initialdir=default_dir)
                        # axle_reaction_force(Y, Yd, dp, t, plot=False, fname=None, ax_arf=None)
                        CDTtools.dyn_core_tools.axle_reaction_force(D['ys'], D['yds'], self.dp, 
                                                                    D['ts'], fname=fname)
                        
            self._p1destroyer()            
            tk.Button(master=self.p1_conditional_controls, 
                      text="Save Load Data", command=sarl).grid(row=0, column=2)
            default_dir = os.path.join(toplevel_path, "saved_designs")
            tk.Label(master=self.p1_conditional_controls, text="number of points").grid(row=0, column=0)
            arlSaveNE = tk.Entry(master=self.p1_conditional_controls)
            arlSaveNE.grid(row=0, column=1)
            arlSaveNE.insert(0, "20")
            self.fig1.clear()
            self.ax1 = self.fig1.add_subplot(111)
            if hasattr(self, "sol"):
                if not self.sol is None:
                    Y = np.array(self.sol['H']['Y'])
                    Yd = np.array(self.sol['H']['Yd'])
                    t = np.array(self.sol['H']['t'])
                    CDTtools.dyn_core_tools.axle_reaction_force(Y, Yd, self.dp, t, plot=True, ax_arf=self.ax1)
                    self.canvas1.draw()   
                               
        else:
            msg = self.msg_text.get() + "Unimplemented plot 1 type: " + self.p1TypeVar.get()
            self.msg_text.set(msg)
        self.prevP1Type = p1T
 

    def _calc_ax2_data(self, s, Y, Yd, t):
        if s == 'time':
            return t
        elif s == 'theta':
            return Y[:,0]
        elif s == 'psi':
            return Y[:,2]
        elif s == 'projectile velocity':
            vp = [CDTtools.dyn_core_tools.vp(Y[i,:], self.dp) for i in range(len(t))]
            return np.array([np.linalg.norm(v) for v in vp])
        elif s == 'sling tension':
            return CDTtools.dyn_core_tools.sling_tension(Y, Yd, t, self.dp)
        elif s == 'arm rotational speed':
            return Y[:,1]
        elif s == 'puller speed':
            theta_hist = (np.array(self.sol['H']['Y']))[:, 0]
            omega_hist = (np.array(self.sol['H']['Y']))[:, 1]
            t_hist = np.array(self.sol['H']['t'])
            R_hist = np.array([self.puller_geometry.R(x) for x in theta_hist])
            Ldot = np.array([omega_hist[i] * R_hist[i] for i in range(len(t_hist))])
            return Ldot
        else:
            self.msg_text.set(self.msg_text.get() + "Unimplemented axis option: " + s)
            return None
            
            
    def _change_p2(self, *args):    
        if hasattr(self, "sol"):
            if not self.sol is None:
                t = np.array(self.sol['H']['t'])
                Y = np.array(self.sol['H']['Y'])
                Yd = np.array(self.sol['H']['Yd'])
                p2x = self.p2xVar.get()
                p2y = self.p2yVar.get()
                xdata = self._calc_ax2_data(p2x, Y, Yd, t)
                ydata = self._calc_ax2_data(p2y, Y, Yd, t)
                    
                if (not xdata is None) and (not ydata is None):
                    self.ax2.clear()
                    self.ax2.plot(xdata, ydata)
                    self.ax2.set_xlabel(p2x)
                    self.ax2.set_ylabel(p2y)
                    self.canvas2.draw()

        
    def _conf_animation(self, *args):
        # def launch_animation(Y, dp, t, axLA=None, figLA=None):
        self._p1destroyer()
        self.play_button = tk.Button(master=self.p1_conditional_controls, text="PLAY", 
                                     command=self._run_animation)
        self.play_button.grid(row=0, column=2)
        tk.Label(master=self.p1_conditional_controls, text="Animation Length (s)").grid(row=0, column=0)
        self.aniLengthE = tk.Entry(master=self.p1_conditional_controls, width=8)
        self.aniLengthE.grid(row=0, column=1)
        self.aniLengthE.insert(0, "2")
        
                    
    def _run_animation(self, *args):
        # Totally impossible to disable the play button while the animation is running
        # it just can't be done, not with after or by any other means, in Tkinter, while
        # keeping the rest of the GUI alive. Only other option is to implement animation directly in
        # Tkinter using after. And that sounds like an unpleasant way to spend a day. The user
        # is just going to have to deal with being able to make the plot glitch by hitting Play again
        # it's not like it crashes the GUI or anything
        self.fig1.clear()
        self.ax1 = self.fig1.add_subplot(111)
        if hasattr(self, "sol"):
            if not self.sol is None:
                h = self.sol['H']
                # need to clean up data so it's evenly spaced in time using hist
                # this also determines how long the animation runs (framerate is constant 25fps)
                nframes = int(1 + 25*float(self.aniLengthE.get()))
                # def y_hist(y0, dt, dp, tp, psif, N=0, platform=True):
                # return {'ts': sample_times, 'ys': ys, 'yds': yds}
                I = CDTtools.dyn_core_tools.y_hist(self.y0, self.dt, self.dp, self.tp, 
                                                          self.sol['yf'][2], N=nframes, platform=self.p)

                CDTtools.dyn_core_tools.launch_animation(I['ys'], self.dp, I['ts'],
                                                         axLA=self.ax1, figLA=self.fig1)
                self.canvas1.draw()
                # DON'T try to make the Play button grey when the animation is running
                # only way to do it is to completely scrap the use of FuncAnimation and
                # roll your own using tkinter's after method
                
                        
    def _check_solveproc(self):
        # as long as the solver is running, checks periodically to see if it's still alive
        if not (self.solve_proc).is_alive():
            self.solver_runningB.set(False)
            with open('logfile', 'a') as f:
                f.write('solve_proc is no longer alive, t = ' + str(time.time() - self.t0) + "\n")
            self.cspcalls = 0 # reset for next simulation run
            self._wrapup() # displays basic results from the simulation in simMsgM
        elif (self.solve_proc).is_alive and (time.time() - self.t0) > float(self.timeoutE.get()):
            # timeout exceeded, stop the run
            with open('logfile', 'a') as f:
                f.write('terminating solve_proc due to timeout, t = ' + str(time.time() - self.t0) + '\n')
            self.solve_proc.terminate() # ran for too long
            self.master.after(100, self._check_solveproc) # next call will find the process dead and return
        else:
            self.solver_runningB.set(True)
            self.cspcalls += 1
            with open('logfile', 'a') as f:
                f.write(str(self.cspcalls) + ' calls after ' + str(time.time() - self.t0) + 's\n')
            self.master.after(500, self._check_solveproc)
        return 0
            
    def _stchange(self, *args):
        # called when simTypeVar changes to update conditional simulation controls
        # needs to take three parameters, but I don't need them, and the docs don't say what they are...
        for w in self.scCondFrame.winfo_children():
            w.grid_remove() # removes all of scCondFrame's children
        stv = self.simTypeVar.get()
        for k in self.sccw[stv].keys():
            self.sccw[stv][k].grid()
            
    def funcDict(self, module):
        M = [t for t in inspect.getmembers(module) if inspect.isfunction(t[1])]
        # list of tuples of form: (namestr, callable)
        return {t[0]: t[1] for t in M} # dictionary of {name: callable}

    def _tcchange(self, *args): 
        tst = self.tsTypeVar.get()
        for w in self.tcCondFrame.winfo_children():
            if not w in self.tccw['common'].values():
                w.grid_remove()
        if tst == 'Configuration A':
            k = 'A'
        elif tst == 'Configuration B':
            k = 'B'
        elif tst == 'r(\u03B8) and F(\u03B8)':
            k = 'C'
        elif tst == '\u03C4(\u03B8) and r(\u03B8)':
            k = 'D'
        for w in self.tccw[k].keys():
            self.tccw[k][w].grid()

            
    def _gen_entries2(self, frame, en, sp, D, entrywidth=None, maxcol=6):
        # uses existing dict, includes labels
        for i in range(len(en)):
            c = math.floor(i/maxcol)
            r = i - (maxcol*c)
            label_string = en[i] + 'L'
            entry_string = en[i] + 'E'
            D[label_string] = tk.Label(master=frame, text=en[i])
            D[label_string].grid(row=sp[0]+r, column=sp[1]+(2*c))
            D[entry_string] = tk.Entry(master=frame)
            if not entrywidth is None:
                D[entry_string].configure(width=entrywidth)
            D[entry_string].grid(row=sp[0]+r, column=sp[1]+1+(2*c))
        return D
    
    def _parameter_entry_initializer(self, D, frame, funcDict, startpos, entrywidth=8, maxcol=6):
        for funcname in funcDict.keys():
            f = funcDict[funcname]
            S = inspect.signature(f)
            i = 0
            SL = [s for s in S.parameters][1:] # L, theta, etc. should not be added
            D_inner = dict()
            D[funcname] = self._gen_entries2(frame, SL, startpos, D_inner, entrywidth=entrywidth, maxcol=maxcol)
        return D
        
    def _entry_change(self, frame, cwd, funcname):
        # changes entries / labels in frame to those matching funcname, as defined in cwd
        for w in frame.winfo_children():
            w.grid_remove()
        if funcname != '':
            # don't try to add any widgets if the function is unspecified
            for k in cwd[funcname].keys():
                cwd[funcname][k].grid()
                
                
    def _calcfit(self):
        tst = self.tsTypeVar.get()
        theta_min = float(self.tccw['common']['\u03B8 minE'].get())
        theta_max = float(self.tccw['common']['\u03B8 maxE'].get())
        self.theta_range = (theta_min, theta_max)
        if (tst == 'Configuration A') or (tst == 'Configuration B'):
            # these two share most of their input structure
            if tst == 'Configuration A':
                # GPFED = {'A': ('r_s', 'd', '\u03B2'), 'B': ('r_w', 'r_c', 'd', '\u03B2')}
                gp = self.tccw['A']['GPFrame'].CW
                iD = {'rs': float(gp['r_sE'].get()), 'd': float(gp['dE'].get()), 'beta0': float(gp['\u03B2E'].get()),
                     'theta0': float(self.theta0E.get())}
                self.puller_geometry = CDTtools.puller_tools.geometry_A(iD)
            elif tst == 'Configuration B':
                gp = self.tccw['B']['GPFrame'].CW
                iD = {'rw': float(gp['r_wE'].get()), 'd': float(gp['dE'].get()),
                     'rc': float(gp['r_cE'].get()), 'beta0': float(gp['\u03B2E'].get()),
                    'theta0': float(self.theta0E.get())}
                self.puller_geometry = CDTtools.puller_tools.geometry_B(iD)
                
            def F(L):
                Ffuncname = self.tcFVar.get()
                flp = self.tccw['A']['FLFrame'].CW[Ffuncname] # widgets containing F(L) parameters
                # this is shared between A and B, two names referring to same object
                args = [flp[k].get() for k in flp.keys() if (flp[k].winfo_class() == 'Entry')]
                argnames = [k for k in flp.keys() if (flp[k].winfo_class() == 'Entry')]
                func = self.FLDict[Ffuncname]
                return func(L, *args)
            # def taugen(FofL, LofTheta, RofTheta):
            self.tau_exact = CDTtools.puller_tools.taugen(F, self.puller_geometry.L, self.puller_geometry.R)
            # self.tau_exact is tau(theta), without consideration to hysteresis
            # i.e., only valid when contracting
            self.L0 = self.puller_geometry.L(float(self.theta0E.get()))
            self.F0 = F(self.L0)
            self.Fmin = min([F(self.puller_geometry.L(t)) for t in np.linspace(*self.theta_range, endpoint=True)])
            
        elif tst == '\u03C4(\u03B8) and r(\u03B8)' or 'r(\u03B8) and F(\u03B8)':
            def R(theta):
                Rfuncname = self.tcRVar.get()
                rtp = self.tccw['C']['RPFrame'].CW[Rfuncname] # same frame shared between C & D
                R_args = [rtp[k].get() for k in rtp.keys() if (rtp[k].winfo_class() == 'Entry')]
                func = self.RThetaDict[Rfuncname]
                return func(theta, *R_args)
            if tst == '\u03C4(\u03B8) and r(\u03B8)':
                def tau(theta):
                    taufuncname = self.tcTauVar.get()
                    ttp = self.tccw['D']['TPFrame'].CW[taufuncname]
                    tau_args = [ttp[k].get() for k in ttp.keys() if (ttp[k].winfo_class() == 'Entry')]
                    func = self.TauThetaDict[self.tcTauVar.get()]
                    return func(theta, *tau_args)
            else:
                def tau(theta):
                    Ffuncname = self.tcF2Var.get()
                    ftp = self.tccw['C']['FPFrame'].CW[Ffuncname] # widgets containing F(theta) entries & labels
                    F_args = [ftp[k].get() for k in ftp.keys() if (ftp[k].winfo_class() == 'Entry')]
                    F_func = self.FThetaDict[Ffuncname]
                    return R(theta)*F_func(theta, *F_args)
            self.puller_geometry = CDTtools.puller_tools.geometry_custom(R)
            self.tau_exact = tau
            theta0 = float(self.theta0E.get())
            def F(theta):
                return tau(theta)/R(theta)
            self.F0 = F(theta0)
            self.Fmin = min([F(t) for t in np.linspace(*self.theta_range, endpoint=True)])
            
        self.tau_fit = CDTtools.puller_tools.torque_fit(self.tau_exact, self.theta_range)
        nc = 3
        tf_msg = "\u03C4(\u03B8) = "
        for i in range(len(self.tau_fit.coef)):
            tf_msg = tf_msg + str(round(self.tau_fit.coef[i], nc)) + "*\u03B8^" + str(i) + " + "
        tf_msg = tf_msg[:-3]
        F0_msg = '\nPuller length at initial configuration: ' + str(round(self.L0, nc))
        if self.Fmin < 0:
            Fmin_warning = "\nWarning: puller in compression. Minimum force " + str(round(self.Fmin, nc))
        else:
            Fmin_warning = ""
        self.torqueMsgM.configure(text=tf_msg+F0_msg+Fmin_warning)
        self._plotTfit()
            
                        
    def _plotTfit(self):
        self.tax.clear()
        xtest = np.linspace(*self.theta_range, endpoint=True) # self.theta_range set by calcfit
        # nothing but calcfit calls plotTfit
        ytest = np.array([self.tau_fit(x) for x in xtest])
        ytrue = np.array([self.tau_exact(x) for x in xtest])
        ymin = min((min(ytest), min(ytrue)))
        self.tax.plot(xtest, ytest, label="interpolation result")
        self.tax.plot(xtest, ytrue, label="true value")
        self.tax.legend(loc='best')
        self.tax.set_xlabel("theta")
        self.tax.set_ylabel("torque")
        if ymin > 0:
            self.tax.set_ylim(0)
        self.tCanvas.draw()
        
    def _gen_dp(self):
        # correct order: (La, Ls, ds, mb, rc, Ia, mp, g)
        def P(S):
            key = S+'E'
            return float(self.dpFrame.CW[key].get())
        self.dp = CDTtools.dyn_core_tools.dyn_params(*[P(S) for S in self.DPEL])
                   
    def _simulate(self):
        # change what's run based on the selected optimization option
        self._gen_dp()
        self._calcfit() # don't want to cause the user headaches because they forgot to hit calculate fit
        # also quietly solves the problem that arises if they change theta0 *after* calculating the torque fit
        c = [ci for ci in self.tau_fit.coef]
        k_w = float(self.tc_kwE.get())
        self.tp = CDTtools.dyn_core_tools.T_params(k_w, *c)
        ENL = ("d\u03B8/dtE", "\u03C8E", "d\u03C8/dtE")
        y0L = [float(self.theta0E.get())] + [float(self.icFrame.CW[en].get()) for en in ENL]
        self.y0 = np.array(y0L)
        self.dt = float(self.dtE.get()) # get this from simulation setup section
        self.p = self.platformVar.get() # is there a platform? (BooleanVar)
        # this is the value of platformVar that the PREVIOUS SIMULATION was run with
        self.solname = "solution.pkl"

        if self.simTypeVar.get() == 'psi':   
            self.prev_solve_type = 'psi'
            psi_release = float(self.sccw['psi']['psif'].get())
            solver_args = (CDTtools.dyn_core_tools.dyn_general, self.solname, self.cps, self.y0, self.dt, 
                           self.dp, self.tp, psi_release)
            solver_kwargs = {'verbose': False, 'platform': self.p, 'history': True}
           
        elif self.simTypeVar.get() == 'launch_angle':
            self.prev_solve_type = 'launch_angle'
            psibounds = (float(self.sccw['launch_angle']['psifmin'].get()), 
                         float(self.sccw['launch_angle']['psifmax'].get()))
            vpa_target = float(self.sccw['launch_angle']['vpa'].get())
            solver_args = (CDTtools.dyn_core_tools.dft_vpastop, self.solname, self.cps, self.y0, self.dt, 
                           self.dp, self.tp, psibounds, vpa_target)
            solver_kwargs = {'verbose': False, 'platform': self.p, 'history': True}
            
        elif self.simTypeVar.get() == 'sling_len_opt':
            self.prev_solve_type = 'sling_len_opt'
            psibounds = (float(self.sccw['sling_len_opt']['psifmin'].get()),
                       float(self.sccw['sling_len_opt']['psifmax'].get()))
            Lsbounds = (float(self.sccw['sling_len_opt']['lsmin'].get()), 
                     float(self.sccw['sling_len_opt']['lsmax'].get()))
            vpa_target = float(self.sccw['sling_len_opt']['vpa'].get())
            theta_target = float(self.sccw['sling_len_opt']['thetaf'].get())
            solver_args = (CDTtools.dyn_core_tools.dft_opt1a, self.solname, self.cps, self.y0, self.dt, 
                           self.dp, self.tp, psibounds, vpa_target, Lsbounds, theta_target)
            solver_kwargs = {'verbose': False, 'platform': self.p, 'history': True}
            
        elif self.simTypeVar.get() == 'arm_inertia_opt':
            self.prev_solve_type = 'arm_inertia_opt'
            psibounds = (float(self.sccw['arm_inertia_opt']['psifmin'].get()),
                       float(self.sccw['arm_inertia_opt']['psifmax'].get()))
            Iabounds = (float(self.sccw['arm_inertia_opt']['aimin'].get()), 
                     float(self.sccw['arm_inertia_opt']['aimax'].get()))
            vpa_target = float(self.sccw['arm_inertia_opt']['vpa'].get())
            theta_target = float(self.sccw['arm_inertia_opt']['thetaf'].get())
            solver_args = (CDTtools.dyn_core_tools.dft_opt2, self.solname, self.cps, self.y0, self.dt, 
                           self.dp, self.tp, psibounds, vpa_target, Iabounds, theta_target)
            solver_kwargs = {'verbose': False, 'platform': self.p, 'history': True}
            
        elif self.simTypeVar.get() == 'max_speed':
            self.prev_solve_type = 'max_speed'
            E = self.sccw['max_speed']
            bounds = np.array([[float(E['psifmin'].get()), float(E['psifmax'].get())],
                               [float(E['lsmin'].get()), float(E['lsmax'].get())],
                               [float(E['aimin'].get()), float(E['aimax'].get())]])
            targets = np.array([float(E['vpa'].get()), float(E['thetaf'].get())])
            solver_args = (CDTtools.dyn_core_tools.dft_opt_vmax, self.solname, self.cps, self.y0, self.dt,
                            self.dp, self.tp, bounds, targets)
            solver_kwargs = {'verbose': False, 'platform': self.p, 'history': True}
            
        self.solve_proc = multiprocessing.Process(target=CDTtools.dyn_core_tools.solver_wrapper,
                                                 args=solver_args, kwargs=solver_kwargs)
        self.solve_proc.start()
        self.t0 = time.time()
        self._check_solveproc()            

        
    def _wrapup(self):
        # display results from the last simulation run and set self.sol for later use
        self.sol = None # keeps the animation from running again if there's no new solution
        if self.cpr.poll(0.1):
            if self.cpr.recv():
                # True if the solver finished without error
                # YAY! We have a solution!
                with open(self.solname, 'rb') as f:
                    self.sol = pickle.load(f) # can't use the pipe directly d.t. potential large size of soln
                # self.sol = self.cpr.recv() # this will be needed elsewhere for postproc
                psif = self.sol['yf'][2]
                Ysol = np.array(self.sol['H']['Y'])
                Ydsol = np.array(self.sol['H']['Yd'])
                dpsol = self.sol['dp']
                tsol = np.array(self.sol['H']['t'])
                adh = CDTtools.dyn_core_tools.arm_design_hardness(Ysol, Ydsol, tsol, dpsol)
                theta_min_true = min(np.array(self.sol['H']['Y'])[:,0])
                theta_min_guess = min(self.theta_range)
                if self.prev_solve_type == 'psi':
                    extra_msg = ("Launch angle: " + 
                             str(round(CDTtools.dyn_core_tools.vp_angle(self.sol['yf'], self.dp), self.rp)) + 
                             "\nArm design hardness: " + str(round(adh, self.rp)) + "\n")
                elif self.prev_solve_type == 'launch_angle':
                    psibounds = (float(self.sccw['launch_angle']['psifmin'].get()), 
                                 float(self.sccw['launch_angle']['psifmax'].get()))
                    extra_msg = ("Psi at release: " + str(round(psif, self.rp)) + 
                                "\nArm design hardness: " + str(round(adh, self.rp)) + "\n")
                    if np.isclose(psif, psibounds[0], atol=1e-3) or np.isclose(psif, psibounds[1], atol=1e-3):
                        extra_msg = (extra_msg + "\nWarning: The target launch angle was not reachable given the"
                                     " specified constraints on psi final\n\n")
                elif self.prev_solve_type == 'sling_len_opt':
                    psibounds = (float(self.sccw['launch_angle']['psifmin'].get()), 
                                 float(self.sccw['launch_angle']['psifmax'].get()))
                    Lsbounds = (float(self.sccw['sling_len_opt']['lsmin'].get()), 
                                 float(self.sccw['sling_len_opt']['lsmax'].get()))
                    self.dp.Ls = self.sol['Ls_opt']
                    extra_msg = ("Psi at release: " + str(round(self.sol['yf'][2], self.rp)) + 
                                 "\nOptimized sling length: " + str(round(self.sol['Ls_opt'], self.rp)) + 
                                 "\nArm design hardness: " + str(round(adh, self.rp)) + "\n")
                    if True in [np.isclose(psif, psibounds[0]), np.isclose(psif, psibounds[1]), 
                               np.isclose(self.dp.Ls, Lsbounds[0]), np.isclose(self.dp.Ls, Lsbounds[1])]:
                        extra_msg = (extra_msg + "\nWarning: The target launch angle and final arm angle were not "
                                     "reachable given the specified constraints on psi final and sling length\n\n")
                elif self.prev_solve_type == 'arm_inertia_opt':
                    psibounds = (float(self.sccw['launch_angle']['psifmin'].get()), 
                                 float(self.sccw['launch_angle']['psifmax'].get()))
                    Iabounds = (float(self.sccw['arm_inertia_opt']['aimin'].get()), 
                                 float(self.sccw['arm_inertia_opt']['aimax'].get()))
                    self.dp.Ia = self.sol['Ia_opt']
                    extra_msg = ("Psi at release: " + str(round(self.sol['yf'][2], self.rp)) + 
                                 "\nOptimized arm inertia: " + str(round(self.sol['Ia_opt'], self.rp)) + 
                                 "\nArm design hardness: " + str(round(adh, self.rp)) + "\n")
                    if True in [np.isclose(psif, psibounds[0]), np.isclose(psif, psibounds[1]), 
                               np.isclose(self.dp.Ia, Iabounds[0]), np.isclose(self.dp.Ia, Iabounds[1])]:
                        extra_msg = (extra_msg + "\nWarning: The target launch angle and final arm angle were not " 
                                     "reachable given the specified constraints on psi final and arm inertia\n\n")
                elif self.prev_solve_type == 'max_speed':
                    self.dp.Ia = self.sol['xopt'][2]
                    self.dp.Ls = self.sol['xopt'][1]
                    extra_msg = ("Psi at release: " + str(round(self.sol['yf'][2], self.rp)) +
                                 "\nOptimized sling length: " + str(round(self.sol['xopt'][1], self.rp)) +
                                 "\nOptimized arm inertia: " + str(round(self.sol['xopt'][2], self.rp)) + 
                                 "\nArm design hardness: " + str(round(adh, self.rp)) + "\n")
                else:
                    extra_msg = "Solver type " + self.prev_solve_type + " NOT YET IMPLEMENTED"
                
                vpf = CDTtools.dyn_core_tools.vp(self.sol['yf'], self.dp) # dp has been updated if necessary
                # crucially, it only gets changed again when simulate is run
                self.vpf = np.linalg.norm(vpf)
                thetaf = self.sol['yf'][0]
                if theta_min_true < theta_min_guess:
                    extra_msg = (extra_msg + 
                                 "\nWarning: Minimum arm angle outside of interpolation range: \u03B8_min = " + 
                                    str(round(theta_min_true, self.rp)) + "\n")
                sim_res_msg = (extra_msg + "Launch speed: " + str(round(self.vpf, self.rp)) + 
                               "\nRelease time: " + str(round(self.sol['tf'], self.rp)) +
                              "\nTheta at release: " + str(round(thetaf, self.rp)))
            else:
                sim_res_msg = "Solver failed. Check the logfile for detailed error information."
                            
        else:
            # womp womp
            sim_res_msg = ("Solver did not finish. This may be caused by excessively short time steps, or the " 
                           "dynamics never encountering a stop condition. Remember that any configuration "
                           "within the search space of an optimizing solver may be tried. It may also be "
                           "helpful to read the logs or increase the timeout")
        self.msg_text.set(sim_res_msg)
        self._change_p2() # updates plot 2
        self._change_p1()

def main():
    global root
    root = tk.Tk()
    gui = CDT_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
