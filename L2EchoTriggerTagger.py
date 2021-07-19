import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import Trigger_Simulation.trigger_simulation as trigsim

class L2Trigger(object):
    def __init__(self,input_pulse_shapes,l1trigsim,plot=False,verbose=False):
        self.input_pulse_shapes = input_pulse_shapes
        self.l1trigsim = l1trigsim
        self.rel_timestamps, self.fir_filtered_tpl = self.MakeFIRfilteredTemplate(self.input_pulse_shapes,self.l1trigsim)
        self.timingOffset = np.argmax(self.fir_filtered_tpl)*16
        self.criticalAmplitude = 0.68*self.GetCriticalAmplitude(self.fir_filtered_tpl)
        self.plot=plot
        self.verbose=verbose
        if self.verbose: print("Critical Amplitude:", self.criticalAmplitude)
        
    def MakeFIRfilteredTemplate(self,input_pulse_shapes,trig):
        input_pulse_shapes_extended = [np.pad(input_pulse_shapes[i],(0,16384),'minimum') \
                                   for i in range(len(input_pulse_shapes))]
        x = input_pulse_shapes_extended[0]*100 #multiplied by 100 to ensure numerical precision when rounding to integers
        input_traces = [(x).astype('int64')] + [np.zeros(len(x),dtype='int64')]*11 + \
                   [np.zeros(4*len(x),dtype='int64')]*4
        pipe = trigsim.pipeline(
            self.l1trigsim.DF,
            self.l1trigsim.LC,
            self.l1trigsim.LC_trunc,
            self.l1trigsim.FIR,
            self.l1trigsim.FIR_trunc,
        )
        self.l1trigsim.reset()
        fir_out = pipe.send(input_traces)
        idx_maxtrig = np.argmax(fir_out,axis=1)
        maxtimestamp= idx_maxtrig*16
        rel_timestamps = [[i*16-maxtimestamp[j] for i in range(len(fir_out[j]))] for j in range(len(fir_out))]
        fir_filtered_tpl = fir_out/np.max(fir_out)#/np.max(fir_out,axis=0)

        return (rel_timestamps, fir_filtered_tpl)
    
    def GetCriticalAmplitude(self,fir_filtered_tpl):
        for scalefactor in range(1,32768,100):
            fir_filtered_tpl_scaled = scalefactor*fir_filtered_tpl
            timesAboveThresh=[0]*4
            criticalScaleFactors=[99999]*4            
            for FIRch in range(1): #Does not work yet with range(3)
                inWindow=False
                for sample, samplevalue in enumerate(fir_filtered_tpl_scaled[FIRch]):
                    if samplevalue > self.l1trigsim.ThL.ThLs[FIRch].activation:
                        if not inWindow:
                            inWindow=True
                            timesAboveThresh[FIRch]+=1
                            if timesAboveThresh[FIRch] > 1:
                                criticalScaleFactors[FIRch]=np.max(fir_filtered_tpl_scaled[FIRch])
                                return np.max(fir_filtered_tpl_scaled[FIRch])
                                break
                    if samplevalue <= self.l1trigsim.ThL.ThLs[FIRch].deactivation:
                        inWindow=False
                
            if np.count_nonzero(timesAboveThresh[FIRch])>1:
                return np.min(criticalScaleFactors)
        
        return 8800
    
    def isEcho(self,ScaledFirFiltTpl,ScaledFirFilTpltTimestamps,ThisCritL1TP,L1TP):
        #If the current trigger is equal to the critical trigger, it is not an echo
        if L1TP == ThisCritL1TP: return False
        #Loop over the 4 FIR channels
        for FIRch in range(3):
            if np.count_nonzero(self.fir_filtered_tpl[FIRch]) == 0: continue #Skip empty FIR channels
            #Window above threshold finding algorithm for the FIR filtered template
            inWindow=False
            for sample in range(len(ScaledFirFiltTpl[FIRch])):
                if not inWindow and ScaledFirFiltTpl[FIRch][sample] > 0.5*self.l1trigsim.ThL.ThLs[FIRch].activation:
                    inWindow=True
                    winlowedge=ScaledFirFilTpltTimestamps[sample]
                if inWindow and ScaledFirFiltTpl[FIRch][sample] < 2*self.l1trigsim.ThL.ThLs[FIRch].deactivation:
                    inWindow=False
                    winhighedge=ScaledFirFilTpltTimestamps[sample]
                    #Check if current trigger is within echo trigger expectation window
                    if L1TP[self.timestamp] > winlowedge and L1TP[self.timestamp] < winhighedge:
                        timestampsInWindow=(ScaledFirFilTpltTimestamps > winlowedge) & (ScaledFirFilTpltTimestamps < winhighedge)
                        echoamp=np.max(ScaledFirFiltTpl[FIRch][timestampsInWindow])
                        #Check if current trigger has amplitude compatible with echo trigger expectation
                        if L1TP[self.amplitude] < 2*echoamp:
                            if self.plot: self.ax.add_patch(Rectangle((winlowedge, 0), winhighedge-winlowedge, 2*echoamp,alpha=0.5,color="red"))
                            return True
        return False
    
    def VetoEchoTriggers(self,L1TPs):
        
        self.amplitude=L1TPs.dtype.names[0]
        self.timestamp=L1TPs.dtype.names[1]
        
        if self.plot: 
            self.fig, self.ax = plt.subplots(dpi=140)
            self.ax.plot(L1TPs[self.timestamp],L1TPs[self.amplitude],linestyle="None",marker="x",
              color="black",label="L1 Trigger Primitives")
        if self.verbose: 
            print("%s L1Trigger primitives before veto:" % len(L1TPs))
            print(L1TPs)
        
        #Find critical triggers with large amplitudes which could cause echo triggers and sort them by amplitude
        CritL1TPs=[L1TP for L1TP in L1TPs if L1TP[self.amplitude] > self.criticalAmplitude]     
        CritL1TPs = np.flip(np.sort(CritL1TPs))
        
        if self.verbose: 
            print("%s Critical L1Trigger primitives before veto:" % len(CritL1TPs))
            print(CritL1TPs)
        
        #If no critical trigger amplitudes are found, we don't need to veto anything
        if len(CritL1TPs) == 0: return(L1TPs)
        
        #Loop over critical amplitude triggers and scale FIR filtered templates by amplitude.
        #Remove triggers which are in echo trigger expectation window according to scaled FIR filtered template.
        while len(CritL1TPs) > 0:
            ScaledFirFiltTpl = self.fir_filtered_tpl*CritL1TPs[0][self.amplitude]
            ScaledFirFilTpltTimestamps = np.arange(CritL1TPs[0][self.timestamp]-self.timingOffset,CritL1TPs[0][self.timestamp]+self.timingOffset,16)
            if self.plot: self.ax.plot(ScaledFirFilTpltTimestamps,ScaledFirFiltTpl[0],
                                       label="FIR-Filtered TPL, "+str(CritL1TPs[0][self.amplitude]))
            L1TPs = [L1TP for L1TP in L1TPs if not self.isEcho(ScaledFirFiltTpl,ScaledFirFilTpltTimestamps,CritL1TPs[0],L1TP)]
            CritL1TPs=[CritL1TP for CritL1TP in CritL1TPs if not self.isEcho(ScaledFirFiltTpl,ScaledFirFilTpltTimestamps,CritL1TPs[0],CritL1TP)]
            CritL1TPs = np.flip(np.sort(CritL1TPs))
            CritL1TPs = np.delete(CritL1TPs,0)
        
        if self.plot:
            self.ax.axhline(self.criticalAmplitude,linestyle="-.",label="Critical Amplitude",color="gray")
            self.ax.axhline(self.l1trigsim.ThL.ThLs[0].activation,linestyle="dashed",label="Activation Threshold",color="gray")
            self.ax.axhline(self.l1trigsim.ThL.ThLs[0].deactivation,linestyle="dotted",label="Deactivation Threshold",color="gray")
            self.ax.add_patch(Rectangle((0, 0), 0, 0,alpha=0.5,color="red",label="Echo Trigger Region"))
            plt.legend()
            plt.title("Echo trigger expectation from scaled FIR-filtered template")
            plt.xlabel("Timestamp")
            plt.ylabel("FIR Out")
        if self.verbose: 
            print("%s L1Trigger primitives after veto:" % len(L1TPs))
            print(L1TPs)            

        return L1TPs 
    
    
"""EXAMPLE USAGE:

import L2EchoTriggerTagger

L2Trig = L2EchoTriggerTagger.L2Trigger(input_pulse_shapes,trig,plot=True,verbose=True)

L1TriggerPrimitivesAfterVeto = L2Trig.VetoEchoTriggers(L1TriggerPrimitives)

"""





"""HOW TO SIMULATE SOME L1TRIGGER PRIMITIVES AND FIR OUT TRACE QUICKLY:

def GetL1TriggerPrimitives(PulseAmplitude,Pulselength,template_phonon,l1trigsim):
    #Extend template to Pulselength:
    delta=Pulselength-len(template_phonon)
    tpl_ext = np.pad(template_phonon,(0,delta),'minimum')
    
    #Get a noise Trace
    NoiseTraces = [DCRC.readout(pchan, Pulselength) for pchan in range(12)]\
                +[DCRC.readout(qchan, 4*Pulselength) for qchan in range(12,16,1)]
    
    #Add template to noise trace
    PulseTraces = [(NoiseTraces[pchan]-np.mean(NoiseTraces[pchan])+tpl_ext*PulseAmplitude).astype('int64') for pchan in range(12)]\
                  +[(NoiseTraces[qchan]-np.mean(NoiseTraces[qchan])).astype('int64') for qchan in range(12,16,1)]

    #Run pulseTrace through L1TrigSim
    pipe = trigsim.pipeline(
        l1trigsim.DF,
        l1trigsim.LC,
        l1trigsim.LC_trunc,
        l1trigsim.FIR,
        l1trigsim.FIR_trunc,
    )
    l1trigsim.reset()
    L1TriggerPrimitives = l1trigsim.send(PulseTraces)
    FIRout = pipe.send(PulseTraces)[0]
    return(L1TriggerPrimitives,FIRout)

#Create some triggers using DCRCnoise, pulse template and L1TrigSim
amp=500
L1TriggerPrimitives,FIRout = GetL1TriggerPrimitives(amp,2*16384,template_phonon,trig)



L2Trig.ax.plot(np.arange(0,32768,16),FIRout,label="Simulated Trace, "+str(np.max(FIRout)))

"""