#########################################################################################
#                            Copyright (c) <2012>                                       #
#        Author: Gautam Varma Mantena (gautam.mantena@research.iiit.ac.in)              #
#        International Institute of Information Technology, Hyderabad, India.           #
#                            All rights reserved.                                       #
#                                                                                       #
#   Redistribution and use in source and binary forms, with or without                  #
#   modification, are permitted provided that the following conditions are met:         #
#      * Redistributions of source code must retain the above copyright                 #
#        notice, this list of conditions and the following disclaimer.                  #
#      * Redistributions in binary form must reproduce the above copyright              #
#        notice, this list of conditions and the following disclaimer in the            #
#        documentation and/or other materials provided with the distribution.           #
#      * Neither the name of the <organization> nor the                                 #
#        names of its contributors may be used to endorse or promote products           #
#        derived from this software without specific prior written permission.          #
#                                                                                       #
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND     #
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED       # 
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE              #
#   DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY                  #
#   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          #
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND         #
#   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT          # 
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS       #
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                        #
#########################################################################################
#  About the package                                                                    #
#      Following package contains functions useful for processing speech signal.        # 
#      Some of the important operations that can be done are Zero Frequency Filtering,  #
#      Speech Segmentation, FIR Filters, Linear Prediction, etc.                        #  
#                                                                                       #
#      All the functions use extensive functionality of numpy and scipy packages. Some  #
#      of the functions like FIR filters do not provide as much flexibility as compared #
#      to scipy.signal package, but are provided for easy usage. Internally they use    #
#       filter response functions provided scipy.                                       #
#########################################################################################


#!/usr/bin/python
import wave,struct,numpy,numpy.fft
import scipy.signal,matplotlib.pyplot

def wavread(wfname,stime=0,etime=0):
    '''
    Returns the contents of the wave file and its sampling frequency.
    Input to the function is the path to the wave file

    wfname is the input wave file path.
    The input for stime and etime should be seconds.
        
    wavread also reads a segment of the wave file.
    stime and etime determines for what sections in the wave file
    should the content be read. By default they are set to zero.
    When both stime and etime are zero the function reads the whole
    wave file.    
    '''
    wavfp = wave.open(wfname,'r')
    fs = wavfp.getframerate() #to obtain the sampling frequency
    sig = []
    wavnframes = wavfp.getnframes()
    sframe = int(stime * fs)                               #input is in seconds and so they are to be convereted
    wavfp.setpos(sframe)                                   #to frame count  
    eframe = int(etime * fs)
    if eframe == 0:
        eframe = wavnframes
    for i in range(sframe,eframe):                          #In wave format 2 bytes are used to represent one value
        hexv = wavfp.readframes(1)                          #readframes is an iterative function which reads 2 bytes
        sig.append(float(struct.unpack('h',hexv)[0]))
    wavfp.close()
    return numpy.array(sig,dtype='float'),fs

def wavwrite(sig,fs,wfname):
    '''
    Following function is used to create a wave file provided
    the signal with its sampling frequency. It creates a
    standard wave audio, PCM, 16 bit, mono with fs Hertz.
    It takes two types of formats
    1. If the amplitude ranges from -1 to 1 it scales the
       amplitudes of the wave files
    2. Else it converts the floats to integers and then writes
       the wave file

    Suggested normalization would be as follows:
    wav = wav/(0.0001 + max(wav))
    '''
    if max(sig) <= 1 and min(sig) >= -1:
        print '[Warning]: Scaling signal magnitudes'
        sampwidth = 2 #nunber of bytes required to store an integer value
        max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)
        sig = sig * max_amplitude
    sig = numpy.array(sig,dtype='int')
    wavfp = wave.open(wfname,'w')
    wavfp.setparams((1,2,fs,sig.size,'NONE','not compressed')) #setting the params for the wave file
    for i in sig:                                              #params: nchannels, sampwidth(bytes), framerate, nframes, comptype, compname
        hvalue = struct.pack('h',i)                            #only accepts 2 bytes => hex value
        wavfp.writeframes(hvalue)
    wavfp.close()

class Cepstrum:
    """
    Following class consists of functions to generate the
    cepstral coefficients given a speech signal.
    Following are the functions it supports
    1. Cepstral Coefficients
    2. LP Cepstral Coefficients    
    """

    def __init__(self):
        self.lpco = LPC()

    def cep(self,sig,corder,L=0):
        """
        Following function require a windowed signal along
        with the cepstral order.
        Following is the process to calculate the cepstral
        coefficients

                 x[n] <--> X[k] Fourier Transform of the
                                signal
                 c = ifft(log(|X[k]|)
             where c is the cepstral coefficients

        For the liftering process two procedures were implemented
        1. Sinusoidal
        c[m] = (1+(L/2)sin(pi*n/L))c[m]
        2. Linear Weighting

        By default it gives linear weighted cepstral coefficients.
        To obtain raw cepstral coefficients input L=None.

        For any other value of L it performs sinusoidal liftering.

        Note: The input signal is a windowed signal,
        typically hamming or hanning
        """
        c = numpy.fft.ifft(numpy.log(numpy.abs(numpy.fft.fft(sig))))
        c = numpy.real(c)
        if len(c) < corder:
            print '[Warning]: Lenght of the windowed signal < cepstral order'
            print '[Warning]: cepstral order set to length of the windowed signal'
            corder = len(sig)
            
        if L == None:           #returning raw cepstral coefficients
            return c[:corder]      
        elif L == 0:      #returning linear weighted cepstral coefficients
            for i in range(0,corder):
                c[i] = c[i] * (i+1)
            return c[:corder]
        
        #cep liftering process as given in HTK
        for i in range(0,corder):
            c[i] = (1 + (float(L)/2) * numpy.sin((numpy.pi * (i+1))/L)) * c[i]
        return c[:corder]

    def cepfeats(self,sig,fs,wlent,wsht,corder,L=None):
        """
        Following function is used to generate the cepstral coefficients
        given a speech signal. Following are the input parameters
        1. sig: input signal
        2. fs: sampling frequency
        3. wlent: window length
        4. wsht: window shift
        5. corder: cepstral order
        """
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000
        
        sig = numpy.append(sig[0],numpy.diff(sig))
        sig = sig + 0.001
        ccs = []
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            c = self.cep(smooth_signal,corder,L)
            ccs.append(c)
        return numpy.array(ccs)

    def lpccfeats(self,sig,fs,wlent,wsht,lporder,corder,L=None):
        '''
        Following fucntion in turn calls the lpccfeats from
        the LPC class to obtain the cepstral coefficients.
        Following are the input parameters
        sig: input speech signal
        fs: its sampling frequency
        wlent: window length in milli seconds
        wsht: window shift in milli seconds.
        lporder: self explanatory
        corder: no. of cepstral coefficients
        (read the lpcc documentation for the description about
        the features)
        L(ceplifter) to lift the cepstral coefficients. Read the
        documentation for the LPCC for the liftering process

        Function returns only the cepstral coefficients
        '''
        Gs,nEs,lpcs,lpccs = self.lpco.lpccfeats(sig,fs,wlent,wsht,lporder,corder,L)
        return lpccs
        

class FIRFilters:
    '''Class consists of building the following FIR filters
       a. Low Pass Filter
       b. High Pass Filter
       c. Band Pass Filter
       d. Band Reject Filter

       In general the filter is denoted as follows

                 jw               -jw            -jmw
          jw  B(e)    b[0] + b[1]e + .... + b[m]e
       H(e) = ---- = ------------------------------------
                 jw               -jw            -jnw
              A(e)    a[0] + a[2]e + .... + a[n]e

       
       Where the roots of the numerator is denoted as zeros
       and that of the denominator as poles. In FIR filters
       are represented only via the zeros. Hence we only
       compute the coefficients "b" as shown in the above
       equation.

       Class also consists of funtions to plot filters to view
       the frequency response
    '''
    def __init__(self):
        pass

    def low_pass(self,M,cfreq,wtype='blackmanharris'):
        """
        Following are the required parameters by the low pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreq is the cutoff frequency. Make sure that the
           cfreq lies between 0 and 1 with 1 being Nyquist
           frequency.
        3. wtype is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
        """
        lb = scipy.signal.firwin(M,cutoff=cfreq,window=wtype)
        return lb

    def high_pass(self,M,cfreq,wtype='blackmanharris'):
        """
        Following are the required parameters by the high pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreq is the cutoff frequency. Make sure that the
           cfreq lies between 0 and 1 with 1 being Nyquist
           frequency.
        3. win_type is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
       The high pass filter is obtained by first obtaining the impulse
       response of a low pass filter. A more detail explanation is given
       Scientists and Engineers Guide to Digital Signal Processing,
       chapter 14-16.
       """
        lb = self.low_pass(M,cfreq,wtype) #to obtain the impulse response using the low pass filter
                                                  #and then reversing
        hb = -1 * lb
        hb[M/2] = 1 + hb[M/2]
        return hb

    def band_reject(self,M,cfreqb,cfreqe,wtype='blackmanharris'):
        """
        Following are the required parameters by the high pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreqb and cfreqe are the frequency ranges that are to be suppressed.
           Make sure that the cfreqb and cfreqe lies between 0 and 1 with 1
           being Nyquist frequency.
        3. wtype is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
       The band reject filter is obtained by first obtaining by combining the
       low pass filter and the high pass filter responses. A more detail explanation
       is given Scientists and Engineers Guide to Digital Signal Processing,
       chapter 14-16.
       """
        lb = self.low_pass(M,cfreqb,wtype) #coefficients from the low pass filter
        hb = self.high_pass(M,cfreqe,wtype) #coefficients from the high pass filter

        brb = lb + hb
        return brb

    def band_pass(self,M,cfreqb,cfreqe,wtype='blackmanharris'):
        """
        Following are the required parameters by the high pass filter
        1. M determines the number of filter taps. M should always be
           even
        2. cfreqb and cfreqe are the frequency ranges that are to be captured.
           Make sure that the cfreqb and cfreqe lies between 0 and 1, with 1
           being Nyquist frequency.
        3. wtype is the type of window to be provided. Supporting
           window types are
           a. blackmanharris
           b. hamming
           c. hanning
       The band pass filter is obtained by using the band reject filter. A more
       detail explanation is given Scientists and Engineers Guide to Digital
       Signal Processing, chapter 14-16.
       """

        brb = self.band_reject(M,cfreqb,cfreqe,wtype)
        bpb = -1 * brb
        bpb[M/2] = 1 + bpb[M/2]
        return bpb
    
    def fsignal(self,sig,b):
        """
        Following function outputs the filtered signal
        using the FIR filter coefficients b.
        """
        fsig = scipy.signal.lfilter(b,[1],sig)
        M = len(b)          #fir filters has a delay of (M-1)/2
        fsig[0:(M-1)/2] = 0 #setting the delay values to zero
        return fsig
        
        
    def plotResponse(self,b):
        """
        Following function plots the amplitude and phase response
        given the impulse response of the FIR filter. The impulse
        response of the FIR filter is nothing but the numerator (b)
        of the transfer function.
        """
        w,h = scipy.signal.freqz(b)
        h_db = 20.0 * numpy.log10(numpy.abs(h))
        ph_angle = numpy.unwrap(numpy.angle(h))

        fig = matplotlib.pyplot.figure()
        subp1 = fig.add_subplot(3,1,1)
        subp1.text(0.05,0.95,'Frequency Response',transform=subp1.transAxes,fontsize=16,fontweight='bold',va='top')
        subp1.plot(w/max(w),numpy.abs(h))
        subp1.set_ylabel('Magnitude')

        
        subp2 = fig.add_subplot(3,1,2)
        subp2.text(0.05,0.95,'Frequency Response',transform=subp2.transAxes,fontsize=16,fontweight='bold',va='top')
        subp2.plot(w/max(w),h_db)
        subp2.set_ylabel('Magnitude (DB)')
        subp2.set_ylim(-150, 5)

        subp3 = fig.add_subplot(3,1,3)
        subp3.text(0.05,0.95,'Phase',transform=subp3.transAxes,fontsize=16,fontweight='bold',va='top')
        subp3.plot(w/max(w),ph_angle)
        subp3.set_xlabel('Normalized Frequency')
        subp3.set_ylabel('Angle (radians)')
        
        fig.show()


class SignalMath:
    '''
    Contains quite commonly used mathematical operations
    needed for signal processing techniques.
    '''
    def __init__(self):
        pass

    def movavg(self,sig,fs,wlent):
        '''
        The following function is a mathematical representation for
        a moving average. The input to the function is the signal and
        the window length in milli seconds.
        Following is the mathematical equation for the moving average.

        y[n] = 1/(2*N+1)sum(sig[i+n]) for i = -N to +N

        y[n], sig[n] are both discrete time signals

        sig is the signal and wlent is the window length in milli seconds
        '''
        wlenf = (wlent * fs)/1000
        window = numpy.array([1] * wlenf)
        avg = numpy.convolve(sig,window,mode='full')
        avg = avg[(window.size/2) - 1:avg.size - (window.size/2)]
        norm = numpy.convolve(window,numpy.array([1] * avg.size),mode='full')
        norm = norm[(window.size/2) - 1:norm.size - (window.size/2)]
        return numpy.divide(avg,norm)


class ZeroFreqFilter:
    '''
    Containg functions required to obtain the Zero Frequncy Filtered Signal.
    Following is the procedure to obtain the Zero Frequency Filtered Signal:

    Let s[n] be the input signal then

    1. Bias Removal:

            x[n] = s[n] - s[n-1]

       Used to remove any dc or low frequency bias that might have been
       captured during recording.

    2. Zero Frequency Filtering:
       Pass the signal twice through a zero frequency resonator. That is
       containes two poles on the unit circle with zero frequency.

           y1[n] = -1 * SUMMATION( a(k) * y[n-k] ) + x[n]   where k = 1,2
                           k
           y2[n] = -1 * SUMMATION( a(k) * y2[n-k] ) + y1[n] where k = 1,2
                           k
                        where a(1) = -2 and a(2) = 1

       The above two operations can be obtained by finding the cumulative
       sum of the signal x[n] four times.

    3. Trend Removal:

           y[n] = y2[n] - (moving average of y2[n] of windlow lenght n)

       The moving average function is clrealy mentioned in the Math class.
       Window length n is important. The choice of the window length is not
       very critical as long as it is in the range of about 1 to 2 times
       the average pitch period.
    '''
    def __init__(self):
        self.sm = SignalMath()

    def getZFFSignal(self,sig,fs,wlent=30,wsht=20,mint=3):
        '''
        Following function returns the Zero Frequency Filtered Signal.
        Following are the steps involved in generating the Zero
        Frequency Filtered Signal:
        1. Bias Removal
        2. Zero Frequency Resonator
        3. Average Pitch Estimation
        4. Trend Removal

        sig is the samples in the wave file.
        fs is the sampling frequency in Hz.
        wlent is the window length in milli-seconds.
        wsht is the window shift in milli-seconds.
        mint is the minimum pitch value in milli-seconds.
        '''
        sig = sig - (sum(sig)/sig.size)
        dsig = numpy.diff(sig) #bias removal
        dsig = numpy.append(dsig,dsig[-1])
        dsig = numpy.divide(dsig,max(abs(dsig))) #normalization
        csig = numpy.cumsum(numpy.cumsum(numpy.cumsum(numpy.cumsum(dsig)))) #zero frequency resonator
        wlenpt = self.__avgpitch(dsig,fs,wlent,wsht,mint) #estimating the average pitch
        wlenpf = int((wlenpt * fs)/1000) #converting pitch in milli seconds to pitch in number of samples
        tr = numpy.subtract(csig,self.sm.movavg(csig,fs,wlenpt)) #trend removal
        tr = numpy.subtract(tr,self.sm.movavg(tr,fs,wlenpt)) #trend removal
        tr = numpy.subtract(tr,self.sm.movavg(tr,fs,wlenpt)) #trend removal
        tr = numpy.subtract(tr,self.sm.movavg(tr,fs,wlenpt)) #trend removal
        for i in range(dsig.size - (wlenpf*3) - 1,dsig.size): # To remove the trailing samples. Without removing the trailing samples
            tr[i] = 0                                        # we cannot view the ZFF signal as they have huge values
        tr = numpy.divide(tr,max(abs(tr)))                   # normalizing     
        return tr

    def __avgpitch(self,sig,fs,wlent,wsht,mint=3):
        '''
        Gets the average pitch from the input signal. The window length
        is for the remove trend function. Window length anything between
        pitch and twice the pitch should be adequate.

        sig is the samples in the wave file.
        fs is the sampling frequency in Hz.
        wlent is the window length in milli-seconds.
        wsht is the window shift in milli-seconds.
        mint is the minimum pitch value in milli-seconds.
        '''
        wlenf = (wlent * fs)/1000
        wshtf = (wsht * fs)/1000
        nof = (sig.size - wlenf)/wshtf
        pitch = []
        for i in range(0,nof):#block processing for obtaining pitch from each of the window
            sigblock = sig[i*wshtf:i*wshtf + wlenf]
            pitch.append(self.__getpitch(sigblock,fs,mint))
        pitch = numpy.array(pitch)
        pbins = numpy.arange(3,20,2)#min pitch is 3 msec and maximum is 20 msec. The bin rages from 3-5,5-7,..
        phist = numpy.histogram(pitch,bins=pbins)[0]#plotting histogram for each of the pitch values
        prange_index = 0                            #to see the most commonly occuring pitch value
        prange_count = 0
        for i in range(0,phist.size):
            if phist[i] > prange_count:
                prange_count = phist[i]
                prange_index = i
        avgpitch = (pbins[prange_index] + pbins[prange_index+1])/2#finding the average pitch value        
        return avgpitch

    def __getpitch(self,sig,fs,mint=3):
        '''
        To find the pitch given a speech signal of some window.
        sig is the samples in the wave file.
        fs is the sampling frequency in Hz.
        mint is the minimum pitch value in milli-seconds.
        '''
        minf = (mint * fs)/1000 #to convert into number of frames/samples
        cor = numpy.correlate(sig,sig,mode='full')
        cor = cor[cor.size/2:]#auto correlation is symmetric about the y axis.
        cor = cor/max(abs(cor))#normalizing the auto correlation values
        dcor = numpy.diff(cor)#finding diff
        dcor[0:minf] = 0#setting values of the frames below mint to be zero
        locmax = numpy.array([1] * dcor.size) * (dcor > 0)
        locmin = numpy.array([1] * dcor.size) * (dcor <= 0)
        locpeaks = numpy.array([2] * (dcor.size - 1)) * (locmax[0:locmax.size - 1] + locmin[1:locmin.size] == 2)#to get the positive peaks
        maxi,maxv = self.__getmax(cor,locpeaks)
        return (maxi * 1000.0)/fs

    def __getmax(self,src,peaks):
        '''
        To get peak which has the maximum value.
        '''
        maxi = 0
        maxv = 0
        for i in range(0,peaks.size):           #diff values will be one sample less than the original signal
            if src[i] > maxv and peaks[i] == 2: #consider only the diff values in the for loop
                maxi = i
                maxv = src[i]
        return maxi,maxv

class SegmentSpeech:
    '''
    Following class provides functions, which are useful segment speech.
    All the functions utilize the Zero Frequency Filtered signal to
    segment the speech.

    Functions of format segmentxc represents a function to segment
    the speech into x categories.
    '''
    def __init__(self):
        self.sm = SignalMath()

    def vnv(self,zfs,fs,theta=2.0,wlent=30):
        '''
        To obtain the voiced regions in the speech segment.
        Following are the input parameters
        1. zfs is the Zero Frequency Filtered Signal
        2. fs is the sampling rate
        3. wlent is the window length required for the moving average.
        '''
        zfse = 1.0 * zfs * zfs #squaring each of the samples: to find the ZFS energy.
        zfse_movavg = numpy.sqrt(self.sm.movavg(zfse,fs,wlent)) #averaging across wlent window
        zfse_movavg = zfse_movavg/max(zfse_movavg) #normalzing
        avg_energy = sum(zfse_movavg)/zfse_movavg.size #average energy across all the window.
        voicereg = zfse_movavg * (zfse_movavg >= avg_energy/theta) #selecting segments whose energy is higher than the average.        
        return voicereg

    def __zerocross(self,zfs):
        '''
        To obtain the postive zero crossing from the
        Zero Frequency Filtered Signal.
        '''
        zc = numpy.array([1]) * (zfs >= 0)
        dzc = numpy.diff(zc)
        zcst = numpy.diff(zfs) * (dzc > 0)        
        return numpy.append(zcst,0)

    def getGCI(self,zfs,voicereg):
        '''
        To obtain the Glottal Closure Instants and the strength of
        excitations. We obtain the Strength of excitation by taking
        the derivative at postive zero crossing from the ZFS.
        Voiced regions are used to remove spurious GCI.
        '''
        gci = self.__zerocross(zfs) * (voicereg > 0)#only considering the GCI in regions where it was
        return gci                   #detected as voiced regions by the vnv function

    def segment2c(self,gci,fs,sildur=300):
        '''
        The following function returns a two category segmentation of
        speech, speech and silence region

        Following are the input parametrs
        1. gci is the glottal closure instants.
        2. sildur is the minimum duration without gci to
        classify the segment as sil
        '''
        stime = 0#starting time
        etime = 0#end time
        wtime = (gci.size * 1000.0)/fs#total time of the wave file
        wlab = []#array containing the lab information for the wave file
        i = 0
        vflag = False#flag to keep track of voiced sounds
        
        while i < gci.size:
            if gci[i] > 0:
                etime = (i * 1000.0)/fs
                if (etime - stime) >= sildur: #to check whether its a silence region or not
                    if stime == 0 and etime == wtime:
                        stime = stime
                        etime = etime
                    elif stime == 0:
                        stime = stime
                        etime = etime - 50
                    elif etime == wtime:
                        stime = stime + 50
                        etime = etime
                    else:
                        stime = stime + 50
                        etime = etime - 50
                    wlab.append((stime,etime,'SIL')) #to make sure that the end unvoiced sounds are
                    stime = etime                    #not classified as silence
                else:
                    stime = etime
            i += 1

        #fixing the trailing silence. Because the trailing silence might not end with an epoch.
        if etime != wtime:
            if (wtime - etime) >= sildur:
                wlab.append((etime+50,wtime,'SIL'))

        #some times there might not be any silence in the wave file
        if len(wlab) == 0:
            wlab.append((0,wtime,'SPH'))

        #fixing the missing time stamps
        #the above loop only puts time stamps for the silence regions
        cwlab = []

        for i in range(0,len(wlab)):
            tlab = wlab[i]
            stime = tlab[0]
            
            etime = tlab[1]
            if len(cwlab) == 0:
                if stime == 0:
                    cwlab.append(tlab[:])
                else:
                    cwlab.append((0,stime,'SPH'))
                    cwlab.append(tlab[:])
            else:
                if cwlab[-1][1] == stime:
                    cwlab.append(tlab[:])
                else:
                    cwlab.append((cwlab[-1][1],stime,'SPH'))
                    cwlab.append(tlab[:])
        if wlab[-1][1] != wtime:
            cwlab.append((wlab[-1][1],wtime,'SPH'))
        return cwlab

    def segmentvnvc(self,gci,fs,uvdur=18):
        '''
        The following function returns a three category segmentation of
        speech, namely, voiced(VOI), unvoiced(UNV).

        Following are the input parametrs
        1. gci is the glottal closure instants.
        2. fs is the sampling rate.
        3. uvdur is the maximum duration between any two GCI
        to classify them at VOI or else they would be
        classified as UNV.
        '''

        stime = 0#start time
        etime = 0#end time
        ptime = stime#to keep track of previous time
        wtime = (gci.size * 1000.0)/fs#total time of the wave file
        wlab = []#array containing the time stamps of the wave file
        i = 0
        vflag = False#to keey track of voiced regions

        while i < gci.size:
            if gci[i] > 0:                
                etime = (i * 1000.0)/fs
                if (etime - ptime) < uvdur:#to check whether VOICED or not
                    ptime = etime          #if its more than uvdur then it is UNVOICED
                    vflag = True                    
                else:                      #to tag it as UNVOICED
                    if vflag:
                        wlab.append((stime,ptime,'VOI'))
                        vflag = False
                    wlab.append((ptime,etime,'UNV'))
                    stime = etime
                    ptime = stime
            i += 1
        #fixing the trailing tags
        if etime != wtime:
            if vflag:
                wlab.append((stime,etime,'VOI'))
            wlab.append((etime,wtime,'UNV'))                
        return wlab

    
    def segment3c(self,gci,fs,uvdur=18,sildur=300):
        '''
        The following function returns a three category segmentation of
        speech, namely, voiced(VOI), unvoiced(UNV) and silence segments(SIL).

        Following are the input parametrs
        1. gci is the glottal closure instants.
        2. fs is the sampling rate.
        3. uvdur is the maximum duration between any two GCI
        to classify them at VOI or else they would be
        classified as UNV.
        4. sildur is the minimum duration without gci to
        classify the segment as SIL.
        '''

        stime = 0#start time
        etime = 0#end time
        ptime = stime#to keep track of previous time
        wtime = (gci.size * 1000.0)/fs#total time of the wave file
        wlab = []#array containing the time stamps of the wave file
        i = 0
        vflag = False#to keey track of voiced regions

        while i < gci.size:
            if gci[i] > 0:                
                etime = (i * 1000.0)/fs
                if (etime - ptime) < uvdur:#to check whether VOICED or not
                    ptime = etime          #if its more than uvdur then it is UNVOICED
                    vflag = True
                elif (etime - ptime) > sildur:#to tag it as SILENCE
                    if vflag:
                        wlab.append((stime,ptime,'VOI'))
                        vflag = False
                    wlab.append((ptime,etime,'SIL'))
                    stime = etime
                    ptime = stime
                else:#to tag it as UNVOICED
                    if vflag:
                        wlab.append((stime,ptime,'VOI'))
                        vflag = False
                    wlab.append((ptime,etime,'UNV'))
                    stime = etime
                    ptime = stime
            i += 1
        #fixing the trailing tags
        #one assumption made is that the trailing speech is most likely
        #contains the SILENCE
        if etime != wtime:
            if vflag:
                wlab.append((stime,etime,'VOI'))
            wlab.append((etime,wtime,'SIL'))                
        return wlab

class LPC:
    '''
    Following class consists functions relating to Linear Prediction
    Analysis.

    The methodology adopted is as given in Linear Prediction:
    A Tutorial Review by John Makhoul.
    This is a preliminary version and the funtions are bound
    to change.

    Following are the functions present
    1. lpc: to calculate the linear prediction coefficients
    2. lpcfeats: to extract the linear prediction coefficients
                 from a wav signal
    3. lpcc: to calculate the cepstral coefficients
    4. lpccfeats: to extract the cepstral coefficients from a
                  wav signal
    5. lpresidual: to obtain the lp residual signal.
    6. plotSpec: to plot the power spectrum of a signal and its 
                 lp spectrum
    '''
    def __init__(self):
        pass

    def __autocor(self,sig,lporder):
        '''
        To calculate the AutoCorrelation Matrix from the input signal.
        Auto Correlation matrix is defined as follows:

        
              N-i-1
        r[i] = sum s[n]s[n+i] for all n
                n

        Following are the input parameters:
        sig: input speech signal
        lporder: self explanatory
        '''
        r = numpy.zeros(lporder + 1)
        for i in range(0,lporder + 1):
            for n in range(0,len(sig) - i):
                r[i] += (sig[n] * sig[n+i])
        return r

    def lpc(self,sig,lporder):
        '''
        Levinson-Durbin algorithm was implemented for
        calculating the lp coefficients. Please refer
        to the Linear Prediction: A Tutorial Review by
        John Makhoul
        
        Following are the input parameters:
        sig: input signal (of fixed window). 
        lporder: self explanatory

        Output of the function is the following:
        1. Gain (E[0])
        2. Normalized Error (E[lporder]/E[0])
        3. LP Coefficients

        Function returns the following
        1. G: Gain
        2. nE: normalized error E[p]/E[0]
        3. LP Coefficients
        '''
        r = self.__autocor(sig,lporder) #Autocorrelation coefficients
        a = numpy.zeros(lporder + 1) #to store the a(k)
        b = numpy.zeros(lporder + 1) #to store the previous values of a(k)
        k = numpy.zeros(lporder + 1) #PARCOR coefficients
        E = numpy.zeros(lporder + 1)
        E[0] = r[0] #Energy of the signal
        for i in range(1,lporder+1):
            Sum = 0.0
            for j in range(1,i):
                Sum += (a[j] * r[i-j])
            k[i] = -(r[i] + Sum)/E[i-1]
            a[i] = k[i]
            for j in range(1,i):
                b[j] = a[j]
            for j in range(1,i):
                a[j] = b[j] + (k[i] * b[i-j])
            E[i] = (1.0 - (k[i]**2)) * E[i-1]
        a[0] = 1
        nE = E[lporder]/E[0] #normalized error
        G = r[0] #gain parameter
        for i in range(1,lporder+1):
            G += a[i] * r[i]
        return G,nE,a #G is the gain

    def lpcfeats(self,sig,fs,wlent,wsht,lporder):
        '''
        Extract the LPC features from the wave file.
        Following are the input parameters
        sig: input speech signal
        fs: its sampling frequency
        wlent: window length in milli seconds
        wsht: window shift in milli seconds.
        lporder: self explanatory
        
        Function returnes the following
        1. G: Gain for each frame
        2. nE: Normalized error for each frame
        3. LP coefficients for each frame.
        '''
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000
        
        sig = numpy.append(sig[0],numpy.diff(sig))
        sig = sig + 0.001
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        lpcs = [] #to store the lp coefficients
        nEs = [] #normalized errors
        Gs = [] #gain values

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            G,nE,a = self.lpc(smooth_signal,lporder)
            lpcs.append(a)
            nEs.append(nE)
            Gs.append(G)
        return numpy.array(Gs),numpy.array(nEs),numpy.array(lpcs)

    def __ersignal(self,sig,a):
        '''
        Returns the error signal provided a set of LP
        coefficients. Error computation is as follows

        e[n] = s[n] + sum a[k]s[n-k] for k = 1,2,..p
                       k

        Following are the input parameters
        1. sig: input signal. Consider using rectangular
                windowed signal, even though the LP
                coefficients were computed on hamming windowed
                signal
        2. a: LP coefficients.
        '''
        residual = numpy.zeros(len(sig))
        for i in range(0,len(sig)):
            for k in range(0,len(a)):
                if (i-k) >= 0:
                    residual[i] += a[k]*sig[i-k]
        return residual

    def lpresidual(self,sig,fs,wlent,wsht,lporder):
        '''
        Computes the LP residual for a given speech signal.
        Signal is windowed using hamming window and LP
        coefficients are computed. Using these LP coefficients
        and the original signal residual for each frame of
        window length wlent is computed (see function error_signal)

        Following are the input parameters:
        1. sig: input speech signal
        2. fs: sampling rate of the signal
        3. wlent: window length in milli seconds
        4. wsht: window shift in milli seconds
        5. lporder: self explanatory.
        '''
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000
        
        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove the dc
        sig = sig + 0.0001 #to make sure that there are no zeros in the signal
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        residual_signal = numpy.zeros(len(sig))
        residual_index = 0

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            G,nE,a = self.lpc(smooth_signal,lporder)
            er = self.__ersignal(window_signal,a)

            for i in range(0,wshf):
                residual_signal[residual_index] = er[i]
                residual_index += 1
        return residual_signal

    def lpcc(self,G,a,corder,L=0):
        """
        Following function returns the cepstral coefficients.
        Following are the input parameters
        1. G is the gain (energy of the signal)
        2. a are the lp coefficients
        3. corder is the cepstral order
        4. L (ceplifter) to lift the cepstral values
        The output of the function will the set of cepstral (optional)
        coefficients with the first value log(G)
        So the number of cepstral coefficients will one more than
        the corder.
        
        For the liftering process two procedures were implemented
        1. Sinusoidal
        c[m] = (1+(L/2)sin(pi*n/L))c[m]
        2. Linear Weighting

        By default it gives linear weighted cepstral coefficients.
        To obtain raw cepstral coefficients input L=None.

        For any other value of L it performs sinusoidal liftering.

        Note that number of cepstral coefficients can be more than
        lporder. Generally it is suggested that corder = (3/2)lporder        
        """
        c = numpy.zeros(corder+1)
        c[0] = numpy.log(G)
        p = len(a) -1 #lp order + 1, a[0] = 1

        if corder <= p: #calculating if the corder is less than the lp order
            for m in range(1,corder+1): 
                c[m] = a[m]
                for k in range(1,m):
                    c[m] -= (float(k)/m) * c[k] * a[m-k]
        else:
            for m in range(1,p+1):
                c[m] = a[m]
                for k in range(1,m):
                    c[m] -= (float(k)/m) * c[k] * a[m-k]

            for m in range(p+1,corder+1):
                for k in range((m-p),m):
                    c[m] -= (float(k)/m) * c[k] * a[m-k]

        if L == None:           #returning raw cepstral coefficients
            return c      
        elif L == 0:      #returning linear weighted cepstral coefficients
            for i in range(1,corder+1):
                c[i] = c[i] * i
            return c
        
        #cep liftering process as given in HTK
        for i in range(1,corder+1):
            c[i] = (1 + (float(L)/2) * numpy.sin((numpy.pi * i)/L)) * c[i]

        return c

    def lpccfeats(self,sig,fs,wlent,wsht,lporder,corder,L=0):
        '''
        Computes the LPCC coefficients from the wave file.
        Following are the input parameters
        sig: input speech signal
        fs: its sampling frequency
        wlent: window length in milli seconds
        wsht: window shift in milli seconds.
        lporder: self explanatory
        corder: no. of cepstral coefficients
        (read the lpcc documentation for the description about
        the features)
        L(ceplifter) to lift the cepstral coefficients. Read the
        documentation for the LPCC for the liftering process

        Function returns the following
        1. G: Gain for each of the frames
        2. nE: Normalized errors for each of the frames
        3. LP Coefficients for each of the frames
        4. LP Cepstral Coefficients for each of the frames
        '''
        wlenf = (wlent * fs)/1000
        wshf = (wsht * fs)/1000
        
        sig = numpy.append(sig[0],numpy.diff(sig)) #to remove dc
        sig = sig + 0.001 #making sure that there are no zeros in the signal
        noFrames = int((len(sig) - wlenf)/wshf) + 1
        lpcs = [] #to store the lp coefficients
        lpccs = []
        nEs = [] #normalized errors
        Gs = [] #gain values

        for i in range(0,noFrames):
            index = i * wshf
            window_signal = sig[index:index+wlenf]
            smooth_signal = window_signal * numpy.hamming(len(window_signal))
            G,nE,a = self.lpc(smooth_signal,lporder)
            c = self.lpcc(G,a,corder,L)
            lpcs.append(a)
            lpccs.append(c)
            nEs.append(nE)
            Gs.append(G)
        return numpy.array(Gs),numpy.array(nEs),numpy.array(lpcs),numpy.array(lpccs)

    def plotSpec(self,sig,G,a,res=0):
        """
        The following function plots the power spectrum of the wave
        signal along with the lp spectrum. This function is primary
        to analyse the lp spectrum
        Input for the function is as follows:
        sig is the input signal
        G is the gain
        a are the lp coefficients
        res is the resolution factor which tell the number of zeros
        to be appended before performing fft on the inverse filter

        Following funcion provides the power spectrum for the following
        Power spectrum of the signal

                          s[n]    <------->     S(w)
                           P(w) = 20 * log(|S(w)|)
                           
                                                
                             P'(w) = 20 * log(G /|A(w)|)
               where A(w) is the inverse filter and is defined as follows

                                           p              -jkw
                             A(z) = 1 + SUMMATION a[k] * e
                                           k=1

        The xaxis in the plots give the frequencies in the rage 0-1,
        where 1 represents the nyquist frequency
        """
        for i in range(0,res):
            a = numpy.insert(a,-1,0) #appending zeros for better resolution
        fftA = numpy.abs(numpy.fft.fft(a))
        Gs = numpy.ones(len(a)) * G
        P1 = 10 * numpy.log10(Gs/fftA)
        P1 = P1[0:len(P1)/2] #power spectrum of the lp spectrum

        P = 10 * numpy.log10(numpy.abs(numpy.fft.fft(sig))) #power spectrum of the signal
        P = P[:len(P)/2]

        x = numpy.arange(0,len(P))
        x = x/float(max(x))

        matplotlib.pyplot.subplot(2,1,1)
        matplotlib.pyplot.title('Power Spectrum of the Signal')
        matplotlib.pyplot.plot(x,P)
        matplotlib.pyplot.xlabel('Frequency')
        matplotlib.pyplot.ylabel('Amplitude (dB)')
        matplotlib.pyplot.subplot(2,1,2)
        matplotlib.pyplot.title('LP Spectrum of the Signal')
        matplotlib.pyplot.plot(x,P1)
        matplotlib.pyplot.xlabel('Frequency')
        matplotlib.pyplot.ylabel('Amplitude (dB)')
        matplotlib.pyplot.show()


            

        

