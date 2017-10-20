import sys
from Queue      import Queue, Empty
from subprocess import call
import binascii
import time
import signal
import scipy
# import matplotlib.mlab as mlab
import numpy as np
# import pandas as pd
import heapq
from scipy import signal
import json
from requests import *
import datetime
import math
from time import sleep
import pygame as pg
# from pyaudio import PyAudio
from constantes import *

def filter_data(data, fs_hz):
    '''
    filter from 2 to 50 Hz, helps remove 50Hz noise and replicates paper
    US : 60Hz, UE : 50Hz
    also helps remove the DC line noise (baseline drift)
    Wn = fc/(fs/2) is the cutoff frequency, frequency at which we lose 3dB.
    For digital filters, Wn is normalized from 0 to 1, where 1 is the Nyquist frequency, pi radians/sample. (Wn is thus in half-cycles / sample.)
    '''

    b, a = scipy.signal.butter(4, [0.5 / (fs_hz / 2.0), 44.0 / (fs_hz / 2.0)], btype='bandpass')
    # f_data = signal.lfilter(b, a, data, axis=0)
    f_data = scipy.signal.filtfilt(b ,a, data)
    # OTHER FILTERS

    # filter the data to remove DC
    # hp_cutoff_hz = 1.0
    # b1, a1 = signal.butter(2, hp_cutoff_hz / (fs_hz / 2.0), 'highpass')  # define the filter
    # ff_data = signal.lfilter(b1, a1, data, 0)  # apply along the zeroeth dimension

    # notch filter the data to remove 50 Hz and 100 Hz
    # notch_freq_hz = np.array([50.0])  # these are the center frequencies
    # for freq_hz in np.nditer(notch_freq_hz):  # loop over each center freq
    #     bp_stop_hz = freq_hz + 3.0 * np.array([-1, 1])  # set the stop band
    #     b, a = signal.butter(3, bp_stop_hz / (fs_hz / 2.0), 'bandstop')  # create the filter
    #     fff_data = signal.lfilter(b, a, f_data, 0)  # apply along the zeroeth dimension

    return f_data

def extract_freqbandmean(N, fe, signal, fmin, fmax):
    #f = np.linspace(0,fe/2,int(np.floor(N/2)))
    fftsig = abs(np.fft.fft(signal))
    # print fftsig.shape
    fftsig = fftsig[fmin+1:fmax+1]
    mean = np.mean(fftsig)
    return mean

def extract_freqband(N, fe, signal, fmin, fmax):
    fftsig = abs(np.fft.fft(signal))
    # print fftsig.shape
    fftsig = fftsig[fmin+1:fmax+1]
    length = len(fftsig)
    # freq = np.fft.fftfreq(N, timestep)
    return fftsig, length # why is there no indication on the fft length ?

def extract_freqbandmin(N, fe, signal, fmin, fmax):
    #f = np.linspace(0,fe/2,int(np.floor(N/2)))
    fftsig = abs(np.fft.fft(signal))
    # print fftsig.shape
    # freq = np.fft.fftfreq(200, timestep)
    fftsig = fftsig[fmin:fmax]
    min = np.min(fftsig)
    return fftsig

def extract_freqbandmax(N, fe, signal, fmin, fmax):
    #f = np.linspace(0,fe/2,int(np.floor(N/2)))
    fftsig = abs(np.fft.fft(signal))
    # print fftsig.shape
    fftsig = fftsig[fmin:fmax]
    max = np.amax(fftsig)
    return max

def neurofeedback_freq(array, freqMax, freqMin):
    last3 = np.average(array[-3:-1])
    max = np.amax(array)
    min = np.min(array)
    a = 1. * (freqMin - freqMax) / (max - min)
    b = freqMin - max  * a
    frequency = a * last3 + b
    return frequency

def neurofeedback_freq_arctan(array, freqMax, freqMin):
    spread_average = np.average(array[-5:-1])
    globalMean = np.average(array)
    frequency = (freqMax-freqMin)/math.pi*np.arctan(spread_average*1e9-globalMean)+freqMax-freqMin    # 1000/Pi * arctan(x-A) + 1000, gives frequency between 500 and 1500
    return frequency

def neurofeedback_volume(array, volMax, volMin):
    last = array[-1]
    max = np.amax(array)
    min = np.min(array)
    a = 1. * (volMin - volMax) / (max - min)
    b = volMin - max  * a
    volume = a * last + b
    return volume

def enqueue_output(out, queue):
    while True:
        lines = out.readline()
        out.flush()
        #if lines != '\n' :
            #queue.put(float(lines))
        queue.put(float(lines))
            #print queue

def sine_tone(freq, duration, bitrate):
    #See http://en.wikipedia.org/wiki/Bit_rate#Audio
    BITRATE = bitrate #number of frames per second/frameset.

    #See http://www.phy.mtu.edu/~suits/notefreqs.html
    FREQUENCY = freq #Hz, waves per second, 261.63=C4-note.
    LENGTH = duration #seconds to play sound

    NUMBEROFFRAMES = int(BITRATE * LENGTH)
    RESTFRAMES = NUMBEROFFRAMES % BITRATE
    WAVEDATA = ''
    # print (type(FREQUENCY))

    for x in xrange(NUMBEROFFRAMES):
        WAVEDATA += chr(int(math.sin(x / ((BITRATE / FREQUENCY) / math.pi)) * 127 + 128))
    #fill remainder of frameset with silence
    for x in xrange(RESTFRAMES):
        WAVEDATA += chr(128)

    p = PyAudio()
    stream = p.open(
        format=p.get_format_from_width(1),
        channels=1,
        rate=BITRATE,
        output=True,
        )
    stream.write(WAVEDATA)
    stream.stop_stream()
    stream.close()
    p.terminate()

def punch(level, levels_images, fond, punch): # function that change the image and increase the level

    level = level + 1
    level_img = pg.image.load(levels_images[level-1]).convert_alpha()
    scaled_level = pg.transform.scale(level_img, (100, 440))
    screen.blit(scaled_level, (700, 100))
    punch_noise.play()
    screen.blit(fond, (0, 0))
    screen.blit(punch, (200+2*x,-9))
    time.sleep(1)

def movePunchinBall(angle, screen, scoreBar, scoreDigit, fond, image):
    animUpLeft = []
    animUpRight = []
    allAnim = []
    for i in range(angle):
        animUpLeft.append(pg.transform.rotate(image, -i))

    for i in range(angle):
        animUpRight.append(pg.transform.rotate(image, i))

    allAnim.append(animUpLeft + list(reversed(animUpLeft)) + animUpRight + list(reversed(animUpRight)))
    # print allAnim
    for j in range(len(allAnim[0])):
        screen.blit(fond, (0, 0))
        # print allAnim[0][1]
        screen.blit(allAnim[0][j], (350, -5))
        # print 1
        # sleep(0.1)
        pg.display.update()

        # screen.blit(scoreBar, (317, 460))
        # screen.blit(scoreDigit, (800, 30))


#TODO function that returns the next position in traininging game
# def newPosition ( ):

def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

def whichButtonHomeV011(mouse, w_display, h_display):
    button = 0

    if (int(mouse[0]) <= 5.* w_display / 6 ) & (int(mouse[0]) >= 1.*w_display / 10) & (int(mouse[1]) <= 6. * h_display / 7 ) & (int(mouse[1]) >= 1. * h_display / 7):
        button = 2

    elif (int(mouse[0]) <= 4.* w_display / 5) & (int(mouse[0]) >= 1.*w_display / 5 ) & (int(mouse[1]) <= h_display-100) & (int(mouse[1]) >= 50):
        button = 3

    # elif (int(mouse[0]) <= 4.*w_display / 5 + 30) & (int(mouse[0]) >= 4.*w_display / 5 - 30) & (int(mouse[1]) <= h_display - 50) & (int(mouse[1]) >= h_display - 100):
    #     button = 4
    # print button
    # return 1
    return button

def whichButtonHome(mouse, w_display, h_display):
    button = 0

    if (int(mouse[0]) <= 1.*w_display / 5 + 30) & (int(mouse[0]) >= 1.* w_display / 5 - 30) & (int(mouse[1]) <= h_display - 50) & (int(mouse[1]) >= h_display - 100):
        button = 1

    elif (int(mouse[0]) <= 2.* w_display / 5 + 30) & (int(mouse[0]) >= 2.*w_display / 5 - 30) & (int(mouse[1]) <= h_display - 50) & (int(mouse[1]) >= h_display - 100):
        button = 2

    elif (int(mouse[0]) <= 3.*w_display / 5 + 30) & (int(mouse[0]) >= 3.*w_display / 5 - 30) & (int(mouse[1]) <= h_display - 50) & (int(mouse[1]) >= h_display - 100):
        button = 3

    elif (int(mouse[0]) <= 4.*w_display / 5 + 30) & (int(mouse[0]) >= 4.*w_display / 5 - 30) & (int(mouse[1]) <= h_display - 50) & (int(mouse[1]) >= h_display - 100):
        button = 4
    # print button
    # return 1
    return button


def whichButtonReturn(mouse, w_display, h_display):
    button = 0
    if (int(mouse[0]) <= 1.* w_display / 6) & (int(mouse[1]) <= 50):
        button = 1
    return button


def mainNeuro(OPB1_bandmean_alpha, OPB1_mean_array_uv, minDisplayY, maxDisplayY):

    newMean_alpha = np.average(OPB1_bandmean_alpha)  # mean of the 4 channels, not the best metric I guess
    OPB1_mean_array_uv.append(newMean_alpha)
    maxAlpha = np.amax(OPB1_mean_array_uv)
    minAlpha = np.min(OPB1_mean_array_uv)

    if newMean_alpha == maxAlpha:
        newPosy = minDisplayY

    elif newMean_alpha == minAlpha:
        newPosy = maxDisplayY

    else:
        a = (maxDisplayY-minDisplayY) * 1. / (minAlpha - maxAlpha)
        b = maxDisplayY - minAlpha * a
        newPosy = a * newMean_alpha + b
    # screen.blit(fond, (0, 0))

    return [newPosy, OPB1_mean_array_uv]

def trainingScore(posY, maxScore, minScore):

    newscore = 1.* (maxScore - minScore)/(minDisplayY - maxDisplayY) * ( posY - minDisplayY) + maxScore
    # print newscore

    if newscore < 0:
        newscore = 0
    elif newscore >= maxScore:
        newscore = 15
    return newscore

def progressionFunc(array, h, w, max_disp, min_disp) :
    if min(array) != max(array):
        a = (max_disp - min_disp) / (min(array) - max(array))
    else :
        a = 0
    b = min_disp - a* max(array)

    new_array = [a*array[i] + b for i in range(len(array))]
    return new_array

def displayNumber(nb, screen, position):
    # print nb
    nb = int(nb)
    if position == 'scoreV011':
        if nb >= 1000 :
            timerSec = pg.image.load(timer[int(str(nb)[3])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerThou = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerThou = pg.transform.scale(timerThou, (int(1. * w_display / 15), int(1. * h_display / 10)))
            screen.blit(timerSec, (25. * w_display / 30, 3. * h_display / 10))
            screen.blit(timerDiz, (23.* w_display / 30, 3. * h_display / 10))
            screen.blit(timerCen, (21.* w_display / 30, 3. * h_display / 10))
            screen.blit(timerThou, (19. * w_display / 30, 3. * h_display / 10))


        elif nb >= 100 and nb <= 999 :
            timerSec = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (25.* w_display / 30, 3.* h_display / 10))
            screen.blit(timerDiz, (23.* w_display / 30, 3.* h_display / 10))
            screen.blit(timerCen, (21. * w_display / 30, 3.* h_display / 10))

        elif nb >= 10 and nb <= 99:
            timerSec = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerSec = pg.transform.scale(timerSec,(int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (25.*w_display/30, 3.* h_display / 10))
            screen.blit(timerDiz, (23.*w_display / 30, 3.* h_display / 10))

        elif nb >= 0  and nb <= 9:
            timerSec = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (25.* w_display / 30, 3.* h_display / 10))

    elif position == 'down':

        if nb >= 1000:
            timerSec = pg.image.load(timer[int(str(nb)[3])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerThou = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerThou = pg.transform.scale(timerThou, (int(1. * w_display / 15), int(1. * h_display / 10)))
            screen.blit(timerSec, (14.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerDiz, (13.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerCen, (12.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerThou, (11. * w_display / 15, 9. * h_display / 10))


        elif nb >= 100:
            timerSec = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (14.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerDiz, (13.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerCen, (12.* w_display / 15, 9.* h_display / 10))

        elif nb >= 10 :
            timerSec = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerSec = pg.transform.scale(timerSec,(int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))

            screen.blit(timerSec, (14.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerDiz, (13.* w_display / 15, 9.* h_display / 10))

        elif nb < 10:
            timerSec = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (14.* w_display / 15, 9.* h_display / 10))

    elif position == 'down_left':

        if nb >= 1000 :
            timerSec = pg.image.load(timer[int(str(nb)[3])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerThou = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerThou = pg.transform.scale(timerThou, (int(1. * w_display / 15), int(1. * h_display / 10)))
            screen.blit(timerSec, (3.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerDiz, (2.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerCen, (1.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerThou, (0, 9. * h_display / 10))


        elif nb >= 100 and nb <= 999 :
            timerSec = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (2.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerDiz, (1.* w_display / 15, 9.* h_display / 10))
            screen.blit(timerCen, (0, 9.* h_display / 10))

        elif nb >= 10 and nb <= 99:
            timerSec = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerSec = pg.transform.scale(timerSec,(int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))

            screen.blit(timerSec, (1.*w_display/15, 9.* h_display / 10))
            screen.blit(timerDiz, (0, 9.* h_display / 10))

        elif nb >= 0  and nb <= 9:
            timerSec = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (0, 9.* h_display / 10))

    elif position == 'timeV011':

        if nb >= 1000 :
            timerSec = pg.image.load(timer[int(str(nb)[3])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerThou = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerThou = pg.transform.scale(timerThou, (int(1. * w_display / 15), int(1. * h_display / 10)))
            screen.blit(timerSec, (25.* w_display / 30, 7.* h_display / 10))
            screen.blit(timerDiz, (23.* w_display / 30, 7.* h_display / 10))
            screen.blit(timerCen, (22.* w_display / 30, 7.* h_display / 10))
            screen.blit(timerThou, (21. * w_display / 30, 7. * h_display / 10))

        elif nb >= 100 and nb <= 999 :
            timerSec = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerCen = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerCen = pg.transform.scale(timerCen, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (25.* w_display / 30, 7.* h_display / 10))
            screen.blit(timerDiz, (23.* w_display / 30, 7.* h_display / 10))
            screen.blit(timerCen, (21.* w_display / 30, 7.* h_display / 10))

        elif nb >= 10 and nb <= 99:
            timerSec = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerSec = pg.transform.scale(timerSec,(int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))

            screen.blit(timerSec, (25.* w_display / 30, 7.* h_display / 10))
            screen.blit(timerDiz, (23.* w_display / 30, 7.* h_display / 10))

        elif nb >= 0  and nb <= 9:
            timerSec = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (25.* w_display / 30, 7.* h_display / 10))

    elif position =='timeRSV011':

        if nb >= 100 and nb <= 999:
            timerSec = pg.image.load(timer[int(str(nb)[2])]).convert()
            timerSec = pg.transform.scale(timerSec,(int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))

            screen.blit(timerSec, (8.* w_display / 30, 1.* h_display / 20))
            screen.blit(timerDiz, (6.* w_display / 30, 1.* h_display / 20))
            screen.blit(timerCen, (4.* w_display / 30, 1.* h_display / 20))

        elif nb >= 10 and nb <= 99:
            timerSec = pg.image.load(timer[int(str(nb)[1])]).convert()
            timerSec = pg.transform.scale(timerSec,(int(1.* w_display / 15), int(1.*h_display / 10)))
            timerDiz = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerDiz = pg.transform.scale(timerDiz, (int(1.* w_display / 15), int(1.*h_display / 10)))

            screen.blit(timerSec, (6.* w_display / 30, 1.* h_display / 20))
            screen.blit(timerDiz, (4.* w_display / 30, 1.* h_display / 20))

        elif nb >= 0  and nb <= 9:
            timerSec = pg.image.load(timer[int(str(nb)[0])]).convert()
            timerSec = pg.transform.scale(timerSec, (int(1.* w_display / 15), int(1.*h_display / 10)))
            screen.blit(timerSec, (4.* w_display / 30, 1.* h_display / 20))

def get_ind_color(score, scoreMax, scoreMin, nbOfColors):

    ind = 1.*nbOfColors/scoreMax*score
    if score >= scoreMax :
        ind = 99
    if ind >= nbOfColors:
        ind = 99
    # print ind
    return int(ind)


def cleanData(cdata, data):

    cdata[0, :] = data[ind_channel_1]
    cdata[1, :] = data[ind_channel_2]
    cdata[2, :] = data[ind_channel_3]
    cdata[3, :] = data[ind_channel_4]
    return cdata

def getfreqmaxband(data, rangefreq, nb_freq):
    # this function finds the peak of the alpha band and returns the freq associated to the peak
    maxBand = 0
    for ind in range(nb_freq): # for each channel, first we need to get the average of each freq during the period
        if np.average(data[:, ind]) >= maxBand :
            ind_freqMax = ind
        else:
            pass
    # maxBand = np.average(band_alphaRS_ch1[ind_freqMax])

    if rangefreq == 'alpha':
        frequencies = [6, 7, 8, 9, 10, 11, 12, 13]
    elif rangefreq == 'delta':
        frequencies = [3, 4]

    return frequencies[ind_freqMax]


def saveData(path, session, kind, channel, data):
    # kind is either 'F', 'PB', or 'RS'
    outfile = path+kind+'_'+channel+'_'+'session'+str(session)+'_'+'.txt'
    np.savetxt(outfile, np.asarray(data), delimiter=',')

def saveAllChannelsData(path, session, kind, data1, data2, data3, data4):
    saveData(path, session, kind, 'ch1', data1)
    saveData(path, session, kind, 'ch2', data2)
    saveData(path, session, kind, 'ch3', data3)
    saveData(path, session, kind, 'ch4', data4)


def mad(a, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    """

    # Median along given axis, but *keeping* the reduced axis so that
    # result can still broadcast against a.
    med = np.median(a)
    mad = np.median(np.absolute(a - med))  # MAD along given axis

    return mad

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
