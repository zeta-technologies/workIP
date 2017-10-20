#!/usr/bin/python
import pygame as pg
from pygame.locals import *
from constantes import *
import os
import sys
import signal
from subprocess import Popen, PIPE
from subprocess import call
from threading  import Thread
from sys import platform
from tempfile import TemporaryFile
from requests import *
import datetime
from functions import *
import os, binascii
from colour import Color
import argparse

print 'Start'
duration = 10
sampleRate = 200
nbLines = 5
bufferT = []
if platform == 'darwin' and sessionRS1 == 0: # mac
    process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE) # for MAC
    '''launch node process'''
    queue = Queue()
    thread = Thread(target=enqueue_output, args=(process.stdout, queue))
    thread.daemon = True
    thread.start()
elif platform == 'linux' or platform == 'linux2' and sessionRS1 == 0: #linux
    process = Popen(['sudo', '/usr/bin/node', 'openBCIDataStream.js'], stdout=PIPE, preexec_fn=os.setsid) # for LINUX
    '''launch node process'''
    queue = Queue()
    thread = Thread(target=enqueue_output, args=(process.stdout, queue))
    thread.daemon = True
    thread.start()
now = time.time()


while time.time() - now <= duration:
    try:
        while len(bufferT) < duration*nbLines*sampleRate:

            bufferT.append(queue.get_nowait())
    except Empty:
        continue  # do stuff
    else :
        str('1')

outfile ='data.txt'
np.savetxt(outfile, np.asarray(bufferT), delimiter=',')
