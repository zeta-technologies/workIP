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

now = time.time()
bufferT = []
process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE)
queue = Queue()
thread = Thread(target=enqueue_output, args=(process.stdout, queue))
thread.daemon = True
thread.start()
print 'Start'

while time.time() - now < 5:
    try:
        while len(bufferT) < 5*900:

            bufferT.append(queue.get_nowait())
    except Empty:
        continue  # do stuff
    else :
        str('1')

outfile ='data.txt'
np.savetxt(outfile, np.asarray(bufferT), delimiter=',')
