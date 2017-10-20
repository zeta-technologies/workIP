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

parser = argparse.ArgumentParser()
parser.add_argument("--test")
args = parser.parse_args()
if args.test :
    durationSessionInit = int(args.test)
    durationSession = durationSessionInit
    # restingStateDuration = int(math.floor(int(args.test)/100))
    restingStateDuration = int(math.floor(int(args.test)))
'''Data initialization '''

dataT = np.zeros((nb_channels, buffersize))
fdataT = np.zeros((nb_channels, buffersize))

dataRS1 = np.zeros((nb_channels, buffersize, restingStateDuration)) # need to store every chunk to reprocess the ratio
fdataRS1 = np.zeros((nb_channels, buffersize, restingStateDuration))

dataRS2 = np.zeros((nb_channels, buffersize, restingStateDuration)) # need to store every chunk to reprocess the ratio
fdataRS2 = np.zeros((nb_channels, buffersize, restingStateDuration))
'''background'''
screen = pg.display.set_mode((w_display, h_display), RESIZABLE)

'''training game'''
red = Color("red")
colors = list(red.range_to(Color("green"), 100))

'''Resting state'''
timerImage = pg.image.load(timer[0])
timerImage = pg.transform.scale(timerImage, (int(math.floor(0.068*w_display)), int(math.floor(0.156*h_display))))
restingDebutImage = pg.image.load(restingStateDebutPath).convert()
restingFinImage = pg.image.load(restingStateFinPath).convert()
restingStateDebutImage = pg.transform.scale(restingDebutImage, (w_display, h_display))
restingStateFinImage = pg.transform.scale(restingFinImage, (w_display, h_display))

# '''Tinnitus questionnaire '''
# questionsSerie1Image = pg.image.load(questionsSerie1)
# questionsSerie1Image = pg.transform.scale(questionsSerie1Image, (w_display, h_display))
'''End Session IMG'''
endSessionImg = pg.image.load(endSessionImg)
endSessionImg = pg.transform.scale(endSessionImg, (w_display, h_display))

'''MAIN LOOP'''
gameOn = 1
now = datetime.datetime.now()
randomId = binascii.b2a_hex(os.urandom(15)) #id is 30 characters long
if args.test:
    sessionName = str('TEST-'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute)+'-'+str(randomId))
else:
    sessionName = str(str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute)+'-'+str(randomId))

'''Loop that reads the text file sessionsNames, and check if the randomId has already been chosen in this folder'''
if os.path.isfile('sessionsNames.txt'):
    sessionsNames = open('sessionsNames.txt', 'r')
    sessionsNamesLines = sessionsNames.readlines()
    for i in range(len(sessionsNamesLines)):
        while randomId == sessionsNamesLines[i][-30:] :
            randomId = binascii.b2a_hex(os.urandom(15))
            sessionName = str(str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute)+'_'+str(randomId))
    sessionsNames.close()
    sessionsNames = open('sessionsNames.txt', 'a+')

else :
    sessionsNames = open('sessionsNames.txt', 'a+')

sessionsNames.write(sessionName+'\n')
sessionsNames.close()

'''check if the directory /data already exists'''
if not os.path.isdir('data'):
    os.mkdir('data')

'''create the paths and folders for the new session'''
if not os.path.isdir('data/'+sessionName):
    os.mkdir('data/'+sessionName)
    os.mkdir('data/'+sessionName+'/training-data')
    os.mkdir('data/'+sessionName+'/RS1-data')
    os.mkdir('data/'+sessionName+'/RS2-data')
    pathT = str('data/'+sessionName+'/training-data/')
    pathRS1 = str('data/'+sessionName+'/RS1-data/')
    pathRS2 = str('data/'+sessionName+'/RS2-data/')

print ' \n  Data will be saved here : ', pathRS2, pathRS1, pathT

'''MAIN LOOP'''

while gameOn:

    #LOAD screen Image
    home = pg.image.load(image_home).convert()
    home = pg.transform.scale(home, (w_display, h_display))
    screen.blit(home, (0,0))

    # launchTraining = 'Lancer la session'
    # launchTrainingSurf, launchTrainingRect = text_objects(launchTraining, buttonText)
    # launchTrainingRect.center = (2.*w_display/5, 3.3*h_display/4)
    background = pg.image.load(backgroundImage).convert()
    background = pg.transform.scale(background, (w_display, h_display))

    nextStep = 'Cliquez ICI pour passer a la suite'
    nextStepSurf, nextStepRect = text_objects(nextStep, buttonTextHuge)
    nextStepRect.center = (5.*w_display/10, 1.*h_display/4)

    progression = 'Accrochez-vous ! Voici votre progression jusque la'
    progressionSurf, progressionRect = text_objects(progression, buttonText)
    progressionRect.center = (4.*w_display/5, 3.3*h_display/4)

    # screen.blit(launchTrainingSurf, launchTrainingRect)
    pg.display.flip()

    # Home window loop
    while homeOn:
        # pg.time.Clock().tick(60)

        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONUP and not(sessionEnded):
                mouseHome = pg.mouse.get_pos()
                choice = whichButtonHomeV011(mouseHome, w_display, h_display)

                if choice == 2: #  Launches Resting State 1
                    homeOn = 0
                    restingState1 = 1

    if sessionEnded:
        progressionMetricSurf, progressionMetricRect = text_objects(progressionMetric, buttonText)
        progressionMetricRect.center = (1.*w_display/2, 1.*h_display/2)
        screen.blit(endSessionImg, (0,0))
        screen.blit(progressionMetricSurf, progressionMetricRect)
        pg.display.flip()
        saveAllChannelsData(pathRS2, sessionRS2, 'RS2-ended', saved_bufferRS2_ch1, saved_bufferRS2_ch2, saved_bufferRS2_ch3, saved_bufferRS2_ch4)

        pg.time.delay(2000)
        # gameOn = 0

    elif restingState1:

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

        secRS1 = 0
        bufferRS1 = []
        band_alphaRS1_ch1 = []
        band_alphaRS1_ch2 = []
        band_alphaRS1_ch3 = []
        band_alphaRS1_ch4 = []
        band_deltaRS1_ch1 = []
        band_deltaRS1_ch2 = []
        band_deltaRS1_ch3 = []
        band_deltaRS1_ch4 = []
        screen.blit(restingStateDebutImage, (0,0))
        displayNumber(0, screen, 'timeRSV011')
        pg.display.flip()
        queue.queue.clear()

    elif progression:
        print progression

    while restingState1:
        # pg.time.Clock().tick(60)

        for event in pg.event.get():
            if event.type == QUIT:
                saveAllChannelsData(pathRS1, sessionRS1, 'RS1-quit', saved_bufferRS1_ch1, saved_bufferRS1_ch2, saved_bufferRS1_ch3, saved_bufferRS1_ch4)
                saved_bufferRS1_ch1 = []
                saved_bufferRS1_ch2 = []
                saved_bufferRS1_ch3 = []
                saved_bufferRS1_ch4 = []
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    restingState1 = 0
            if event.type == MOUSEBUTTONUP:
                mouseReturn = pg.mouse.get_pos()
                if whichButtonReturn(mouseReturn, w_display, h_display):
                    homeOn = 1
                    training = 0
                    restingState1 = 0
                    bufferRS1 = []
                    queue.queue.clear()
                    saveAllChannelsData(pathRS1, sessionRS1, 'RS1-return', saved_bufferRS1_ch1, saved_bufferRS1_ch2, saved_bufferRS1_ch3, saved_bufferRS1_ch4)
                    saved_bufferRS1_ch1 = []
                    saved_bufferRS1_ch2 = []
                    saved_bufferRS1_ch3 = []
                    saved_bufferRS1_ch4 = []

        if secRS1 == restingStateDuration :
            # np.zeros(nb_freq_alpha)
            band_alphaRS1_ch1 = np.asarray(band_alphaRS1_ch1)
            band_alphaRS1_ch2 = np.asarray(band_alphaRS1_ch2)
            band_alphaRS1_ch3 = np.asarray(band_alphaRS1_ch3)
            band_alphaRS1_ch4 = np.asarray(band_alphaRS1_ch4)
            # print 'band_alphaRS_ch1', band_alphaRS_ch1
            # print 'band_alphaRS_ch1[:, 0]', np.average(band_alphaRS_ch1[:,0])
            freqMaxAlphaCh1 = getfreqmaxband(band_alphaRS1_ch1, 'alpha', nb_freq_alpha)
            freqMaxAlphaCh2 = getfreqmaxband(band_alphaRS1_ch2, 'alpha', nb_freq_alpha)
            freqMaxAlphaCh3 = getfreqmaxband(band_alphaRS1_ch3, 'alpha', nb_freq_alpha)
            freqMaxAlphaCh4 = getfreqmaxband(band_alphaRS1_ch4, 'alpha', nb_freq_alpha)

            freqMaxAlpha = int(np.average([freqMaxAlphaCh1, freqMaxAlphaCh2, freqMaxAlphaCh3, freqMaxAlphaCh4]))

            for chunk in range(restingStateDuration):
                ratios_ch1.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS1[0,:, chunk], freqMaxAlphaCh1-2, freqMaxAlphaCh1+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS1[0,:, chunk], 3, 4)[0])))
                ratios_ch2.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS1[1,:, chunk], freqMaxAlphaCh2-2, freqMaxAlphaCh2+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS1[1,:, chunk], 3, 4)[0])))
                ratios_ch3.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS1[2,:, chunk], freqMaxAlphaCh3-2, freqMaxAlphaCh3+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS1[2,:, chunk], 3, 4)[0])))
                ratios_ch4.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS1[3,:, chunk], freqMaxAlphaCh4-2, freqMaxAlphaCh4+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS1[3,:, chunk], 3, 4)[0])))

            median_ratio_ch1 = np.median(ratios_ch1)
            median_ratio_ch2 = np.median(ratios_ch2)
            median_ratio_ch3 = np.median(ratios_ch3)
            median_ratio_ch4 = np.median(ratios_ch4)

            mad_ch1 = mad(ratios_ch1)
            mad_ch2 = mad(ratios_ch2)
            mad_ch3 = mad(ratios_ch3)
            mad_ch4 = mad(ratios_ch4)

            madRatioAlphaOverDelta = np.average([mad_ch4, mad_ch3, mad_ch2, mad_ch1])
            medianratioAlphaoverDelta = np.average([median_ratio_ch1, median_ratio_ch2, median_ratio_ch3, median_ratio_ch4])
            minRatioAlphaOverDelta = medianratioAlphaoverDelta - 3 * madRatioAlphaOverDelta
            maxRatioAlphaOverDelta = medianratioAlphaoverDelta + 3 * madRatioAlphaOverDelta

            screen.blit(nextStepSurf, nextStepRect)
            pg.display.flip()

            saveAllChannelsData(pathRS1, sessionRS1, 'RS1-end', saved_bufferRS1_ch1, saved_bufferRS1_ch2, saved_bufferRS1_ch3, saved_bufferRS1_ch4)
            saved_bufferRS1_ch1 = []
            saved_bufferRS1_ch2 = []
            saved_bufferRS1_ch3 = []
            saved_bufferRS1_ch4 = []

            ''' END OF THE SESSION, WAITING FOR THE USER TO CLICK ON THE TEXT BUTTON '''
            for event in pg.event.get():
                if event.type == MOUSEBUTTONUP:
                    mouseRS1 = pg.mouse.get_pos()

                    RS1choice = whichButtonHomeV011(mouseRS1, w_display, h_display)
                    if RS1choice == 2:
                        homeOn = 0
                        training = 1
                        restingState1 = 0
                        bufferRS1 = []
                        bufferT = [] # init Training session
                        queue.queue.clear()
                        sessionT == 0

        elif secRS1 < restingStateDuration:
            try:
                while len(bufferRS1) < buffersize * nb_lines:

                    bufferRS1.append(queue.get_nowait())
                    # print bufferRS1

                if len(bufferRS1) == nb_lines*buffersize:

                    bufferRS1_array = np.asarray(bufferRS1)

                    dataRS1[0, :, secRS1] = bufferRS1_array[ind_channel_1]
                    dataRS1[1, :, secRS1] = bufferRS1_array[ind_channel_2]
                    dataRS1[2, :, secRS1] = bufferRS1_array[ind_channel_3]
                    dataRS1[3, :, secRS1] = bufferRS1_array[ind_channel_4]

                    saved_bufferRS1_ch1.extend(dataRS1[0, :, secRS1])
                    saved_bufferRS1_ch2.extend(dataRS1[1, :, secRS1])
                    saved_bufferRS1_ch3.extend(dataRS1[2, :, secRS1])
                    saved_bufferRS1_ch4.extend(dataRS1[3, :, secRS1])

                    fdataRS1[0, :, secRS1] = filter_data(dataRS1[0, :, secRS1], fs_hz)
                    fdataRS1[1, :, secRS1] = filter_data(dataRS1[1, :, secRS1], fs_hz)
                    fdataRS1[2, :, secRS1] = filter_data(dataRS1[2, :, secRS1], fs_hz)
                    fdataRS1[3, :, secRS1] = filter_data(dataRS1[3, :, secRS1], fs_hz)

                    band_alphaRS1_ch1.append(extract_freqband(200, fs_hz, fdataRS1[0,:, secRS1], 6, 13)[0])
                    band_alphaRS1_ch2.append(extract_freqband(200, fs_hz, fdataRS1[1,:, secRS1], 6, 13)[0])
                    band_alphaRS1_ch3.append(extract_freqband(200, fs_hz, fdataRS1[2,:, secRS1], 6, 13)[0])
                    band_alphaRS1_ch4.append(extract_freqband(200, fs_hz, fdataRS1[3,:, secRS1], 6, 13)[0])

                    nb_freq_alpha = extract_freqband(200, fs_hz, fdataRS1[0,:], 6, 13)[1]

                    band_deltaRS1_ch1.append(extract_freqband(200, fs_hz, fdataRS1[0,:, secRS1], 3, 4)[0])
                    band_deltaRS1_ch2.append(extract_freqband(200, fs_hz, fdataRS1[1,:, secRS1], 3, 4)[0])
                    band_deltaRS1_ch3.append(extract_freqband(200, fs_hz, fdataRS1[2,:, secRS1], 3, 4)[0])
                    band_deltaRS1_ch4.append(extract_freqband(200, fs_hz, fdataRS1[3,:, secRS1], 3, 4)[0])

                    nb_freq_delta = extract_freqband(200, fs_hz, fdataRS1[3,:], 3, 4)[1]

                    bufferRS1 = []
                    displayNumber(secRS1, screen, 'timeRSV011')
                    # checkImp() # TODO  check impedances function
                    pg.display.update()
                    secRS1 = secRS1 + 1
                    # TODO add a queue.queue.clear() and a time study to find out where does the program take so long

            except Empty:
                continue  # do stuff
            else:
                str(bufferRS1)
                # sys.stdout.write(char)


    while training:
        # delta2 = datetime.datetime.now() - now
        # print 'NOW training loop begins, line 337', delta2
        # pg.time.Clock().tick(60)
        for event in pg.event.get():
            if event.type == QUIT:
                saveAllChannelsData(pathT, sessionT, 'T-quit', saved_bufferT_ch1, saved_bufferT_ch2, saved_bufferT_ch3, saved_bufferT_ch4)
                bufferT = []

                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    saveAllChannelsData(pathT, sessionT, 'T-esc', saved_bufferT_ch1, saved_bufferT_ch2, saved_bufferT_ch3, saved_bufferT_ch4)
                    bufferT = []

                    training = 0
            elif event.type == MOUSEBUTTONUP:
                mouseReturn = pg.mouse.get_pos()
                if whichButtonReturn(mouseReturn, w_display, h_display):
                    homeOn = 1
                    training = 0
                    restingState1 = 0
                    saveAllChannelsData(pathT, sessionT, 'T-return', saved_bufferT_ch1, saved_bufferT_ch2, saved_bufferT_ch3, saved_bufferT_ch4)
                    bufferT = []


        if durationSession > 0:
            # delta3 = datetime.datetime.now() - now
            # print 'NOW duration Session loop begins line 364', delta3
            try:

                while len(bufferT) < buffersize * nb_channels:

                    if len(bufferT) % int(math.floor(1.*buffersize/5)) == 0:
                        screen.blit(background, (0,0))
                        indColor = get_ind_color(trainingScore(newPosy, maxScore, minScore), 10,0, len(colors))
                        if indColor > 100 :
                            indColor = 100
                        color = (colors[indColor].rgb[0]*255,colors[indColor].rgb[1]*255,colors[indColor].rgb[2]*255)
                        if (veryOldPosy + 1.*(oldPosy - veryOldPosy)/steps) <= minDisplayY + 10 :
                            positionY = minDisplayY + 10
                        elif (veryOldPosy + 1.*(oldPosy - veryOldPosy)/steps) >= maxDisplayY :
                            positionY = maxDisplayY
                        else :
                            positionY = veryOldPosy + 1.*(oldPosy - veryOldPosy)/steps

                        pg.draw.rect(screen, color , (2. * w_display / 12, positionY, 100, 10 ))
                        displayNumber(math.floor(scoreT), screen, 'scoreV011')
                        displayNumber(durationSession, screen, 'timeV011')
                        veryOldPosy += 1.*(oldPosy - veryOldPosy)/steps
                        pg.display.flip()

                    bufferT.append(queue.get_nowait())

                if len(bufferT) == 800 :
                    queue.queue.clear()
                    bufferT_array = np.asarray(bufferT)

                    dataT[0, :] = bufferT_array[ind_channel_1]
                    dataT[1, :] = bufferT_array[ind_channel_2]
                    dataT[2, :] = bufferT_array[ind_channel_3]
                    dataT[3, :] = bufferT_array[ind_channel_4]

                    saved_bufferT_ch1.extend(dataT[0, :]) # extend instead of append, it does save correctly after that, with extend it does not
                    saved_bufferT_ch2.extend(dataT[1, :])
                    saved_bufferT_ch3.extend(dataT[2, :])
                    saved_bufferT_ch4.extend(dataT[3, :])

                    fdataT[0, :] = filter_data(dataT[0, :], fs_hz)
                    fdataT[1, :] = filter_data(dataT[1, :], fs_hz)
                    fdataT[2, :] = filter_data(dataT[2, :], fs_hz)
                    fdataT[3, :] = filter_data(dataT[3, :], fs_hz)

                    bandmean_alphaT = np.zeros(nb_channels)
                    bandmax_alphaT = np.zeros(nb_channels)
                    bandmin_alphaT = np.zeros(nb_channels)

                    bandmean_deltaT = np.zeros(nb_channels)
                    bandmax_deltaT = np.zeros(nb_channels)
                    bandmin_deltaT = np.zeros(nb_channels)
                    ratioT = np.zeros(nb_channels)

                    for channel in range(nb_channels):
                        bandmean_alphaT[channel] = extract_freqbandmean(200, fs_hz, fdataT[channel,:], freqMaxAlpha-2, freqMaxAlpha+2)
                        bandmean_deltaT[channel] = extract_freqbandmean(200, fs_hz, fdataT[channel,:], 3, 4)
                        ratioT[channel] = 1.* bandmean_alphaT[channel] / bandmean_deltaT[channel]

                    # maximiser alpha/delta
                    ''' Get the mean, min and max of the last reslt of all channels'''
                    newMean_alphaT = np.average(bandmean_alphaT)
                    # maxalphaT = np.amax(mean_array_uvT)
                    # minalphaT = np.min(mean_array_uvT)

                    medRatioT = np.median(ratioT)
                    mean_array_uvT.append(medRatioT)

                    if medRatioT == maxRatioAlphaOverDelta:
                        newPosy = minDisplayY

                    elif medRatioT == minRatioAlphaOverDelta:
                        newPosy = maxDisplayY

                    else:
                        a = (maxDisplayY - minDisplayY) * 1. / (minRatioAlphaOverDelta - maxRatioAlphaOverDelta)
                        b = maxDisplayY - minRatioAlphaOverDelta * a
                        newPosy = a * medRatioT + b

                    scoreT = scoreT + trainingScore(newPosy, maxScore, minScore)

                    durationSession = durationSession -  1

            except Empty:
                continue  # do stuff
            else:
                str(1)
                # sys.stdout.write(char)
            veryOldPosy = oldPosy
            oldPosy = newPosy
            bufferT = []

        elif durationSession == 0 :
            # print saved_bufferT_ch1
            if sessionT == 0:
                saveAllChannelsData(pathT, sessionT, 'T-end', saved_bufferT_ch1, saved_bufferT_ch2, saved_bufferT_ch3, saved_bufferT_ch4)
                sessionT += 1

            screen.blit(restingStateFinImage, (0,0))
            screen.blit(nextStepSurf, nextStepRect)
            pg.display.flip()

            for event in pg.event.get():
                if event.type == MOUSEBUTTONUP:
                    mouseRS2 = pg.mouse.get_pos()
                    choiceRS2 = whichButtonHomeV011(mouseRS2, w_display, h_display)
                    if choiceRS2 == 2:
                        queue.queue.clear()
                        training = 0
                        secRS2 = 0
                        sessionRS2 = 0 #when it's 1, it wont save data anymore
                        restingState2 = 1
                        bufferRS2 = []
                        band_alphaRS2_ch1 = []
                        band_alphaRS2_ch2 = []
                        band_alphaRS2_ch3 = []
                        band_alphaRS2_ch4 = []

                        band_deltaRS2_ch1 = []
                        band_deltaRS2_ch2 = []
                        band_deltaRS2_ch3 = []
                        band_deltaRS2_ch4 = []
                        screen.blit(restingStateFinImage, (0,0))
                        displayNumber(0, screen, 'timeRSV011')
                        pg.display.flip()


    while restingState2:
        # pg.time.Clock().tick(60)

        for event in pg.event.get():
            if event.type == QUIT:
                saveAllChannelsData(pathRS2, sessionRS2, 'RS2-quit', saved_bufferRS2_ch1, saved_bufferRS2_ch2, saved_bufferRS2_ch3, saved_bufferRS2_ch4)
                saved_bufferRS2_ch1 = []
                saved_bufferRS2_ch2 = []
                saved_bufferRS2_ch3 = []
                saved_bufferRS2_ch4 = []
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    restingState2 = 0
            elif event.type == MOUSEBUTTONUP:
                mouseReturn = pg.mouse.get_pos()
                if whichButtonReturn(mouseReturn, w_display, h_display):
                    homeOn = 1
                    training = 0
                    restingState2 = 0
                    questionnaire = 0
                    bufferRS2 = []
                    queue.queue.clear()
                    saveAllChannelsData(pathRS2, sessionRS2, 'RS2-return', saved_bufferRS2_ch1, saved_bufferRS2_ch2, saved_bufferRS2_ch3, saved_bufferRS2_ch4)
                    saved_bufferRS2_ch1 = []
                    saved_bufferRS2_ch2 = []
                    saved_bufferRS2_ch3 = []
                    saved_bufferRS2_ch4 = []

        if secRS2 == restingStateDuration :

            band_alphaRS2_ch1 = np.asarray(band_alphaRS2_ch1)
            band_alphaRS2_ch2 = np.asarray(band_alphaRS2_ch2)
            band_alphaRS2_ch3 = np.asarray(band_alphaRS2_ch3)
            band_alphaRS2_ch4 = np.asarray(band_alphaRS2_ch4)
            freqMaxAlphaCh1 = getfreqmaxband(band_alphaRS2_ch1, 'alpha', nb_freq_alpha)
            freqMaxAlphaCh2 = getfreqmaxband(band_alphaRS2_ch2, 'alpha', nb_freq_alpha)
            freqMaxAlphaCh3 = getfreqmaxband(band_alphaRS2_ch3, 'alpha', nb_freq_alpha)
            freqMaxAlphaCh4 = getfreqmaxband(band_alphaRS2_ch4, 'alpha', nb_freq_alpha)

            freqMaxAlpha = int(np.average([freqMaxAlphaCh1, freqMaxAlphaCh2, freqMaxAlphaCh3, freqMaxAlphaCh4]))

            for chunk in range(restingStateDuration):
                ratios_ch1.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS2[0,:, chunk], freqMaxAlphaCh1-2, freqMaxAlphaCh1+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS2[0,:, chunk], 3, 4)[0])))
                ratios_ch2.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS2[1,:, chunk], freqMaxAlphaCh2-2, freqMaxAlphaCh2+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS2[1,:, chunk], 3, 4)[0])))
                ratios_ch3.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS2[2,:, chunk], freqMaxAlphaCh3-2, freqMaxAlphaCh3+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS2[2,:, chunk], 3, 4)[0])))
                ratios_ch4.append(1.*np.median((extract_freqband(200, fs_hz, fdataRS2[3,:, chunk], freqMaxAlphaCh4-2, freqMaxAlphaCh4+2)[0]))/np.median((extract_freqband(200, fs_hz, fdataRS2[3,:, chunk], 3, 4)[0])))

            median_ratio_ch1 = np.median(ratios_ch1)
            median_ratio_ch2 = np.median(ratios_ch2)
            median_ratio_ch3 = np.median(ratios_ch3)
            median_ratio_ch4 = np.median(ratios_ch4)

            mad_ch1 = mad(ratios_ch1)
            mad_ch2 = mad(ratios_ch2)
            mad_ch3 = mad(ratios_ch3)
            mad_ch4 = mad(ratios_ch4)

            madRatioAlphaOverDeltaEnd = np.average([mad_ch4, mad_ch3, mad_ch2, mad_ch1])
            medianratioAlphaoverDeltaEnd = np.average([median_ratio_ch1, median_ratio_ch2, median_ratio_ch3, median_ratio_ch4])
            minRatioAlphaOverDelta = medianratioAlphaoverDelta - 3 * madRatioAlphaOverDelta
            maxRatioAlphaOverDelta = medianratioAlphaoverDelta + 3 * madRatioAlphaOverDelta

            metric = (medianratioAlphaoverDeltaEnd - medianratioAlphaoverDelta)
            if metric < 0 :
                metric = 0

            displayedMetric = metric * progressionCoeff
            if sessionRS2 == 0 :
                sessionRS2 += 1
                dailyProgressionFile = open('dailyProgression.txt', 'a+')
                dailyProgressionFile.write(str(metric)+'***_Progression Metric_'+sessionName+'\n') #*** is the marker i use to stop reading the line which means "it's not the metric anymore"
                dailyProgressionFile.close()
                dailyProgressionMetrics = [line.rstrip('\n') for line in open('dailyProgression.txt')]
                dailyProgressionMetrics = [float(i.split('***')[0]) for i in dailyProgressionMetrics] # we take the 0th element beacause '***' can be in the middle of number : for instance '569348278***706958' returns ['569348278','706958']
                # dailyProgressionMetrics = [ i * progressionCoeff for i in dailyProgressionMetrics]

                screen.blit(endSessionImg, (0,0))
                # print dailyProgressionMetrics

                dailyProgressionMetricsDisplayedArray = progressionFunc(dailyProgressionMetrics, h_display, w_display, h_display * 3/4, h_display*1/4)
                saveAllChannelsData(pathRS2, sessionRS2, 'RS2-end', saved_bufferRS2_ch1, saved_bufferRS2_ch2, saved_bufferRS2_ch3, saved_bufferRS2_ch4)

                for s in range(len(dailyProgressionMetrics)-1): # we dont take the last one, it's today's and we want to print it bigger
                # displayedMetric = dailyProgressionMetrics[s]
                # print ((s+1)*w_display/(len(dailyProgressionMetrics)+1), dailyProgressionMetricsDisplayedArray[s]), ((s+2)*w_display/(len(dailyProgressionMetrics)+1), dailyProgressionMetricsDisplayedArray[s+1])
                    pg.draw.line(screen, 0x4F89D8, ((s+1)*w_display/(len(dailyProgressionMetrics)+1), dailyProgressionMetricsDisplayedArray[s]), ((s+2)*w_display/(len(dailyProgressionMetrics)+1), dailyProgressionMetricsDisplayedArray[s+1]), 1 )
                    pg.display.flip()

            time.sleep(2)
            if print1 < 4:
                print print1
                print1 +=1
            elif print1 == 4 :
                pg.quit()

                    # if  metric >= 0 :
                    #     progressionText = 'JOUR ' + str(s) + ' :' +  str(displayedMetric)[0]+ '.' + str(displayedMetric)[2:5]
                    #     print progressionText
                    # elif metric < 0 :
                    #     progressionText = 'JOUR ' + str(s) + ' :' + str(displayedMetric)[1] + '.' + str(displayedMetric)[3:6]
                    #     print progressionText

                    # progressionMetricSurf, progressionMetricRect = text_objects(progressionText, progressionTextFont)
                    # progressionMetricRect.center = (s*w_display/(len(dailyProgressionMetrics)-1), 1.*h_display/2)
                    # screen.blit(progressionMetricSurf, progressionMetricRect)



        elif secRS2 < restingStateDuration:
            try:
                while len(bufferRS2) < buffersize * nb_channels:
                    bufferRS2.append(queue.get_nowait())

                if len(bufferRS2) == 800:
                    bufferRS2_array = np.asarray(bufferRS2)

                    dataRS2[0, :, secRS2] = bufferRS2_array[ind_channel_1]
                    dataRS2[1, :, secRS2] = bufferRS2_array[ind_channel_2]
                    dataRS2[2, :, secRS2] = bufferRS2_array[ind_channel_3]
                    dataRS2[3, :, secRS2] = bufferRS2_array[ind_channel_4]

                    saved_bufferRS2_ch1.extend(dataRS2[0, :, secRS2])
                    saved_bufferRS2_ch2.extend(dataRS2[1, :, secRS2])
                    saved_bufferRS2_ch3.extend(dataRS2[2, :, secRS2])
                    saved_bufferRS2_ch4.extend(dataRS2[3, :, secRS2])

                    fdataRS2[0, :, secRS2] = filter_data(dataRS2[0, :, secRS2], fs_hz)
                    fdataRS2[1, :, secRS2] = filter_data(dataRS2[1, :, secRS2], fs_hz)
                    fdataRS2[2, :, secRS2] = filter_data(dataRS2[2, :, secRS2], fs_hz)
                    fdataRS2[3, :, secRS2] = filter_data(dataRS2[3, :, secRS2], fs_hz)

                    band_alphaRS2_ch1.append(extract_freqband(200, fs_hz, fdataRS2[0,:, secRS2], 6, 13)[0])
                    band_alphaRS2_ch2.append(extract_freqband(200, fs_hz, fdataRS2[1,:, secRS2], 6, 13)[0])
                    band_alphaRS2_ch3.append(extract_freqband(200, fs_hz, fdataRS2[2,:, secRS2], 6, 13)[0])
                    band_alphaRS2_ch4.append(extract_freqband(200, fs_hz, fdataRS2[3,:, secRS2], 6, 13)[0])

                    nb_freq_alpha = extract_freqband(200, fs_hz, fdataRS2[0,:], 6, 13)[1]

                    band_deltaRS2_ch1.append(extract_freqband(200, fs_hz, fdataRS2[0,:, secRS2], 3, 4)[0])
                    band_deltaRS2_ch2.append(extract_freqband(200, fs_hz, fdataRS2[1,:, secRS2], 3, 4)[0])
                    band_deltaRS2_ch3.append(extract_freqband(200, fs_hz, fdataRS2[2,:, secRS2], 3, 4)[0])
                    band_deltaRS2_ch4.append(extract_freqband(200, fs_hz, fdataRS2[3,:, secRS2], 3, 4)[0])

                    nb_freq_delta = extract_freqband(200, fs_hz, fdataRS2[3,:], 3, 4)[1]

                    bufferRS2 = []
                    displayNumber(secRS2, screen, 'timeRSV011')
                    pg.display.update()
                    secRS2 = secRS2 + 1

            except Empty:
                continue  # do stuff
            else:
                str(bufferRS2)
                # sys.stdout.write(char)
