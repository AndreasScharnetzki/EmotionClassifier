import librosa
import numpy as np
import soundfile as sf
import os
from random import randint

"""[method for augmenting audio samples, use for increasing corpus size]
    Uses pitch shifting to augment sample, will not override source file.

        dir [String]            -> path to audio files
        range [unsigned int]    -> sets the range of semitone steps within the samples pitch will be shifted (postive/negative), 
                                   number of steps will be choosen randomly for each sample, default is set to 3 semitone steps
                  
    src: https://www.kaggle.com/huseinzol05/sound-augmentation-librosa

    ! make sure to use <r> infront of PathString or use double back-slash, otherwise this will result in invalid path error !
    ! start script from within the folder where the files you want to augment are located, due to this script will store the changed files where it was started !

    examplary call: aug_p(r'D:\folder', 5) or aug_p('D:\\foo', 2)"""
def aug_p(dirIN, dirOUT, range=3):
    #syntax adjustments to provide a working path variable
    directory = r'{}'.format(dirIN)
    #determines into how many steps an octave should be divided (12 is used in the tonal system of western music)
    bins_per_octave = 12   
    for soundfile in os.scandir(directory): 
        #temporarily storing file name for purpose of referencing it on export
        temp_name = os.path.basename(soundfile.name).replace('.wav', '_pitched.wav')
        exportPath = os.path.join(r'{}'.format(dirOUT), temp_name)
        #loading audiosignal and converting it to mono in case it was provided in stereo
        #f = soundfile currently opened, sr = samplerate -> explicitely set to None to avoid downsampling
        f, sr = librosa.load(soundfile, mono=True, sr=None)
        #choosing amount of semitone steps by random to be used for pitch shifting
        sts = getRandSemitoneStep(range)
        #apply pitch shift in given range
        p = librosa.effects.pitch_shift(f.astype('float64'), sr, n_steps=sts, bins_per_octave=bins_per_octave)
        #store augmented file to drive under new name
        sf.write(exportPath, p, sr, format='wav')

#Helper method to return a random number for given number of semitones, excluding 0 due to this wouldnt cause pitch shifting but result in duplicates
#src: https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers
def getRandSemitoneStep(range):
  exclude=[0]
  randInt = randint(-range, range)
  return getRandSemitoneStep(range) if randInt in exclude else randInt 

"""[method for augmenting audio samples, use for increasing corpus size]
    Uses gaussian white noise to augment sample, will not override source file
    
        dir [String]            -> path to audio files
        amount [float]          -> amount of noise one wants to add, recommended is a value between [0.005, 0.5], the greater the number the more noise will be added
                                   default is set to 0.025
        ! beware that adding to the original signal will also effect the signals amplitude, hence it would be better to use this before normalization !
        ! make sure to use <r> infront of PathString or use double back-slash, otherwise this will result in invalid path error !

    examplary call: aug_n(r'D:\folder', 0.05)"""
def aug_n(dirIN, dirOUT, amount=0.025):
    #syntax adjustments to provide a working path variable
    directory = r'{}'.format(dirIN)
    for soundfile in os.scandir(directory):
        #temporarily storing file name for purpose of referencing it on export
        temp_name = os.path.basename(soundfile.name).replace('.wav', '_noise.wav')
        exportPath = os.path.join(r'{}'.format(dirOUT), temp_name)
        #loading audiosignal and converting it to mono in case it was provided in stereo
        #f = soundfile currently opened, sr = samplerate -> explicitely set to None to avoid downsampling
        f, sr = librosa.load(soundfile, mono=True, sr=None)
        #calculates the amount of gaussian noise to be addded to the source, takes max peak in account for reasonable Signal-Noise-Ratio
        noise_amp = amount*np.random.uniform()*np.amax(f)
        #adds a small, random ammount of gaussian white noise to source
        n = f.astype('float64') + noise_amp * np.random.normal(size=f.shape[0])
        #store augmented file to drive under new name
        sf.write(exportPath, n, sr, format='wav')
