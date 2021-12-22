import soundfile as sf
import os
import soundfile as sf
from random import randint
from pydub import AudioSegment

"""[METHODE I - Gap Filling]
    The duration of the longest sample will be established as standard for the remaining corpus.
    Differences in duration between longest and every other sample will be handled by repeatedly appending shorter samples with themselves.

        dir [String]        -> path to audio files

    ! make sure to use <r> infront of PathString or use double back-slash, otherwise this will result in invalid path error !

    examplary call: fill(r'D:\folder') or fill('D:\\foo')"""
def fill(dirIN, dirOUT):
    #syntax adjustments to provide a working path variable
    directory = r'{}'.format(dirIN)
    #Find duration of longest sample in dataset
    longest_duration = float(0.0)
    for soundfile in os.scandir(directory):
        temp_duration = getDur(soundfile)
        if(longest_duration<temp_duration):
            longest_duration = temp_duration
            
    #append shorter samples with repetitions of itself until it matches longest duration 
    for soundfile in os.scandir(directory):               
        #temporarily storing file name for purpose of referencing it on export
        temp_name = os.path.basename(soundfile.name.replace('.wav', '_m1.wav'))
        #creating export path
        exportPath = os.path.join(r'{}'.format(dirOUT), temp_name)
        current_duration = getDur(soundfile)
        #opening current file as audio object
        current_file = AudioSegment.from_file(soundfile)
        #ignore files of same lenght as longest sample
        if(current_duration == longest_duration):
            current_file.export(exportPath, format='wav')
            continue    
        #counter variable for number of repetitions
        i=2
        while(current_duration < longest_duration):
            # repeat shorter sample i times, appending the file with each iteration with a copy of itself 
            repeated_file = current_file * i
            current_duration = repeated_file.duration_seconds
            i=i+1
        # getting rid of excess length by slicing repeated audio until longest duration + 1ms -> ! beware that it is unlikely, yet possible to encounter a nullpointer
        adjust_dur = repeated_file[:int(longest_duration*1000)]
        #dmwls = duration matched with longest sample
        adjust_dur.export(exportPath, format='wav')

"""[Helper Method to get the duration of the sample passed as argument]
    
        soundfile [audio object] -> the sample one wants to know the duration of (in sec.)
         
    src: https://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration"""
def getDur(soundfile):
    f = sf.SoundFile(soundfile)
    result = len(f) / f.samplerate
    f.close()
    return result
