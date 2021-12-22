import soundfile as sf
import os
from pydub import AudioSegment

"""[METHODE II Variation - Audio Cropping, Sliding Window]
    The duration of the shortest sample will be established as standard for the remaining corpus.
    In a "sliding-window" approach longer samples will be cut into n snippets matching the length of the shortest sample

        dir [String]        -> path to audio files
        verbose [Boolean]   -> if set to TRUE, this method will also create an extra sample from the very end of a longer sample -> beware that this is will 
                               increase computational time, due to its necessary to reload the sample because 
                               AudioSegment objects do not allow for negative indexing (see: https://github.com/jiaaro/pydub/blob/master/pydub/audio_segment.py,
                               accessed: 15.07.2021) when using the slicing operator.                  
                               default is set to FALSE
        stride [int]        -> duration in ms of how much the time window (= shortest duration) should be moved along the longer samples before cropping, 
                               default is set to a quarter a second (~ one word)

    ! make sure to use <r> infront of PathString or use double back-slash, otherwise this will result in invalid path error !
    ! If <verbose> argument is set to false, a small amount of data will be lost during the process 
      (in case the remaining bit is shorter in duration than the window size), but the amount of extra samples created by this method should compensate for that :) !

    examplary call: crop_sw(r'D:\folder', True, 300) or crop_sw('D:\\foo')"""
def crop_sw(dirIN, dirOUT, verbose=False, stride = 250):
    #syntax adjustments to provide a working path variable
    directory = r'{}'.format(dirIN)
    #initialize with 10 min so it will get undercut on initial cycle
    shortest_duration = float(600.0)
    #Find duration of shortest sample in dataset
    for soundfile in os.scandir(directory):
        temp_duration = getDur(soundfile)
        if(temp_duration<shortest_duration):
            shortest_duration = temp_duration
    #this on the same time represents the length of the window that will be slid across longer samples            
    window_size = int(shortest_duration*1000)

    #take random snipped from longer samples matching duration of shortest sample
    for soundfile in os.scandir(directory):               
        #temporarily storing file name for purpose of referencing it on export
        temp_name = os.path.basename(soundfile.name)
        #creating export path
        exportPath = os.path.join(r'{}'.format(dirOUT), temp_name)     
        #opening current file as audio object
        current_file = AudioSegment.from_file(soundfile)
        #calculating the remaining duration of longer sample
        remaining_duration = len(current_file)
        #counter variable for naming purpose        
        i=1
        #splitting longer samples into n items of duration of shortest sample
        while(remaining_duration>window_size):
            current_file[:window_size].export(exportPath.replace('.wav', '_snipped{number}.wav').format(number=i), format='wav')
            if(remaining_duration-stride < window_size):
                break
            else:
                current_file = current_file[stride:]
                remaining_duration = len(current_file)
            i=i+1

        if(verbose):
            #reloading file 
            current_file = AudioSegment.from_file(soundfile)
            #calculating the remaining duration of longer sample
            starting_point = len(current_file)-window_size
            #exporting an additional sample measured from the end of the longer sample
            current_file[starting_point:].export(exportPath.replace('.wav', '_extraSnipped.wav'), format='wav')

"""[Helper Method to get the duration of the sample passed as argument]
    
        soundfile [audio object] -> the sample one wants to know the duration of (in sec.)
         
    src: https://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration"""
def getDur(soundfile):
    f = sf.SoundFile(soundfile)
    result = len(f) / f.samplerate
    f.close()
    return result