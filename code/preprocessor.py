import librosa
import soundfile as sf
import os

"""[preprocessing method, used to convert signal to mono + trim silence at begin and end of signal + normalize volume of each sample]
    use args to modify method behaviour:
        
        dir [String]        -> path to audio files
        normalize [Boolean] -> if TRUE method will lift volume of sample up until loudest peak of sample is at -0.1 decibel (this will not affect audio dynamics)
        trim [Boolean]      -> if TRUE method will trim silence before audio starts and after it has ended (silence is everything below <gate>)
        gate [unsigned int] -> threshold (in decibels) below reference to consider as silence, default is set to 20 dB

    src: http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.effects.trim.html   

    ! make sure to use <r> infront of PathString or use double back-slash, otherwise this will result in invalid path error !
    ! Be aware, that normalization needs to take place after additive augmention methods as e.g. adding noise to avoid risk of causing clipping !
   
    examplary call: pp("C:\\folder\\where\\files\\are\\stored", override=True, 40) or  pp(r"C:\folder\where\files\are\stored")"""
def pp(dirIN, dirOUT, normalize=False, trim=True, gate=20):

    #determine how to append file name or if method should run at all
    if(not trim and not normalize):
        print("ERROR: Process was stopped due to no preprocessing was enabled via arguemnts.")
        return  
    if(trim and not normalize):
        indicator = '_t.wav'
    if(not trim and normalize):
        indicator = '_n.wav'
    if(trim and normalize):
        indicator = '_t_n.wav'

    #syntax adjustments to provide a working path variable
    directory = r'{}'.format(dirIN)
 
    for soundfile in os.scandir(directory):
        temp_name  = os.path.basename(soundfile).replace('.wav', indicator)  
        exportPath = os.path.join(r'{}'.format(dirOUT), temp_name)
        #loading audiosignal and converting it to mono in case it was provided in stereo
        #f = soundfile currently opened, sr = samplerate -> explicitely set to None to avoid downsampling
        result, sr = librosa.load(soundfile, mono=True, sr=None)

        if(trim):
            result, _ = librosa.effects.trim(result, top_db=gate)

        if(normalize):
            result = librosa.util.normalize(result)   

        sf.write(exportPath, result, sr, format='wav')