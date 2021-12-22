import librosa
from librosa.core.audio import get_duration
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

"""[This method will transform audio signals into coresponding log-mel-spectrograms and crop and scale them to appropriate size to function as input for the VGG16]

        dirIn [String]          -> path to audio files
        dirIn [String]          -> path where to export spectograms to
        mel_bins [unsigned int] -> number of how many evenly spaced frequency bins the input signal should be divided into ('evenly' according to human perception)
                                   visually this can be interpreted as the resolution of the spectrogram on the y-axis
                                   default is set to 128 mel bins (high resolution).
        n_fft [unsigned int]    -> "number of samples per frame in STFT-like analyses"*, note that this is NOT per se equal to the size of the 
                                    underlying STFT window (although they mostly coincide)                                            
                                    default is set to 2048 (high resolution of the frequency domain), 
                                    this is due to (n_fft/2)+1 = number of frequency bins into which the freqency spectrum will be devided
                                        example: 
                                            (4096/2)+1 = 2049 bins for the frequency range of (0, sampling rate/2) <- Nyquist freq.
        hop_size [unisgned int] -> "the number of samples between frames"*, this determines, how many samples the FT-window should be moved 
                                    before another DFT will be applied, use this parameter to tune overlapping default is set to 512 
                                    (high resolution, but computationally expensive -> increase for speedup)
                                    this is due to the number of frames taken into consideration is defined by:
                                        (total number of samples in signal - frame size) / (hop size + 1)

src: https://github.com/musikalkemist/AudioSignalProcessingForML/blob/master/18%20-%20Extracting%20Mel%20Spectrograms%20with%20Python/Extracting%20Mel%20Spectrograms.ipynb
     https://stackoverflow.com/questions/37963042/python-librosa-what-is-the-default-frame-size-used-to-compute-the-mfcc-feature
     https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
     https://librosa.org/librosa_gallery/auto_examples/plot_presets.html#sphx-glr-auto-examples-plot-presets-py *

    ! beware that adding to the original signal will also effect the signals amplitude, hence it would be better to use this BEFORE normalization !
    ! make sure to use <r> infront of PathString or use double back-slash, otherwise this will result in invalid path error !
    ! Note that a greater window_size (more samples per frame taken into consideration when applying FFT) comes with better frequency resolution (this is
      due to now there are more frequency bins), but lower time resolution because more samples will be looked at with what is considered a frame in time 
      -> imagine this as a sample of a group of samples !
"""
def a2s(dirIN, dirOUT, mel_bins=128, window_size=2048, hop_length=512):
    #syntax adjustments to provide a working path variable
    directory = r'{}'.format(dirIN)    

    for soundfile in os.scandir(directory):
        #by using this figure size the resulting spectrogram will have a size of 224 x 224 pixels without having to rescale/crop it, regardless of the duration
        plt.figure(figsize=(2.9, 2.91))
        #by doing so cropping is not necessary
        plt.axis('off')
        #temporarily storing file name for purpose of referencing it on export
        temp_name = os.path.basename(soundfile.name).replace('.wav','')
        #creating export path
        exportPath = os.path.join(r'{}'.format(dirOUT), temp_name)
        #loading audiosignal and converting it to mono in case it was provided in stereo
        #f = soundfile currently opened, sr = samplerate -> explicitely set to None to avoid downsampling
        f, sr = librosa.load(soundfile, mono=True, sr=None)
        if(librosa.get_duration(f, sr) == 0.0):
            continue
        # create spectrogram for given audio, using Hann Window function (bell shaped) to avoid discontinuation on overlapping segments
        # restricting the area of interesst to (90, 6800Hz) because outside of this intervall there is no to not much information
        S = librosa.feature.melspectrogram(f, sr=sr, hop_length=hop_length, n_fft=window_size, n_mels=mel_bins, fmin=90, fmax=6800)
        # transforming linear representation of amplitude to logarithmic one 
        # (this is done due to sounds carry very little physical energy, hence scaling them logarithmically improves visualization)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # creating and storing spectrogram
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)
        plt.savefig(exportPath, bbox_inches='tight', pad_inches=0.0)
        #avoid rance-condition like errors
        plt.close('all')