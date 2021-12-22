# Emotion classification using paraverbal speech analysis 

### TL&DR

In this project a pretrained Convolution Neural Network (CNN) is used to classify spoken voice samples among seven different emotions.

Due to the circumstance, that CNN use a fixed-size input and aren't per se suited to process sequential data, the varying length within a data set introduces a temporal bias to the model that can drastically influence its performance for the worse or its ability to work sufficiently at all.

Multiple methods of audio-based augmentation and techniques to tackle the problem of variable length of input data have been developed and tested separately/combined on three different corpora and in various model configurations.

Some of them are presented here, together with some insights gained along the way.

---

### Exhaustive explanation

The voice is a multidimensional information carrier medium. Apart from *what* is being said, a lot of information is transported by the way *how* it is being expressed with respect to the audible (paraverbal) aspect. Intercultural language studies [[1]](https://www.pnas.org/content/107/6/2408), [[2]](https://ieeexplore.ieee.org/document/7051419), [[3]](https://ieeexplore.ieee.org/document/8616972) have not only proven that a set of so called *basic emotions* is shared by all humans, but that they are expressed in a similar way and can be reliably distinguished in a cross-cultural context by the means of the voice.

These *basic emotions* (anger, disgust, fear, joy, sadness, surprise) are expressed in ways that share some commonalities with regards to paraverbal features and can be visualized by using a special form of the Fourier Transformation to generate so called *spectrogrammes* - mapping time domain events (left) into the frequency domain (right).

![spectrogram](img/time_freq_dom.png)

With this form of visualization it is possible to see what frequency range received what specific amount of energy at what point in time. These images can be used as an input by Convolution Neural Networks. Within this project a modified VGG16 architecture is used, pretrained on the ImageNet Dataset. Although the type of imagery used to train the model initially differs widely from the kind data dealt within this project, the low-level feautures developed during this process are sufficient for this task and not having to train them from scratch shortens the training process siginificantly. 

The problem is that almost no sample within a given dataset will have the same length (in fact within some corpora used within this project they diverge over 400%). This is problematic due to all *spectrograms* fed into the model have to have the exact same size, resulting in longer samples being compressed, while shorter once get streched out. Possible features like patterns within the melody of the voice will become way harder to detect for the model within the training process, diminishing the chance of it to generalize.

Another issue is that most labeled data sets of emotional speech are relatively small (the biggest one used within this project contained 2.8k samples) or show little diversity with regards to the audible qualities of the voice, making it hard to train a model reliably or extrapolate from the results gained during testing. In order to approach this, four different methods of audio-based augmentation where introduced to increase the size of the corpora and result in enhanced performance in regards of accuracy and robustness.

The main goal of this project was to investigate, what impact different approaches to eliminating the temporal bias have on the overall, intra-corporal performance and ability to inter-corporal generalization of various model configurations.

After having tested over 70 different constellations on three different data sets of acted speech (see list below) it can be reported, that within this given setting excluding the temporal bias from the training process significantly improves the performance of the model on two of the three datasets <sup>[1](#fn1)</sup>, while no indications for an improvement of inter-corporal generalization ability could be found. The latter could be related to the different nature and size of the corpora and the trade-offs that had to be made during the visualization process in order to establish comparability between the different data sets. Yet these findings are promising and worth further investigations in a setting of natural speech and with a much greater sample size. 

![result](img/cm_RAVDESS_raw.png)

---

<a name="fn1">1</a>: no meaningful results in this respect could be derived from the *TESS* corpus, due to the model correctly identified all samples in the evaluation set in its standard configuration

---

COMPONENT DESCRIPTION:

audio_augmentor:

- used to augment audio files by either:
	- adding a random amount of gaussian white noise
	- pitch-shift the given audio file within a given range (chosen randomly)
	
audio_to_spectrogram:
- used to transform audio signals into coresponding log-mel-spectrograms and crop and scale them to appropriate size to function as input for the VGG16

log:
- helper method, used to create a logfile containing numeric data about the model performance measures, can be used to avoid NaN Values, resulting from "devision-by-zero" related errors in case the model underfits or fails to generalize

method1:
- used to equal the duration of audio samples within a given folder by:
	- repeatingly append shorter samples with themselves until they meet the duration of longest sample (will cut overlap from the end)

method4:
- used to equal the duration of audio samples within a given folder by:
	- creating new samples by splitting longer samples into one, matching the duration of the shortest sample of the corpus (stride can be adjusted)

preprocessor:
- used to convert signal to mono and trim silence at begin and end of signal and normalize volume of each sample

unit_tests:
- test cases for the methods to harmonize the duration of samples

vgg:
- main project, used to train, validate and test the VGG16-model on given spectrogram data, concludes with a section that plots/stores the confusion matrix, training&validaiton progress and creates an additional numeric logfile if desired

---

REQUIREMENTS:

L i b r a r i e s:

- numpy:
	-> numpy.org/install/

- sklearn:
	-> scikit-learn.org/stable/install.html

- mathplotlib:
	-> matplotlib.org/stable/users/installing.html

- librosa:
	-> librosa.org/doc/latest/install.html

- daze:
	-> pypi.org/project/daze/

- soundfile:
	-> pypi.org/project/SoundFile/

- pydub:
	-> pypi.org/project/pydub/

- PyTorch:
	-> pytorch.org/get-started/locally/

C o r p o r a:

- Berlin Database of Emotional Speech (Emo-DB):
	-> kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb

- Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):	
	-> zenodo.org/record/1188976

- Torronto emotional speech set (TESS):					
	-> dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
