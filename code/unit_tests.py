from method1 import fill
from method4 import crop_sw

import soundfile as sf
import unittest, os, shutil
from pydub import AudioSegment
from pydub.generators import Sine
import warnings

"""
[Unit testing the methods for harmonisation of the sample duration]
"""
class TestMethods(unittest.TestCase):

    def test_method1(self):        
        #pydub keeps a reference of exported and opened files alive, which cause resource warnings if they aren't assigned to a variable and closed after usage
        #due to the tearDown() method this can be ignored, because it will delete all data used for testin purpose after process is finished
        warnings.simplefilter("ignore", ResourceWarning)

        #create a testfolder within current work directory
        test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test")
        #creating an export path for testing purpose, gotta do it here, due to creating this one immedeately before method call, this can result in a race condition
        export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "m1")
        try: 
            os.mkdir(test_path)
            os.mkdir(export_path)
        except OSError as error: 
            print(error)            
            
        #initializing three tones of duration 1,2,3 second(s) 
        gen = Sine(440)
        short  = gen.to_audio_segment(duration=1000)
        medium = gen.to_audio_segment(duration=2000)
        long = gen.to_audio_segment(duration=3000)
        
        #save test tones as files within the test folder
        short.export(os.path.join(test_path, "short.wav"), format="wav")
        medium.export(os.path.join(test_path, "medium.wav"), format="wav")
        long.export(os.path.join(test_path, "long.wav"), format="wav")

        #sanity check: get duration of recently created sound files + opening soundfiles from folders works as intended
        self.assertEqual(short.duration_seconds, 1.0)
        self.assertEqual(AudioSegment.from_file(os.path.join(test_path,"long.wav")).duration_seconds, 3.0)

        #testing method 1
        expected = AudioSegment.from_file(os.path.join(test_path,"long.wav")).duration_seconds
        fill(test_path, export_path)
        self.assertEqual(AudioSegment.from_file(os.path.join(export_path,"long_m1.wav")).duration_seconds, expected)
        self.assertEqual(AudioSegment.from_file(os.path.join(export_path,"medium_m1.wav")).duration_seconds, expected)
        self.assertEqual(AudioSegment.from_file(os.path.join(export_path,"short_m1.wav")).duration_seconds, expected)

        #tear down to ensure tests can run in random order
        try:
            shutil.rmtree(test_path)
            shutil.rmtree(export_path)
        except:
            print("ERROR: Path, folder or file does not exist.")
    
    def test_method4(self):        
        #pydub keeps a reference of exported and opened files alive, which cause resource warnings if they aren't assigned to a variable and closed after usage
        #due to the tearDown() method this can be ignored, because it will delete all data used for testin purpose after process is finished
        warnings.simplefilter("ignore", ResourceWarning)

        #create a testfolder within current work directory
        test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test")
        #creating an export path for testing purpose, gotta do it here, due to creating this one immedeately before method call, this can result in a race condition
        export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "m4")
        try: 
            os.mkdir(test_path)
            os.mkdir(export_path)
        except OSError as error: 
            print(error)            
            
        #initializing three tones of duration 1,2,3 second(s) 
        gen = Sine(440)
        short  = gen.to_audio_segment(duration=1000)
        medium = gen.to_audio_segment(duration=2000)
        long = gen.to_audio_segment(duration=3000)
        
        #save test tones as files within the test folder
        short.export(os.path.join(test_path, "short.wav"), format="wav")
        medium.export(os.path.join(test_path, "medium.wav"), format="wav")
        long.export(os.path.join(test_path, "long.wav"), format="wav")

        #sanity check: get duration of recently created sound files + opening soundfiles from folders works as intended
        self.assertEqual(short.duration_seconds, 1.0)
        self.assertEqual(AudioSegment.from_file(os.path.join(test_path,"long.wav")).duration_seconds, 3.0)
        list_of_samples = os.listdir(test_path)
        number_files_in_test_path = len(list_of_samples)
        self.assertEqual(number_files_in_test_path, 3)

        #testing method 4
        expected = AudioSegment.from_file(os.path.join(test_path,"short.wav")).duration_seconds
        crop_sw(test_path, export_path, stride=500)
        for soundfile in os.scandir(export_path):
            temp_duration = AudioSegment.from_file(soundfile).duration_seconds
            self.assertEqual(temp_duration, expected)

        #proving the method increased the number of samples
        list_of_split_samples = os.listdir(export_path)
        number_files_in_export_path = len(list_of_split_samples)
        self.assertEqual(number_files_in_export_path, 6)

        #tear down to ensure tests can run in random order
        try:
            shutil.rmtree(test_path)
            shutil.rmtree(export_path)
        except:
            print("ERROR: Path, folder or file does not exist.")            
 
#allows to run the script from command line without providing further arguments
if __name__ == '__main__':
    unittest.main()
