import os
import pandas as pd
import librosa
import numpy as np


class DataLoader:

    def __init__(self,lang='ta'):
        self.lang = lang
        self.DATA_DIR = './data/'
        self.metadata = 'metadata.csv'
        self.output_hf_file=f'{lang}_hf_dataset'
        self.metadata_file = 'metadata.csv'

    def _load_metadata(self, data_dir=None, metadata=None):
        """
        Load metadata CSV file.
        Expected columns: [audio_path, prompt_path, lyrics_path]
        """
        try:
            if data_dir and os.path.exists(data_dir):
                self.DATA_DIR = data_dir

            if metadata and os.path.exists(os.path.join(self.DATA_DIR, metadata)):
                self.metadata_file = metadata

            metadata_path = os.path.join(self.DATA_DIR, self.metadata_file)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            df = pd.read_csv(metadata_path)
            print(f"Metadata loaded successfully from {metadata_path}")
            return df

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None

    def _load_audio(self, audio_path ,sample_rate=22050,audio_load=True):

        """
            => Read an audio from the audio_path ,
        """
        if not audio_load:
            return None,None
        
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None

            audio , sr = librosa.load(audio_path, sr=sample_rate)
            return audio,sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
        
    def _load_lyrics(self,lyric_path):
        """
                => Read a song lyrics from the lyric_path ,
                
        """
        
        lyrics=None
        try:    
            if os.path.exists(lyric_path):
                with open(lyric_path, 'r', encoding='utf8') as file:
                    lyrics = file.read()
            else:
                print(f"Lyrics file not found:{lyric_path}")
        except Exception as e:
            print(f"Error loading transcript: {e}")
        return lyrics
        
        
        
    def _load_prompt(self ,prompt_path):
        """
            => Read a song prompt from the prompt_path,
        """
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r' ,encoding='utf8') as file:
                    prompt = file.read()
            return prompt
        except Exception as e:
            print(f"Error loading prompt: {e}")
            return None

    def _load_speaker(self):
        """
            It's optional 
        """
        try:
            return None
        except Exception as e:
            print(f"Error loading speaker: {e}")
            return None
    
    def __call__(self, data_dir=None , metadata=None,sample_rate=22050,audio_load=True):

        print("load dataset")
        df = self._load_metadata(data_dir, metadata)

        if df is None:
            return None
        
        for idx , row in df.iterrows():
            
            try:
                
                audio_path = os.path.join(self.DATA_DIR, row['audio_path'])
                lyric_path = os.path.join(self.DATA_DIR, row['lyrics_path'])
                prompt_path = os.path.join(self.DATA_DIR, row['prompt_path'])

                audio, sr = self._load_audio(audio_path, sample_rate , audio_load=audio_load)

                lyrics = self._load_lyrics(lyric_path)
                prompt = self._load_prompt(prompt_path)
                speaker = self._load_speaker()
    
                if  lyrics is not None and prompt is not None:
                    yield {
                        'audio':audio,
                        'sr':sr,
                        'lyrics':lyrics,
                        'prompt':prompt,
                        'speaker':speaker
                    }
                
                else:
                    print(f"Skipping row {idx} due to missing data")

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
            

