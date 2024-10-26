import torch
from transformers import AutoProcessor, BarkModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import io
from pydub import AudioSegment
import ast

class TTSGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize Parler TTS for Speaker 1
        self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        ).to(self.device)
        self.parler_tokenizer = AutoTokenizer.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        )
        
        # Initialize Bark for Speaker 2
        self.bark_processor = AutoProcessor.from_pretrained("suno/bark")
        self.bark_model = BarkModel.from_pretrained(
            "suno/bark",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.speaker1_description = """
        Laura's voice is expressive and dramatic in delivery, speaking at a 
        moderately fast pace with a very close recording that has almost no 
        background noise.
        """
        
    def generate_speaker1_audio(self, text):
        """Generate audio using ParlerTTS for Speaker 1"""
        input_ids = self.parler_tokenizer(
            self.speaker1_description,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        prompt_input_ids = self.parler_tokenizer(
            text,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        generation = self.parler_model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids
        )
        
        audio_arr = generation.cpu().numpy().squeeze()
        return audio_arr, self.parler_model.config.sampling_rate
    
    def generate_speaker2_audio(self, text):
        """Generate audio using Bark for Speaker 2"""
        inputs = self.bark_processor(
            text,
            voice_preset="v2/en_speaker_6"
        ).to(self.device)
        
        speech_output = self.bark_model.generate(
            **inputs,
            temperature=0.9,
            semantic_temperature=0.8
        )
        
        audio_arr = speech_output[0].cpu().numpy()
        return audio_arr, 24000  # Bark sampling rate
    
    def numpy_to_audio_segment(self, audio_arr, sampling_rate):
        """Convert numpy array to AudioSegment"""
        audio_int16 = (audio_arr * 32767).astype(np.int16)
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sampling_rate, audio_int16)
        byte_io.seek(0)
        return AudioSegment.from_wav(byte_io)
    
    def generate_audio(self, script):
        """Generate full podcast audio from script"""
        final_audio = None
        script_tuples = ast.literal_eval(script)
        
        for speaker, text in script_tuples:
            if speaker == "Speaker 1":
                audio_arr, rate = self.generate_speaker1_audio(text)
            else:
                audio_arr, rate = self.generate_speaker2_audio(text)
            
            audio_segment = self.numpy_to_audio_segment(audio_arr, rate)
            
            if final_audio is None:
                final_audio = audio_segment
            else:
                final_audio += audio_segment
        
        # Save and return the path
        output_path = "temp/podcast.mp3"
        final_audio.export(
            output_path,
            format="mp3",
            bitrate="192k",
            parameters=["-q:a", "0"]
        )
        
        return output_path