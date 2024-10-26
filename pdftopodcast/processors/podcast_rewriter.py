import torch
from transformers import pipeline

class PodcastRewriter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        self.system_prompt = """
        You are an Oscar-winning screenwriter who works with podcasters. Rewrite this transcript
        for an AI Text-To-Speech pipeline. Make it engaging and natural.
        
        Remember:
        - Speaker 1's lines should be clean without fillers
        - Speaker 2 can use [sigh], [laughs], "umm", and "hmm"
        - Keep it conversational with realistic anecdotes
        
        Return the script as a list of tuples: [("Speaker 1", "text"), ("Speaker 2", "text")]
        """
        
    def rewrite_transcript(self, transcript):
        """Rewrite the transcript for TTS"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": transcript}
        ]
        
        outputs = self.model(
            messages,
            max_new_tokens=8126,
            temperature=1,
        )
        
        return outputs[0]["generated_text"][-1]['content']