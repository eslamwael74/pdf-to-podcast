import torch
from transformers import pipeline

class TranscriptWriter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-70B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        self.system_prompt = """
        You are a world-class podcast writer who has worked with Joe Rogan, Lex Fridman, 
        and other top podcasters. Write an engaging conversation based on this content.
        
        Speaker 1: Leads the conversation and teaches Speaker 2, uses great analogies
        Speaker 2: Asks follow-up questions, shows curiosity, and occasionally goes on interesting tangents
        
        Make it engaging and natural. Add "umm" and "hmm" for Speaker 2 only.
        Start directly with Speaker 1's opening line.
        """
        
    def generate_transcript(self, text):
        """Generate podcast transcript from cleaned text"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]
        
        outputs = self.model(
            messages,
            max_new_tokens=8126,
            temperature=1,
        )
        
        return outputs[0]["generated_text"][-1]['content']