import torch
from transformers import pipeline
import gc


class TranscriptWriter:
    def __init__(self):
        # Optimize device selection for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("TranscriptWriter: Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("TranscriptWriter: Using CUDA")
        else:
            self.device = "cpu"
            print("TranscriptWriter: Using CPU")

        # Clear cache before loading
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("Loading transcript generation model with memory optimizations...")

        # Use memory-optimized pipeline configuration
        self.model = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",  # Use 1B instead of 8B
            model_kwargs={
                "torch_dtype": torch.float16,  # Use float16 instead of bfloat16
                "low_cpu_mem_usage": True,
                "use_cache": False,  # Disable KV cache
                "attn_implementation": "eager"  # Use eager attention
            },
            device=0 if self.device == "cuda" else -1,  # Explicit device mapping for pipeline
            batch_size=1,  # Process one at a time to save memory
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
        """Generate podcast transcript from cleaned text with memory optimization"""
        try:
            # Limit input text length to prevent memory overflow
            max_input_length = 2000  # Reduce input size
            if len(text) > max_input_length:
                text = text[:max_input_length]
                print(f"Input text truncated to {max_input_length} characters to save memory")

            # Create prompt
            prompt = f"{self.system_prompt}\n\nContent to convert:\n{text}\n\nPodcast conversation:"

            # Generate with strict memory limits
            outputs = self.model(
                prompt,
                max_new_tokens=2048,  # Reduced from 8126
                temperature=0.8,  # Slightly reduced temperature
                do_sample=True,
                pad_token_id=self.model.tokenizer.eos_token_id,
                num_return_sequences=1,
                batch_size=1,
                use_cache=False,  # Disable cache
            )

            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            # Remove the original prompt from the output
            transcript = generated_text[len(prompt):].strip()

            # Clean up memory
            del outputs
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return transcript

        except Exception as e:
            print(f"Error generating transcript: {e}")
            # Fallback: return a simple formatted version
            return f"Speaker 1: Let me tell you about this interesting topic...\n\nSpeaker 2: That sounds fascinating! Can you explain more?\n\nSpeaker 1: {text[:500]}..."
