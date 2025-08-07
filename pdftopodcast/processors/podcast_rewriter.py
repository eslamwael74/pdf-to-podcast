import torch
from transformers import pipeline
import gc
import ast


class PodcastRewriter:
    def __init__(self):
        # Optimize device selection for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("PodcastRewriter: Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("PodcastRewriter: Using CUDA")
        else:
            self.device = "cpu"
            print("PodcastRewriter: Using CPU")

        # Clear cache before loading
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("Loading podcast rewriter model with memory optimizations...")

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
        You are an Oscar-winning screenwriter who works with podcasters. Rewrite this transcript
        for an AI Text-To-Speech pipeline. Make it engaging and natural.

        Remember:
        - Speaker 1's lines should be clean without fillers
        - Speaker 2 can use [sigh], [laughs], "umm", and "hmm"
        - Keep it conversational with realistic anecdotes

        Return the script as a list of tuples: [("Speaker 1", "text"), ("Speaker 2", "text")]
        """

    def rewrite_transcript(self, transcript):
        """Rewrite the transcript for TTS with memory optimization"""
        try:
            # Limit input text length to prevent memory overflow
            max_input_length = 1500  # Reduce input size for rewriting
            if len(transcript) > max_input_length:
                transcript = transcript[:max_input_length]
                print(f"Transcript truncated to {max_input_length} characters to save memory")

            # Create prompt
            prompt = f"{self.system_prompt}\n\nTranscript to rewrite:\n{transcript}\n\nRewritten script:"

            # Generate with strict memory limits
            outputs = self.model(
                prompt,
                max_new_tokens=1024,  # Reduced from 8126
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.model.tokenizer.eos_token_id,
                num_return_sequences=1,
                batch_size=1,
                use_cache=False,  # Disable cache
            )

            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            # Remove the original prompt from the output
            rewritten_script = generated_text[len(prompt):].strip()

            # Clean up memory
            del outputs
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Try to parse the script as a list of tuples, fallback if it fails
            try:
                # Look for list-like structure in the output
                if "[(" in rewritten_script and ")]" in rewritten_script:
                    # Extract the list part
                    start_idx = rewritten_script.find("[")
                    end_idx = rewritten_script.rfind("]") + 1
                    list_str = rewritten_script[start_idx:end_idx]
                    script_list = ast.literal_eval(list_str)
                    return script_list
                else:
                    # Fallback: create a simple structure
                    lines = rewritten_script.split('\n')
                    script_list = []
                    current_speaker = "Speaker 1"
                    for line in lines:
                        if line.strip():
                            if "Speaker 2" in line or "speaker 2" in line:
                                current_speaker = "Speaker 2"
                            elif "Speaker 1" in line or "speaker 1" in line:
                                current_speaker = "Speaker 1"

                            # Clean the line
                            clean_line = line.replace("Speaker 1:", "").replace("Speaker 2:", "").strip()
                            if clean_line:
                                script_list.append((current_speaker, clean_line))
                                # Alternate speakers
                                current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"

                    return script_list if script_list else [("Speaker 1", rewritten_script)]

            except Exception as parse_error:
                print(f"Error parsing script format: {parse_error}")
                # Ultimate fallback: return simple format
                return [("Speaker 1", rewritten_script)]

        except Exception as e:
            print(f"Error rewriting transcript: {e}")
            # Fallback: return original transcript in simple format
            return [("Speaker 1", transcript[:500] + "...")]
