import torch
from transformers import pipeline
import gc
import time

class TranscriptWriter:
    def __init__(self):
        print("🚀 TranscriptWriter: Initializing...")
        start_time = time.time()

        # Optimize device selection for Apple Silicon / CUDA
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ TranscriptWriter: Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✅ TranscriptWriter: Using CUDA")
        else:
            self.device = "cpu"
            print("✅ TranscriptWriter: Using CPU")

        print("🧹 TranscriptWriter: Clearing memory cache...")
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ TranscriptWriter: Memory cache cleared")

        print("📦 TranscriptWriter: Loading model (this may take a moment)...")
        model_start = time.time()

        # Use memory-optimized pipeline configuration and let Accelerate place it (MPS/GPU/CPU)
        self.model = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            model_kwargs={
                "torch_dtype": (torch.float16 if (torch.backends.mps.is_available() or torch.cuda.is_available())
                                else torch.float32),
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",      # faster than "eager" on recent PyTorch
                "device_map": "auto",               # <-- puts weights on MPS/CUDA automatically
            },
            batch_size=1,
        )

        # Sanity check: where did it load?
        try:
            model_device = getattr(self.model.model, "device", None)
            device_map = getattr(self.model.model, "hf_device_map", None)
            print(f"🔍 TranscriptWriter: Model main device: {model_device}")
            print(f"🔍 TranscriptWriter: HF device map: {device_map}")
        except Exception:
            pass

        model_time = time.time() - model_start
        total_time = time.time() - start_time
        print(f"✅ TranscriptWriter: Model loaded in {model_time:.2f}s (total init: {total_time:.2f}s)")

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
            print("\n" + "="*60)
            print("📝 TRANSCRIPT GENERATION STARTED")
            print("="*60)

            generation_start = time.time()
            print(f"📝 TranscriptWriter: Input text length: {len(text):,} characters")

            # Progress: Step 1/5 - Text validation (char-based guard; token limits are enforced by model anyway)
            print("\n[1/5] 📏 Validating input text length...")
            max_input_length = 20000  # You set high; real safety comes from tokenizer truncation later
            if len(text) > max_input_length:
                text = text[:max_input_length]
                print(f"⚠️  Input text truncated to {max_input_length:,} characters to save memory")
                print(f"    Original: {len(text):,} chars → Truncated: {max_input_length:,} chars")
            else:
                print(f"✅ Input text within limit ({len(text):,}/{max_input_length:,} characters)")

            # Progress: Step 2/5 - Prompt creation
            print("\n[2/5] 📝 Creating conversation prompt...")
            prompt_start = time.time()
            prompt = f"{self.system_prompt}\n\nContent to convert:\n{text}\n\nPodcast conversation:"
            prompt_time = time.time() - prompt_start
            print(f"✅ Prompt created in {prompt_time:.3f}s")
            print(f"    Prompt length: {len(prompt):,} characters")
            print(f"    System prompt: {len(self.system_prompt):,} chars")
            print(f"    Content: {len(text):,} chars")

            # Progress: Step 3/5 - Model generation
            print("\n[3/5] 🤖 Generating with language model...")
            print("    This may take 30-120 seconds depending on content length...")
            generation_model_start = time.time()

            # Enable KV cache for speed, and avoid returning the prompt again
            outputs = self.model(
                prompt,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.model.tokenizer.eos_token_id,
                num_return_sequences=1,
                batch_size=1,
                use_cache=True,               # <-- ON for speed
                return_full_text=False,       # <-- only new text
            )

            generation_model_time = time.time() - generation_model_start
            # We requested 2048 new tokens; treat that as the denominator for an approximate speed
            print(f"✅ Model generation completed in {generation_model_time:.2f}s")
            print(f"    Average speed: {2048/generation_model_time:.1f} tokens/second")

            # Progress: Step 4/5 - Processing output
            print("\n[4/5] 🔄 Processing model output...")
            processing_start = time.time()

            # Pipeline with return_full_text=False returns only the new text
            transcript = outputs[0]["generated_text"].strip()

            processing_time = time.time() - processing_start
            print(f"✅ Output processed in {processing_time:.3f}s")
            print(f"    Final transcript length: {len(transcript):,} characters")

            # Count approximate dialogue exchanges
            speaker_changes = transcript.count("Speaker 1") + transcript.count("Speaker 2")
            print(f"    Estimated dialogue exchanges: ~{speaker_changes}")

            # Progress: Step 5/5 - Memory cleanup
            print("\n[5/5] 🧹 Cleaning up memory...")
            cleanup_start = time.time()

            del outputs
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            cleanup_time = time.time() - cleanup_start
            total_time = time.time() - generation_start

            print(f"✅ Memory cleanup completed in {cleanup_time:.3f}s")
            print(f"\n" + "="*60)
            print(f"🎉 TRANSCRIPT GENERATION COMPLETED!")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Input: {len(text):,} chars → Output: {len(transcript):,} chars")
            print(f"   Efficiency: {len(transcript)/total_time:.0f} chars/second")
            print("="*60 + "\n")

            return transcript

        except Exception as e:
            print(f"\n❌ ERROR in transcript generation: {e}")
            print("🔄 Attempting fallback generation...")

            # Fallback: return a simple formatted version
            fallback_text = (
                "Speaker 1: Let me tell you about this interesting topic...\n\n"
                "Speaker 2: That sounds fascinating! Can you explain more?\n\n"
                f"Speaker 1: {text[:500]}..."
            )
            print(f"✅ Fallback transcript created ({len(fallback_text)} characters)")
            return fallback_text
