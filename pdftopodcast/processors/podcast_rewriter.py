import torch
from transformers import pipeline
import gc
import ast
import json
import time
import re

class PodcastRewriter:
    def __init__(self):
        print("🚀 PodcastRewriter: Initializing...")
        start_time = time.time()

        # Optimize device selection for Apple Silicon / CUDA
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ PodcastRewriter: Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✅ PodcastRewriter: Using CUDA")
        else:
            self.device = "cpu"
            print("✅ PodcastRewriter: Using CPU")

        print("🧹 PodcastRewriter: Clearing memory cache...")
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ PodcastRewriter: Memory cache cleared")

        print("📦 PodcastRewriter: Loading model (this may take a moment)...")
        model_start = time.time()

        # Use memory-optimized pipeline configuration and let Accelerate place it (MPS/CUDA/CPU)
        self.model = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            model_kwargs={
                "torch_dtype": (torch.float16 if (torch.backends.mps.is_available() or torch.cuda.is_available())
                                else torch.float32),
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",   # faster than "eager" on recent PyTorch
                "device_map": "auto",            # <-- puts weights on MPS/CUDA automatically
            },
            batch_size=1,
        )

        # Ensure tokenizer has pad token
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        # Sanity check: where did it load?
        try:
            model_device = getattr(self.model.model, "device", None)
            device_map = getattr(self.model.model, "hf_device_map", None)
            print(f"🔍 PodcastRewriter: Model main device: {model_device}")
            print(f"🔍 PodcastRewriter: HF device map: {device_map}")
        except Exception:
            pass

        model_time = time.time() - model_start
        total_time = time.time() - start_time
        print(f"✅ PodcastRewriter: Model loaded in {model_time:.2f}s (total init: {total_time:.2f}s)")

        self.system_prompt = """
        You are an Oscar-winning screenwriter who works with podcasters. Rewrite this transcript
        for an AI Text-To-Speech pipeline. Make it engaging and natural.

        Remember:
        - Speaker 1's lines should be clean without fillers
        - Speaker 2 can use [sigh], [laughs], "umm", and "hmm"
        - Keep it conversational with realistic anecdotes

        Return the script as a list of tuples: [("Speaker 1", "text"), ("Speaker 2", "text")]
        """.strip()

    # --- helper: trim long inputs by tokens (safer than characters) ---
    def _truncate_by_tokens(self, text: str, max_input_tokens: int) -> str:
        tok = self.model.tokenizer
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=max_input_tokens).input_ids
        return tok.decode(ids, skip_special_tokens=True)

    # --- helper: try to parse list-of-tuples robustly ---
    def _parse_as_list_of_tuples(self, s: str):
        # Normalize fancy quotes and whitespace
        s = s.strip()
        s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("`", "'")

        # If it's valid JSON (e.g., [["Speaker 1","..."], ["Speaker 2","..."]]), convert to tuples
        try:
            js = json.loads(s)
            if isinstance(js, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in js):
                return [(str(x[0]), str(x[1])) for x in js]
        except Exception:
            pass

        # If it looks like a Python list literal, try ast.literal_eval safely
        if s.startswith("[") and s.endswith("]"):
            try:
                res = ast.literal_eval(s)
                if isinstance(res, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in res):
                    return [(str(x[0]), str(x[1])) for x in res]
            except Exception:
                pass

        # Try to extract the first [...] block if the model wrapped content
        if "[(" in s and ")]" in s:
            start_idx = s.find("[")
            end_idx = s.rfind("]") + 1
            candidate = s[start_idx:end_idx]
            try:
                res = ast.literal_eval(candidate)
                if isinstance(res, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in res):
                    return [(str(x[0]), str(x[1])) for x in res]
            except Exception:
                pass

        # Last-ditch regex: lines like '("Speaker 1","text")'
        tuple_re = re.compile(r'\(\s*["\'](Speaker 1|Speaker 2)["\']\s*,\s*["\'](.+?)["\']\s*\)')
        matches = tuple_re.findall(s)
        if matches:
            return [(spk, txt) for spk, txt in matches]

        return None

    def rewrite_transcript(self, transcript):
        """Rewrite the transcript for TTS with memory optimization"""
        try:
            print("\n" + "="*60)
            print("🔄 PODCAST REWRITING STARTED")
            print("="*60)

            rewrite_start = time.time()
            print(f"🔄 PodcastRewriter: Input transcript length: {len(transcript):,} characters")

            # Progress: Step 1/6 - Token-aware validation
            print("\n[1/6] 📏 Validating transcript length (token-aware)...")
            # Leave headroom for system prompt + response. For a 1B Llama variant, assume ~4k ctx.
            max_ctx = getattr(self.model.model.config, "max_position_embeddings", 4096)
            max_new_tokens = 1024
            safety = 64
            max_input_tokens = max(256, max_ctx - max_new_tokens - safety)

            # Tokenize & truncate by tokens
            original_len = len(transcript)
            truncated_transcript = self._truncate_by_tokens(transcript, max_input_tokens)
            if truncated_transcript != transcript:
                print(f"⚠️  Transcript truncated by tokens to fit context window")
                print(f"    Approx chars before: {original_len:,} → after: {len(truncated_transcript):,}")
                transcript = truncated_transcript
            else:
                print(f"✅ Transcript fits context window (≤ ~{max_input_tokens} tokens)")

            # Progress: Step 2/6 - Prompt creation
            print("\n[2/6] 📝 Creating rewrite prompt...")
            prompt_start = time.time()
            prompt = f"{self.system_prompt}\n\nTranscript to rewrite:\n{transcript}\n\nRewritten script:"
            prompt_time = time.time() - prompt_start
            print(f"✅ Prompt created in {prompt_time:.3f}s")
            print(f"    Prompt length: {len(prompt):,} characters")
            print(f"    System prompt: {len(self.system_prompt):,} chars")
            print(f"    Transcript: {len(transcript):,} chars")

            # Progress: Step 3/6 - Model generation
            print("\n[3/6] 🤖 Generating rewritten script...")
            print("    This may take 20-60 seconds depending on transcript length...")
            generation_start = time.time()

            outputs = self.model(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.model.tokenizer.eos_token_id,
                num_return_sequences=1,
                batch_size=1,
                use_cache=True,                # <-- ON for speed
                return_full_text=False,        # <-- only new text
            )

            generation_time = time.time() - generation_start
            print(f"✅ Model generation completed in {generation_time:.2f}s")
            print(f"    Average speed: {max_new_tokens/generation_time:.1f} tokens/second")

            # Progress: Step 4/6 - Processing output
            print("\n[4/6] 🔄 Processing model output...")
            processing_start = time.time()

            rewritten_script = outputs[0]["generated_text"].strip()

            processing_time = time.time() - processing_start
            print(f"✅ Output processed in {processing_time:.3f}s")
            print(f"    Final script length: {len(rewritten_script):,} characters")
            print(f"    Compression ratio: {len(rewritten_script)/max(1,len(transcript)):.2f}x")

            # Progress: Step 5/6 - Memory cleanup
            print("\n[5/6] 🧹 Cleaning up memory...")
            cleanup_start = time.time()

            del outputs
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            cleanup_time = time.time() - cleanup_start
            print(f"✅ Memory cleanup completed in {cleanup_time:.3f}s")

            # Progress: Step 6/6 - Parsing script
            print("\n[6/6] 🎯 Parsing script into dialogue format...")
            parsing_start = time.time()

            parsed = self._parse_as_list_of_tuples(rewritten_script)
            if parsed is not None and len(parsed) > 0:
                parsing_time = time.time() - parsing_start
                print(f"✅ Script parsed successfully as list of tuples in {parsing_time:.3f}s")
                print(f"    Dialogue entries: {len(parsed)}")

                speaker1_count = sum(1 for spk, _ in parsed if spk == "Speaker 1")
                speaker2_count = sum(1 for spk, _ in parsed if spk == "Speaker 2")
                print(f"    Speaker 1 lines: {speaker1_count}")
                print(f"    Speaker 2 lines: {speaker2_count}")

                total_time = time.time() - rewrite_start
                print(f"\n" + "="*60)
                print(f"🎉 PODCAST REWRITING COMPLETED!")
                print(f"   Total time: {total_time:.2f}s")
                print(f"   Input: {len(transcript):,} chars → Output: {len(parsed)} dialogue entries")
                print(f"   Efficiency: {len(parsed)/total_time:.1f} entries/second")
                print("="*60 + "\n")

                return parsed

            print("⚠️  Script not in expected list format, converting to simple structure...")
            # Fallback: turn line-by-line into alternating speakers
            lines = [ln.strip() for ln in rewritten_script.split("\n") if ln.strip()]
            script_list = []
            current_speaker = "Speaker 1"
            for i, line in enumerate(lines):
                # Detect explicit speaker markers
                if line.lower().startswith("speaker 1:"):
                    current_speaker = "Speaker 1"
                    line = line.split(":", 1)[1].strip()
                elif line.lower().startswith("speaker 2:"):
                    current_speaker = "Speaker 2"
                    line = line.split(":", 1)[1].strip()

                if line:
                    script_list.append((current_speaker, line))
                    current_speaker = "Speaker 2" if current_speaker == "Speaker 1" else "Speaker 1"

                if i > 0 and i % 10 == 0:
                    print(f"    Processed {i}/{len(lines)} lines...")

            parsing_time = time.time() - parsing_start
            total_time = time.time() - rewrite_start

            print(f"✅ Script converted to simple structure in {parsing_time:.3f}s")
            print(f"    Dialogue entries: {len(script_list)}")
            print(f"\n" + "="*60)
            print(f"🎉 PODCAST REWRITING COMPLETED!")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Input: {len(transcript):,} chars → Output: {len(script_list)} dialogue entries")
            print(f"   Efficiency: {len(script_list)/total_time:.1f} entries/second")
            print("="*60 + "\n")

            return script_list if script_list else [("Speaker 1", rewritten_script)]

        except Exception as e:
            print(f"\n❌ ERROR in podcast rewriting: {e}")
            print("🔄 Attempting fallback rewrite...")

            fallback_script = [("Speaker 1", transcript[:500] + "...")]
            print(f"✅ Fallback script created ({len(fallback_script)} entries)")
            return fallback_script
