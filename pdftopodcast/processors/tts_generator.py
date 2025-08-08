# processors/tts_generator.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import json
import re
import ast
import time
import numpy as np
import torch
from scipy.io import wavfile
from pydub import AudioSegment, effects
from transformers import AutoProcessor, BarkModel, AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


class TTSGenerator:
    def __init__(self):
        print("🚀 TTSGenerator: Initializing...")

        # -------- Device selection --------
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ TTSGenerator: Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✅ TTSGenerator: Using CUDA")
        else:
            self.device = "cpu"
            print("✅ TTSGenerator: Using CPU")

        # -------- Load Parler-TTS (Speaker 1) --------
        print("📦 Loading Parler-TTS (Speaker 1)...")
        t0 = time.time()
        self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1"
        ).to(self.device)
        self.parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        self.speaker1_description = (
            "Laura's voice is expressive and dramatic in delivery, speaking at a "
            "moderately fast pace with a very close recording that has almost no "
            "background noise."
        )
        print(f"✅ Parler-TTS loaded in {time.time()-t0:.2f}s")

        # -------- Load Bark (Speaker 2) via Transformers (float32) --------
        print("📦 Loading Bark (Speaker 2)...")
        t0 = time.time()
        # IMPORTANT: keep Bark in float32 on MPS to avoid dtype mismatch errors
        bark_dtype = torch.float32
        self.bark_processor = AutoProcessor.from_pretrained("suno/bark")
        self.bark_model = BarkModel.from_pretrained(
            "suno/bark",
            torch_dtype=bark_dtype
        ).to(self.device)
        print(f"✅ Bark loaded in {time.time()-t0:.2f}s")

        # Output config
        self.target_rate = 24000  # Bark-native; we’ll resample Parler to this
        self.inter_line_ms = 200  # short pause between lines

    # ---------------- Parsing helpers ----------------
    def _normalize_quotes(self, s: str) -> str:
        return (
            s.replace("“", '"').replace("”", '"')
             .replace("’", "'").replace("`", "'")
        )

    def _coerce_script(self, script):
        """Return a list of (speaker, text). Accepts list/tuples or string."""
        print("📜 Step 1/4: Parsing podcast script...")

        if isinstance(script, list):
            print("   → Script already a list, verifying format...")
            out = []
            for item in script:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((str(item[0]), str(item[1])))
            if out:
                print(f"   ✅ Parsed as list of tuples ({len(out)} entries)")
                return out

        if isinstance(script, str):
            s = self._normalize_quotes(script).strip()

            # JSON list
            try:
                js = json.loads(s)
                if isinstance(js, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in js):
                    print(f"   ✅ Parsed as JSON list ({len(js)} entries)")
                    return [(str(x[0]), str(x[1])) for x in js]
            except Exception:
                pass

            # Python literal
            try:
                lit = ast.literal_eval(s)
                if isinstance(lit, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in lit):
                    print(f"   ✅ Parsed as Python literal ({len(lit)} entries)")
                    return [(str(x[0]), str(x[1])) for x in lit]
            except Exception:
                pass

            # Extract [ ... ] block and try again
            if "[" in s and "]" in s:
                try:
                    start = s.find("[")
                    end = s.rfind("]") + 1
                    block = s[start:end]
                    lit = ast.literal_eval(block)
                    if isinstance(lit, list) and all(isinstance(x, (list, tuple)) and len(x) >= 2 for x in lit):
                        print(f"   ✅ Extracted and parsed list block ({len(lit)} entries)")
                        return [(str(x[0]), str(x[1])) for x in lit]
                except Exception:
                    pass

            # Regex fallback
            tuple_re = re.compile(r'\(\s*["\'](Speaker 1|Speaker 2)["\']\s*,\s*["\'](.+?)["\']\s*\)')
            matches = tuple_re.findall(s)
            if matches:
                print(f"   ✅ Parsed via regex tuple match ({len(matches)} entries)")
                return [(spk, txt) for spk, txt in matches]

            # Alternate speakers line-by-line
            lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
            out = []
            current = "Speaker 1"
            for ln in lines:
                ll = ln.lower()
                if ll.startswith("speaker 1:"):
                    current = "Speaker 1"
                    ln = ln.split(":", 1)[1].strip()
                elif ll.startswith("speaker 2:"):
                    current = "Speaker 2"
                    ln = ln.split(":", 1)[1].strip()
                out.append((current, ln))
                current = "Speaker 2" if current == "Speaker 1" else "Speaker 1"
            if out:
                print(f"   ⚠️ Fallback: Alternating speakers ({len(out)} entries)")
                return out

        print("   ⚠️ Parsing failed — using single-line fallback")
        return [("Speaker 1", str(script))]

    # ---------------- Audio helpers ----------------
    def _numpy_to_segment(self, audio_arr: np.ndarray, rate: int) -> AudioSegment:
        """Convert numpy waveform to pydub segment and resample to target_rate."""
        if audio_arr.dtype != np.float32:
            audio_arr = audio_arr.astype(np.float32)
        audio_int16 = (np.clip(audio_arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        bio = io.BytesIO()
        wavfile.write(bio, rate, audio_int16)
        bio.seek(0)
        seg = AudioSegment.from_wav(bio)
        if seg.frame_rate != self.target_rate:
            seg = seg.set_frame_rate(self.target_rate)
        seg = seg.set_channels(1)
        return seg

    def _silence(self, ms: int) -> AudioSegment:
        return AudioSegment.silent(duration=ms, frame_rate=self.target_rate)

    # ---------------- TTS backends ----------------
    def generate_speaker1_audio(self, text):
        """Parler-TTS (Speaker 1) — always pass BOTH attention masks."""
        # Tokenize voice description
        desc = self.parler_tokenizer(
            self.speaker1_description,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = desc.input_ids.to(self.device)
        attention_mask = (desc.attention_mask if hasattr(desc, "attention_mask") and desc.attention_mask is not None
                          else torch.ones_like(input_ids)).to(self.device)

        # Tokenize prompt text
        prm = self.parler_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        prompt_input_ids = prm.input_ids.to(self.device)
        prompt_attention_mask = (prm.attention_mask if hasattr(prm, "attention_mask") and prm.attention_mask is not None
                                 else torch.ones_like(prompt_input_ids)).to(self.device)

        # Be explicit about pad token id to avoid other warnings
        pad_id = getattr(self.parler_model.config, "pad_token_id", None)
        if pad_id is None and hasattr(self.parler_tokenizer, "eos_token_id"):
            pad_id = self.parler_tokenizer.eos_token_id

        with torch.inference_mode():
            generation = self.parler_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # ✅ main mask
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,  # ✅ prompt mask
                pad_token_id=pad_id
            )

        audio_arr = generation.cpu().numpy().squeeze()
        rate = getattr(self.parler_model.config, "sampling_rate", 22050)
        return audio_arr, rate

    def generate_speaker2_audio(self, text):
        """Bark (Speaker 2) on MPS: float32 for floats, long for ids."""
        inputs = self.bark_processor(
            text,
            voice_preset="v2/en_speaker_6"
        ).to(self.device)

        # Fix dtypes precisely:
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                continue
            # input_ids / attention-style masks should be integer
            if "input_ids" in k or v.dtype in (torch.int64, torch.int32):
                inputs[k] = v.to(dtype=torch.long)  # embeddings expect long
            else:
                # Only cast *floating* tensors to float32
                if v.dtype.is_floating_point:
                    inputs[k] = v.to(dtype=torch.float32)

        with torch.inference_mode():
            speech_output = self.bark_model.generate(
                **inputs,
                temperature=0.9,
                semantic_temperature=0.8
            )

        audio_arr = speech_output[0].cpu().numpy()
        return audio_arr, 24000  # Bark native

    # ---------------- Main ----------------
    def generate_audio(self, script):
        print("\n" + "="*60)
        print("🎤 PODCAST AUDIO GENERATION STARTED")
        print("="*60)

        # Step 1: Parse script
        entries = self._coerce_script(script)
        if not entries:
            raise ValueError("Empty podcast script after parsing.")
        print(f"🔊 Step 2/4: Generating audio for {len(entries)} dialogue entries...")

        final = AudioSegment.silent(duration=0, frame_rate=self.target_rate)

        # Step 2: Synthesize per-line (resilient)
        for i, (speaker, text) in enumerate(entries, 1):
            speaker = "Speaker 1" if str(speaker).strip().lower().startswith("speaker 1") else "Speaker 2"
            text = str(text).strip()
            if not text:
                print(f"   • [{i}/{len(entries)}] {speaker}: (empty) — skipped")
                final += self._silence(self.inter_line_ms)
                continue

            print(f"   • [{i}/{len(entries)}] {speaker}: {text[:60]}{'…' if len(text)>60 else ''}")

            try:
                if speaker == "Speaker 1":
                    audio_arr, rate = self.generate_speaker1_audio(text)
                else:
                    audio_arr, rate = self.generate_speaker2_audio(text)
            except Exception as e:
                print(f"   ⚠️  TTS failed on line {i} ({speaker}): {e}. Skipping.")
                final += self._silence(self.inter_line_ms)
                continue

            seg = self._numpy_to_segment(audio_arr, rate)
            seg = effects.normalize(seg)
            final += seg + self._silence(self.inter_line_ms)

        # Step 3: Export MP3
        output_path = "temp/podcast.mp3"
        final.export(output_path, format="mp3", bitrate="192k", parameters=["-q:a", "0"])

        # Step 4: Done
        duration_sec = len(final) / 1000.0
        print(f"\n✅ Step 4/4: Audio file saved to {output_path} ({duration_sec:.1f}s total)")
        print("="*60 + "\n")

        return output_path
