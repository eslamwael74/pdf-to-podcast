import gc
import time
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class PDFProcessor:
    def __init__(self, chunk_size_tokens=700, max_new_tokens=256):
        print("🚀 PDFProcessor: Initializing...")
        start_time = time.time()

        self.chunk_size_tokens = chunk_size_tokens
        self.max_new_tokens = max_new_tokens

        # -------- Device select --------
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ PDFProcessor: Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✅ PDFProcessor: Using CUDA")
        else:
            self.device = "cpu"
            print("✅ PDFProcessor: Using CPU")

        # -------- Clean caches --------
        print("🧹 PDFProcessor: Clearing memory cache...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ PDFProcessor: Memory cache cleared")

        print("📦 PDFProcessor: Loading Llama model with memory optimizations...")
        model_start = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=(torch.float16 if (torch.backends.mps.is_available() or torch.cuda.is_available())
                         else torch.float32),
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",   # Faster attention
            use_cache=True,               # ON for big speedup
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            use_fast=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not hasattr(self.model, "hf_device_map") and self.device != "cpu":
            self.model.to(self.device)

        self.model.eval()
        self.max_ctx = getattr(self.model.config, "max_position_embeddings", 4096)
        self.safety = 48

        model_time = time.time() - model_start
        total_time = time.time() - start_time
        print(f"✅ PDFProcessor: Model loaded in {model_time:.2f}s (total init: {total_time:.2f}s)")

        self.system_prompt = (
            "You are a world class text pre-processor. Clean up this PDF text to make it suitable "
            "for a podcast transcript. Remove LaTeX, unnecessary newlines, and any fluff that "
            "wouldn't be useful in a podcast. Be aggressive in cleaning but preserve the key content. "
            "Start your response directly with the cleaned text."
        )

    # -------- PDF extraction --------
    def extract_text(self, pdf_path: str) -> str:
        print("📄 Step 1/4: Extracting text from PDF...")
        t0 = time.time()
        try:
            import fitz  # PyMuPDF
            text_parts = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text_parts.append(page.get_text("text"))
            text = "\n".join(text_parts)
            print(f"✅ PDF text extracted with PyMuPDF in {time.time()-t0:.2f}s")
            return text
        except Exception as e:
            print(f"⚠️ PyMuPDF failed ({e}), falling back to PyPDF2...")
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join((p.extract_text() or "") for p in reader.pages)
            print(f"✅ PDF text extracted with PyPDF2 in {time.time()-t0:.2f}s")
            return text

    # -------- Token chunking --------
    def _token_chunks(self, text: str):
        print("✂️ Step 2/4: Creating token-bounded text chunks...")
        max_input_tokens = max(256, self.max_ctx - self.max_new_tokens - self.safety)
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = enc["input_ids"]
        window = min(self.chunk_size_tokens, max_input_tokens)
        chunks = []
        for i in range(0, len(ids), window):
            chunk_text = self.tokenizer.decode(ids[i:i+window], skip_special_tokens=True)
            chunks.append(chunk_text)
        print(f"✅ Created {len(chunks)} chunks (~{self.chunk_size_tokens} tokens each)")
        return chunks

    # -------- One chunk clean --------
    def _clean_chunk(self, chunk_text: str, idx: int, total: int) -> str:
        print(f"🧼 Cleaning chunk {idx}/{total} (len={len(chunk_text):,} chars)...")
        prompt = f"{self.system_prompt}\n\n{chunk_text}"
        max_input_tokens = max(256, self.max_ctx - self.max_new_tokens - self.safety)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            padding=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        cleaned_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        del inputs, output, new_tokens
        return cleaned_text

    # -------- Main pipeline --------
    def process_pdf(self, pdf_path: str) -> str:
        print("\n" + "="*60)
        print("📝 PDF CLEANING PROCESS STARTED")
        print("="*60)

        try:
            raw_text = self.extract_text(pdf_path)
            chunks = self._token_chunks(raw_text)

            cleaned_chunks = []
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
                cleaned = self._clean_chunk(chunk, i+1, len(chunks))
                cleaned_chunks.append(cleaned)

                if (i + 1) % 10 == 0:
                    print("🧹 Periodic cache cleanup...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()

            cleaned_text = " ".join(cleaned_chunks)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            print(f"\n✅ PDF cleaning completed. Final length: {len(cleaned_text):,} chars")
            print("="*60 + "\n")
            return cleaned_text

        except Exception as e:
            print(f"❌ Error in process_pdf: {e}")
            return self.extract_text(pdf_path)
