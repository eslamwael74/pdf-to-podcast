import PyPDF2
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc


class PDFProcessor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

        # Optimize device selection for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA")
        else:
            self.device = "cpu"
            print("Using CPU")

        # Clear any existing cache
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("Loading Llama model with memory optimizations...")

        # Initialize the cleaning model with aggressive memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better compatibility
            device_map="auto",  # Let transformers handle device mapping
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            use_cache=False,  # Disable KV cache to save memory
            attn_implementation="eager",  # Use eager attention instead of flash attention
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            use_fast=True,  # Use fast tokenizer
            padding_side="left"  # Set padding side
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move model to device if not already done by device_map
        if self.device != "auto":
            self.model = self.model.to(self.device)

        # Enable memory efficient features
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully on {self.device}")

        self.system_prompt = """
        You are a world class text pre-processor. Clean up this PDF text to make it suitable
        for a podcast transcript. Remove LaTeX, unnecessary newlines, and any fluff that
        wouldn't be useful in a podcast. Be aggressive in cleaning but preserve the key content.
        Start your response directly with the cleaned text.
        """

    def extract_text(self, pdf_path):
        """Extract raw text from PDF"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def create_chunks(self, text):
        """Split text into word-bounded chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def clean_chunk(self, chunk):
        """Clean a single chunk of text using the LLM with memory optimization"""
        try:
            # Limit input length to prevent memory issues
            max_input_length = 512
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True,
                padding=False
            ).to(self.device)

            # Generate with strict memory limits
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced from 512
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disable KV cache
                    num_beams=1,  # Use greedy search instead of beam search
                )

            # Decode only the new tokens
            new_tokens = output[0][inputs['input_ids'].shape[1]:]
            cleaned_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean up memory after each generation
            del inputs, output, new_tokens
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return cleaned_text

        except Exception as e:
            print(f"Error processing chunk: {e}")
            # Fallback: return original chunk if processing fails
            return chunk

    def process_pdf(self, pdf_path):
        """Main processing pipeline with memory monitoring"""
        try:
            # Extract raw text
            print("Extracting text from PDF...")
            raw_text = self.extract_text(pdf_path)

            # Split into smaller chunks to reduce memory usage
            print("Creating text chunks...")
            chunks = self.create_chunks(raw_text)
            print(f"Created {len(chunks)} chunks")

            # Process each chunk with memory cleanup
            cleaned_chunks = []
            for i, chunk in enumerate(tqdm(chunks, desc="Cleaning text")):
                print(f"Processing chunk {i + 1}/{len(chunks)}")
                cleaned_chunk = self.clean_chunk(chunk)
                cleaned_chunks.append(cleaned_chunk)

                # Periodic memory cleanup
                if (i + 1) % 5 == 0:  # Every 5 chunks
                    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            # Combine cleaned chunks
            cleaned_text = " ".join(cleaned_chunks)

            # Final cleanup
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return cleaned_text

        except Exception as e:
            print(f"Error in process_pdf: {e}")
            # Fallback: return raw text if processing fails
            return self.extract_text(pdf_path)
