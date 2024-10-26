import PyPDF2
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the cleaning model
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-1B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-1B-Instruct"
        )
        
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
        """Clean a single chunk of text using the LLM"""
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": chunk}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False
        )
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=0.7,
                max_new_tokens=512
            )
            
        cleaned_text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )[len(prompt):].strip()
        
        return cleaned_text
    
    def process_pdf(self, pdf_path):
        """Main processing pipeline"""
        # Extract raw text
        raw_text = self.extract_text(pdf_path)
        
        # Split into chunks
        chunks = self.create_chunks(raw_text)
        
        # Process each chunk
        cleaned_chunks = []
        for chunk in tqdm(chunks, desc="Cleaning text"):
            cleaned_chunk = self.clean_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
            
        # Combine cleaned chunks
        cleaned_text = " ".join(cleaned_chunks)
        
        return cleaned_text