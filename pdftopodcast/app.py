# app.py
import gradio as gr
import torch
from pathlib import Path
from processors.pdf_processor import PDFProcessor
from processors.transcript_writer import TranscriptWriter
from processors.podcast_rewriter import PodcastRewriter 
from processors.tts_generator import TTSGenerator
import tempfile
import os

class PodcastApp:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pdf_processor = PDFProcessor()
        self.transcript_writer = TranscriptWriter()
        self.podcast_rewriter = PodcastRewriter()
        self.tts_generator = TTSGenerator()
        
        # Create temp directory
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
    def process_pdf(self, pdf_file, progress=gr.Progress()):
        """Main processing pipeline"""
        try:
            progress(0, desc="Extracting text from PDF...")
            extracted_text = self.pdf_processor.process_pdf(pdf_file.name)
            
            progress(0.25, desc="Generating initial transcript...")
            transcript = self.transcript_writer.generate_transcript(extracted_text)
            
            progress(0.5, desc="Rewriting for podcast format...")
            podcast_script = self.podcast_rewriter.rewrite_transcript(transcript)
            
            progress(0.75, desc="Generating audio...")
            output_path = self.tts_generator.generate_audio(podcast_script)
            
            progress(1.0, desc="Done!")
            return (
                output_path,  # Audio file
                "Processing completed successfully! You can now play the generated podcast."
            )
            
        except Exception as e:
            return None, f"Error: {str(e)}"

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 📚 PDF to Podcast Converter
            Upload a PDF document and convert it into an engaging podcast conversation between two speakers!
            """)
            
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                    )
                    process_btn = gr.Button("Convert to Podcast", variant="primary")
                    
            with gr.Row():
                with gr.Column():
                    audio_output = gr.Audio(
                        label="Generated Podcast",
                        type="filepath"
                    )
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            # Handle the conversion process
            process_btn.click(
                fn=self.process_pdf,
                inputs=[pdf_input],
                outputs=[audio_output, status_output]
            )
            
            gr.Markdown("""
            ### How it works:
            1. **Upload PDF**: Choose any PDF document you want to convert
            2. **Processing**: The app will:
               - Extract and clean the text
               - Generate a natural conversation script
               - Create audio using different voices
            3. **Result**: Listen to your PDF as an engaging podcast!
            
            Note: Processing may take a few minutes depending on the PDF size.
            """)
            
        return interface

def main():
    app = PodcastApp()
    interface = app.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()