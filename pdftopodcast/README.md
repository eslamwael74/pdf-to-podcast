# PDF to Podcast Converter 🎙️

Convert any PDF document into an engaging podcast-style conversation between two speakers using state-of-the-art AI models.

## 🌟 Features

- **Smart Text Extraction**: Cleanly extracts and processes text from PDF documents
- **Natural Conversations**: Converts academic/technical content into engaging dialogues
- **Dual Voice Generation**: Uses two different voice models for a dynamic conversation
- **User-Friendly Interface**: Simple Gradio web interface for easy interaction
- **High-Quality Audio**: Professional-grade audio output with natural-sounding voices

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) for faster processing
- At least 16GB RAM
- Approximately 20GB free disk space for models

### Setup

1. Clone the repository:
```bash
git clone https://github.com/h9-tect/pdf-to-podcast
cd pdf-to-podcast
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:7860`)

3. Use the interface to:
   - Upload your PDF file
   - Click "Convert to Podcast"
   - Wait for processing
   - Download or play the generated podcast

## 📂 Project Structure

```
pdf-to-podcast/
├── app.py                 # Main application file
├── processors/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF text extraction and cleaning
│   ├── transcript_writer.py  # Initial transcript generation
│   ├── podcast_rewriter.py  # Script adaptation for TTS
│   └── tts_generator.py   # Audio generation
├── temp/                  # Temporary file storage
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## 🔧 Advanced Configuration

The application uses several AI models that can be configured:

### PDF Processing
- Model: Llama-3.1-1B-Instruct
- Chunk size: 1000 words (configurable in PDFProcessor)

### Transcript Generation
- Model: Llama-3.1-70B-Instruct
- Temperature: 1.0 (higher for more creative outputs)

### Voice Generation
- Speaker 1: Parler TTS Mini v1
- Speaker 2: Suno Bark
- Audio quality: 192kbps MP3

To modify these settings, edit the respective class initializations in the processor files.

## 🎯 Use Cases

This tool is perfect for:
- Converting academic papers into educational content
- Making technical documentation more accessible
- Creating engaging audio content from written materials
- Learning complex topics through dialogue
- Making content more accessible for audio learners

## ⚠️ Limitations

- Processing time depends on PDF length and GPU availability
- Very large PDFs (100+ pages) may need to be split
- Some technical symbols and equations may not translate perfectly
- GPU memory requirements can be high for longer documents

## 🔍 Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   - Reduce chunk size in PDFProcessor
   - Process smaller sections of the PDF
   - Use a GPU with more VRAM

2. **Slow Processing**
   - Ensure GPU is being utilized
   - Reduce PDF size
   - Consider using smaller models

3. **Audio Quality Issues**
   - Check available disk space
   - Ensure all models are properly downloaded
   - Try regenerating the problematic section

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Llama AI Models](https://ai.meta.com/llama/)
- [Parler TTS](https://github.com/parler-ai/parler-tts)
- [Suno Bark](https://github.com/suno-ai/bark)
- [Gradio Documentation](https://gradio.app/docs/)
