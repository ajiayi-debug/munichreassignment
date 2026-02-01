# Public Health RAG Pipeline

A multi-OCR RAG (Retrieval-Augmented Generation) system for querying the book "Principles of Public Health" by Thomas D. Tuttle (1910).

---

## Overall Approach and System Design

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. **Extracts** text from a historical PDF using multiple OCR methods
2. **Chunks** the extracted text using different strategies for comparison
3. **Embeds** chunks using sentence transformers and stores in a vector database
4. **Retrieves** relevant chunks via semantic similarity search
5. **Generates** answers using an LLM (Gemini) grounded in the retrieved context

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   PDF Input     │     │  Google Colab   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│    EasyOCR      │     │   Qwen3 VL      │
│  (Local, CPU)   │     │  (GPU, T4/L4)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Page Chunks    │     │ Page + Paragraph│
│    (~118)       │     │  Chunks (~561)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────┐
         │     ChromaDB        │
         │  (all-mpnet-base-v2)│
         │   Cosine Similarity │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Streamlit Chatbot  │
         │  + Gemini 2.5 Flash │
         └─────────────────────┘
```

---

## OCR Methods and Preprocessing

### EasyOCR (Local)

- **Library**: EasyOCR with English language model
- **Input**: PDF converted to images at **300 DPI** using pdf2image/poppler
- **Processing**:
  1. Convert each PDF page to a high-resolution image
  2. Run OCR to extract text blocks with bounding boxes and confidence scores
  3. Sort text blocks by position (top-to-bottom, left-to-right reading order)
  4. Clean text: remove URLs, timestamps, Project Gutenberg headers, normalize whitespace
- **Output**: Full text per page with metadata
- **Limitation**: Cannot extract information from images/figures—only recognizes text. Any diagrams, charts, or illustrations are completely ignored.

### Qwen3 VL (Vision-Language Model on Colab)

- **Model**: `Qwen/Qwen3-VL-8B-Instruct` via Hugging Face Transformers
- **Input**: PDF pages converted to images
- **Processing**:
  1. Each page image is passed to the VLM with a structured extraction prompt
  2. Model extracts text, tables (as markdown), headings, lists, formulas, and image descriptions
  3. Output is structured in markdown format preserving document hierarchy
- **Advantages**: Better handling of tables, complex layouts, and contextual understanding
- **Image Handling**: The extraction prompt explicitly asks the model to describe any images and explain their relevance to the surrounding text. This captures visual information that EasyOCR would miss.

> **Note**: In this particular PDF (Principles of Public Health, 1910), most images and figures are already described in the accompanying text, so the impact of EasyOCR's image blindness is minimal. However, for PDFs with standalone diagrams or charts without textual descriptions, Qwen VL would capture significantly more information.

---

## Chunking and Embedding Strategy

### Chunking Approaches

| Source | Strategy | Description | Chunk Count |
|--------|----------|-------------|-------------|
| EasyOCR | **Page-based** | One chunk per page, preserves page boundaries | ~118 |
| Qwen VL (paragraph) | **Sliding window** | 3 paragraphs per chunk with 1 paragraph overlap | ~442 |
| Qwen VL (page) | **Page-based** | One chunk per page for comparison | ~119 |

### Why Multiple Strategies?

- **Page-based**: Preserves full context within a page, good for questions about specific topics
- **Paragraph-based**: Finer granularity, better for precise fact retrieval, overlap prevents losing context at boundaries

### Embedding

- **Model**: `all-mpnet-base-v2` (SentenceTransformers)
  - 768-dimensional dense vectors
  - Trained on 1B+ sentence pairs
  - Strong performance on semantic similarity tasks
- **Device**: MPS (Apple Silicon) or CPU
- **Storage**: ChromaDB with **cosine similarity** metric

---

## Retrieval and LLM Prompt Construction

### Retrieval

1. User query is embedded using the same `all-mpnet-base-v2` model
2. ChromaDB performs approximate nearest neighbor search
3. Top-K chunks (default: 5) are retrieved based on cosine similarity
4. Chunks can be filtered by OCR source (EasyOCR, Qwen VL paragraph, Qwen VL page, or all)

### Confidence Scoring

- **Retrieval Confidence**: Average similarity score of top-K retrieved chunks (0-100%)
- **LLM Confidence**: Self-reported confidence extracted from the model's response (1-10 scale)

### LLM Prompt

The prompt is designed to:
- Ground answers strictly in the retrieved excerpts
- Prevent hallucination by explicitly forbidding external information
- Encourage uncertainty highlighting with `[UNCERTAIN: reason]` markers
- Request self-assessment confidence scores

---

## Overview

This project extracts text from a PDF using two OCR methods:
- **EasyOCR** - Local OCR extraction (runs on your machine)
- **Qwen3 VL** - Vision-language model extraction (runs on Google Colab with GPU)

The extracted text is chunked, embedded, and stored in ChromaDB for semantic search. A Streamlit chatbot interface allows querying the book with answers generated by Gemini.

## Requirements

### API Keys

You need the following API keys:

| Key | Purpose | Where to Get |
|-----|---------|--------------|
| `Gemini` | Gemini API for answer generation | [Google AI Studio](https://aistudio.google.com/apikey) |
| `HF_TOKEN` | Hugging Face token for Qwen3 VL model (Colab only) | [Hugging Face Tokens](https://huggingface.co/settings/tokens) |

Create a `.env` file in the project root:
```
Gemini=your_gemini_api_key_here
```

### System Dependencies

- Python 3.9+
- Poppler (for PDF to image conversion)

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py                      # Streamlit chatbot interface
├── main.py                     # Main pipeline orchestrator
├── combine_ocr_sources.py      # Combines OCR sources into ChromaDB
├── setup_vector_db.py          # Vector database setup utilities
├── extraction_pipeline_easyOCR/
│   ├── ocr_extraction.py       # EasyOCR extraction
│   ├── cleanup_text.py         # Text cleanup utilities
│   └── data/                   # Place your PDF here
├── qwen3_VL_extraction/
│   ├── extraction.ipynb        # Colab notebook for Qwen3 VL
│   ├── cleanup_qwen.py         # Qwen output processing
│   └── full_extraction.txt     # Qwen extraction output
└── chroma_db/                  # Vector database storage
```

## Running the Pipeline

### Option 1: Full Pipeline (EasyOCR + Qwen VL)

**Step 1: Run Qwen3 VL extraction on Google Colab**

1. Open `qwen3_VL_extraction/extraction.ipynb` in Google Colab
2. Upload your PDF to Google Drive root (`MyDrive/`)
3. Add your Hugging Face token to Colab secrets (name it `HF_TOKEN`)
4. Run all cells
5. Download the output `full_extraction.txt` to place in local `qwen3_VL_extraction/`

**Step 2: Run the main pipeline**

```bash
python main.py --pdf "extraction_pipeline_easyOCR/data/Principles of Public Health.pdf"
```

This will:
- Run EasyOCR on the PDF (OCR then chunking by page)
- Process the Qwen VL extraction (chunking by page and paragraph)
- Combine both sources into ChromaDB

### Option 2: EasyOCR Only

```bash
python main.py --pdf "extraction_pipeline_easyOCR/data/Principles of Public Health.pdf" --skip-qwen
```

### Option 3: Qwen VL Only

```bash
python main.py --skip-easyocr
```

### Option 4: Rebuild Vector DB from Existing Chunks

```bash
python main.py --skip-ocr --skip-cleanup
```

## Running the Streamlit Chatbot

```bash
streamlit run app.py
```

The chatbot provides:
- **OCR Source Selection** - Choose between EasyOCR, Qwen VL (paragraph), Qwen VL (page), or compare all
- **Confidence Scoring** - Shows retrieval and LLM confidence
- **Uncertainty Highlighting** - Marks uncertain parts of answers
- **Export** - Download Q&A history as JSON

### Screenshots of Answers

![Question 1](images/Screenshot%202026-02-01%20at%204.23.43%20PM.png)

![Question 2](images/Screenshot%202026-02-01%20at%204.23.52%20PM.png)

![Question 3](images/Screenshot%202026-02-01%20at%204.24.01%20PM.png)

### Chatbot Prompt

```
prompt = f"""You are an expert assistant helping to answer questions about the book "Principles of Public Health".

Based ONLY on the following excerpts from the book, answer the question below. Your answer should be:
- Clear and concise
- Faithful to the content in the book excerpts
- Grounded in the evidence provided
- Do NOT add information not present in the excerpts

Book Excerpts:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY information from the excerpts above
2. Be specific and cite relevant details from the text
3. If the excerpts don't contain enough information to fully answer, say so
4. Keep your answer focused and well-organized
5. **UNCERTAINTY HIGHLIGHTING**: Mark any uncertain or inferred parts with [UNCERTAIN: reason]. For example: "The book suggests X [UNCERTAIN: text is partially unclear]" or "This likely means Y [UNCERTAIN: inferred from context]"
6. At the END of your answer, add a self-assessment line: "CONFIDENCE: X/10" where X is your confidence (1-10) that your answer is correct and well-supported by the excerpts.

Answer:"""
```

## Google Colab Notebook (Qwen3 VL)

The notebook `qwen3_VL_extraction/extraction.ipynb` runs the Qwen3-VL-8B-Instruct model for high-quality OCR extraction.

### Setup Requirements

1. **GPU Runtime**: Change runtime to L4 GPU (Runtime → Change runtime type → L4 GPU)
2. **Hugging Face Token**: Add to Colab secrets as `HF_TOKEN`
3. **PDF File**: Upload to Google Drive root

### What the Notebook Does

1. Mounts Google Drive
2. Installs dependencies (poppler, transformers, pdf2image)
3. Loads Qwen3-VL-8B-Instruct model
4. Converts PDF to images
5. Extracts text from each page with structured formatting
6. Saves output to `full_extraction.txt`

**You will need to download `full_extraction.txt` and place it in /qwen3_VL_extraction**

### Extraction Prompt

The notebook extracts:
- All text content in reading order
- Tables (as markdown)
- Headings and hierarchy
- Lists and bullet points
- Formulas/equations
- Image descriptions with context

```
prompt = """Please extract all the text content from this PDF page image.

Extract the following information:
1. All text content in reading order
2. Tables (format as markdown tables)
3. Headings and their hierarchy
4. Lists and bullet points
5. Any formulas or equations
6. Any Images, describe the images and what you can extract from them in terms of relevancy to the text

Organize the output in a clear, structured format that maintains the original document's layout and hierarchy. Return in markdown format"""
```

## Chunking Strategies

The system uses different chunking strategies:

| Source | Strategy | Chunks |
|--------|----------|--------|
| EasyOCR | Page-based | ~118 |
| Qwen VL (paragraph) | 3 paragraphs + 1 overlap | ~442 |
| Qwen VL (page) | One chunk per page | ~119 |

## Troubleshooting

### SSL Certificate Errors
The app includes SSL certificate fixes. If issues persist:
```bash
pip install --upgrade certifi
```

### ChromaDB Errors
Delete and rebuild the database:
```bash
rm -rf chroma_db/
python main.py --skip-ocr --skip-cleanup
```

### Colab Session Timeout
The Qwen VL extraction takes 12 hours for a 120-page PDF. Keep the Colab tab active or use Colab Pro.

---

## Assumptions, Limitations, and Known Failure Cases

### Assumptions

1. **English text only**: EasyOCR is configured for English; the book is in English
2. **Standard PDF format**: PDFs with unusual encodings or DRM may fail
3. **Sufficient disk space**: ChromaDB and extracted images require ~500MB+
4. **Network access**: Required for Gemini API calls and Hugging Face model downloads

### Limitations

1. **OCR Quality on Historical Text**
   - Old typography, faded text, and unusual fonts reduce OCR accuracy
   - EasyOCR may struggle with decorative chapter headers and marginalia
   - Qwen VL performs better but is slower and requires GPU

2. **Chunking Trade-offs**
   - Page-based chunking may split related content across chunk boundaries
   - Paragraph chunking may lose broader context needed for complex questions

3. **Retrieval Limitations**
   - Semantic search may miss relevant chunks if query phrasing differs significantly from source text
   - Fixed top-K retrieval may include irrelevant chunks or miss relevant ones

4. **LLM Constraints**
   - Gemini may occasionally hallucinate despite grounding instructions
   - Context window limits the amount of retrieved text that can be included

### Known Failure Cases

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Blank or garbled text | Low-quality scan or unusual font | Try Qwen VL instead of EasyOCR |
| Missing table data | EasyOCR doesn't preserve table structure | Use Qwen VL (extracts as markdown tables) |
| Irrelevant chunks retrieved | Query semantically distant from source text | Rephrase query or increase top-K |
| "I don't have enough information" | Relevant content not in top-K chunks | Try different OCR source or increase top-K |
| SSL certificate errors | macOS certificate issues | Run `pip install --upgrade certifi` |
| ChromaDB dimension mismatch | Switching embedding models | Delete `chroma_db/` and rebuild |

### Performance Notes

- **EasyOCR**: ~2-3 minutes for 120-page PDF (CPU)
- **Qwen VL**: ~12 hours on Colab T4 GPU
- **Embedding**: ~30 seconds for 600+ chunks
- **Query latency**: <2 seconds (retrieval + generation)
