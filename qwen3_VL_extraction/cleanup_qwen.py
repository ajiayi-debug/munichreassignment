import re
import json
import argparse
from pathlib import Path


def clean_qwen_text(text: str) -> str:
    text = re.sub(r'```markdown\n?', '', text)
    text = re.sub(r'```\n?', '', text)
    text = re.sub(r'1/21/26,?\s*2:50\s*PM\s*The Project Gutenberg eBook of Principles of Public Health,?\s*by Thomas Tuttle\.?\s*\n?', '', text)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'<!--\s*Image[^>]*-->', '', text)
    text = re.sub(r'<!--\s*Footnotes?\s*-->', '', text)
    text = re.sub(r'<!--[^>]*-->', '', text)
    text = re.sub(r'\n\d+/123\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK \d+\s*\*{3}', '', text)
    text = re.sub(r'\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK \d+\s*\*{3}', '', text)
    text = re.sub(r'\[Pg\s+[ivxlcdm\d]+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\[\d+\]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\[\d+\]\n', '\n', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s+\n', '\n\n', text)
    return text.strip()


def parse_qwen_pages(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by page markers
    page_pattern = r'={60,}\nPAGE (\d+)\n={60,}'
    
    # Find all page markers
    matches = list(re.finditer(page_pattern, content))
    
    pages = []
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        
        # Find end (next page or end of file)
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(content)
        
        page_text = content[start:end].strip()
        
        # Clean the page text
        cleaned_text = clean_qwen_text(page_text)
        
        if cleaned_text and len(cleaned_text) > 20:
            pages.append((page_num, cleaned_text))
    
    return pages


def chunk_by_paragraph(pages: list, paragraphs_per_chunk: int = 3, overlap_paragraphs: int = 1) -> list:
    chunks = []
    
    for page_num, text in pages:
        # Split into paragraphs (double newline or markdown headers)
        # Keep headers attached to their content
        paragraphs = re.split(r'\n\n+', text)
        
        # Clean and filter paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
        
        if not paragraphs:
            continue
        
        # Sliding window
        step = max(1, paragraphs_per_chunk - overlap_paragraphs)
        chunk_index = 0
        
        for i in range(0, len(paragraphs), step):
            chunk_paragraphs = paragraphs[i:i + paragraphs_per_chunk]
            
            if not chunk_paragraphs:
                continue
            
            chunk_text = "\n\n".join(chunk_paragraphs)
            
            # Skip if too short
            if len(chunk_text) < 100:
                continue
            
            chunk_id = f"qwen_para_page{page_num}_chunk{chunk_index}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": chunk_text,
                "source": "qwen_vl_paragraph"
            })
            
            chunk_index += 1
    
    return chunks


def chunk_by_page(pages: list) -> list:
    chunks = []
    
    for page_num, text in pages:
        # Skip if too short
        if len(text.strip()) < 50:
            continue
        
        chunk_id = f"qwen_page{page_num}"
        
        chunks.append({
            "chunk_id": chunk_id,
            "page": page_num,
            "text": text.strip(),
            "source": "qwen_vl_page"
        })
    
    return chunks


def process_qwen_extraction(
    input_file: str,
    output_dir: str = "qwen3_VL_extraction",
    paragraphs_per_chunk: int = 3,
    overlap_paragraphs: int = 1
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🤖 QWEN VL EXTRACTION PROCESSING")
    print("=" * 70)
    
    # Step 1: Parse pages
    print("\n Parsing pages...")
    pages = parse_qwen_pages(input_file)
    print(f"✓ Found {len(pages)} pages")
    
    # Step 2: Save cleaned text
    cleaned_file = output_path / "cleaned_extraction.txt"
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        for page_num, text in pages:
            f.write(f"---Page {page_num} ---\n")
            f.write(text)
            f.write("\n\n")
    print(f"✓ Saved cleaned text: {cleaned_file}")
    
    # Step 3a: Chunk by paragraph
    print(f"\n Chunking by PARAGRAPH ({paragraphs_per_chunk} paragraphs, {overlap_paragraphs} overlap)...")
    para_chunks = chunk_by_paragraph(pages, paragraphs_per_chunk, overlap_paragraphs)
    print(f"✓ Created {len(para_chunks)} paragraph chunks")
    
    # Save paragraph chunks
    para_chunks_file = output_path / "chunks_paragraph.json"
    with open(para_chunks_file, 'w', encoding='utf-8') as f:
        json.dump(para_chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {para_chunks_file}")
    
    # Also save as chunks.json for backward compatibility
    chunks_file = output_path / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(para_chunks, f, indent=2, ensure_ascii=False)
    
    # Step 3b: Chunk by page
    print("\n Chunking by PAGE (one chunk per page)...")
    page_chunks = chunk_by_page(pages)
    print(f"✓ Created {len(page_chunks)} page chunks")
    
    # Save page chunks
    page_chunks_file = output_path / "chunks_page.json"
    with open(page_chunks_file, 'w', encoding='utf-8') as f:
        json.dump(page_chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {page_chunks_file}")
    
    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)
    print(f"  Paragraph chunks: {len(para_chunks)} (chunks_paragraph.json)")
    print(f"  Page chunks:      {len(page_chunks)} (chunks_page.json)")
    
    return para_chunks, page_chunks


def main():
    parser = argparse.ArgumentParser(description='Process Qwen VL OCR output')
    parser.add_argument('--input', type=str, default='qwen3_VL_extraction/full_extraction.txt',
                        help='Input file path')
    parser.add_argument('--output-dir', type=str, default='qwen3_VL_extraction',
                        help='Output directory')
    parser.add_argument('--paragraphs', type=int, default=3,
                        help='Paragraphs per chunk')
    parser.add_argument('--overlap', type=int, default=1,
                        help='Overlap paragraphs')
    
    args = parser.parse_args()
    
    para_chunks, page_chunks = process_qwen_extraction(
        args.input,
        args.output_dir,
        args.paragraphs,
        args.overlap
    )
    
    print("\n✅ Processing complete!")
    print(f"   - Paragraph chunks: {len(para_chunks)}")
    print(f"   - Page chunks: {len(page_chunks)}")


if __name__ == "__main__":
    main()
