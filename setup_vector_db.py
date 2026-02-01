import json
import torch
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse

DB_PATH = "./chroma_db"
COLLECTION_NAME = "public_health"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def chunk_text(extracted_txt_path: str, paragraphs_per_chunk: int = 3, overlap_paragraphs: int = 1) -> list:
    print(f"Loading extracted text from: {extracted_txt_path}")
    
    with open(extracted_txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Split by page markers
    import re
    page_pattern = r'---Page (\d+) ---'
    
    # Find all page markers and their positions
    page_splits = list(re.finditer(page_pattern, full_text))
    
    chunks = []
    
    # Process each page
    for idx, match in enumerate(page_splits):
        page_num = int(match.group(1))
        
        # Get text from this page marker to the next (or end of file)
        start_pos = match.end()
        if idx + 1 < len(page_splits):
            end_pos = page_splits[idx + 1].start()
        else:
            end_pos = len(full_text)
        
        text = full_text[start_pos:end_pos].strip()
        
        # Skip empty pages
        if not text or len(text.strip()) < 10:
            continue
        
        # Split into paragraphs (double newline or single newline with significant spacing)
        paragraphs = re.split(r'\n\s*\n|\n(?=[A-Z])', text)
        
        # Clean and filter paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
        
        # Skip if no valid paragraphs
        if not paragraphs:
            continue
        
        # Sliding window chunking by paragraphs
        step = paragraphs_per_chunk - overlap_paragraphs
        if step < 1:
            step = 1  # Ensure we make progress
        
        chunk_index = 0
        for i in range(0, len(paragraphs), step):
            chunk_paragraphs = paragraphs[i:i + paragraphs_per_chunk]
            
            # Skip very small chunks
            if len(chunk_paragraphs) == 0:
                continue
            
            chunk_text = "\n\n".join(chunk_paragraphs)
            
            # Skip if chunk is too short
            if len(chunk_text) < 100:
                continue
            
            # Create a unique ID
            chunk_id = f"page_{page_num}_chunk_{chunk_index}"
            
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": chunk_text
            })
            
            chunk_index += 1
    
    print(f"✓ Created {len(chunks)} chunks from {len(page_splits)} pages")
    print(f"  Using {paragraphs_per_chunk} paragraphs per chunk with {overlap_paragraphs} paragraph overlap")
    return chunks


def setup_vector_db(chunks: list, reset: bool = True):
    print(f"🚀 Running on device: {DEVICE}")
    
    print("Loading embedding model (all-mpnet-base-v2)...")
    embed_model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)

    print(f"Initializing ChromaDB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass
    
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Created collection: {COLLECTION_NAME}")

    print("Embedding and indexing chunks...")
    
    batch_size = 32
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = chunks[i:i+batch_size]
        
        documents = [c['text'] for c in batch]
        metadatas = [{"chunk_id": c['chunk_id'], "page": c['page']} for c in batch]
        ids = [c['chunk_id'] for c in batch]
        
        embeddings = embed_model.encode(documents, show_progress_bar=False).tolist()
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  Batch {batch_num}/{total_batches} indexed ({len(batch)} chunks)")
    
    print(f"✓ Successfully indexed {len(chunks)} chunks in ChromaDB")
    
    return collection, embed_model


def save_chunks(chunks: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(chunks)} chunks to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Chunk text and create vector database')
    parser.add_argument('txt_file', type=str,
                        help='Path to cleaned TXT file from OCR extraction')
    parser.add_argument('--paragraphs-per-chunk', type=int, default=3,
                        help='Number of paragraphs per chunk (default: 3)')
    parser.add_argument('--overlap-paragraphs', type=int, default=1,
                        help='Number of overlapping paragraphs (default: 1)')
    parser.add_argument('--save-chunks', type=str, 
                        default='extracted_text_easyOCR/chunks.json',
                        help='Path to save chunks JSON (default: extracted_text_easyOCR/chunks.json)')
    parser.add_argument('--no-reset', action='store_true',
                        help='Do not reset the database collection')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.txt_file).exists():
        print(f"Error: TXT file not found: {args.txt_file}")
        return
    
    print("="*60)
    print("CHUNKING AND VECTOR DATABASE SETUP")
    print("  Strategy: Paragraph-based sliding window")
    print("="*60)
    
    # Step 1: Chunk the text
    chunks = chunk_text(
        args.txt_file,
        paragraphs_per_chunk=args.paragraphs_per_chunk,
        overlap_paragraphs=args.overlap_paragraphs
    )
    
    # Step 2: Save chunks for reference
    save_chunks(chunks, args.save_chunks)
    
    # Step 3: Set up vector database
    collection, embed_model = setup_vector_db(
        chunks,
        reset=not args.no_reset
    )
    
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Database path: {DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Chunks saved to: {args.save_chunks}")
    print("\nYou can now use this database for RAG queries.")


if __name__ == "__main__":
    main()
