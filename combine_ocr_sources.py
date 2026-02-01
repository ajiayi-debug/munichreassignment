import json
import torch
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse

DB_PATH = "./chroma_db"
COLLECTION_NAME = "public_health"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_chunks(chunks_file: str, source_name: str = None) -> list:
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    if source_name:
        for chunk in chunks:
            if 'source' not in chunk:
                chunk['source'] = source_name
    
    return chunks


def setup_combined_vector_db(
    easyocr_chunks_file: str = None,
    qwen_para_chunks_file: str = None,
    qwen_page_chunks_file: str = None,
    reset: bool = True
):
    all_chunks = []
    
    if easyocr_chunks_file and Path(easyocr_chunks_file).exists():
        easyocr_chunks = load_chunks(easyocr_chunks_file, "easyocr")
        print(f"📄 Loaded {len(easyocr_chunks)} chunks from EasyOCR (page-based)")
        all_chunks.extend(easyocr_chunks)
    
    if qwen_para_chunks_file and Path(qwen_para_chunks_file).exists():
        qwen_para_chunks = load_chunks(qwen_para_chunks_file, "qwen_vl_paragraph")
        print(f"🤖 Loaded {len(qwen_para_chunks)} chunks from Qwen VL (paragraph-based)")
        all_chunks.extend(qwen_para_chunks)
    
    if qwen_page_chunks_file and Path(qwen_page_chunks_file).exists():
        qwen_page_chunks = load_chunks(qwen_page_chunks_file, "qwen_vl_page")
        print(f"📑 Loaded {len(qwen_page_chunks)} chunks from Qwen VL (page-based)")
        all_chunks.extend(qwen_page_chunks)
    
    if not all_chunks:
        print("❌ No chunks loaded!")
        return None, None, 0
    
    print(f"\n📊 Total chunks: {len(all_chunks)}")
    
    print(f"\n🚀 Running on device: {DEVICE}")
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
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection: {COLLECTION_NAME} (cosine similarity)")
    
    print("\nEmbedding and indexing chunks...")
    batch_size = 32
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_chunks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = all_chunks[i:i+batch_size]
        
        documents = [c['text'] for c in batch]
        metadatas = [{
            "chunk_id": c['chunk_id'], 
            "page": c['page'],
            "source": c.get('source', 'unknown')
        } for c in batch]
        ids = [c['chunk_id'] for c in batch]
        
        embeddings = embed_model.encode(documents, show_progress_bar=False).tolist()
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"  Batch {batch_num}/{total_batches} indexed ({len(batch)} chunks)")
    
    print(f"\n✓ Successfully indexed {len(all_chunks)} chunks in ChromaDB")
    
    return collection, embed_model, len(all_chunks)


def main():
    parser = argparse.ArgumentParser(description='Combine OCR sources into vector DB')
    parser.add_argument('--easyocr', type=str, default='extracted_text_easyOCR/chunks.json',
                        help='Path to EasyOCR chunks JSON')
    parser.add_argument('--qwen-para', type=str, default='qwen3_VL_extraction/chunks_paragraph.json',
                        help='Path to Qwen paragraph chunks JSON')
    parser.add_argument('--qwen-page', type=str, default='qwen3_VL_extraction/chunks_page.json',
                        help='Path to Qwen page chunks JSON')
    parser.add_argument('--no-easyocr', action='store_true',
                        help='Skip EasyOCR chunks')
    parser.add_argument('--no-qwen-para', action='store_true',
                        help='Skip Qwen paragraph chunks')
    parser.add_argument('--no-qwen-page', action='store_true',
                        help='Skip Qwen page chunks')
    parser.add_argument('--no-reset', action='store_true',
                        help='Do not reset the collection')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMBINING OCR SOURCES INTO VECTOR DATABASE")
    print("=" * 70)
    
    easyocr_file = None if args.no_easyocr else args.easyocr
    qwen_para_file = None if args.no_qwen_para else args.qwen_para
    qwen_page_file = None if args.no_qwen_page else args.qwen_page
    
    collection, embed_model, total = setup_combined_vector_db(
        easyocr_chunks_file=easyocr_file,
        qwen_para_chunks_file=qwen_para_file,
        qwen_page_chunks_file=qwen_page_file,
        reset=not args.no_reset
    )

    print(f"Total chunks indexed: {total}")
    print(f"Database path: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print("\nRun the chatbot: streamlit run app.py")


if __name__ == "__main__":
    main()
