import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "extraction_pipeline_easyOCR"))
sys.path.insert(0, str(Path(__file__).parent / "qwen3_VL_extraction"))

from extraction_pipeline_easyOCR.ocr_extraction import PDFOCRExtractor
from extraction_pipeline_easyOCR.cleanup_text import clean_ocr_text
from qwen3_VL_extraction.cleanup_qwen import process_qwen_extraction
from combine_ocr_sources import setup_combined_vector_db


def run_easyocr_pipeline(
    pdf_path: str,
    output_dir: str = "extracted_text_easyOCR",
    skip_ocr: bool = False,
    skip_cleanup: bool = False
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    extracted_txt = output_path / f"{pdf_name}_extracted.txt"
    cleaned_txt = output_path / f"{pdf_name}_extracted_cleaned.txt"
    chunks_json = output_path / "chunks.json"
    
    print("\n" + "=" * 70)
    print("📄 EASYOCR PIPELINE")
    print("=" * 70)
    
    if not skip_ocr:
        print("\n🔍 Running EasyOCR extraction...")
        
        if not Path(pdf_path).exists():
            print(f"❌ Error: PDF not found: {pdf_path}")
            return None
        
        extractor = PDFOCRExtractor(languages=['en'], gpu=True)
        result = extractor.process_pdf(
            pdf_path=pdf_path,
            output_dir=str(output_path)
        )
        print(f"✓ EasyOCR: {len(result['pages'])} pages extracted")
    else:
        print("⏭️  Skipping EasyOCR extraction")
    
    if not skip_cleanup:
        print("\n🧹 Cleaning EasyOCR text...")
        
        if not extracted_txt.exists():
            print(f"❌ Error: {extracted_txt} not found")
            return None
        
        with open(extracted_txt, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        cleaned = clean_ocr_text(raw_text)
        
        with open(cleaned_txt, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f"✓ Cleaned: {len(raw_text)} → {len(cleaned)} chars")
    else:
        print("⏭️  Skipping EasyOCR cleanup")
    
    print("\n✂️  Chunking EasyOCR text...")
    from setup_vector_db import chunk_text, save_chunks
    
    if not cleaned_txt.exists():
        print(f"❌ Error: {cleaned_txt} not found")
        return None
    
    chunks = chunk_text(str(cleaned_txt), paragraphs_per_chunk=3, overlap_paragraphs=1)
    
    for chunk in chunks:
        chunk['source'] = 'easyocr'
    
    save_chunks(chunks, str(chunks_json))
    print(f"✓ Created {len(chunks)} EasyOCR chunks")
    
    return chunks_json


def run_qwen_pipeline(
    qwen_input: str = "qwen3_VL_extraction/full_extraction.txt",
    output_dir: str = "qwen3_VL_extraction",
    skip_cleanup: bool = False
) -> tuple:
    output_path = Path(output_dir)
    para_chunks_json = output_path / "chunks_paragraph.json"
    page_chunks_json = output_path / "chunks_page.json"
    
    print("\n" + "=" * 70)
    print("🤖 QWEN VL PIPELINE")
    print("=" * 70)
    
    if not Path(qwen_input).exists():
        print(f"❌ Error: Qwen extraction not found: {qwen_input}")
        print("   (Run Qwen extraction in Google Colab first)")
        return None, None
    
    if not skip_cleanup:
        print("\n🧹 Processing Qwen VL extraction...")
        para_chunks, page_chunks = process_qwen_extraction(
            qwen_input,
            output_dir,
            paragraphs_per_chunk=3,
            overlap_paragraphs=1
        )
        print(f"✓ Created {len(para_chunks)} paragraph chunks")
        print(f"✓ Created {len(page_chunks)} page chunks")
    else:
        print("⏭️  Using existing Qwen chunks")
    
    return para_chunks_json, page_chunks_json


def run_full_pipeline(
    pdf_path: str = None,
    easyocr_output: str = "extracted_text_easyOCR",
    qwen_input: str = "qwen3_VL_extraction/full_extraction.txt",
    qwen_output: str = "qwen3_VL_extraction",
    skip_easyocr: bool = False,
    skip_qwen: bool = False,
    skip_ocr: bool = False,
    skip_cleanup: bool = False
):
    print("=" * 70)
    print("🚀 MULTI-OCR RAG PIPELINE")
    print("=" * 70)
    print(f"EasyOCR: {'SKIP' if skip_easyocr else 'ENABLED'}")
    print(f"Qwen VL: {'SKIP' if skip_qwen else 'ENABLED'}")
    print("=" * 70)
    
    easyocr_chunks = None
    qwen_para_chunks = None
    qwen_page_chunks = None
    
    if not skip_easyocr:
        if pdf_path and Path(pdf_path).exists():
            easyocr_chunks = run_easyocr_pipeline(
                pdf_path=pdf_path,
                output_dir=easyocr_output,
                skip_ocr=skip_ocr,
                skip_cleanup=skip_cleanup
            )
        elif Path(f"{easyocr_output}/chunks.json").exists():
            print("\n📄 Using existing EasyOCR chunks")
            easyocr_chunks = Path(f"{easyocr_output}/chunks.json")
        else:
            print("\n⚠️  No PDF provided and no existing EasyOCR chunks found")
    
    if not skip_qwen:
        if Path(qwen_input).exists():
            qwen_para_chunks, qwen_page_chunks = run_qwen_pipeline(
                qwen_input=qwen_input,
                output_dir=qwen_output,
                skip_cleanup=skip_cleanup
            )
        elif Path(f"{qwen_output}/chunks_paragraph.json").exists():
            print("\n🤖 Using existing Qwen chunks")
            qwen_para_chunks = Path(f"{qwen_output}/chunks_paragraph.json")
            qwen_page_chunks = Path(f"{qwen_output}/chunks_page.json")
        else:
            print(f"\n⚠️  Qwen extraction not found: {qwen_input}")
    
    print("\n" + "=" * 70)
    print("📊 COMBINING INTO VECTOR DATABASE")
    print("=" * 70)
    
    easyocr_file = str(easyocr_chunks) if easyocr_chunks else None
    qwen_para_file = str(qwen_para_chunks) if qwen_para_chunks else None
    qwen_page_file = str(qwen_page_chunks) if qwen_page_chunks else None
    
    collection, embed_model, total = setup_combined_vector_db(
        easyocr_chunks_file=easyocr_file,
        qwen_para_chunks_file=qwen_para_file,
        qwen_page_chunks_file=qwen_page_file,
        reset=True
    )
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nTotal chunks indexed: {total}")
    print(f"  • EasyOCR (page):       {easyocr_chunks if easyocr_chunks else 'skipped'}")
    print(f"  • Qwen VL (paragraph):  {qwen_para_chunks if qwen_para_chunks else 'skipped'}")
    print(f"  • Qwen VL (page):       {qwen_page_chunks if qwen_page_chunks else 'skipped'}")
    print(f"\nVector DB: ./chroma_db")
    print("\nNext steps:")
    print("  → Run chatbot: streamlit run app.py")
    print("  → Query CLI:   python query_rag.py --answer-all")


def main():
    parser = argparse.ArgumentParser(
        description='Run the Multi-OCR RAG pipeline (EasyOCR + Qwen VL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with both OCR sources
  python main.py --pdf "extraction_pipeline_easyOCR/data/Principles of Public Health.pdf"

  # Use existing extractions only (no new OCR)
  python main.py --skip-ocr

  # Only EasyOCR (skip Qwen)
  python main.py --pdf path/to/file.pdf --skip-qwen

  # Only Qwen (skip EasyOCR)  
  python main.py --skip-easyocr

  # Rebuild vector DB from existing chunks
  python main.py --skip-ocr --skip-cleanup
"""
    )
    
    parser.add_argument('--pdf', type=str, default=None,
                        help='Path to PDF for EasyOCR extraction')
    parser.add_argument('--easyocr-output', type=str, default='extracted_text_easyOCR',
                        help='EasyOCR output directory')
    parser.add_argument('--qwen-input', type=str, default='qwen3_VL_extraction/full_extraction.txt',
                        help='Qwen extraction input file')
    parser.add_argument('--qwen-output', type=str, default='qwen3_VL_extraction',
                        help='Qwen output directory')
    parser.add_argument('--skip-easyocr', action='store_true',
                        help='Skip EasyOCR entirely')
    parser.add_argument('--skip-qwen', action='store_true',
                        help='Skip Qwen VL entirely')
    parser.add_argument('--skip-ocr', action='store_true',
                        help='Skip OCR extraction (use existing files)')
    parser.add_argument('--skip-cleanup', action='store_true',
                        help='Skip text cleanup (use existing chunks)')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        pdf_path=args.pdf,
        easyocr_output=args.easyocr_output,
        qwen_input=args.qwen_input,
        qwen_output=args.qwen_output,
        skip_easyocr=args.skip_easyocr,
        skip_qwen=args.skip_qwen,
        skip_ocr=args.skip_ocr,
        skip_cleanup=args.skip_cleanup
    )


if __name__ == "__main__":
    main()
