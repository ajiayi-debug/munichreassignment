import os
import json
import easyocr
import numpy as np
import re
from pathlib import Path
from pdf2image import convert_from_path
from typing import List, Dict
import argparse


class PDFOCRExtractor:
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        print(f"Initializing EasyOCR with languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR initialized successfully!")
        
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List:
        print(f"Converting PDF to images at {dpi} DPI: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} pages")
        return images
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}[:.]\d{2}\s*[AP]M', '', text)
        text = re.sub(r'The Project Gutenberg eBook of[^.]*\.', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([.,;:!?])', r'\1', text)
        text = re.sub(r'^\d+\s+', '', text)
        return text.strip()
    
    def extract_text_from_image(self, image) -> Dict:
        image_np = np.array(image)
        results = self.reader.readtext(image_np)
        extracted_data = {
            'full_text': '',
            'text_blocks': []
        }
        sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
        
        for bbox, text, confidence in sorted_results:
            cleaned_text = self.clean_text(text)
            if not cleaned_text or len(cleaned_text) < 2:
                continue
            bbox_list = [[float(x), float(y)] for x, y in bbox]
            extracted_data['text_blocks'].append({
                'text': cleaned_text,
                'bbox': bbox_list,
                'confidence': float(confidence)
            })
            extracted_data['full_text'] += cleaned_text + ' '
        
        extracted_data['full_text'] = self.clean_text(extracted_data['full_text'])
        
        return extracted_data
    
    def process_pdf(self, pdf_path: str, output_dir: str = 'extracted_text_easyOCR') -> Dict:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        images = self.pdf_to_images(pdf_path)
        
        all_pages = []
        full_document_text = ""
        
        for page_num, image in enumerate(images, start=1):
            print(f"\nProcessing page {page_num}/{len(images)}...")
            page_data = self.extract_text_from_image(image)
            page_data['page_number'] = page_num
            all_pages.append(page_data)
            full_document_text += f"\n\n--- Page {page_num} ---\n\n"
            full_document_text += page_data['full_text']
            print(f"Page {page_num}: Extracted {len(page_data['text_blocks'])} text blocks")
        
        output_data = {
            'source_pdf': str(pdf_path),
            'total_pages': len(images),
            'pages': all_pages,
            'full_document_text': full_document_text.strip()
        }
        
        json_output_path = output_path / f"{Path(pdf_path).stem}_extracted.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved JSON output: {json_output_path}")
        
        txt_output_path = output_path / f"{Path(pdf_path).stem}_extracted.txt"
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(full_document_text)
        print(f"✓ Saved text output: {txt_output_path}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF using EasyOCR')
    parser.add_argument('pdf_path', type=str, help='Path to PDF file')
    parser.add_argument('--output-dir', type=str, default='extracted_text_easyOCR', 
                        help='Output directory for extracted text')
    parser.add_argument('--dpi', type=int, default=300, 
                        help='DPI for PDF to image conversion (default: 300)')
    parser.add_argument('--no-gpu', action='store_true', 
                        help='Disable GPU acceleration')
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                        help='Languages for OCR (e.g., en, zh, fr)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return
    
    extractor = PDFOCRExtractor(
        languages=args.languages,
        gpu=not args.no_gpu
    )
    
    print(f"\n{'='*60}")
    print(f"Starting OCR extraction for: {args.pdf_path}")
    print(f"{'='*60}\n")
    
    result = extractor.process_pdf(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"Total pages processed: {result['total_pages']}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
