import re
import argparse
from pathlib import Path


def clean_ocr_text(text: str) -> str:
    text = re.sub(r'https?\s*[(\[]?[Il1l]w+w+[^\s]*', '', text)
    text = re.sub(r'hltps[^\s]*', '', text)
    text = re.sub(r'htm\?utm_source[^\s]*', '', text)
    text = re.sub(r'\d+/\d+', '', text)
    text = re.sub(r'The Project Gu[tl]enberg.*?by Thomas Tu[tl]+le\.', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\bt0\b', 'to', text)
    text = re.sub(r'\b0f\b', 'of', text)
    text = re.sub(r'\b0n\b', 'on', text)
    text = re.sub(r'\b0r\b', 'or', text)
    text = re.sub(r'\bYOu\b', 'you', text)
    text = re.sub(r'\bYOur\b', 'your', text)
    text = re.sub(r'\bs0\b', 'so', text)
    text = re.sub(r'\bd0\b', 'do', text)
    text = re.sub(r'\bg0\b', 'go', text)
    text = re.sub(r'\bn0\b', 'no', text)
    text = re.sub(r'\btO0\b', 'too', text)
    text = re.sub(r'\bbcen\b', 'been', text)
    text = re.sub(r'\bbam\b', 'barn', text)
    text = re.sub(r'\bcOw\b', 'cow', text)
    text = re.sub(r'\bcOws\b', 'cows', text)
    text = re.sub(r'\bCOW\b', 'cow', text)
    text = re.sub(r'\bbotlle\b', 'bottle', text)
    text = re.sub(r'\bpublie\b', 'public', text)
    text = re.sub(r'\bgrOw\b', 'grow', text)
    text = re.sub(r'\byOu\b', 'you', text)
    
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s*[\[\]]\s*', ' ', text)
    text = re.sub(r';(\s+[a-z])', r',\1', text)
    text = re.sub(r'\s+\d+/\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'CHAPTER\s*J', 'CHAPTER', text)
    text = re.sub(r'CHAPTERJ', 'CHAPTER', text)
    text = re.sub(r'\bFiG\b', 'Fig', text)
    text = re.sub(r'\bFIG\b', 'Fig', text)
    text = re.sub(r'-\s+', '-', text)
    
    return text.strip()


def process_file(input_file: str, output_file: str = None):
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    print(f"Reading file: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Cleaning text...")
    cleaned_text = clean_ocr_text(text)
    
    # Determine output path
    if output_file:
        output_path = Path(output_file)
    else:
        # Create new filename with "_cleaned" suffix
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    print(f"Writing cleaned text to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print("✓ Cleaning complete!")
    print(f"Original length: {len(text)} characters")
    print(f"Cleaned length: {len(cleaned_text)} characters")


def main():
    parser = argparse.ArgumentParser(description='Clean up OCR-extracted text')
    parser.add_argument('input_file', type=str, help='Path to input text file')
    parser.add_argument('--output', '-o', type=str, help='Path to output file (default: input_cleaned.txt)')
    
    args = parser.parse_args()
    
    process_file(args.input_file, args.output)


if __name__ == "__main__":
    main()
