import os, argparse
from core.processor import AI_OCR

def main():
    parser = argparse.ArgumentParser(description="Run AI OCR on a directory of documents.")
    parser.add_argument("input_dir", help="Path to the input directory containing documents")
    args = parser.parse_args()

    processor = AI_OCR()
    for file in os.listdir(args.input_dir):
        processor.process_document(os.path.join(args.input_dir, file))

if __name__ == "__main__":
    main()
