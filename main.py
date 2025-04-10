import os, argparse
from tesserat.processor import Tesserat_OCR

def tesserat():
    parser = argparse.ArgumentParser(description="Run AI OCR on a directory of documents.")
    parser.add_argument("input_dir", help="Path to the input directory containing documents")
    args = parser.parse_args()

    processor = Tesserat_OCR()
    for file in os.listdir(args.input_dir):
        processor.process_document(os.path.join(args.input_dir, file))

def openai():
    pass

if __name__ == "__main__":
    tesserat()
    
    input("AI starting... ")
    
    openai()