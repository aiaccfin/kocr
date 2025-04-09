# from kevin email. Sent: April 7, 2025 11:30 AM

import os, re, json, cv2, numpy, pytesseract, spacy, logging, argparse
from pathlib import Path
from pdf2image import convert_from_path
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

class AI_OCR:
    def __init__(self, tesseract_path: Optional[str] = None, output_dir: str = "output"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self._check_tesseract()

    def _check_tesseract(self):
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except:
            logger.error("Tesseract not found or misconfigured.")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            return self._process_image(file_path)
        else:
            return {"error": "Unsupported file type."}

    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        images = convert_from_path(pdf_path)
        results = [self._process_image_data(numpy.array(img)) for img in images]
        combined = self._combine_results(results)
        combined["file_path"] = pdf_path
        combined["total_pages"] = len(images)
        self._save_result(combined, os.path.basename(pdf_path))
        return combined

    def _process_image(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Image not readable."}
        result = self._process_image_data(image)
        result["file_path"] = image_path
        self._save_result(result, os.path.basename(image_path))
        return result

    def _process_image_data(self, image: numpy.ndarray) -> Dict[str, Any]:
        processed = self._preprocess_image(image)
        text = pytesseract.image_to_string(processed)
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        confidences = [
            int(c) for c in data.get("conf", [])
            if isinstance(c, (int, float)) and c >= 0
        ]
        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

        fields = self._extract_fields(text)
        
        return {
            "text": text,
            "fields": fields,
            "ocr_data": data,
            "confidence": avg_conf
        }


    # def _process_image_data(self, image: numpy.ndarray) -> Dict[str, Any]:
    #     # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #     processed = self._preprocess_image(image)
    #     text = pytesseract.image_to_string(processed)
    #     data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    #     fields = self._extract_fields(text)
    #     return {
    #         "text": text,
    #         "fields": fields,
    #         "ocr_data": data
    #     }

    def _preprocess_image(self, image: numpy.ndarray) -> numpy.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        return denoised

    def _extract_fields(self, text: str) -> Dict[str, Any]:
        return {
            "vendor": self._extract_vendor(text),
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "line_items": self._extract_line_items(text)
        }

    def _extract_vendor(self, text: str) -> str:
        if not nlp:
            return text.split('\n')[0].strip()
        doc = nlp(text[:1000])
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        return orgs[0] if orgs else ""

    def _extract_dates(self, text: str) -> List[str]:
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        ]
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text))
        return dates

    def _extract_amounts(self, text: str) -> List[float]:
        matches = re.findall(r'\$\s*\d+[,.]?\d*', text)
        return [float(m.replace("$", "").replace(",", "")) for m in matches if m]

    def _extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        lines = text.split('\n')
        items = []
        for line in lines:
            if re.search(r'\$\s*\d+\.\d{2}', line):
                items.append({"description": line.strip()})
        return items

    # def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     text = "\n".join([r["text"] for r in results])
    #     all_items = sum([r["fields"].get("line_items", []) for r in results], [])
    #     return {
    #         "text": text,
    #         "fields": {
    #             "vendor": results[0]["fields"].get("vendor", ""),
    #             "dates": sum([r["fields"].get("dates", []) for r in results], []),
    #             "amounts": sum([r["fields"].get("amounts", []) for r in results], []),
    #             "line_items": all_items
    #         }
    #     }


    def _combine_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = "\n".join([r["text"] for r in results])
        all_items = sum([r["fields"].get("line_items", []) for r in results], [])
        all_dates = sum([r["fields"].get("dates", []) for r in results], [])
        all_amounts = sum([r["fields"].get("amounts", []) for r in results], [])
        confidences = [r.get("confidence", 0.0) for r in results]

        avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

        return {
            "text": text,  # raw text (with \n)
            "prett1y_text": text.strip().splitlines(),
            "fields": {
                "vendor": results[0]["fields"].get("vendor", ""),
                "dates": all_dates,
                "amounts": all_amounts,
                "line_items": all_items
            },
            "confidence": avg_conf
        }
    
    def _save_result(self, result: Dict[str, Any], filename: str):
        out_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_result.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved result to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI OCR on a directory of documents.")
    parser.add_argument("input_dir", help="Path to the input directory containing documents")
    args = parser.parse_args()

    print("Processing directory:", args.input_dir)
    processor = AI_OCR()
    for file in os.listdir(args.input_dir):
        processor.process_document(os.path.join(args.input_dir, file))
