import os, cv2, numpy, cv2

from pathlib import Path
from tesserat.vision import image_utils, ocr_engine
from tesserat.extractors import field_extractors, nlp_utils
from . import  result_handler
from .config import logger

class Tesserat_OCR:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

    def process_document(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            return self._process_image(file_path)
        return {"error": "Unsupported file type."}

    def _process_pdf(self, pdf_path: str):
        images = image_utils.pdf_to_images(pdf_path)
        results = [self._process_image_data(numpy.array(img)) for img in images]
        combined = result_handler.combine_results(results)
        combined["file_path"] = pdf_path
        combined["total_pages"] = len(images)
        result_handler.save_result(combined, os.path.basename(pdf_path), self.output_dir)
        return combined

    def _process_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Image not readable."}
        result = self._process_image_data(image)
        result["file_path"] = image_path
        result_handler.save_result(result, os.path.basename(image_path), self.output_dir)
        return result

    def _process_image_data(self, image):
        ocr_result = ocr_engine.run_ocr(image)
        fields = field_extractors.extract_fields(ocr_result["text"])
        return {**ocr_result, "fields": fields}
