# Unified AI Document Processing Router
# Separates AI OCR and AI Classifier into modular, updatable units

import cv2
from pathlib import Path
from document_ocr import DocumentOCR  # your custom OCR-only module
from ai_ocr import AI_OCR             # your full OCR system with GPT-4 fallback
from document_classifier import DocumentClassifier  # your classification model

class DocumentRouter:
    def __init__(self):
        self.ocr_engine = DocumentOCR()            # Basic OCR for text preview
        self.full_ocr = AI_OCR()                   # Full OCR with AI features
        self.classifier = DocumentClassifier()     # Predict document type

    def auto_process(self, file_path: str) -> dict:
        """
        Automatically determines how to handle a document:
        - Uses light OCR to preview text
        - Classifies the document type
        - Runs full OCR if needed
        - Returns result as dictionary
        """
        preview_text = self.ocr_engine.extract_text(file_path)
        doc_type = self.classifier.classify(preview_text)

        result = {
            "file_name": Path(file_path).name,
            "document_type": doc_type,
            "text_preview": preview_text[:300]
        }

        if doc_type in [
            'invoice', 'receipt', 'bill', 'vendor_statement', 'payroll',
            'bank_statement', 'credit_card', 't4', 't2', 'purchase_order', 'reimbursement'
        ]:
            image = cv2.imread(file_path)
            detailed_result = self.full_ocr.process_image_data(image, file_path)
            result.update(detailed_result)
        else:
            result['message'] = "No advanced OCR applied (not required)."

        return result

# Example usage:
# router = DocumentRouter()
# result = router.auto_process("sample_documents/bank_statement.png")
# print(result)
