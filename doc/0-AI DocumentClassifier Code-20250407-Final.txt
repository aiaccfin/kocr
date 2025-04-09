# Document Classification Module
# Clean and structured version of 'Classify Module-20250407-T'

import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

class DocumentClassifier:
    def __init__(self, model_path: str = "model/document_classifier.pkl"):
        self.model_path = model_path
        self.rules = self._load_rules()
        self.model = self._load_model()

    def _load_rules(self) -> List[Tuple[str, List[str]]]:
        return [
            ("invoice", ["invoice", "inv no", "invoice number", "amount due"]),
            ("bill", ["utility", "gas", "hydro", "billing date"]),
            ("receipt", ["receipt", "cashier", "thank you", "change"]),
            ("invoice_receipt", ["invoice", "receipt"]),
            ("bill_receipt", ["bill", "receipt"]),
            ("vendor_statement", ["vendor", "balance due", "statement"]),
            ("bank_statement", ["bank statement", "transactions", "opening balance"]),
            ("credit_card", ["credit card", "payment due", "credit limit"]),
            ("payroll", ["pay period", "net pay", "gross pay", "deductions"]),
            ("t4", ["t4", "employment income", "box 14"]),
            ("t2", ["t2", "corporation income", "tax return"]),
            ("financial_statement", ["balance sheet", "income statement", "financial position"]),
            ("sales_order", ["sales order", "so no", "customer"]),
            ("purchase_order", ["purchase order", "po no", "supplier"]),
            ("reimbursement", ["reimbursement", "claim", "expense"]),
            ("insurance", ["insurance", "policy", "premium"]),
            ("investment", ["investment", "dividend", "portfolio", "gains"]),
            ("miscellaneous", ["miscellaneous", "note", "confirmation"])
        ]

    def _load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def classify(self, text: str) -> str:
        text_lower = text.lower()
        for label, keywords in self.rules:
            if any(re.search(rf"\b{kw}\b", text_lower) for kw in keywords):
                return label
        if self.model:
            return self.model.predict([text])[0]
        return "unknown"

    def train_model(self, texts: List[str], labels: List[str]):
        pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
        pipeline.fit(texts, labels)
        joblib.dump(pipeline, self.model_path)
        self.model = pipeline

    def available_labels(self) -> List[str]:
        return [label for label, _ in self.rules]

# Example usage:
# classifier = DocumentClassifier()
# label = classifier.classify("Invoice No. 1234 | Total: $500.00")
# print(label)  # Output: 'invoice'
#
# classifier.train_model(sample_texts, sample_labels)  # Optional: training custom model