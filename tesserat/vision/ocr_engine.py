import pytesseract
from .image_utils import preprocess_image
from typing import Dict, Any
import numpy as np

import re
from collections import defaultdict
from typing import List, Dict, Any

def extract_lines_with_confidences(ocr_data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    lines = defaultdict(list)

    # Group words by line number (use a compound key if needed: block_num + par_num + line_num)
    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        conf = ocr_data["conf"][i]
        line_id = (ocr_data["block_num"][i], ocr_data["par_num"][i], ocr_data["line_num"][i])

        if word and conf != '-1':
            try:
                conf_val = int(conf)
            except ValueError:
                continue
            lines[line_id].append((word, conf_val))

    results = []

    for _, word_list in lines.items():
        line_text = " ".join(word for word, _ in word_list)
        date_conf = None
        amount_conf = None

        for word, conf in word_list:
            if re.match(r"\d{2}/\d{2}", word) and date_conf is None:
                date_conf = conf
            elif re.match(r"\d{1,3}(?:,\d{3})*\.\d{2}", word) and amount_conf is None:
                amount_conf = conf

        results.append({
            "text": line_text,
            "conf_date": date_conf if date_conf is not None else 0,
            "conf_amount": amount_conf if amount_conf is not None else 0
        })

    return results


def run_ocr(image: np.ndarray) -> Dict[str, Any]:
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    confidences = [int(c) for c in data.get("conf", []) if str(c).isdigit() or isinstance(c, int)]
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    
    line_details = extract_lines_with_confidences(data)

    return {
        "text": text,
        "ocr_data": data,
        "confidence": avg_conf,
        "lines": line_details
    }
