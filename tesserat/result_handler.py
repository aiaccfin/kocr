import os, json
from typing import List, Dict, Any

def combine_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = "\n".join([r["text"] for r in results])
    avg_conf = round(sum(r.get("confidence", 0.0) for r in results) / len(results), 2)

    combined_lines = sum([r.get("lines", []) for r in results], [])
    
    return{"lines": combined_lines,}

    # return {
    #     "text": text,
    #     "pretty_text": text.strip().splitlines(),
    #     "fields": {
    #         "vendor": results[0]["fields"].get("vendor", ""),
    #         "dates": sum([r["fields"]["dates"] for r in results], []),
    #         "amounts": sum([r["fields"]["amounts"] for r in results], []),
    #         "line_items": sum([r["fields"]["line_items"] for r in results], [])
    #     },
    #     "lines": combined_lines,  # ⬅️ Added line-level detail
    #     "confidence": avg_conf
    # }

# def combine_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
#     text = "\n".join([r["text"] for r in results])
#     avg_conf = round(sum(r.get("confidence", 0.0) for r in results) / len(results), 2)

#     return {
#         "text": text,
#         "pretty_text": text.strip().splitlines(),
#         "fields": {
#             "vendor": results[0]["fields"].get("vendor", ""),
#             "dates": sum([r["fields"]["dates"] for r in results], []),
#             "amounts": sum([r["fields"]["amounts"] for r in results], []),
#             "line_items": sum([r["fields"]["line_items"] for r in results], [])
#         },
#         "confidence": avg_conf
#     }

def save_result(result: Dict[str, Any], filename: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
