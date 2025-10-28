from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from colorama import Fore


import json, re
from typing import List, Dict
from litellm import completion
from generated_prompt import prompt_template


def clean_and_parse(raw_output: str) -> List[Dict]:
    """
    Cleans LLM output that may contain markdown fences, multiple objects,
    or malformed formatting. Always returns a list of dicts with
    'question' and 'answer' keys. If nothing valid is found, returns [].
    """
    raw = raw_output.strip()

    # --- Remove markdown fences like ```json ... ```
    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.startswith("```"):
        raw = raw[len("```"):].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    # --- Remove JS-style comments ---
    raw = re.sub(r"//.*", "", raw)

    # --- Normalize multiple objects into an array ---
    if raw.startswith("{") and not raw.startswith("["):
        if "}\n{" in raw or "},\n{" in raw or "}, {" in raw:
            raw = "[" + raw + "]"

    # --- Try strict JSON parse ---
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [{"question": parsed.get("question", ""), "answer": parsed.get("answer", "")}]
        elif isinstance(parsed, list):
            return [
                {"question": item.get("question", ""), "answer": item.get("answer", "")}
                for item in parsed if isinstance(item, dict)
            ]
    except Exception:
        pass

    # --- Fallback: regex extraction of question/answer pairs ---
    qa_pairs = []
    pattern = re.compile(
        r'["\']?question["\']?\s*:\s*["\'](?P<q>.*?)["\']\s*,\s*["\']?answer["\']?\s*:\s*["\'](?P<a>.*?)["\']',
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(raw):
        q = match.group("q").strip()
        a = match.group("a").strip()
        qa_pairs.append({"question": q, "answer": a})

    return qa_pairs  # may be [] if nothing found

def llm_call(data: str, num_records: int = 2) -> List[Dict]:
    stream = completion(
        model="ollama_chat/qwen2.5:3b",
        messages=[{"role": "user", "content": prompt_template(data, num_records)}],
        stream=True,
        options={"num_predict": 2},
    )

    buffer = ""
    for x in stream:
        delta = x['choices'][0]["delta"].get("content")
        if delta:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            buffer += delta

    return clean_and_parse(buffer)


if __name__ == "__main__":


    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)})
    doc = converter.convert("Agreement.pdf").document

    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    all_records: List[Dict] = []

    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
        #enriched_text = chunker.contextualize(chunk=chunk)
        #print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text:\n{enriched_text[:300]}…" + Fore.RESET)

        records = llm_call(chunk.text)
        all_records.extend(records)

        if i >= 2:  # limit for testing
            break

    # Save everything as one big JSON array
    with open("data/QA.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
