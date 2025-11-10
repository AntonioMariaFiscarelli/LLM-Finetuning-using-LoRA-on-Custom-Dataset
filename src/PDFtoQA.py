import yaml
import json, re
from typing import List, Dict
from colorama import Fore
from litellm import completion

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker


def prompt_template(data: str, num_records: int = 5):

    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
    the provided data chunk into diverse question and answer (Q&A) pairs that will be used to fine-tune a language model. 

    For each of the {num_records} entries, generate one or two well-structured questions that reflect different aspects of the information in the chunk. 
    Ensure a mix of longer and shorter questions, with shorter ones typically containing 1-2 sentences and longer ones spanning up to 3-4 sentences. Each 
    Q&A pair should be concise yet informative, capturing key insights from the data.

    Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:

        "question": "Your question here...",
        "answer": "Your answer here..."

    Focus on creating clear, relevant, and varied questions that encourage the model to learn from diverse perspectives. Avoid any sensitive or biased 
    content, ensuring answers are accurate and neutral.

    Example:
    
        "question": "What is the primary purpose of this dataset?",
        "answer": "This dataset serves as training data for fine-tuning a language model."
    

    By following these guidelines, you'll contribute to a robust and effective dataset that enhances the model's performance."

    ---

    **Explanation:**

    - **Clarity and Specificity:** The revised prompt clearly defines the role of the assistant and the importance of the task, ensuring alignment with the 
    project goals.
    - **Quality Standards:** It emphasizes the need for well-formulated Q&A pairs, specifying the structure and content of each question and answer.
    - **Output Format:** An example JSON structure is provided to guide the format accurately.
    - **Constraints and Biases:** A note on avoiding sensitive or biased content ensures ethical considerations are met.
    - **Step-by-Step Guidance:** The prompt breaks down the task into manageable steps, making it easier for the assistant to follow.

    This approach ensures that the generated data is both high-quality and meets the specific requirements of the machine learning project.
    
    Data
    {data}
    """

class PDFtoQA:
    def __init__(self, config_path="config.yaml", params_path="params.yaml"):
        # Load configs
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.config = self.config["PDFtoQA"]
        self.params = self.params["PDFtoQA"]

        self.pdf_path = self.config["pdf_path"]
        self.output_path = self.config["output_path"]
        self.model_id = self.config["model_id"]

        self.num_records = self.params["num_records"]
        self.chunk_limit = self.params["chunk_limit"]

        self.pdf_options = PdfPipelineOptions()
        self.pdf_options.do_ocr = False
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self.pdf_options)}
        )
        self.chunker = HybridChunker()


    @staticmethod
    def clean_and_parse(raw_output: str) -> List[Dict]:
        raw = raw_output.strip()
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
        if raw.startswith("```"):
            raw = raw[len("```"):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        raw = re.sub(r"//.*", "", raw)
        if raw.startswith("{") and not raw.startswith("["):
            if "}\n{" in raw or "},\n{" in raw or "}, {" in raw:
                raw = "[" + raw + "]"
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
        pattern = re.compile(
            r'["\']?question["\']?\s*:\s*["\'](?P<q>.*?)["\']\s*,\s*["\']?answer["\']?\s*:\s*["\'](?P<a>.*?)["\']',
            re.DOTALL | re.IGNORECASE,
        )
        return [{"question": m.group("q").strip(), "answer": m.group("a").strip()} for m in pattern.finditer(raw)]

    def llm_call(self, data: str) -> List[Dict]:
        stream = completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt_template(data, self.num_records)}],
            stream=True,
            options={"num_predict": 2},
        )
        buffer = ""
        for x in stream:
            delta = x['choices'][0]["delta"].get("content")
            if delta:
                print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
                buffer += delta
        return self.clean_and_parse(buffer)

    def run(self) -> List[Dict]:
        doc = self.converter.convert(self.pdf_path).document
        chunks = self.chunker.chunk(dl_doc=doc)

        all_records: List[Dict] = []
        print(self.chunk_limit)
        for i, chunk in enumerate(chunks):
            print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
            records = self.llm_call(chunk.text)
            all_records.extend(records)
            if self.chunk_limit and i >= self.chunk_limit:
                break

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)

        return all_records


if __name__ == "__main__":
    pdftoqua = PDFtoQA()
    results = pdftoqua.run()
    print(Fore.GREEN + f"Saved {len(results)} QA pairs to {pdftoqua.output_path}" + Fore.RESET)
