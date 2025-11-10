# main.py
import multiprocessing
from colorama import Fore

# Import classes from src/
from src.PDFtoQA import PDFtoQA
from src.finetuning import FineTuning


def run_pdf_to_qa():
    print(Fore.CYAN + "=== Running PDF → QA extraction ===" + Fore.RESET)
    pdf_pipeline = PDFtoQA()
    qa_records = pdf_pipeline.run()
    print(Fore.GREEN + f"Extracted {len(qa_records)} QA pairs" + Fore.RESET)


def run_finetuning():
    print(Fore.CYAN + "=== Running fine‑tuning ===" + Fore.RESET)
    ft = FineTuning()
    ft.run()
    print(Fore.GREEN + "Fine‑tuning complete" + Fore.RESET)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # safe on Windows
    run_pdf_to_qa()
    run_finetuning()