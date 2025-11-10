# LLM Finetuning using LoRA on Custom Dataset

This repository demonstrates how to fine‑tune large language models (LLMs) using **LoRA (Low‑Rank Adaptation)** on a custom dataset. LoRA is a parameter‑efficient technique that allows you to adapt powerful pretrained models (like Mistral, LLaMA, or GPT‑style architectures) without retraining the entire network. This makes fine‑tuning faster, cheaper, and more accessible.

---

## 🚀 Features
- Fine‑tune state‑of‑the‑art LLMs with LoRA adapters
- Train on your own custom dataset (e.g., Q&A pairs, domain‑specific text)
- Save and load adapters separately from the base model

---

## 📂 Project Structure
- `main.py` – entry point for training and evaluation
- `src/` – helper scripts (data preprocessing, finetuning, prompt generation)
- `data/` – place your custom dataset files (e.g., `QA.json`)
- `workspace/` – model checkpoints and outputs (ignored in `.gitignore`)
- `.gitignore` – ensures large files (weights, cache, PDFs) are not pushed to GitHub


### 📝 PDFtoQA.py
- Converts a PDF into text and splits it into smaller chunks  
- Builds a prompt that instructs an LLM to generate Q&A pairs from each chunk  
- Streams responses back from the LLM in real time  
- Cleans and parses the responses into valid JSON with `"question"` and `"answer"` fields  
- Collects all Q&A pairs and saves them into a JSON file for later use  


### 🧠 finetuning.py
- **Dataset preparation**  
  - Loads a dataset containing `question` and `answer` fields  
  - Formats each entry into a chat‑style template (system, user, assistant roles)  
  - Produces text samples suitable for instruction‑tuning  

- **Tokenizer setup**  
  - Loads the tokenizer for the chosen base model  
  - Applies the custom chat template to dataset entries  

- **Model setup with quantization**  
  - Loads the base language model with 4‑bit quantization (`BitsAndBytesConfig`) to save memory  
  - Enables gradient checkpointing for efficiency  
  - Prepares the model for low‑bit training  

- **LoRA configuration**  
  - Defines LoRA adapter parameters (rank, alpha, dropout, target modules, task type)  
  - Wraps the model with LoRA for parameter‑efficient fine‑tuning  

- **Trainer setup and training**  
  - Uses `SFTTrainer` (Supervised Fine‑Tuning) from TRL to train the model  
  - Configures training (epochs, logging, disables checkpoint saving)  
  - Runs the training loop on the prepared dataset  

- **Model saving and publishing**  
  - Saves the trained model locally (`complete_checkpoint`, `final_model`)  
  - Optionally pushes the model and tokenizer to Hugging Face Hub if a `repo_id` is 



---

## ⚙️ Requirements
Install dependencies:
pip install -r requirements.txt




# Run training

python main.py



## ✨ Author

Developed by Antonio Maria Fiscarelli This repository is a personal project exploring efficient fine‑tuning of LLMs.