# LLM Finetuning using LoRA and Quantization on Custom Dataset

This project provides an implementation for fine‑tuning large language models (LLMs) using **LoRA (Low‑Rank Adaptation)** on a custom dataset. LoRA is a parameter‑efficient technique that allows you to adapt powerful pretrained models (like Mistral, LLaMA, or GPT‑style architectures) without retraining the entire network. This makes fine‑tuning faster, cheaper, and more accessible.  

In addition, the project leverages **4‑bit quantization (BitsAndBytes)** to reduce memory usage and enable training on consumer‑grade GPUs. By combining quantization with LoRA, the workflow achieves efficient fine‑tuning without sacrificing model performance, making large‑scale adaptation practical even on limited hardware.  

A key part of this project is the **automatic dataset generation pipeline**: PDFs are converted into text, then an LLM is used to generate a Q&A dataset that will be used to finetune the model. This produces a high‑quality, domain‑specific dataset for instruction tuning without requiring manual annotation.

---

## 🚀 Features

- Fine‑tune state‑of‑the‑art LLMs with **LoRA adapters**
- Automatically generate a **Q&A dataset** from PDFs using an LLM
- Train on your own custom dataset (e.g., Q&A pairs, domain‑specific text)
- Leverage **4‑bit quantization (BitsAndBytes)** to reduce memory usage and enable efficient training on consumer‑grade GPUs


---

## 📂 Project Structure
- `main.py` – entry point for training and evaluation
- `src/` – PDFtoQA.py, finetuning.py
- `data/` – place your custom dataset files (e.g., `QA.json`)
- `workspace/` – model checkpoints and outputs
- `config.yaml` – configuration file (paths, model IDs, dataset references)
- `params.yaml` – parameters file (training, quantization, LoRA settings)
- `requirements.txt` – requirements file
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


## ⚙️ Installation & Setup

### ⚙️ Environment
- **Create  and activate environment**
  ```bash
    conda create -p venv python==3.13 -y
    conda activate venv
    ```
### ⚙️ Requirements
- **Install dependencies**
  ```bash
    pip install -r requirements.txt
    ```


### 🐋  Ollama
- **Download and install Ollama**
  ```bash
    curl -fsSL https://ollama.com/install.sh | sh -- Linux
    ```
  ```bash
    brew install ollama -- Mac
    ```
- **Run Ollama**
  ```bash
    ollama serve -- keeps Ollama running
    ```
- **Download model**
  ```bash
    ollama pull qwen2.5:3b
    ```
### 🤝 Hugging Face Hub
- **Install the HuggingFace hub client**
  ```bash
    pip install huggingface_hub
    ```
- **Login with your token (must be fine‑grained and enable gated repo access):**
  ```bash
    huggingface-cli login
    ```
- **Enter your huggingface token when prompted.**

## ▶️ Run the Project
To launch the application, simply execute:
```bash
    python main.py
```

## ✨ Author

Developed by Antonio Maria Fiscarelli This repository is a personal project exploring efficient fine‑tuning of LLMs.