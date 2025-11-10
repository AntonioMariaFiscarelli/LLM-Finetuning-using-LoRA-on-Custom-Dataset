import yaml
import torch
from datasets import load_dataset
from colorama import Fore
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training


def format_chat_template(batch, tokenizer):
    system_prompt = """You are a helpful, honest and harmless assistant designed to help engineers. 
    Think through each question logically and provide an answer. 
    Don't make things up; if you're unable to answer a question, advise the user that it's outside your scope."""

    samples = []
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    return {"instruction": questions, "response": answers, "text": samples}


class FineTuning:
    def __init__(self, config_path="config.yaml", params_path="params.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.config = self.config["finetuning"]
        self.params = self.params["finetuning"]

        self.dataset_path = self.config["dataset_path"]
        self.base_model = self.config["base_model"]
        self.output_dir = self.config["output_dir"]
        self.repo_id = self.config.get("repo_id")

        self.training_params = self.params["training"]
        self.quant_params = self.params["quantization"]
        self.lora_params = self.params["lora"]

    def run(self):
        dataset = load_dataset(self.dataset_path, split="train")
        print(Fore.YELLOW + str(dataset[2]) + Fore.RESET)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)

        train_dataset = dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            num_proc=self.training_params["num_proc"],
            batched=True,
            batch_size=self.training_params["batch_size"],
        )
        print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.quant_params["load_in_4bit"],
            bnb_4bit_use_double_quant=self.quant_params["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=self.quant_params["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, self.quant_params["bnb_4bit_compute_dtype"]),
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="cuda:0",
            quantization_config=quant_config,
            cache_dir="./workspace",
        )

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=self.lora_params["r"],
            lora_alpha=self.lora_params["lora_alpha"],
            lora_dropout=self.lora_params["lora_dropout"],
            target_modules=self.lora_params["target_modules"],
            task_type=self.lora_params["task_type"],
        )

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            args=SFTConfig(
                output_dir=self.output_dir, 
                num_train_epochs=self.training_params["num_train_epochs"],
                save_strategy="no",          # disables checkpoint saving
                logging_strategy="steps",    # still logs metrics
                save_total_limit=0           # no checkpoints kept
                ),
            peft_config=peft_config,
        )

        trainer.train()
        trainer.save_model("complete_checkpoint")
        trainer.model.save_pretrained("final_model")
        
        if self.repo_id:
            trainer.model.push_to_hub(self.repo_id)
            tokenizer.push_to_hub(self.repo_id)
            print(Fore.GREEN + f"Model pushed to https://huggingface.co/{self.repo_id}" + Fore.RESET)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    ft = FineTuning()
    ft.run()
