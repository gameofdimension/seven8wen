import random
from os.path import join

import torch
import torch.nn as nn
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import AutoModel, TrainingArguments, AutoModelForCausalLM
from transformers import Trainer

from data_prep import load_dataset, inference_alpaca, inference_adgen


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def get_base_model(model_name, v100):
    # need to set load_in_8bit=False in gpu v100
    if "baichuan-inc/baichuan-7B" == model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=not v100, trust_remote_code=True, device_map='auto')
    else:
        model = AutoModel.from_pretrained(
            model_name, load_in_8bit=not v100, trust_remote_code=True, device_map='auto')
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    return model


def build_peft_model(model: str):
    # see site-packages/peft/utils/other.py
    if 'chatglm' in model.lower():
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']
    elif 'baichuan' in model.lower():
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']
    else:
        target_modules = None
    peft_config = LoraConfig(
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    return model


def data_collator(tokenizer, features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
                [-100] * (seq_len - 1) + ids[(seq_len - 1):] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss


def peft_train(tokenizer, model, train_dataset, max_steps):
    training_args = TrainingArguments(
        "output",
        fp16=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        max_steps=max_steps,
        logging_steps=50,
        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        group_by_length=False,
    )

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=lambda features: data_collator(tokenizer, features),
    )
    trainer.train()


# save_tunable_parameters(model, os.path.join("output", "chatglm-lora.pt"))
def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def finetune(model_name, base_dir, v100: bool):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    short_name = model_name.split('/')[-1]
    save_path = join(base_dir, short_name, "adgen/train/")
    train_num = 1500
    train_data = load_dataset(save_path, train_num)

    model = get_base_model(model_name, v100=v100)
    model.eval()
    random.seed(42)
    inference_adgen(tokenizer, model, join(base_dir, "adgen/AdvertiseGen/dev.json"), 10)
    model.train()

    model = build_peft_model(model)
    peft_train(tokenizer, model, train_data, max_steps=train_num)
    save_tunable_parameters(model, join(base_dir, short_name, "model-lora.pt"))
    model.eval()
    random.seed(42)
    inference_adgen(tokenizer, model, join(base_dir, "adgen/AdvertiseGen/dev.json"), 10)
    model.train()


if __name__ == '__main__':
    finetune(model_name="THUDM/chatglm-6b", base_dir='/content/drive/MyDrive/corpus/', v100=False)
