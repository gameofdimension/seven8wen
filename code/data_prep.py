import json
import random

import datasets
import torch
import transformers


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def format_adgen(example: dict) -> dict:
    return {"context": example['content'], "target": example['summary']}


def preprocess(tokenizer, eos_token_id, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def build_pipeline(max_seq_length, tokenizer, eos_token_id, formatter):
    def mapping(example: dict):
        formatted = formatter(example)
        feature = preprocess(tokenizer, eos_token_id, formatted, max_seq_length)
        feature["input_ids"] = feature["input_ids"][:max_seq_length]
        return feature

    return mapping


def prepare_alpaca_data(model_name, tokenizer, data_path, save_path):
    max_seq_length = 128
    # skip_overlength = False
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    mapping = build_pipeline(max_seq_length, tokenizer, config.eos_token_id, format_example)

    def tokenize(path):
        with open(path) as f:
            examples = json.load(f)
        for example in examples:
            feature = mapping(example)
            yield feature

    dataset = datasets.Dataset.from_generator(lambda: tokenize(data_path))
    dataset.save_to_disk(save_path)


def prepare_math_data(model_name, tokenizer, data_path, save_path):
    max_seq_length = 128
    # skip_overlength = False
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    mapping = build_pipeline(max_seq_length, tokenizer, config.eos_token_id, format_example)

    def tokenize(path):
        with open(path) as fp:
            # examples = json.load(f)
            for example in fp:
                if example.strip() == '':
                    continue
                example = json.loads(example)
                feature = mapping(example)
                yield feature

    dataset = datasets.Dataset.from_generator(lambda: tokenize(data_path))
    dataset.save_to_disk(save_path)


def prepare_adgen_data(model_name, tokenizer, data_path, save_path):
    max_seq_length = 128
    # skip_overlength = False
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    mapping = build_pipeline(max_seq_length, tokenizer, config.eos_token_id, format_adgen)

    def tokenize(path):
        with open(path) as fp:
            for example in fp:
                if example.strip() == '':
                    continue
                example = json.loads(example)
                feature = mapping(example)
                yield feature

    dataset = datasets.Dataset.from_generator(lambda: tokenize(data_path))
    dataset.save_to_disk(save_path)


def load_dataset(dataset_path, train_num):
    dataset = datasets.load_from_disk(dataset_path)
    mini_train_dataset = datasets.Dataset.from_dict(dataset[:train_num])
    return mini_train_dataset


def inference_alpaca(tokenizer, model, data_path, num):
    with open(data_path) as fp:
        instructions = json.load(fp)
    instructions = list(instructions)
    inference(tokenizer, model, format_example, instructions, num)


def inference_math(tokenizer, model, data_path, num):
    instructions = []
    with open(data_path) as fp:
        for example in fp:
            if example.strip() == '':
                continue
            example = json.loads(example)
            instructions.append(example)
    inference(tokenizer, model, format_example, instructions, num)


def inference_adgen(tokenizer, model, data_path, num):
    instructions = []
    with open(data_path) as fp:
        for example in fp:
            if example.strip() == '':
                continue
            example = json.loads(example)
            instructions.append(example)
    inference(tokenizer, model, format_adgen, instructions, num)


def inference(tokenizer, model, formatter, instructions, num):
    random.shuffle(instructions)
    with torch.no_grad():
        for idx, item in enumerate(instructions[:num]):
            feature = formatter(item)
            input_text = feature["context"]
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            out = model.generate(
                input_ids=input_ids,
                max_length=150,
                temperature=0
            )
            answer = tokenizer.decode(out[0])
            print("prompt->\n", input_text)
            print("prediction->\n", answer)
            print("golden->\n", feature["target"])


def main():
    model_name = "THUDM/chatglm-6b"
    for data_path, save_path, func in [
        ["data/AdvertiseGen/train.json", "data/adgen/train/", prepare_adgen_data],
        ["data/AdvertiseGen/dev.json", "data/adgen/dev/", prepare_adgen_data],
        ["data/school_math_0.25M.json", "data/math/", prepare_math_data],
        ["data/alpaca_data.json", "data/alpaca/", prepare_alpaca_data],
    ]:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        func(model_name, tokenizer, data_path, save_path)
        train_data = load_dataset(save_path, 10)

        for row in train_data:
            print(row)


if __name__ == '__main__':
    main()
