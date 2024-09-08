# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

import os

from llama import Dialog, Llama

from datasets import load_dataset

def prepare_data(dataset, num_examples: int = -1):
    dialogs = []
    labels = []

    for i, example in enumerate(dataset):
        if example['label'] == -1:
            continue
        dialog = [{"role": "system", "content": "Respond with 'Entailment', 'Neutral', or 'Contradiction' only. This is a natural language inference task, and your role is to be an expert for labelling examples. Determine the relationship between these two statements."},
                  {"role": "user", "content": f"Premise: '{example['premise']}', Hypothesis: '{example['hypothesis']}'. "}]
        dialogs.append(dialog)
        labels.append(example['label'])

        if i != -1:  
            if i == num_examples-1:
                break

    return dialogs, labels

def process_batches(generator, batch_size, dialogs, max_gen_len, temperature, top_p):
    results = []
    for i in range(0, len(dialogs), batch_size):
        batch_dialogs = dialogs[i : i + batch_size]
        batch_results = generator.chat_completion(batch_dialogs,
                                        max_gen_len=max_gen_len,
                                        temperature=temperature,
                                        top_p=top_p)
        results.extend(batch_results)
    return results
        

def load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(ckpt_dir=ckpt_dir,
                            tokenizer_path=tokenizer_path,
                            max_seq_len=max_seq_len,
                            max_batch_size=max_batch_size)
    return generator



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    print("> Loading SNLI Test Set...")
    
    snli_test = load_dataset("stanfordnlp/snli", split="test")

    print("> Preparing Dialogs...")
    
    num_test_examples = 100
    dialogs, true_labels = prepare_data(snli_test, num_test_examples)

    print("> Building LLaMA-3-8b-Instruct Model...")


    generator = load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    

    print("> Generating Responses...")

    num_repetitions = 3
    max_gen_len = 3

    all_results = []

    for i in range(num_repetitions):
        results = process_batches(generator, max_batch_size, dialogs, max_gen_len, temperature, top_p)
        all_results.append(results)


    print("> Writing Output Files...")


    output_dir = 'saved_outputs'

    preds_file = f"test.txt"
    full_output_path = os.path.join(output_dir, preds_file)
    with open(full_output_path, 'w') as file:
        for result in results:
            file.write(f"{result['generation']['content']}\n")
    

    print("> Process Complete")

if __name__ == "__main__":
    fire.Fire(main)
