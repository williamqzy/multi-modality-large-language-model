import multiprocessing
import argparse
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer
from model import PALME_Tokenizer
import torch


class Config:
    SEED = 42
    SEQ_LEN = 8192
    NUM_CPU = multiprocessing.cpu_count()
    HF_ACCOUNT_REPO = "YOUR HF ACCOUNT"
    TOKENIZER = "EleutherAI/gpt-neox-20b"
    DATASET_NAME = "EleutherAI/the_pile_deduplicated"


def prepare_sample(sample):
    question = sample["question"]
    multiple_choice_answer = sample["multiple_choice_answer"]
    answers = sample["answers"]
    image_id = sample["image_id"]
    answer_type = sample["answer_type"]
    question_id = sample["question_id"]
    image = sample["image"]

    text = f"Question: {question} Multiple Choice Answer: {multiple_choice_answer} Answers: {answers} Answer Type: {answer_type} Question ID: {question_id} Image ID: {image_id}"
    
    return {
        "image": image,
        "target_text": text
    }


def main(args):
    tokenizer = PALME_Tokenizer()

    train_dataset = load_dataset(Config.DATASET_NAME, split="train", streaming=True)

    def preprocess_and_group_texts(samples):
        processed_samples = []
        for sample in samples:
            prepared_sample = prepare_sample(sample)
            text = prepared_sample["target_text"]
            image = prepared_sample["image"]

            text_tokens, _ = tokenizer.tokenize_texts([text + tokenizer.eos_token])
            image_tokens = tokenizer.tokenize_images([image])

            merged_tokens = torch.cat((text_tokens, image_tokens), dim=-1)

            processed_samples.append(merged_tokens)

        concatenated_examples = list(chain(*processed_samples))

        total_length = len(concatenated_examples)
        if total_length >= Config.SEQ_LEN:
            total_length = (total_length // Config.SEQ_LEN) * Config.SEQ_LEN
        
        result = [t[i : i + Config.SEQ_LEN] for i in range(0, total_length, Config.SEQ_LEN)]
        return result

    train_tokenized_dataset = train_dataset.map(
        preprocess_and_group_texts,
        batched=True,
    )
    
    train_tokenized_dataset.push_to_hub(Config.HF_ACCOUNT_REPO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=Config.SEQ_LEN, help="Sequence length for processing")
    parser.add_argument("--hf_account", type=str, default=Config.HF_ACCOUNT_REPO, help="Hugging Face account name and repo")
    parser.add_argument("--tokenizer", type=str, default=Config.TOKENIZER, help="Tokenizer model to use")
    parser.add_argument("--dataset_name", type=str, default=Config.DATASET_NAME, help="Name of the dataset to process")
    args = parser.parse_args()
    main(args)
