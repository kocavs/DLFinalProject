from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def dataloader(name, token_name, train_length, batch_size):
    # Method used to tokenize dataset
    def tokenize_function(examples):
        tokenizer = AutoTokenizer.from_pretrained(token_name)
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    # Download dataset
    datasets = load_dataset(name, cache_dir="./dataset")
    # Tokenizing dataset
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # Select dataset
    train_datasets = tokenized_datasets["train"].select(range(train_length))
    test_datasets = tokenized_datasets["test"]

    # Here we shuffle our train dataloader
    train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_datasets, batch_size=batch_size)

    return train_dataloader, test_dataloader, train_datasets, test_datasets
