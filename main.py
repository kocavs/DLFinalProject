# Import function and class from our own files
from DataLoad import dataloader
import torch
import argparse
import time
import os
from transformers import  AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calcuate_accuracy(preds, labels):
    # Calculate the correct numbers
    idx_max = torch.argmax(preds, dim=-1)
    n_correct = (idx_max == labels).sum().item()

    return n_correct


def train(model, train_loader, optimizer, scheduler, rank=None, mixed=False):
    model.train()
    total_loss = 0
    num_correct = 0
    num_total = 0
    # Enable mixed single precision or not
    scaler = amp.GradScaler() if mixed else None
    # Start Training by batch size
    for batch in train_loader:
        if rank:
            labels = batch['labels'].cuda(rank)
            input_ids = batch['input_ids'].cuda(rank)
            attention_mask = batch['attention_mask'].cuda(rank)
        else:
            labels = batch['labels'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        # Save training time with mixed method
        if mixed:
            with amp.autocast():
                outputs = model(input_ids, attention_mask).logits
                loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # Update parameters
            scheduler.step()  
            optimizer.zero_grad()
        else:
            # Get predict result
            outputs = model(input_ids, attention_mask).logits
            # Calculate loss
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            # Update parameters
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
        # Calculate correct labels
        num_correct += calcuate_accuracy(outputs, labels)
        total_loss += loss.item()
        num_total += labels.size(0)
    # Calculate loss and accuracy
    avg_train_loss = total_loss / num_total
    avg_train_acc = num_correct / num_total * 100.0

    return avg_train_loss, avg_train_acc


def evaluate(model, test_loader, rank=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_total = 0
    with torch.no_grad():
        for batch in test_loader:
            if rank:
                labels = batch['labels'].cuda(rank)
                input_ids = batch['input_ids'].cuda(rank)
                attention_mask = batch['attention_mask'].cuda(rank) 
            else:
                labels = batch['labels'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            # Get predict result
            output = model(input_ids, attention_mask).logits
            loss = torch.nn.CrossEntropyLoss()(output, labels)
            # Calculate correct labels
            total_loss += loss.item()
            total_correct += calcuate_accuracy(output, labels)
            total_samples += labels.size(0)
            num_total += labels.size(0)
    # Calculate loss and accuracy
    average_loss = total_loss / num_total
    accuracy = (total_correct / num_total) * 100.0

    return average_loss, accuracy


def main(rank=None, world_size=None, opts=None):
    best_acc = 0
    # Get dataset and model
    if opts.dataset == 'emotion':
        train_loader, test_loader, train_datasets, test_datasets = dataloader(name=opts.dataset,
                                                                              token_name=opts.pretrained_model_name,
                                                                              train_length=3200,
                                                                              batch_size=opts.batch_size)
    else:
        train_loader, test_loader, train_datasets, test_datasets = dataloader(name=opts.dataset,
                                                                              token_name=opts.pretrained_model_name,
                                                                              train_length=10000,
                                                                              batch_size=opts.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(opts.pretrained_model_name, num_labels=opts.num_classes)
    # if using data parallel
    if opts.DP:
        device_ids = [i for i in range(world_size)]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # Send model to GPU for speed-up training
    model.to(device)
    
    # Set the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.0001)
    num_epochs = opts.epoch
    num_training_steps = num_epochs * len(train_loader)
    # Adjust the warmup steps as needed
    num_warmup_steps = int(0.1 * num_training_steps)
    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    for epoch in range(opts.epoch):
        # Calculate time for training the model
        start_time = time.time()
        # Training process
        avg_train_loss, avg_train_acc = train(model, train_loader, optimizer, scheduler, rank=rank, mixed=opts.mixed)
        end_time = time.time()
        # Testing process
        avg_test_loss, avg_test_acc = evaluate(model, test_loader, rank)
        epoch_time = end_time - start_time
        # Print out results
        print("Epoch: ", (epoch+1))
        print(f'\tTrain Loss: {avg_train_loss:.5f} | Train Acc: {avg_train_acc:.2f}%')
        print(f'\tTest. Loss: {avg_test_loss:.5f} |  Test Acc: {avg_test_acc:.2f}%')
        print(f"\tTime: {epoch_time:.2f} seconds")
        # Save the best model
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            torch.save(model, "./" + opts.dataset + ".pt")
    

if __name__ == "__main__":
    # Get GPU numbers
    num_gpu = torch.cuda.device_count()
    print(f"You are using {num_gpu} GPUS!")
    # Get the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased', help='Name of the pre-trained BERT model')
    parser.add_argument('--dataset', type=str, default='emotion', help='Name of dataset')
    parser.add_argument('--epoch', type=int, default=5, help='Number of training epoches')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--lr', type=int, default=2e-5, help='Learning Rate')
    parser.add_argument('--DP', action="store_true", help='enable data parallel')
    parser.add_argument('--mixed', action="store_true", help='enable mixed-precision training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and testing')
    opts = parser.parse_args()
    
    # enable Distributed Data parallel
    if opts.DP:
        main(world_size=num_gpu, opts=opts)
    else:
        main(world_size=1, opts=opts)
