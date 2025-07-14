import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, save_checkpoint, load_checkpoint


# Memory-efficient dataset class using np.memmap
class MemoryEfficientDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.memmap(data_path, dtype='float32', mode='r')
        self.labels = np.memmap(label_path, dtype='int', mode='r')
        assert len(self.data) == len(self.labels), "Data and labels length mismatch"
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.int64)
        return x, y


# Main training function
def train(args):
    # Initialize the model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(args.device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Load the dataset
    train_dataset = MemoryEfficientDataset(args.train_data_path, args.train_label_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Load checkpoint if specified
    if args.checkpoint:
        iteration = load_checkpoint(args.checkpoint, model, optimizer)
    else:
        iteration = 0
    
    # Set up logging (e.g., with Weights and Biases, can be defined as needed)
    
    # Training loop
    for epoch in range(args.num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            if iteration % args.save_interval == 0:
                save_checkpoint(model, optimizer, iteration, args.save_path)

            if iteration % args.log_interval == 0:
                print(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss.item()}")
            
            iteration += 1

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a TransformerLM model with AdamW optimizer.')
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size.')
    parser.add_argument('--context_length', type=int, required=True, help='Context length.')
    parser.add_argument('--d_model', type=int, required=True, help='Dimension of model embeddings.')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers in the model.')
    parser.add_argument('--num_heads', type=int, required=True, help='Number of attention heads.')
    parser.add_argument('--d_ff', type=int, required=True, help='Dimension of feed forward network.')
    parser.add_argument('--rope_theta', type=float, required=True, help='Rope theta parameter value.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data.')
    parser.add_argument('--train_label_path', type=str, required=True, help='Path to the training labels.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 value for optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 value for optimizer.')
    parser.add_argument('--save_path', type=str, default='checkpoint.pth', help='Path to save the checkpoint.')
    parser.add_argument('--checkpoint', type=str, help='Path to a checkpoint file to resume training.')
    parser.add_argument('--log_interval', type=int, default=10, help='How frequently to log progress.')
    parser.add_argument('--save_interval', type=int, default=100, help='How frequently to save checkpoints.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)
