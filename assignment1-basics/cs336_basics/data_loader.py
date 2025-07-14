import numpy as np
import torch
import numpy.typing as npt

def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    
    # Randomly sample start indices for each sequence in the batch
    indices = np.random.randint(0, n - context_length, batch_size)
    
    # Generate input sequences and their corresponding target sequences
    input_sequences = np.array([dataset[i:i + context_length] for i in indices])
    target_sequences = np.array([dataset[i + 1:i + context_length + 1] for i in indices])
    
    # Convert to PyTorch tensors and move to specified device
    input_tensor = torch.tensor(input_sequences, dtype=torch.int64).to(device)
    target_tensor = torch.tensor(target_sequences, dtype=torch.int64).to(device)
    
    return input_tensor, target_tensor

# Example usage:
# input_tensor, target_tensor = data_loader('data_file_path', batch_size=32, context_length=128, device_str='cuda:0')