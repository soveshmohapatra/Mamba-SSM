import torch
import torch.nn as nn
import torch.optim as optim
import time
from simple_ssm import Simple1DSSM

class MambaBlock(nn.Module):
    """
    A full Mamba-style block that wraps the 1D SSM for sequence modeling.
    """
    def __init__(self, d_model: int, expand: int = 2, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner)
        self.gate_proj = nn.Linear(d_model, self.d_inner)
        self.ssm_channels = nn.ModuleList([
            Simple1DSSM(d_state=d_state) for _ in range(self.d_inner)
        ])
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_proj = self.in_proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        
        ssm_outputs = []
        for i, ssm in enumerate(self.ssm_channels):
            channel_x = x_proj[..., i]
            channel_y = ssm(channel_x)
            ssm_outputs.append(channel_y)
            
        ssm_out = torch.stack(ssm_outputs, dim=-1)
        y = ssm_out * gate
        return self.out_proj(y)


class SequenceClassifier(nn.Module):
    """Wraps a block (Mamba or Transformer) for sequence classification."""
    def __init__(self, block, d_model):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.block = block
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)
        emb = self.embedding(x)
        
        # Pass through sequence block
        out = self.block(emb)
        
        # Take the last token's representation for classification
        last_token_out = out[:, -1, :]
        return self.classifier(last_token_out).squeeze(-1)


def generate_copy_task_data(batch_size, seq_len):
    """
    Synthetic Long-Range Memory Task:
    The sequence consists of random noise (N(0,1)).
    The label is whether the FIRST token is strictly positive (>0) or not.
    The model must route information from step 0 to step L-1 to predict correctly.
    """
    x = torch.randn(batch_size, seq_len)
    # The label is 1 if the very first token is > 0, else 0
    y = (x[:, 0] > 0).float()
    return x, y


def train_and_eval(model, seq_len, steps=25):
    """Train for a few steps and return the final training accuracy as a proxy."""
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for step in range(steps):
        x, y = generate_copy_task_data(16, seq_len)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
    # Eval on a fresh batch
    model.eval()
    with torch.no_grad():
        x, y = generate_copy_task_data(128, seq_len)
        logits = model(x)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean().item()
        
    return acc * 100

def benchmark_accuracy():
    # Set to use 6 CPU cores
    torch.set_num_threads(7)
    
    print("--- BENCHMARKING LONG-RANGE ACCURACY ---")
    print("Task: Selective Copying (Predicting the first token's class at the end of the sequence)")
    print("This strictly tests the memory routing capability over L steps.")
    print("We train tiny proxy models for a fixed number of steps to observe learning degradation.")
    print("---------------------------------------------------------")
    print(f"{'Seq Len (L)':<12} | {'Mamba Accuracy':<20} | {'Transformer Accuracy':<20}")
    print("-" * 57)
    
    seq_lengths = [256, 512, 1024, 2048]
    d_model = 16
    
    for L in seq_lengths:
        # 1. Init Mamba
        mamba_block = MambaBlock(d_model=d_model, expand=2, d_state=8)
        mamba_model = SequenceClassifier(mamba_block, d_model)
        
        # 2. Init Transformer 
        # (Standard unoptimized attention, matching parameters)
        tf_block = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=2, 
            dim_feedforward=d_model*2, 
            batch_first=True
        )
        tf_model = SequenceClassifier(tf_block, d_model)
        
        # Train and Evaluate (Keeping it to 50 steps so the script finishes in reasonable time)
        # Note: In a real environment you train to convergence. We use proxy step-counts here.
        mamba_acc = train_and_eval(mamba_model, L, steps=10)
        tf_acc = train_and_eval(tf_model, L, steps=10)
        
        # Formatting limits for displaying results nicely
        print(f"{L:<12} | {mamba_acc:>15.1f}%     | {tf_acc:>16.1f}%")

if __name__ == "__main__":
    benchmark_accuracy()
