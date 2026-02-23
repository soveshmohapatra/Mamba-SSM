import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple1DSSM(nn.Module):
    """
    A minimal, educational 1-Dimensional Linear Time-Invariant (LTI) 
    State Space Model layer in pure PyTorch.

    This module maps a 1D sequence of inputs x to an output sequence y via 
    a hidden state h. It demonstrates continuous parameter initialization, 
    ZOH discretization, fast sequence convolution training, and O(1) memory 
    recurrent inference.
    """
    
    def __init__(self, d_state: int):
        super().__init__()
        self.d_state = d_state
        
        # 1. Continuous-time parameters
        # A: (d_state), assuming a diagonal state matrix for efficiency.
        # Initialized with negative values to ensure stable memory decay dynamics.
        self.A = nn.Parameter(-torch.rand(d_state) - 1.0)
        
        # B: (d_state, 1) mapping scalar input to N-dimensional state
        self.B = nn.Parameter(torch.randn(d_state, 1))
        
        # C: (1, d_state) mapping N-dimensional state to scalar output
        self.C = nn.Parameter(torch.randn(1, d_state))
        
        # Delta: The step size for discretization.
        # Logged parameter to strictly enforce positivity via exponentiation.
        self.log_delta = nn.Parameter(torch.randn(1))
        
    def discretize(self):
        """
        Converts the continuous dynamical parameters into discrete sequence
        transitions using the Zero-Order Hold (ZOH) technique.
        """
        # Enforce delta > 0
        delta = torch.exp(self.log_delta)
        
        # Matrix exponential simplifies to element-wise operations 
        # because the state matrix A is physically modeled as a diagonal vector.
        A_bar = torch.exp(delta * self.A) 
        
        # The algebraic simplification of B_bar for a diagonal A matrix
        # B_bar = (A * delta)^-1 * (A_bar - I) * (delta * B) -> (A_bar - I) / A * B
        B_bar = ((A_bar - 1.0) / self.A).unsqueeze(-1) * self.B
        
        return A_bar, B_bar
        
    def forward(self, x):
        """
        Convolutional execution path.
        Optimized for massive parallel training over full sequences.
        
        Args:
            x (Tensor): Input sequence of shape (batch, L)
            
        Returns:
            y (Tensor): Target output sequence of shape (batch, L)
        """
        batch, L = x.shape
        A_bar, B_bar = self.discretize()
        
        # Calculate the global convolutional kernel that defines
        # the model's unrolled response across the entire sequence.
        # K = [C * B_bar, C * A_bar * B_bar, C * (A_bar^2) * B_bar, ...]
        K = torch.zeros(L, device=x.device)
        
        # A_pow tracks cumulative state transitions
        A_pow = torch.ones_like(self.A) 
        
        for k in range(L):
            # Compute the linear response at delay k
            K[k] = (self.C @ (A_pow.unsqueeze(-1) * B_bar)).squeeze()
            
            # Transition the underlying state forward by one step
            A_pow = A_pow * A_bar
            
        # Apply the global response via FFT-backed convolution.
        # Reshape to (batch, in_channels, sequence_length) -> (batch, 1, L)
        K = K.view(1, 1, L)
        x = x.view(batch, 1, L)
        
        # Apply padding to prevent future data bleeding into the past
        # Token x_T should only be influenced by {x_0, ... x_T}.
        # Note: PyTorch's F.conv1d computes cross-correlation, not true convolution.
        # We must reverse/flip the kernel to make it a causal convolution!
        K_flipped = torch.flip(K, dims=(-1,))
        y = F.conv1d(x, K_flipped, padding=L-1)
        
        # The convolution inflates to size 2L-1 due to padding.
        # Slice it eagerly back to the original strict sequence length.
        y = y[..., :L]
        
        return y.squeeze(1)

    def step(self, x_k, h_prev):
        """
        Recurrent execution path.
        Optimized for memory-flat, autoregressive generation at deployment.
        
        Args:
            x_k (Tensor): The single current token of shape (batch, 1)
            h_prev (Tensor): The rolling hidden state matrix of shape (batch, d_state)
            
        Returns:
            y_k (Tensor): Current generated scalar prediction
            h_k (Tensor): The newly updated hidden state matrix to roll forward
        """
        A_bar, B_bar = self.discretize()
        
        # 1. State Update: Compress new info and decay old info
        # h_t = (A_bar * h_{prev}) + (B_bar * x_t)
        h_k = (A_bar * h_prev) + (B_bar.squeeze(-1) * x_k)
        
        # 2. Project State back to arbitrary Token vocabulary/scalar
        # y_t = C * h_t
        y_k = (self.C * h_k).sum(dim=-1, keepdim=True)
        
        return y_k, h_k


if __name__ == "__main__":
    # Extremely quick sanity check
    batch_size = 4
    seq_len = 128
    hidden_dim = 16
    
    print("Initializing Simple1DSSM...")
    model = Simple1DSSM(d_state=hidden_dim)
    
    # Generate random sequence dataset
    dataset_x = torch.randn(batch_size, seq_len)
    
    # Run the unrolled 1D Fast Convolution (Training Mode)
    print("\n[Mode A] Parallel Convolution / Training")
    fast_conv_y = model(dataset_x)
    print(f"Input Shape:  {dataset_x.shape}")
    print(f"Output Shape: {fast_conv_y.shape}")
    
    # Run the Autoregressive Recurrence (Inference Mode)
    print("\n[Mode B] Recurrent Step-by-Step / Inference")
    # State initializes to zero!
    rolling_hidden = torch.zeros(batch_size, hidden_dim) 
    recurrent_y_sequence = []
    
    for t in range(seq_len):
        # Ingest the t-th token from across the batch
        x_t = dataset_x[:, t].unsqueeze(-1)
        
        # Step the State Space Model
        y_t, rolling_hidden = model.step(x_t, rolling_hidden)
        
        recurrent_y_sequence.append(y_t)
        
    recurrent_y = torch.cat(recurrent_y_sequence, dim=-1)
    print(f"Input Shape (Accumulated steps):  {dataset_x.shape}")
    print(f"Output Shape (Accumulated steps): {recurrent_y.shape}")
    
    print("\nVerifying Mathematical LTI Duality...")
    # The output vectors of the two operations should be roughly equal (minus float drift)
    difference = (fast_conv_y - recurrent_y).abs().max().item()
    print(f"Max Absolute Error between operations: {difference:.6f}")
    if difference < 1e-4:
        print("[SUCCESS] Fast Convolution and O(1) Recurrence are mathematically identical!")
    else:
        print("[WARNING] Huge divergence detected!")
