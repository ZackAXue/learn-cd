import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Import the diffusion components from cleandiffuser
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp

# Import your JdmContinuousDiffusionSDE implementation
from cleandiffuser.diffusion import JdmContinuousDiffusionSDE


# Simple synthetic dataset for task and motion planning
class SimpleTampDataset(Dataset):
    def __init__(self, num_samples=200):
        """
        Creates a simple synthetic dataset for task and motion planning:
        - High-level: discrete action sequences represented as binary bits
        - Low-level: continuous motion trajectories
        """
        self.num_samples = num_samples
        
        # Define shapes
        self.hl_horizon = 8
        self.hl_bits = 6
        self.ll_horizon = 48
        self.ll_dims = 3
        
        # Generate synthetic data
        self.hl_data = torch.zeros((num_samples, self.hl_horizon, self.hl_bits))
        self.ll_data = torch.zeros((num_samples, self.ll_horizon, self.ll_dims))
        
        # Create initial and goal states
        self.init_states = torch.zeros((num_samples, 11, 6))
        self.goal_states = torch.zeros((num_samples, 11, 6))
        self.init_coords = torch.rand((num_samples, 5, 2)) * 2 - 1  # Range: [-1, 1]
        self.goal_coords = torch.rand((num_samples, 5, 2)) * 2 - 1  # Range: [-1, 1]
        
        # Generate high-level discrete actions (one-hot encoded)
        for i in range(num_samples):
            for h in range(self.hl_horizon):
                bit_idx = np.random.randint(0, self.hl_bits)
                self.hl_data[i, h, bit_idx] = 1.0
                
                # Also populate initial and goal states
                if h < 11:
                    init_bit = np.random.randint(0, 6)
                    goal_bit = np.random.randint(0, 6)
                    self.init_states[i, h, init_bit] = 1.0
                    self.goal_states[i, h, goal_bit] = 1.0
        
        # Generate continuous motion trajectories
        for i in range(num_samples):
            # Time component (first dimension)
            self.ll_data[i, :, 0] = torch.linspace(0, 1, self.ll_horizon)
            
            # Create x, z motion patterns that correlate with the actions
            for h in range(self.hl_horizon):
                start_idx = h * (self.ll_horizon // self.hl_horizon)
                end_idx = (h + 1) * (self.ll_horizon // self.hl_horizon)
                
                # Use action pattern to influence trajectory
                action_idx = self.hl_data[i, h].argmax().item()
                
                # Map action to a specific motion pattern
                amplitude = 0.5 + 0.1 * action_idx
                frequency = 1 + 0.2 * action_idx
                phase = 0.2 * action_idx
                
                # Generate sine/cosine patterns for x, z
                t = torch.linspace(0, 1, end_idx - start_idx)
                self.ll_data[i, start_idx:end_idx, 1] = amplitude * torch.sin(frequency * 2 * np.pi * t + phase)
                self.ll_data[i, start_idx:end_idx, 2] = amplitude * torch.cos(frequency * 2 * np.pi * t + phase)
        
        # Create masks for fixed portions (initial conditions)
        self.fix_mask_hl = torch.zeros((self.hl_horizon, self.hl_bits))
        self.fix_mask_ll = torch.zeros((self.ll_horizon, self.ll_dims))
        
        # Fix first action and first few motion points
        self.fix_mask_hl[0] = 1.0
        self.fix_mask_ll[:6] = 1.0
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "obs": {
                "init_discrete": self.init_states[idx],
                "goal_discrete": self.goal_states[idx],
                "init_coords_block": self.init_coords[idx],
                "goal_coords_block": self.goal_coords[idx]
            },
            "hl_discrete_action_seq": self.hl_data[idx],
            "ll_traj": self.ll_data[idx],
            "segment_idx": [(i*6, (i+1)*6) for i in range(8)]  # 8 segments of 6 timesteps each
        }


def test_jdm_diffusion():
    """
    Comprehensive test function for JdmContinuousDiffusionSDE class.
    Tests initialization, training, evaluation, and sampling.
    """
    print("Starting JdmContinuousDiffusionSDE test...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "jdm_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # ======= DATA SETUP =======
    print("\n[1/7] Creating synthetic dataset...")
    dataset = SimpleTampDataset(num_samples=200)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"HL data shape: {dataset.hl_data.shape}, LL data shape: {dataset.ll_data.shape}")
    
    # ======= MODEL SETUP =======
    print("\n[2/7] Creating diffusion model...")
    
    # Define dimensions
    hl_dim = dataset.hl_horizon * dataset.hl_bits
    ll_dim = dataset.ll_horizon * dataset.ll_dims
    emb_dim = 128
    hidden_dim = 512
    
    # Create neural networks using cleandiffuser components
    # High-level diffusion model (for discrete actions)
    nn_diffusion_hl = PearceMlp(
        act_dim=hl_dim, 
        To=1, 
        emb_dim=emb_dim, 
        hidden_dim=hidden_dim, 
        timestep_emb_type="untrainable_fourier"
    )
    
    # Low-level diffusion model (for motion trajectories)
    nn_diffusion_ll = PearceMlp(
        act_dim=ll_dim, 
        To=1, 
        emb_dim=emb_dim, 
        hidden_dim=hidden_dim, 
        timestep_emb_type="untrainable_fourier"
    )
    
    # Conditioning networks
    nn_condition_ll_to_hl = PearceObsCondition(
        obs_dim=ll_dim, 
        emb_dim=emb_dim, 
        flatten=True, 
        dropout=0.1
    )
    
    nn_condition_hl_to_ll = PearceObsCondition(
        obs_dim=hl_dim, 
        emb_dim=emb_dim, 
        flatten=True, 
        dropout=0.1
    )
    
    # Prepare masks and constraints
    fix_mask_hl = dataset.fix_mask_hl.clone()  # Reshape to (1, hl_dim)
    fix_mask_ll = dataset.fix_mask_hl.clone()  # Reshape to (1, ll_dim)
    
    # Create JdmContinuousDiffusionSDE model
    jdm_model = JdmContinuousDiffusionSDE(
        # Neural networks
        nn_diffusion_hl=nn_diffusion_hl,
        nn_diffusion_ll=nn_diffusion_ll,
        nn_condition_ll_to_hl=nn_condition_ll_to_hl,
        nn_condition_hl_to_ll=nn_condition_hl_to_ll,
        
        # Masks and weights
        fix_mask_hl=fix_mask_hl,
        fix_mask_ll=fix_mask_ll,
        
        # Hierarchical parameters
        steps_per_action=6,  # 6 motion points per action token
        
        # Training parameters
        ema_rate=0.999,
        optim_params_hl={"lr": 1e-4},
        optim_params_ll={"lr": 1e-4},
        
        # Diffusion parameters
        noise_schedule="cosine",
        
        # Data constraints
        x_max_hl=torch.ones(hl_dim),
        x_min_hl=torch.zeros(hl_dim),
        
        # Device
        device=device
    )
    
    
    print(f"Model created successfully")
    print(f"High-level model parameters: {sum(p.numel() for p in jdm_model.model_hl['diffusion'].parameters())}")
    print(f"Low-level model parameters: {sum(p.numel() for p in jdm_model.model_ll['diffusion'].parameters())}")
    
    # ======= TEST FORWARD PROCESS =======
    print("\n[3/7] Testing forward process (add_noise)...")
    
    # Get a batch of data
    batch = next(iter(train_dataloader))
    x0_hl = batch["hl_discrete_action_seq"].to(device)  # Shape: [batch_size, horizon, bit_dim]
    x0_ll = batch["ll_traj"].to(device)  # Shape: [batch_size, horizon, features]
    
    # Test add_noise function
    xt_hl, t_hl, eps_hl, xt_ll, t_ll, eps_ll = jdm_model.add_noise(x0_hl, x0_ll)
    
    print(f"High-level noise added: shape={xt_hl.shape}, time steps={t_hl.shape}")
    print(f"Low-level noise added: shape={xt_ll.shape}, time steps={t_ll.shape}")
    
    # ======= TEST LOSS CALCULATION =======
    print("\n[4/7] Testing loss calculation...")
    
    # Calculate losses
    loss_hl, loss_ll = jdm_model.loss(x0_hl, x0_ll)
    
    print(f"High-level loss: {loss_hl.item():.4f}")
    print(f"Low-level loss: {loss_ll.item():.4f}")
    
    # ======= TEST TRAINING UPDATE =======
    print("\n[5/7] Testing model update...")
    
    # Set model to training mode
    jdm_model.train()
    
    # Perform one update
    log = jdm_model.update(x0_hl, x0_ll)
    
    print(f"Update log: {log}")
    
    # Perform mini-training loop
    num_batches = 5
    print(f"\nTraining for {num_batches} batches...")
    
    for i, batch in enumerate(train_dataloader):
        if i >= num_batches:
            break
            
        x0_hl = batch["hl_discrete_action_seq"].reshape(batch_size, -1).to(device)
        x0_ll = batch["ll_traj"].reshape(batch_size, -1).to(device)
        
        log = jdm_model.update(x0_hl, x0_ll)
        print(f"Batch {i+1}/{num_batches}, HL Loss: {log['loss_hl']:.4f}, LL Loss: {log['loss_ll']:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, "jdm_model.pt")
    jdm_model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # ======= TEST MODEL LOADING =======
    print("\n[6/7] Testing model loading...")
    
    jdm_model.load(model_path)
    print(f"Model loaded successfully")
    
    # ======= TEST SAMPLING =======
    print("\n[7/7] Testing sampling process...")
    
    # Set model to evaluation mode
    jdm_model.eval()
    
    # Create priors for sampling
    n_samples = 4
    
    # Get reference samples from validation set
    val_batch = next(iter(val_dataloader))
    ref_hl = val_batch["hl_discrete_action_seq"][0].reshape(1, -1).to(device)
    ref_ll = val_batch["ll_traj"][0].reshape(1, -1).to(device)
    
    # Create priors with fixed initial conditions
    prior_hl = torch.zeros((n_samples, hl_dim), device=device)
    prior_ll = torch.zeros((n_samples, ll_dim), device=device)
    
    # Apply fixed masks to priors
    for i in range(n_samples):
        prior_hl[i] = prior_hl[i] * (1 - fix_mask_hl) + ref_hl * fix_mask_hl
        prior_ll[i] = prior_ll[i] * (1 - fix_mask_ll) + ref_ll * fix_mask_ll
    
    # Sample with different temperatures
    temperatures = [0.8, 1.0, 1.2, 1.5]
    all_samples_hl = []
    all_samples_ll = []
    
    print("Generating samples with different temperatures...")
    for i, temp in enumerate(temperatures):
        print(f"  Temperature = {temp}")
        
        samples_hl, samples_ll, sample_log = jdm_model.sample(
            prior_hl=prior_hl[i:i+1],
            prior_ll=prior_ll[i:i+1],
            sample_steps=20,
            temperature=temp,
            preserve_history=True
        )
        
        all_samples_hl.append(samples_hl)
        all_samples_ll.append(samples_ll)
    
    all_samples_hl = torch.cat(all_samples_hl, dim=0)
    all_samples_ll = torch.cat(all_samples_ll, dim=0)
    
    print(f"Generated samples: HL shape={all_samples_hl.shape}, LL shape={all_samples_ll.shape}")
    
    # Visualize the samples
    print("\nCreating visualizations...")
    
    # Reshape samples for visualization
    hl_samples_vis = all_samples_hl.reshape(n_samples, dataset.hl_horizon, dataset.hl_bits).cpu().numpy()
    ll_samples_vis = all_samples_ll.reshape(n_samples, dataset.ll_horizon, dataset.ll_dims).cpu().numpy()
    
    # Visualize high-level action sequences
    plt.figure(figsize=(12, 8))
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)
        plt.imshow(hl_samples_vis[i], aspect='auto', cmap='Blues')
        plt.colorbar(label='Value')
        plt.title(f"Action Sequence (Temp={temperatures[i]})")
        plt.ylabel("Action Step")
        plt.xlabel("Action Bit")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hl_action_sequences.png"))
    
    # Visualize low-level motion trajectories
    plt.figure(figsize=(12, 10))
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)
        
        # Extract x, z coordinates
        x = ll_samples_vis[i, :, 1]
        z = ll_samples_vis[i, :, 2]
        
        # Plot trajectory
        plt.scatter(x, z, c=range(len(x)), cmap='viridis', s=30, alpha=0.8)
        plt.plot(x, z, 'k-', alpha=0.3)
        
        # Mark start and end points
        plt.scatter(x[0], z[0], c='green', s=100, marker='o', label='Start')
        plt.scatter(x[-1], z[-1], c='red', s=100, marker='x', label='End')
        
        # Add action segment boundaries
        for seg_idx in range(1, dataset.hl_horizon):
            boundary_idx = seg_idx * (dataset.ll_horizon // dataset.hl_horizon)
            plt.axvline(x=x[boundary_idx], color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f"Motion Trajectory (Temp={temperatures[i]})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ll_motion_trajectories.png"))
    
    print(f"\nTest completed successfully! Outputs saved to {output_dir}")
    return jdm_model


if __name__ == "__main__":
    test_jdm_diffusion()