import os
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.nn_diffusion import PearceObsCondition
from cleandiffuser.utils import report_parameters
from utils import set_seed

# Import our JDM implementation and BlocksWorld dataset
from cleandiffuser.diffusion import JdmContinuousDiffusionSDE
from cleandiffuser.dataset.blocks_world_dataset import BlocksWorldDataset

@hydra.main(config_path="../configs/diffuser/blocks_world", config_name="blocks_world", version_base=None)
def pipeline(args):
    """
    Pipeline for training and evaluating the JDM diffusion model on the BlocksWorld environment.
    This model jointly learns discrete actions and continuous motion planning.
    """
    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    dataset = BlocksWorldDataset(
        data_path=args.task.data_path,
        horizon=args.task.hl_horizon,
        steps_per_action=args.task.steps_per_action,
        max_predicates=args.task.max_predicates,
        max_blocks=args.task.max_blocks,
        bit_dim=args.task.bit_dim,
        tokenizer_save_path=args.task.tokenizer_path,
        do_normalize=args.task.normalize_data,
        discount=args.discount
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    
    # ------------------- Set Dimensions & Shapes -------------------
    # High-level dimensions (discrete symbolic actions)
    hl_bit_dim = args.task.bit_dim
    hl_horizon = args.task.hl_horizon
    hl_dim = hl_bit_dim
    
    # Low-level dimensions (continuous motion trajectories)
    ll_horizon = args.task.hl_horizon * args.task.steps_per_action  # Total motion steps
    ll_dim = 3  # (time, x, z) for each motion step
    
    # Input dimensions for conditions
    init_discrete_dim = args.task.max_predicates * hl_bit_dim
    goal_discrete_dim = args.task.max_predicates * hl_bit_dim
    block_coords_dim = args.task.max_blocks * 2
    
    # --------------- Network Architecture (High-Level) -----------------
    # Network for high-level diffusion (discrete actions)
    nn_diffusion_hl = JannerUNet1d(
        hl_horizon * hl_dim,  # Input dimension: (horizon, bit_dim)
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult_hl,
        timestep_emb_type="positional",
        attention=True,
        kernel_size=5
    )
    
    # Condition network for LL -> HL (motion to action conditioning)
    nn_condition_ll_to_hl = PearceObsCondition(
        obs_dim=ll_horizon * ll_dim,  # Motion trajectory
        emb_dim=args.model_dim,
        flatten=True,
        dropout=0.1
    )
    
    # Classifier for HL guidance (optional)
    nn_classifier_hl = None
    if args.use_classifier_guidance:
        nn_classifier_hl = HalfJannerUNet1d(
            hl_horizon,
            hl_dim,
            out_dim=1,
            model_dim=args.model_dim,
            emb_dim=args.model_dim,
            dim_mult=args.task.dim_mult_hl,
            timestep_emb_type="positional",
            kernel_size=3
        )
    
    # --------------- Network Architecture (Low-Level) -----------------
    # Network for low-level diffusion (motion trajectories)
    nn_diffusion_ll = JannerUNet1d(
        ll_horizon * ll_dim,  # Input dimension: (ll_horizon, 3)
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult_ll,
        timestep_emb_type="positional",
        attention=True,
        kernel_size=5
    )
    
    # Condition network for HL -> LL (action to motion conditioning)
    nn_condition_hl_to_ll = PearceObsCondition(
        obs_dim=hl_horizon * hl_dim,  # Action sequence
        emb_dim=args.model_dim,
        flatten=True,
        dropout=0.1
    )
    
    # Classifier for LL guidance (optional)
    nn_classifier_ll = None
    if args.use_classifier_guidance:
        nn_classifier_ll = HalfJannerUNet1d(
            ll_horizon,
            ll_dim,
            out_dim=1,
            model_dim=args.model_dim,
            emb_dim=args.model_dim,
            dim_mult=args.task.dim_mult_ll,
            timestep_emb_type="positional",
            kernel_size=3
        )
    
    # Report model parameters
    print(f"======================= Parameter Report of HL Diffusion Model =======================")
    report_parameters(nn_diffusion_hl)
    print(f"======================= Parameter Report of LL Diffusion Model =======================")
    report_parameters(nn_diffusion_ll)
    if nn_classifier_hl is not None:
        print(f"======================= Parameter Report of HL Classifier =======================")
        report_parameters(nn_classifier_hl)
    if nn_classifier_ll is not None:
        print(f"======================= Parameter Report of LL Classifier =======================")
        report_parameters(nn_classifier_ll)
    print(f"==============================================================================")
    
    # --------------- Classifier Guidance (if enabled) --------------------
    classifier_hl = None
    classifier_ll = None
    
    if args.use_classifier_guidance and nn_classifier_hl is not None:
        classifier_hl = CumRewClassifier(nn_classifier_hl, device=args.device)
    
    if args.use_classifier_guidance and nn_classifier_ll is not None:
        classifier_ll = CumRewClassifier(nn_classifier_ll, device=args.device)
    
    # ----------------- Masking for Conditioning -------------------
    # High-level: fix initial and goal states as conditions
    fix_mask_hl = torch.zeros((hl_horizon, hl_dim))
    # We don't need to mask any part of HL actions, as we'll use the init/goal 
    # states as separate conditions through the condition network
    
    # Low-level: fix initial and goal robot end-effector positions
    fix_mask_ll = torch.zeros((ll_horizon, ll_dim))
    # Set first and last positions as fixed (or can be handled via conditioning)
    # The first position of each segment is the initial position for that action
    for i in range(0, ll_horizon, args.task.steps_per_action):
        fix_mask_ll[i, :] = 1.0  # Fix the first position of each segment
    
    # Loss weights
    loss_weight_hl = torch.ones((hl_horizon, hl_dim))
    loss_weight_ll = torch.ones((ll_horizon, ll_dim))
    
    # --------------- JDM Diffusion Model --------------------
    agent = JdmContinuousDiffusionSDE(
        # Neural networks
        nn_diffusion_hl=nn_diffusion_hl,
        nn_diffusion_ll=nn_diffusion_ll,
        nn_condition_hl_to_ll=nn_condition_hl_to_ll,
        nn_condition_ll_to_hl=nn_condition_ll_to_hl,
        
        # Masks
        fix_mask_hl=fix_mask_hl,
        fix_mask_ll=fix_mask_ll,
        loss_weight_hl=loss_weight_hl,
        loss_weight_ll=loss_weight_ll,
        
        # Classifiers
        classifier_hl=classifier_hl,
        classifier_ll=classifier_ll,
        
        # Hierarchical parameters
        steps_per_action=args.task.steps_per_action,
        
        # Training parameters
        grad_clip_norm=args.grad_clip_norm,
        ema_rate=args.ema_rate,
        
        # Diffusion parameters
        noise_schedule=args.noise_schedule,
        predict_noise=args.predict_noise,
        
        # Device
        device=args.device
    )
    
    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Set up schedulers for learning rates
        diffusion_hl_lr_scheduler = CosineAnnealingLR(agent.optimizer_hl, args.diffusion_gradient_steps)
        diffusion_ll_lr_scheduler = CosineAnnealingLR(agent.optimizer_ll, args.diffusion_gradient_steps)
        
        if classifier_hl is not None:
            classifier_hl_lr_scheduler = CosineAnnealingLR(agent.classifier_hl.optim, args.classifier_gradient_steps)
        if classifier_ll is not None:
            classifier_ll_lr_scheduler = CosineAnnealingLR(agent.classifier_ll.optim, args.classifier_gradient_steps)
        
        agent.train()
        
        n_gradient_step = 0
        log = {
            "avg_loss_hl": 0.,
            "avg_loss_ll": 0.,
            "avg_loss_combined": 0.,
            "avg_loss_classifier_hl": 0.,
            "avg_loss_classifier_ll": 0.,
        }
        
        for batch in loop_dataloader(dataloader):
            # Process batch data
            # 1. High-level data: discrete actions
            hl_actions = batch["hl_discrete_action_seq"].to(args.device)  # (B, 8, 6)
            
            # Conditions for HL
            init_discrete = batch["obs"]["init_discrete"].to(args.device)  # (B, 11, 6)
            goal_discrete = batch["obs"]["goal_discrete"].to(args.device)  # (B, 11, 6)
            
            # 2. Low-level data: motion trajectories
            ll_traj = batch["ll_traj"].to(args.device)  # (B, 48, 3)
            
            # Conditions for LL
            init_coords_block = batch["obs"]["init_coords_block"].to(args.device)  # (B, 5, 2)
            goal_coords_block = batch["obs"]["goal_coords_block"].to(args.device)  # (B, 5, 2)
            
            # Concatenate init and goal states for conditioning
            condition_hl = torch.cat([
                init_discrete.reshape(init_discrete.shape[0], -1),
                goal_discrete.reshape(goal_discrete.shape[0], -1)
            ], dim=1)  # (B, 11*6 + 11*6)
            
            condition_ll = torch.cat([
                init_coords_block.reshape(init_coords_block.shape[0], -1),
                goal_coords_block.reshape(goal_coords_block.shape[0], -1)
            ], dim=1)  # (B, 5*2 + 5*2)
            
            # ----------- Gradient Step ------------
            # Update both HL and LL diffusion models
            update_log = agent.update(
                x0_hl=hl_actions,
                x0_ll=ll_traj,
                condition_hl=condition_hl,
                condition_ll=condition_ll,
                update_ema=True,
                hl_weight=args.hl_loss_weight,
                ll_weight=args.ll_loss_weight,
                update_both=True
            )
            
            # Update learning rate schedulers
            diffusion_hl_lr_scheduler.step()
            diffusion_ll_lr_scheduler.step()
            
            # Update classifiers if they exist
            if args.use_classifier_guidance and n_gradient_step <= args.classifier_gradient_steps:
                classifier_log = agent.update_classifiers(
                    x0_hl=hl_actions,
                    x0_ll=ll_traj,
                    condition_hl=condition_hl,
                    condition_ll=condition_ll
                )
                
                if classifier_hl is not None:
                    classifier_hl_lr_scheduler.step()
                    log["avg_loss_classifier_hl"] += classifier_log.get("classifier_hl", {}).get("loss", 0)
                
                if classifier_ll is not None:
                    classifier_ll_lr_scheduler.step()
                    log["avg_loss_classifier_ll"] += classifier_log.get("classifier_ll", {}).get("loss", 0)
            
            # Update running averages for logging
            log["avg_loss_hl"] += update_log["loss_hl"]
            log["avg_loss_ll"] += update_log["loss_ll"]
            log["avg_loss_combined"] += update_log["loss_combined"]
            
            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_hl"] /= args.log_interval
                log["avg_loss_ll"] /= args.log_interval
                log["avg_loss_combined"] /= args.log_interval
                
                if args.use_classifier_guidance:
                    log["avg_loss_classifier_hl"] /= args.log_interval
                    log["avg_loss_classifier_ll"] /= args.log_interval
                
                print(log)
                
                # Reset running averages
                log = {
                    "avg_loss_hl": 0.,
                    "avg_loss_ll": 0.,
                    "avg_loss_combined": 0.,
                    "avg_loss_classifier_hl": 0.,
                    "avg_loss_classifier_ll": 0.,
                }
            
            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save_checkpoint(save_path + f"jdm_diffusion", n_gradient_step + 1, log)
            
            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break
        
        # Save final model
        agent.save_checkpoint(save_path + f"jdm_diffusion", n_gradient_step, log)
    
    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        # Load trained model
        agent.load_checkpoint(save_path + f"jdm_diffusion_step{args.ckpt}.pt")
        agent.eval()
        
        # Get a test batch
        test_loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True, num_workers=0)
        test_batch = next(iter(test_loader))
        
        # Prepare conditions
        init_discrete = test_batch["obs"]["init_discrete"].to(args.device)
        goal_discrete = test_batch["obs"]["goal_discrete"].to(args.device)
        init_coords_block = test_batch["obs"]["init_coords_block"].to(args.device)
        goal_coords_block = test_batch["obs"]["goal_coords_block"].to(args.device)
        
        # Create prior tensors with fixed conditions
        prior_hl = torch.zeros((args.num_samples, hl_horizon, hl_dim), device=args.device)
        prior_ll = torch.zeros((args.num_samples, ll_horizon, ll_dim), device=args.device)
        
        # Set initial conditions in the prior masks
        # For HL: init and goal discrete states could be added as fixed parts
        # For LL: init and goal positions could be added as fixed parts
        
        # Sample from the model
        hl_samples, ll_samples, sample_log = agent.sample(
            prior_hl=prior_hl,
            prior_ll=prior_ll,
            solver=args.solver,
            n_samples=args.num_samples,
            sample_steps=args.sampling_steps,
            use_ema=args.use_ema,
            temperature=args.temperature,
            w_cfg_hl=args.task.w_cfg_hl,
            w_cfg_ll=args.task.w_cfg_ll,
            preserve_history=True
        )
        
        # Process and evaluate samples
        # This will depend on your specific evaluation metrics
        print(f"Generated {args.num_samples} samples.")
        
        # Save samples
        samples_dict = {
            "hl_samples": hl_samples.cpu().numpy(),
            "ll_samples": ll_samples.cpu().numpy(),
            "init_discrete": init_discrete.cpu().numpy(),
            "goal_discrete": goal_discrete.cpu().numpy(),
            "init_coords_block": init_coords_block.cpu().numpy(),
            "goal_coords_block": goal_coords_block.cpu().numpy(),
            "sample_history_hl": sample_log.get("sample_history_hl"),
            "sample_history_ll": sample_log.get("sample_history_ll"),
        }
        
        np.save(save_path + f"samples_{args.ckpt}.npy", samples_dict)
        print(f"Samples saved to {save_path}samples_{args.ckpt}.npy")
        
        # Visualize samples (if applicable)
        # This will depend on your specific visualization needs
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()