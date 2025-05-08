import os
import hydra
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb, uuid
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.nn_condition import PearceObsCondition, PearceObsConditionV2
from cleandiffuser.utils import report_parameters, set_seed, pad_to_next_power_of_2
from cleandiffuser.dataset.blocks_world_dataset import BlocksWorldDataset

from cleandiffuser.diffusion import JdmContinuousDiffusionSDE
from cleandiffuser.render.blocks_world_render import BlocksWorldRender
from omegaconf import OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true" if you want to enable it
# ---------------------- Define Dimensions ----------------------
# High-level input dimensions: (11 + 11 + 8 )*6
# High-level mask: (11 + 11 )*6 [part of HL input]
# Low-level input dimensions: (5 + 5 + 2 + 48)*3
# Low-level mask: (5 + 5 + 2)*3 [part of LL input]
# --------------- Network Architecture -----------------
@hydra.main(config_path="../configs/diffuser/blocks_world", config_name="blocks_world", version_base=None)
def pipeline(args):
    """
    Main pipeline for training and evaluating the JDM diffuser on the Blocks World task.
    """
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    
    save_path = f'results/{args.name}/{args.task.env_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.mode == "train":
        config_path = os.path.join(save_path, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(args, f)
        print(f"[BlcoksWorldTrain] Configuration saved to {config_path}")

    # ---------------------- Create Dataset ----------------------
    dataset = BlocksWorldDataset(
        data_path=args.task.data_path,
        horizon=args.task.horizon,
        steps_per_action=args.task.steps_per_action,
        max_predicates=args.task.max_predicates,
        max_blocks=args.task.max_blocks,
        bit_dim=args.task.bit_dim,
        do_normalize=True,
        discount=args.discount,
        tokenizer_save_path=args.task.tokenizer_save_path,
    )
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )

    # ---------------------- Define Dimensions ----------------------
    # High-level dimensions
    hl_obs_dim = (args.task.max_predicates * 2)  # init + goal predicates
    hl_act_dim = args.task.horizon # HL discrete actions (bits)
    hl_overall_dim = hl_obs_dim + hl_act_dim  # Overall high-level dimension
    hl_horizon = args.task.horizon + args.task.max_predicates * 2
    next_power_of_2 = 2**(hl_horizon - 1).bit_length()
    # For jannner unet, we need to pad the input to the next power of 2
    hl_horizon = next_power_of_2

    # Low-level dimensions
    ll_obs_dim = (args.task.max_blocks * 2 + 2)  # init + goal blocks (adding time dim) + ee coords
    ll_act_dim = args.task.horizon * args.task.steps_per_action  # LL motion trajectory (time, x, z)
    ll_overall_dim = ll_obs_dim + ll_act_dim  # Overall low-level dimension
    ll_horizon = args.task.max_blocks * 2 + 2 + args.task.horizon * args.task.steps_per_action
    next_power_of_2 = 2**(ll_horizon - 1).bit_length()
    # For jannner unet, we need to pad the input to the next power of 2
    ll_horizon = next_power_of_2
    # ---------------------- Network Architecture ----------------------
    if args.nn == "pearce":
        # High-level diffusion network (for discrete symbolic actions)
        nn_diffusion_hl = JannerUNet1d(
            args.task.bit_dim, model_dim=args.model_dim, emb_dim=args.model_dim, 
            dim_mult=args.task.dim_mult_hl,
            timestep_emb_type="positional", attention=True, kernel_size=5
        )
        
        # Low-level diffusion network (for continuous motion)
        nn_diffusion_ll = JannerUNet1d(
            args.task.motion_dim, model_dim=args.model_dim, emb_dim=args.model_dim, 
            dim_mult=args.task.dim_mult_ll,
            timestep_emb_type="positional", attention=True, kernel_size=5
        )
    elif args.nn == "dit":
        from cleandiffuser.nn_diffusion import DiT1d
        nn_diffusion_hl = DiT1d(
            args.task.bit_dim, emb_dim=args.model_dim, d_model=320, n_heads=args.n_heads_HL, depth=args.depth_HL, timestep_emb_type="fourier").to(args.device)
        nn_diffusion_ll = DiT1d(
            args.task.motion_dim, emb_dim=args.model_dim, d_model=320, n_heads=args.n_heads_LL, depth=args.depth_LL, timestep_emb_type="fourier").to(args.device)
    
    # Cross-attention condition networks
    # x_ll | x_hl_emb
    nn_condition_hl_to_ll = PearceObsConditionV2(
        obs_dim=args.task.bit_dim, emb_dim=args.model_dim, in_horizon=32 ,out_horizon=1, flatten=True, dropout=0.1
    )
    # x_hl | x_ll_emb
    nn_condition_ll_to_hl = PearceObsConditionV2(
        obs_dim=args.task.motion_dim, emb_dim=args.model_dim, in_horizon=64, out_horizon=1, flatten=True, dropout=0.1
    )
    # TODO: make the encoder larger for the condition networks
    # Print model parameter summaries
    print(f"==== Parameter Report of High-Level Diffusion Model ====")
    report_parameters(nn_diffusion_hl)
    print(f"==== Parameter Report of Low-Level Diffusion Model ====")
    report_parameters(nn_diffusion_ll)
    print(f"==== Parameter Report of HL->LL Condition Network ====")
    report_parameters(nn_condition_hl_to_ll, topk=4)
    print(f"==== Parameter Report of LL->HL Condition Network ====")
    report_parameters(nn_condition_ll_to_hl, topk=4)
    print(f"======================================================")

    # ---------------------- Create Masks for Fixed Components ----------------------
    # High-level: fix initial and goal predicates
    fix_mask_hl = torch.zeros((hl_horizon, args.task.bit_dim), device=args.device)
    fix_mask_hl[:hl_obs_dim] = 1.0  # Fix all observation components (init + goal)
    
    # Low-level: fix initial and goal coordinates
    fix_mask_ll = torch.zeros((ll_horizon, args.task.motion_dim), device=args.device)
    fix_mask_ll[:ll_obs_dim + 2] = 1.0  # Fix all observation components (block coords + ee_init + ee_goal)

    # ---------------------- Loss Weights ----------------------
    # Add higher weight to action components if needed
    loss_weight_hl = torch.ones((hl_horizon, args.task.bit_dim), device=args.device)
    # loss_weight_hl[hl_obs_dim:] = args.hl_action_loss_weight

    
    loss_weight_ll = torch.ones((ll_horizon, args.task.motion_dim), device=args.device)
    # loss_weight_ll[ll_obs_dim:] = args.ll_action_loss_weight

    # ---------------------- Create JDM Diffusion Model ----------------------
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
        
        # Hierarchical parameters
        steps_per_action=args.task.steps_per_action,
        
        # Training parameters
        grad_clip_norm=args.grad_clip_norm,
        ema_rate=args.ema_rate,
        optim_params_hl={"lr": args.lr_hl, "weight_decay": args.weight_decay},
        optim_params_ll={"lr": args.lr_ll, "weight_decay": args.weight_decay},
        expected_sample_steps=args.expected_sample_steps,
        
        # Diffusion parameters
        epsilon=1e-3,
        noise_schedule=args.noise_schedule,
        predict_noise=args.predict_noise,
        device=args.device
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Create learning rate schedulers
        diffusion_hl_lr_scheduler = CosineAnnealingLR(agent.optimizer_hl, args.max_train_steps)
        diffusion_ll_lr_scheduler = CosineAnnealingLR(agent.optimizer_ll, args.max_train_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_hl": 0., "avg_loss_ll": 0., "avg_loss_combined": 0.}
        
        #load model from checkpoint if provided
        if args.load_checkpoint_train:
            agent.load(save_path + f"jdm_diffusion_step{args.ckpt}.pt" if args.ckpt else save_path + "jdm_diffusion_final.pt")
            n_gradient_step = args.ckpt
            print(f"[BlocksWorldTrain] Loaded checkpoint from {save_path + f'jdm_diffusion_step{args.ckpt}.pt' if args.ckpt else save_path + 'jdm_diffusion_final.pt'}")

        for batch in loop_dataloader(dataloader):
            # -------------------- Process Batch --------------------
            # Process high-level components
            init_discrete = batch["obs"]["init_discrete"].to(args.device)  # (B, 11, 6)
            goal_discrete = batch["obs"]["goal_discrete"].to(args.device)  # (B, 11, 6)
            hl_actions = batch["hl_discrete_action_seq"].to(args.device)  # (B, 8, 6)
            
            # Process low-level components
            init_coords_block = batch["obs"]["init_coords_block"].to(args.device)  # (B, 5, 2)
            goal_coords_block = batch["obs"]["goal_coords_block"].to(args.device)  # (B, 5, 2)
            init_coords_ee = batch["obs"]["init_coords_ee"].to(args.device).unsqueeze(1)  # (B, 1, 2)
            goal_coords_ee = batch["obs"]["goal_coords_ee"].to(args.device).unsqueeze(1)  # (B, 1, 2)
            ll_traj = batch["ll_traj"].to(args.device)  # (B, 48, 3)
            
            # Process segment indices (might be needed for future conditioning)
            segment_idx = batch["segment_idx"].to(args.device)  # (B, 8, 2)
            
            # -------------------- Prepare Inputs --------------------
            batch_size = init_discrete.shape[0]

            # High-level: (B, 11, 6) + (B, 11, 6) + (B, 8, 6) ==> (B, 30, 6)
            # CHANGED ↓↓↓ -- no flatten, no unsqueeze
            x0_hl = torch.cat([init_discrete, goal_discrete, hl_actions], dim=1)  # shape: (B, 11+11+8=30, 6)

            # Low-level: we want (B, 60, 3) total
            #   each part is (B, #segments, 3), then cat along dim=1
            time_dim = torch.zeros(batch_size, args.task.max_blocks, 1, device=args.device)
            # (B, 5, 2) -> add time => (B, 5, 3)
            init_coords_block_with_time = torch.cat([time_dim, init_coords_block], dim=2)
            goal_coords_block_with_time = torch.cat([time_dim, goal_coords_block], dim=2)

            # (B, 1, 2) -> add time => (B, 1, 3)
            init_coords_ee_with_time = torch.cat([
                torch.zeros(batch_size, 1, 1, device=args.device),
                init_coords_ee
            ], dim=2)
            # we use time = 0 for the goal ee coords
            goal_coords_ee_with_time = torch.cat([
                torch.zeros(batch_size, 1, 1, device=args.device),
                goal_coords_ee
            ], dim=2)

            # CHANGED ↓↓↓ -- no flatten, no unsqueeze
            # Now simply cat along dim=1 to get (B, 60, 3)
            x0_ll = torch.cat([
                init_coords_block_with_time,    # (B, 5, 3)
                goal_coords_block_with_time,    # (B, 5, 3)
                init_coords_ee_with_time,       # (B, 1, 3)
                goal_coords_ee_with_time,       # (B, 1, 3)
                ll_traj                         # (B, 50, 3)
            ], dim=1)  # => shape (B, 60, 3)
            # -------------------- Update Model --------------------
            # Compute weights for this batch (can adjust based on training progress)
            hl_weight = args.hl_weight
            ll_weight = args.ll_weight
            
            # Perform gradient update (we're using inpainting via masks, so no explicit condition)
            update_log = agent.update(
                x0_hl=x0_hl, 
                x0_ll=x0_ll,
                condition_hl=None,  # Using inpainting for fixed parts
                condition_ll=None,  # Using inpainting for fixed parts
                update_ema=True,
                hl_weight=hl_weight,
                ll_weight=ll_weight,
                update_both=True
            )
            
            # Update learning rate schedulers
            diffusion_hl_lr_scheduler.step()
            diffusion_ll_lr_scheduler.step()
            
            # Update logs
            log["avg_loss_hl"] += update_log["loss_hl"]
            log["avg_loss_ll"] += update_log["loss_ll"]
            log["avg_loss_combined"] += update_log["loss_combined"]
            
            # -------------------- Logging and Saving --------------------
            n_gradient_step += 1
            
            # Log periodically
            if n_gradient_step % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step
                log["avg_loss_hl"] /= args.log_interval
                log["avg_loss_ll"] /= args.log_interval
                log["avg_loss_combined"] /= args.log_interval
                print(f"Step {n_gradient_step}: {log}")
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                log = {"avg_loss_hl": 0., "avg_loss_ll": 0., "avg_loss_combined": 0.}
            
            # Save periodically
            if n_gradient_step % args.save_interval == 0:
                agent.save_checkpoint(save_path + f"jdm_diffusion", n_gradient_step, log)
            
            # Break if we've reached the maximum number of gradient steps
            if n_gradient_step >= args.max_train_steps:
                print(f"Training completed after {n_gradient_step} gradient steps.")
                break
        
        # Save final model
        agent.save(save_path + f"jdm_diffusion_final.pt")
        print(f"Training completed after {n_gradient_step} gradient steps. Model saved to {save_path}")

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        # Load the trained model
        agent.load(save_path + f"jdm_diffusion_step{args.ckpt}.pt" if args.ckpt else save_path + "jdm_diffusion_final.pt")
        agent.eval()

        # Create result directory if it doesn't exist
        result_dir = os.path.join(save_path, "inference_results")
        os.makedirs(result_dir, exist_ok=True)

        # Sample from the validation set
        val_dataset = BlocksWorldDataset(
            data_path=args.task.val_data_path if hasattr(args.task, 'val_data_path') else args.task.data_path,
            horizon=args.task.horizon,
            steps_per_action=args.task.steps_per_action,
            max_predicates=args.task.max_predicates,
            max_blocks=args.task.max_blocks,
            bit_dim=args.task.bit_dim,
            do_normalize=True,
            discount=args.discount
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )

        # Create renderer for visualizing results
        renderer = BlocksWorldRender(
        tokenizer_save_path=args.task.tokenizer_save_path,
        n_bits=args.task.bit_dim
    )
        
        # Run inference on samples
        success_rate = 0.0
        hl_success_rate = 0.0
        ll_success_rate = 0.0
        total_samples = 0
        batch_size = args.eval_batch_size

        # Create dictionaries to store results for later analysis
        all_results = {
            "init_discrete": [],
            "goal_discrete": [],
            "gt_actions": [],
            "pred_actions": [],
            "gt_motion": [],
            "pred_motion": [],
            "hl_errors": [],
            "ll_errors": [],
            "hl_success": [],
            "ll_success": [],
            "combined_success": []
        }
        
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= args.eval_num_batches:
                break
                
            # Process batch similar to training
            # -------------------- Process Batch --------------------
            # Process high-level components
            init_discrete = batch["obs"]["init_discrete"].to(args.device)  # (B, 11, 6)
            goal_discrete = batch["obs"]["goal_discrete"].to(args.device)  # (B, 11, 6)
            gt_hl_actions = batch["hl_discrete_action_seq"].to(args.device)  # (B, 8, 6)
            
            # Process low-level components
            init_coords_block = batch["obs"]["init_coords_block"].to(args.device)  # (B, 5, 2)
            goal_coords_block = batch["obs"]["goal_coords_block"].to(args.device)  # (B, 5, 2)
            init_coords_ee = batch["obs"]["init_coords_ee"].to(args.device).unsqueeze(1)  # (B, 1, 2)
            # we use time = 0 for the goal ee coords
            goal_coords_ee = batch["obs"]["goal_coords_ee"].to(args.device).unsqueeze(1)  # (B, 1, 2)
            gt_ll_traj = batch["ll_traj"].to(args.device)  # (B, 48, 3)
            
            # Process segment indices (might be needed for future conditioning)
            segment_idx = batch["segment_idx"].to(args.device)  # (B, 8, 2)
            
            # -------------------- Prepare Inputs --------------------
            # Prepare HL prior: (B, 32, 6)
            #  - We only know init (B,11,6) + goal (B,11,6), so we fill zeros for (B,8,6).
            hl_zeros = torch.zeros(batch_size, args.task.horizon, args.task.bit_dim, device=args.device)
            prior_hl = torch.cat([init_discrete, goal_discrete, hl_zeros], dim=1)  # => (B, 30, 6)
            #  - For jannner unet, we need to pad the input to the next power of 2
            prior_hl = pad_to_next_power_of_2(prior_hl)  # Pad to next power of 2
            # copy the last element for dim 1 of prior_hl to the next 2^n
            # Prepare LL prior: (B, 64, 3)
            #  - We only know init+goal blocks (B,5,2), ee init+goal (B,1,2), so we fill zeros for (B,48,3) trajectory.

            time_dim = torch.zeros(batch_size, args.task.max_blocks, 1, device=args.device)  # => (B,5,1)
            init_coords_block_with_time = torch.cat([time_dim, init_coords_block], dim=2)  # => (B,5,3)
            goal_coords_block_with_time = torch.cat([time_dim, goal_coords_block], dim=2)  # => (B,5,3)

            init_coords_ee_with_time = torch.cat([
                torch.zeros(batch_size, 1, 1, device=args.device), 
                init_coords_ee
            ], dim=2)  # => (B,1,3)
            goal_coords_ee_with_time = torch.cat([
                torch.zeros(batch_size, 1, 1, device=args.device), 
                goal_coords_ee
            ], dim=2)  # => (B,1,3)

            ll_zeros = torch.zeros(batch_size, 48, 3, device=args.device)  # (B,48,3)
            prior_ll = torch.cat([
                init_coords_block_with_time,  # (B,5,3)
                goal_coords_block_with_time,  # (B,5,3)
                init_coords_ee_with_time,     # (B,1,3)
                goal_coords_ee_with_time,     # (B,1,3)
                ll_zeros                      # (B,48,3)
            ], dim=1)  # => (B,60,3)
            #  - For jannner unet, we need to pad the input to the next power of 2
            prior_ll = pad_to_next_power_of_2(prior_ll)
            # copy the last element for dim 1 of prior_ll to the next 2^n
            #  - We need to pad the input to the next power of 2
            #  - For jannner unet, we need to pad the input to the next power of 2
            
            # -------------------- Sample from the model --------------------
            with torch.no_grad():
                # For each batch, generate multiple samples and pick the best one
                all_samples_hl = []
                all_samples_ll = []
                all_logs = []
                
                for _ in range(args.num_samples_per_inference):
                    samples_hl, samples_ll, log = agent.sample(
                        prior_hl=prior_hl,
                        prior_ll=prior_ll,
                        solver=args.solver,
                        n_samples=batch_size,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema,
                        temperature=args.temperature,
                        w_cfg_hl=args.w_cfg_hl,
                        w_cfg_ll=args.w_cfg_ll,
                        preserve_history=args.preserve_history
                    )
                    
                    all_samples_hl.append(samples_hl)
                    all_samples_ll.append(samples_ll)
                    all_logs.append(log)
                
                # TODO: Select the best sample for each batch item (e.g., based on some metric)
                # For this demo, we'll just use the first sample
                best_samples_hl = all_samples_hl[0]
                best_samples_ll = all_samples_ll[0]
                
                # Extract the action part from the HL samples
                hl_action_start = hl_obs_dim
                # TODO: revise the horizon in reshape to more general
                sampled_hl_actions = best_samples_hl[:, hl_action_start:].reshape(batch_size, 10, args.task.bit_dim)
                
                # Extract the trajectory part from the LL samples
                ll_act_start = ll_obs_dim
                # TODO: revise the horizon in reshape to more general
                # 52 = 48 + 4[PAD]
                sampled_ll_traj = best_samples_ll[:, ll_act_start:].reshape(batch_size, 52, args.task.motion_dim)
                
                # slice sampled_hl_actions and sampled_ll_traj to the original size
                sampled_hl_actions = sampled_hl_actions[:, :hl_horizon, :]
                sampled_ll_traj = sampled_ll_traj[:, :ll_horizon, :]

                # Evaluate success (for demonstration, using a simple criterion)
                # In a real implementation, you'd evaluate against task-specific success metrics
                hl_errors = torch.mean((sampled_hl_actions - gt_hl_actions) ** 2, dim=(1, 2))
                ll_errors = torch.mean((sampled_ll_traj - gt_ll_traj) ** 2, dim=(1, 2))
                
                # Count successes (using arbitrary thresholds for demonstration)
                hl_success = (hl_errors < args.hl_success_threshold).float()
                ll_success = (ll_errors < args.ll_success_threshold).float()
                
                # Combined success requires both HL and LL to succeed
                combined_success = hl_success * ll_success
                
                # Update statistics
                success_rate += combined_success.sum().item()
                hl_success_rate += hl_success.sum().item()
                ll_success_rate += ll_success.sum().item()
                total_samples += batch_size

                # Store results for later analysis
                all_results["init_discrete"].append(init_discrete.cpu().numpy())
                all_results["goal_discrete"].append(goal_discrete.cpu().numpy())
                all_results["gt_actions"].append(gt_hl_actions.cpu().numpy())
                all_results["pred_actions"].append(sampled_hl_actions.cpu().numpy())
                all_results["gt_motion"].append(gt_ll_traj.cpu().numpy())
                all_results["pred_motion"].append(sampled_ll_traj.cpu().numpy())
                all_results["hl_errors"].append(hl_errors.cpu().numpy())
                all_results["ll_errors"].append(ll_errors.cpu().numpy())
                all_results["hl_success"].append(hl_success.cpu().numpy())
                all_results["ll_success"].append(ll_success.cpu().numpy())
                all_results["combined_success"].append(combined_success.cpu().numpy())
                    
                # Print batch results
                print(f"Batch {batch_idx+1}/{min(len(val_dataloader), args.eval_num_batches)}:")
                print(f"  HL Success Rate: {hl_success.mean().item():.4f}")
                print(f"  LL Success Rate: {ll_success.mean().item():.4f}")
                print(f"  Combined Success Rate: {combined_success.mean().item():.4f}")
                
                # Save visualization samples if requested
                if args.save_samples and batch_idx < args.num_vis_batches:
                    batch_dir = os.path.join(result_dir, f"batch_{batch_idx}")
                    os.makedirs(batch_dir, exist_ok=True)
                    
                    # Prepare batch data for visualization
                    batch_data = {
                        "init_discrete": init_discrete.cpu().numpy(),
                        "goal_discrete": goal_discrete.cpu().numpy(),
                        "actions_discrete": gt_hl_actions.cpu().numpy(),
                        "motion_data": gt_ll_traj.cpu().numpy()
                    }
                    # Prepare prediction data
                    predictions = {
                        "actions_discrete": sampled_hl_actions.cpu().numpy(),
                        "motion_data": sampled_ll_traj.cpu().numpy()
                    }
                    # Visualize up to max_vis_samples_per_batch samples
                    max_samples = min(batch_size, args.max_vis_samples_per_batch)
                    
                    # Directly use the renderer to visualize each sample without concatenation
                    for sample_idx in range(max_samples):
                        sample_dir = os.path.join(batch_dir, f"sample_{sample_idx}")
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        # Save metrics for this sample
                        with open(os.path.join(sample_dir, "metrics.txt"), "w") as f:
                            f.write(f"HL Error: {hl_errors[sample_idx].item():.6f}\n")
                            f.write(f"LL Error: {ll_errors[sample_idx].item():.6f}\n")
                            f.write(f"HL Success: {bool(hl_success[sample_idx].item())}\n")
                            f.write(f"LL Success: {bool(ll_success[sample_idx].item())}\n")
                            f.write(f"Combined Success: {bool(combined_success[sample_idx].item())}\n")
                        
                        # Visualize this sample with the enhanced renderer
                        # BUG: There is a bug on HL actions visualization, all the decoded token is None
                        renderer.visualize_sample(
                            os.path.join(sample_dir, "visualization"),
                            init_discrete[sample_idx].cpu().numpy(),
                            goal_discrete[sample_idx].cpu().numpy(),
                            gt_hl_actions[sample_idx].cpu().numpy(),
                            gt_ll_traj[sample_idx].cpu().numpy(),
                            sampled_hl_actions[sample_idx].cpu().numpy(),
                            sampled_ll_traj[sample_idx].cpu().numpy()
                        )

        # Save overall statistics
        with open(os.path.join(result_dir, "overall_results.txt"), "w") as f:
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"HL Success Rate: {hl_success_rate / total_samples:.4f}\n")
            f.write(f"LL Success Rate: {ll_success_rate / total_samples:.4f}\n")
            f.write(f"Combined Success Rate: {success_rate / total_samples:.4f}\n")
    
        # Print overall results
        if total_samples > 0:
            print(f"Overall HL Success Rate: {hl_success_rate / total_samples:.4f}")
            print(f"Overall LL Success Rate: {ll_success_rate / total_samples:.4f}")
            print(f"Overall Combined Success Rate: {success_rate / total_samples:.4f}")
            if args.enable_wandb:
                wandb.log({
                    "overall_hl_success_rate": hl_success_rate / total_samples,
                    "overall_ll_success_rate": ll_success_rate / total_samples,
                    "overall_combined_success_rate": success_rate / total_samples
                })
        if args.enable_wandb:
                wandb.finish()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()