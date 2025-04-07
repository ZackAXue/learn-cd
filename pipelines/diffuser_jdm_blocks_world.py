import os
import hydra
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.utils import report_parameters, set_seed
from cleandiffuser.dataset.blocks_world_dataset import BlocksWorldDataset

from cleandiffuser.diffusion import JdmContinuousDiffusionSDE
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
    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    hl_obs_dim = (args.task.max_predicates * 2) * args.task.bit_dim  # init + goal predicates
    hl_act_dim = args.task.horizon * args.task.bit_dim  # HL discrete actions (bits)
    hl_overall_dim = hl_obs_dim + hl_act_dim  # Overall high-level dimension

    # Low-level dimensions
    ll_obs_dim = (args.task.max_blocks * 2 + 2) * 3  # init + goal blocks (adding time dim) + ee coords
    ll_act_dim = args.task.horizon * args.task.steps_per_action * 3  # LL motion trajectory (time, x, z)
    ll_overall_dim = ll_obs_dim + ll_act_dim  # Overall low-level dimension
    # ---------------------- Network Architecture ----------------------
    # High-level diffusion network (for discrete symbolic actions)
    nn_diffusion_hl = JannerUNet1d(
        hl_overall_dim, model_dim=args.model_dim, emb_dim=args.model_dim, 
        dim_mult=args.task.dim_mult_hl,
        timestep_emb_type="positional", attention=True, kernel_size=5
    )
    
    # Low-level diffusion network (for continuous motion)
    nn_diffusion_ll = JannerUNet1d(
        ll_overall_dim, model_dim=args.model_dim, emb_dim=args.model_dim, 
        dim_mult=args.task.dim_mult_ll,
        timestep_emb_type="positional", attention=True, kernel_size=5
    )
    # Cross-attention condition networks
    nn_condition_hl_to_ll = PearceObsCondition(
        obs_dim=hl_overall_dim, emb_dim=args.model_dim, flatten=True, dropout=0.1
    )
    
    nn_condition_ll_to_hl = PearceObsCondition(
        obs_dim=ll_overall_dim, emb_dim=args.model_dim, flatten=True, dropout=0.1
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
    fix_mask_hl = torch.zeros((hl_overall_dim), device=args.device)
    fix_mask_hl[:hl_obs_dim] = 1.0  # Fix all observation components (init + goal)
    fix_mask_hl = fix_mask_hl.reshape(1, -1)  # Reshape for broadcasting
    
    # Low-level: fix initial and goal coordinates
    fix_mask_ll = torch.zeros((ll_overall_dim), device=args.device)
    fix_mask_ll[:ll_obs_dim + 2] = 1.0  # Fix all observation components (block coords + ee_init + ee_goal)
    fix_mask_ll = fix_mask_ll.reshape(1, -1)  # Reshape for broadcasting
    import pdb
    # ---------------------- Loss Weights ----------------------
    # Add higher weight to action components if needed
    loss_weight_hl = torch.ones((hl_obs_dim + hl_act_dim), device=args.device)
    loss_weight_hl[hl_obs_dim:] = args.hl_action_loss_weight
    loss_weight_hl = loss_weight_hl.reshape(1, -1)  # Reshape for broadcasting
    
    loss_weight_ll = torch.ones((ll_obs_dim + ll_act_dim), device=args.device)
    loss_weight_ll[ll_obs_dim:] = args.ll_action_loss_weight
    loss_weight_ll = loss_weight_ll.reshape(1, -1)  # Reshape for broadcasting

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
        
        # Diffusion parameters
        epsilon=1e-3,
        noise_schedule=args.noise_schedule,
        predict_noise=args.predict_noise,
        device=args.device
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Create learning rate schedulers
        diffusion_hl_lr_scheduler = CosineAnnealingLR(agent.optimizer_hl, args.diffusion_gradient_steps)
        diffusion_ll_lr_scheduler = CosineAnnealingLR(agent.optimizer_ll, args.diffusion_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_hl": 0., "avg_loss_ll": 0., "avg_loss_combined": 0.}

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
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
                
                # Prepare high-level input: concatenate init+goal predicates and actions
                init_discrete_flat = init_discrete.reshape(batch_size, -1)  # (B, 11*6)
                goal_discrete_flat = goal_discrete.reshape(batch_size, -1)  # (B, 11*6)
                hl_actions_flat = hl_actions.reshape(batch_size, -1)  # (B, 8*6)
                
                x0_hl = torch.cat([init_discrete_flat, goal_discrete_flat, hl_actions_flat], dim=1)  # (B, (11+11+8)*6)
                # add one dimention at dim=1 for sequence length
                x0_hl = x0_hl.unsqueeze(1)  # (B, 1, (11+11+8)*6)
                # Prepare low-level input: concatenate block coordinates, ee coords, and motion trajectory
                # Add time dimension to block coordinates (zeros for simplicity)
                time_dim = torch.zeros(batch_size, args.task.max_blocks, 1, device=args.device)
                
                init_coords_block_with_time = torch.cat([time_dim, init_coords_block], dim=2)  # (B, 5, 3)
                goal_coords_block_with_time = torch.cat([time_dim, goal_coords_block], dim=2)  # (B, 5, 3)
                
                # Add time dimension to ee coordinates
                init_coords_ee_with_time = torch.cat([torch.zeros(batch_size, 1, 1, device=args.device), 
                                                      init_coords_ee], dim=2)  # (B, 1, 3)
                goal_coords_ee_with_time = torch.cat([torch.zeros(batch_size, 1, 1, device=args.device), 
                                                      goal_coords_ee], dim=2)  # (B, 1, 3)
                
                # Flatten and concatenate all low-level components
                init_coords_block_flat = init_coords_block_with_time.reshape(batch_size, -1)  # (B, 5*3)
                goal_coords_block_flat = goal_coords_block_with_time.reshape(batch_size, -1)  # (B, 5*3)
                init_coords_ee_flat = init_coords_ee_with_time.reshape(batch_size, -1)  # (B, 1*3)
                goal_coords_ee_flat = goal_coords_ee_with_time.reshape(batch_size, -1)  # (B, 1*3)
                ll_traj_flat = ll_traj.reshape(batch_size, -1)  # (B, 48*3)
                
                x0_ll = torch.cat([
                    init_coords_block_flat, goal_coords_block_flat,
                    init_coords_ee_flat, goal_coords_ee_flat,
                    ll_traj_flat
                ], dim=1)  # (B, (5+5+1+1)*3 + 48*3)
                # add one dimention at dim=1 for sequence length
                x0_ll = x0_ll.unsqueeze(1)  # (B, 1, (5+5+1+1)*3 + 48*3)
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
                    log = {"avg_loss_hl": 0., "avg_loss_ll": 0., "avg_loss_combined": 0.}
                
                # Save periodically
                if n_gradient_step % args.save_interval == 0:
                    agent.save_checkpoint(save_path + f"jdm_diffusion", n_gradient_step, log)
                
                # Break if we've reached the maximum number of gradient steps
                if n_gradient_step >= args.diffusion_gradient_steps:
                    break
            
            # Break if we've reached the maximum number of gradient steps
            if n_gradient_step >= args.diffusion_gradient_steps:
                break
        
        # Save final model
        agent.save(save_path + f"jdm_diffusion_final.pt")
        print(f"Training completed after {n_gradient_step} gradient steps. Model saved to {save_path}")

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        # Load the trained model
        agent.load(save_path + f"jdm_diffusion_ckpt_{args.ckpt}.pt" if args.ckpt else save_path + "jdm_diffusion_final.pt")
        agent.eval()
        
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
        
        # Run inference on samples
        success_rate = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= args.eval_num_batches:
                break
                
            # Process batch similar to training
            init_discrete = batch["obs"]["init_discrete"].to(args.device)  # (B, 11, 6)
            goal_discrete = batch["obs"]["goal_discrete"].to(args.device)  # (B, 11, 6)
            
            init_coords_block = batch["obs"]["init_coords_block"].to(args.device)  # (B, 5, 2)
            goal_coords_block = batch["obs"]["goal_coords_block"].to(args.device)  # (B, 5, 2)
            init_coords_ee = batch["obs"]["init_coords_ee"].to(args.device)  # (B, 1, 2)
            goal_coords_ee = batch["obs"]["goal_coords_ee"].to(args.device)  # (B, 1, 2)
            
            # Ground truth for evaluation
            gt_hl_actions = batch["hl_discrete_action_seq"].to(args.device)  # (B, 8, 6)
            gt_ll_traj = batch["ll_traj"].to(args.device)  # (B, 48, 3)
            
            batch_size = init_discrete.shape[0]
            
            # Prepare priors (the known parts to condition on)
            # HL prior: init and goal predicates, zeros for actions
            init_discrete_flat = init_discrete.reshape(batch_size, -1)  # (B, 11*6)
            goal_discrete_flat = goal_discrete.reshape(batch_size, -1)  # (B, 11*6)
            hl_zeros = torch.zeros(batch_size, args.task.horizon * args.task.bit_dim, device=args.device)
            
            prior_hl = torch.cat([init_discrete_flat, goal_discrete_flat, hl_zeros], dim=1)
            
            # LL prior: block and ee coordinates, zeros for trajectory
            time_dim = torch.zeros(batch_size, args.task.max_blocks, 1, device=args.device)
            
            init_coords_block_with_time = torch.cat([time_dim, init_coords_block], dim=2)  # (B, 5, 3)
            goal_coords_block_with_time = torch.cat([time_dim, goal_coords_block], dim=2)  # (B, 5, 3)
            
            init_coords_ee_with_time = torch.cat([torch.zeros(batch_size, 1, 1, device=args.device), 
                                                 init_coords_ee], dim=2)  # (B, 1, 3)
            goal_coords_ee_with_time = torch.cat([torch.zeros(batch_size, 1, 1, device=args.device), 
                                                 goal_coords_ee], dim=2)  # (B, 1, 3)
            
            init_coords_block_flat = init_coords_block_with_time.reshape(batch_size, -1)  # (B, 5*3)
            goal_coords_block_flat = goal_coords_block_with_time.reshape(batch_size, -1)  # (B, 5*3)
            init_coords_ee_flat = init_coords_ee_with_time.reshape(batch_size, -1)  # (B, 1*3)
            goal_coords_ee_flat = goal_coords_ee_with_time.reshape(batch_size, -1)  # (B, 1*3)
            
            ll_zeros = torch.zeros(batch_size, args.task.horizon * args.task.steps_per_action * 3, device=args.device)
            
            prior_ll = torch.cat([
                init_coords_block_flat, goal_coords_block_flat,
                init_coords_ee_flat, goal_coords_ee_flat,
                ll_zeros
            ], dim=1)
            
            # Sample from the model
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
                
                # Select the best sample for each batch item (e.g., based on some metric)
                # For this demo, we'll just use the first sample
                best_samples_hl = all_samples_hl[0]
                best_samples_ll = all_samples_ll[0]
                
                # Extract the action part from the HL samples
                hl_action_start = hl_obs_dim
                sampled_hl_actions = best_samples_hl[:, hl_action_start:].reshape(batch_size, args.task.horizon, args.task.bit_dim)
                
                # Extract the trajectory part from the LL samples
                ll_act_start = ll_obs_dim
                sampled_ll_traj = best_samples_ll[:, ll_act_start:].reshape(batch_size, args.task.horizon * args.task.steps_per_action, 3)
                
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
                total_samples += batch_size
                
                # Print batch results
                print(f"Batch {batch_idx+1}/{min(len(val_dataloader), args.eval_num_batches)}:")
                print(f"  HL Success Rate: {hl_success.mean().item():.4f}")
                print(f"  LL Success Rate: {ll_success.mean().item():.4f}")
                print(f"  Combined Success Rate: {combined_success.mean().item():.4f}")
                
                # Save visualization samples if requested
                if args.save_samples and batch_idx < args.num_vis_batches:
                    # Here you would implement visualization logic
                    # For example, saving the predicted actions and trajectories
                    pass
        
        # Print overall results
        if total_samples > 0:
            print(f"Overall Success Rate: {success_rate / total_samples:.4f}")
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()