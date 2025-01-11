import hydra
import os
import sys
import warnings


import gym
import pathlib
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from utils import set_seed, Logger

from cleandiffuser.env import kitchen
from cleandiffuser.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.dataset.kitchen_dataset import KitchenDataset, KitchenMjlDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters

warnings.filterwarnings("ignore")


def make_env(args, idx):
    def thunk():
        env = gym.make(args.env_name, use_abs_action=args.abs_action)
        env = KitchenLowdimWrapper(env=env, init_qpos=None, init_qvel=None, render_hw=(240, 360))
        video_recorder = VideoRecorder.create_h264(
            fps=12.5, codec="h264", input_pix_fmt="rgb24", crf=22, thread_type="FRAME", thread_count=1
        )
        env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=1)
        env = MultiStepWrapper(
            env, n_obs_steps=args.obs_steps, n_action_steps=args.action_steps, max_episode_steps=args.max_episode_steps
        )
        env.seed(args.seed + idx)
        print("Env seed: ", args.seed + idx)
        return env

    return thunk


def inference(args, envs, dataset, agent, logger):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_kit_success = []

    if args.diffusion == "ddpm":
        solver = None
    elif args.diffusion == "ddim":
        solver = "ddim"
    elif args.diffusion == "dpm":
        solver = "ode_dpmpp_2"
    elif args.diffusion == "edm":
        solver = "euler"

    for i in range(args.eval_episodes // args.num_envs):
        ep_reward = [0.0] * args.num_envs
        obs, done, t = envs.reset(), False, 0

        # initialize video stream
        if args.save_video:
            logger.video_init(envs.envs[0], enable=True, video_id=str(i))  # save videos

        while t < args.max_episode_steps:
            obs_seq = obs.astype(np.float32)  # (num_envs, obs_steps, obs_dim)
            # normalize obs
            nobs = dataset.normalizer["obs"]["state"].normalize(obs_seq)
            nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)

            with torch.no_grad():
                if args.nn == "pearce_mlp":
                    condition = nobs
                    # run sampling (num_envs, action_dim)
                    prior = torch.zeros((args.num_envs, args.action_dim), device=args.device)
                elif args.nn == "dit":
                    # reshape observation to (num_envs, obs_horizon*obs_dim)
                    condition = nobs.flatten(start_dim=1)
                    # run sampling (num_envs, args.action_steps, action_dim)
                    prior = torch.zeros((args.num_envs, args.action_steps, args.action_dim), device=args.device)
                else:
                    ValueError("fatal nn")
                if not args.diffusion_x:
                    naction, _ = agent.sample(
                        prior=prior,
                        n_samples=args.num_envs,
                        sample_steps=args.sample_steps,
                        solver=solver,
                        condition_cfg=condition,
                        w_cfg=1.0,
                        use_ema=True,
                    )
                else:
                    naction, _ = agent.sample_x(
                        prior=prior,
                        n_samples=args.num_envs,
                        sample_steps=args.sample_steps,
                        solver=solver,
                        condition_cfg=condition,
                        w_cfg=1.0,
                        use_ema=True,
                        extra_sample_steps=args.extra_sample_steps,
                    )
            # unnormalize prediction
            naction = naction.detach().to("cpu").clip(-1.0, 1.0).numpy()  # (num_envs, action_steps, action_dim)
            action_pred = dataset.normalizer["action"].unnormalize(naction)
            action = action_pred.reshape(args.num_envs, 1, args.action_dim)  # (num_envs, 1, action_dim)
            obs, reward, done, info = envs.step(action)
            ep_reward += reward
            t += args.action_steps

        success = [len(info[i]["completed_tasks"][0]) for i in range(args.num_envs)]
        kit_success = []
        for num in success:
            sublist = [1 if i < num else 0 for i in range(7)]
            kit_success.append(sublist)
        ep_reward = np.around(np.array(ep_reward), 2)
        print(
            f"[Episode {1+i*(args.num_envs)}-{(i+1)*(args.num_envs)}] reward: {ep_reward} success:{success} kit_success:{kit_success}"
        )
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)
        episode_kit_success.append(kit_success)

    eval_metrics = {
        "mean_step": np.nanmean(episode_steps),
        "mean_reward": np.nanmean(episode_rewards),
        "mean_success": np.nanmean(episode_success),
    }
    print(
        f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}"
    )
    mean_kit_success = np.mean(np.array(episode_kit_success), axis=(0, 1))
    kit_metrics = {}
    for i in range(7):
        kit_metrics[f"p_{i+1}"] = mean_kit_success[i]
    print(kit_metrics)
    eval_metrics.update(kit_metrics)
    return eval_metrics


@hydra.main(config_path="../configs/dbc/kitchen/pearce_mlp", config_name="kitchen")
def pipeline(args):
    # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # ---------------- Create Environment ----------------
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, idx) for idx in range(args.num_envs)],
    )

    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    if args.abs_action:
        dataset = KitchenMjlDataset(
            dataset_path, horizon=args.horizon, pad_before=args.obs_steps - 1, pad_after=args.action_steps - 1
        )
    else:
        dataset = KitchenDataset(
            dataset_path, horizon=args.horizon, pad_before=args.obs_steps - 1, pad_after=args.action_steps - 1
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True,
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "pearce_mlp":
        from cleandiffuser.nn_condition import PearceObsCondition
        from cleandiffuser.nn_diffusion import PearceMlp

        nn_diffusion = PearceMlp(act_dim=args.action_dim, To=args.obs_steps, emb_dim=128, hidden_dim=512).to(
            args.device
        )
        nn_condition = PearceObsCondition(obs_dim=args.obs_dim, dropout=0.0).to(args.device)
    elif args.nn == "dit":
        from cleandiffuser.nn_condition import MLPCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256, d_model=384, n_heads=12, depth=6, timestep_emb_type="fourier"
        ).to(args.device)
        nn_condition = MLPCondition(
            in_dim=args.obs_steps * args.obs_dim,
            out_dim=256,
            hidden_dims=[
                256,
            ],
            act=nn.ReLU(),
            dropout=0.25,
        ).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion, topk=3)
    print(f"==============================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition, topk=3)
    print(f"==============================================================================")

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM

        agent = DDPM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            diffusion_steps=args.sample_steps,
            ema_rate=0.9999,
            optim_params={"lr": args.lr},
        )
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM

        agent = EDM(
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device, optim_params={"lr": args.lr}
        )
    else:
        raise NotImplementedError

    if args.mode == "train":
        # ----------------- Training ----------------------
        n_gradient_step = 0
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            # preprocess
            nobs = batch["obs"]["state"].to(args.device)  # (B, horizon, obs_dim)
            naction = batch["action"].to(args.device)  # (B, horizon, action_dim)

            # diffusionBC
            # |o|o|
            # | |a|
            condition = nobs[:, : args.obs_steps, :]  # (B, obs_horizon, obs_dim)
            if args.nn == "pearce_mlp":
                naction = naction[:, -1, :]  # (B, action_dim)
            elif args.nn == "dit":
                condition = condition.flatten(start_dim=1)  # (B, obs_horizon*obs_dim)
                naction = naction[:, -args.action_steps :, :]  # (B, action_steps, action_dim)
            else:
                ValueError("fatal nn")

            # update diffusion
            diffusion_loss = agent.update(naction, condition)["loss"]
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    "step": n_gradient_step,
                    "total_time": time.time() - start_time,
                    "avg_diffusion_loss": np.mean(diffusion_loss_list),
                }
                logger.log(metrics, category="train")
                diffusion_loss_list = []

            if n_gradient_step % args.save_freq == 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)

            if n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                metrics = {"step": n_gradient_step}
                metrics.update(inference(args, envs, dataset, agent, logger))
                logger.log(metrics, category="inference")
                agent.model.train()
                agent.model_ema.train()

            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                # finish
                logger.finish(agent)
                break
    elif args.mode == "inference":
        # ----------------- Inference ----------------------
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for inference")
        agent.model.eval()
        agent.model_ema.eval()

        metrics = {"step": 0}
        metrics.update(inference(args, envs, dataset, agent, logger))
        logger.log(metrics, category="inference")

    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
