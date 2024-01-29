# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from collections import OrderedDict
from pathlib import Path

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
from omegaconf import OmegaConf

from sear import utils
from sear.environments import dmc
from sear.environments.adroit_dm_env import make_adroit
from sear.environments.distracting_dmc import make_distracting_dmc
from sear.environments.kitchen_dm_env import make_kitchen
from sear.environments.metaworld_dm_env import make_metaworld
from sear.logger import Logger
from sear.replay_buffer import ReplayBufferStorage, make_replay_loader
from sear.video import FrameRecorder, TrainVideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, agent_cfg, pretrain_cfg):
    assert (
        "pixels" in obs_spec
    ), "Observation spec passed to make_agent must contain a observation named 'pixels'"

    agent_cfg.obs_shape = obs_spec["pixels"].shape
    agent_cfg.action_shape = action_spec.shape
    agent = hydra.utils.instantiate(agent_cfg)
    if "path" in pretrain_cfg:
        agent.load_pretrained_weights(pretrain_cfg.path,
                                      pretrain_cfg.just_encoder_decoders)
    return agent


def make_env(cfg, is_eval):
    if cfg.task_name.split("_", 1)[0] == "metaworld":
        env = make_metaworld(
            cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
            cfg.discount, cfg.seed, cfg.camera_name,
            cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
            cfg.slim_mask_cfg)
    elif cfg.task_name.split("_", 1)[0] == "adroit":
        env = make_adroit(
            cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
            cfg.discount, cfg.seed, cfg.camera_name,
            cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
            cfg.slim_mask_cfg)
    elif cfg.task_name.split("_", 1)[0] == "kitchen":
        env = make_kitchen(
            cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
            cfg.discount, cfg.seed, cfg.camera_name,
            cfg.add_segmentation_to_obs, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
            cfg.slim_mask_cfg)
    elif cfg.task_name.split("_", 1)[0] == "distracting":
        background_dataset_videos = "val" if is_eval else "train"
        env = make_distracting_dmc(
            cfg.task_name.split("_", 1)[1], cfg.frame_stack, cfg.action_repeat,
            cfg.seed, cfg.add_segmentation_to_obs, cfg.distraction.difficulty,
            cfg.distraction.types, cfg.distraction.dataset_path,
            background_dataset_videos, cfg.noisy_mask_drop_prob, cfg.use_rgbm,
            cfg.slim_mask_cfg)
    else:
        env = dmc.make(cfg.task_name, cfg.frame_stack, cfg.action_repeat,
                       cfg.seed)
    return env


class Workspace:

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)

        self.train_env = make_env(self.cfg, is_eval=False)
        self.eval_env = make_env(self.cfg, is_eval=True)

        if self.cfg.has_success_metric:
            reward_spec = OrderedDict([
                ("reward", specs.Array((1,), np.float32, "reward")),
                ("success", specs.Array((1,), np.int16, "reward"))
            ])
        else:
            reward_spec = specs.Array((1,), np.float32, "reward")

        discount_spec = specs.Array((1,), np.float32, "discount")
        data_specs = {
            "observation": self.train_env.observation_spec(),
            "action": self.train_env.action_spec(),
            "reward": reward_spec,
            "discount": discount_spec
        }
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer", self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_buffer_snapshot, self.cfg.nstep, self.cfg.discount,
            self.cfg.has_success_metric)
        self._replay_iter = None

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent,
                                self.cfg.pretrain)

        self.video_recorder = FrameRecorder(
            self.work_dir if self.cfg.save_video else None,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

        if self.cfg.use_wandb:
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            wandb.init(project=self.cfg.wandb.project_name,
                       config=cfg_dict,
                       name=self.cfg.wandb.run_name)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        if self.cfg.has_success_metric:
            mean_max_success, mean_mean_success, mean_last_success = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            if self.cfg.has_success_metric:
                current_episode_max_success = 0
                current_episode_mean_success = 0
                current_episode_last_success = 0
            current_episode_step = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.agent,
                                     time_step.observation,
                                     enabled=(episode == 0))

            while not time_step.last():
                current_episode_step += 1
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.agent, time_step.observation)
                if self.cfg.has_success_metric:
                    total_reward += time_step.reward["reward"]
                    success = int(time_step.reward["success"])
                    current_episode_max_success = max(
                        current_episode_max_success, success)
                    current_episode_last_success = success
                    current_episode_mean_success += success
                else:
                    total_reward += time_step.reward
                step += 1
            if self.cfg.has_success_metric:
                mean_max_success += current_episode_max_success
                mean_last_success += current_episode_last_success
                mean_mean_success += current_episode_mean_success / current_episode_step
            episode += 1
            self.video_recorder.save(f"{self.global_frame}",
                                     step=self.global_frame)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            if self.cfg.has_success_metric:
                log("max_success", mean_max_success / episode)
                log("last_success", mean_last_success / episode)
                log("mean_success", mean_mean_success / episode)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        if self.cfg.has_success_metric:
            mean_success = 0
            max_success = 0
            last_success = 0

        # Score is mean success if task has success metric, else it is episode reward
        best_episode_score = -np.inf

        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty="train") as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)
                        if self.cfg.has_success_metric:
                            log("mean_success", mean_success / episode_step)
                            log("max_success", max_success)
                            log("last_success", last_success)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)

                if self.cfg.has_success_metric:
                    episode_score = mean_success / episode_step
                else:
                    episode_score = episode_reward

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot(file_name="latest.pt")

                    if episode_score > best_episode_score:
                        self.save_snapshot(file_name="best.pt")

                best_episode_score = max(episode_score, best_episode_score)

                episode_step = 0
                episode_reward = 0
                if self.cfg.has_success_metric:
                    mean_success = 0
                    max_success = 0
                    last_success = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log("eval_total_time", self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            if self.cfg.has_success_metric:
                episode_reward += time_step.reward["reward"]
                success = int(time_step.reward["success"])
                max_success = max(max_success, success)
                last_success = success
                mean_success += success
            else:
                episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self, file_name="snapshot.pt"):
        snapshot = self.work_dir / file_name
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        cfg_keys_to_save = [
            "has_success_metric", "task_name", "frame_stack", "action_repeat",
            "discount", "add_segmentation_to_obs"
        ]
        if "camera_name" in self.cfg:
            cfg_keys_to_save.append("camera_name")
        payload.update({k: self.cfg[k] for k in cfg_keys_to_save})

        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="train_config")
def main(cfg):
    from train import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
