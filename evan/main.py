import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="evan_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_evan(config)


def run_evan(config) -> None:
    if not ray.is_initialized():
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(), ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        ray.get(TaskRunner.remote().run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        from evan.trainer import RayEVANTrainer
        from evan.worker import EVANActorRolloutRefWorker

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        role_worker_mapping = {Role.ActorRollout: ray.remote(EVANActorRolloutRefWorker)}
        mapping = {Role.ActorRollout: "global_pool"}
        if config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(EVANActorRolloutRefWorker)
            mapping[Role.RefPolicy] = "global_pool"
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping=mapping,
        )

        trainer = RayEVANTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=load_reward_manager(config, tokenizer, 0),
            val_reward_fn=load_reward_manager(config, tokenizer, 1),
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
