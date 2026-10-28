import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.tracking import Tracking

ACTOR_KEYS = ["responses", "response_mask", "input_ids", "attention_mask", "position_ids", "old_log_probs"]


def token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def is_correct(batch: DataProto) -> np.ndarray:
    return np.asarray(batch.non_tensor_batch["accuracy"], dtype=np.float32) > 0


class RayEANTrainer(RayPPOTrainer):
    """DAPO trainer that appends online evidence-ablated negatives to the actor update."""

    def _compute_log_probs(self, batch: DataProto, timing_raw: dict, name: str) -> tuple[DataProto, torch.Tensor]:
        batch.batch["response_mask"] = compute_response_mask(batch)
        padded, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
        with marked_timer(name, timing_raw, "blue"):
            output = unpad_dataproto(self.actor_rollout_wg.compute_log_prob(padded), pad_size=pad_size)
        entropys = output.batch.pop("entropys")
        batch = batch.union(output)
        if self.use_reference_policy:
            with marked_timer(f"{name}_ref", timing_raw, "olive"):
                batch = batch.union(unpad_dataproto(self.ref_policy_wg.compute_ref_log_prob(padded), pad_size=pad_size))
        return batch, entropys

    def _generate_ablated_rollouts(self, batch: DataProto, prompt_indices: list[int], timing_raw: dict) -> DataProto:
        prompts = batch[prompt_indices]
        prompts.non_tensor_batch["raw_prompt"] = prompts.non_tensor_batch.pop("adv_raw_prompt")
        gen_batch = self._get_gen_batch(prompts).repeat(repeat_times=self.config.ean.rollout_n, interleave=True)
        size = len(gen_batch)
        gen_batch.padding(-size % len(self.async_rollout_manager.agent_loop_workers), padding_candidate="last")
        with marked_timer("ean_gen", timing_raw, "red"):
            output = self.async_rollout_manager.generate_sequences(gen_batch)[:size]
        for key in ("uid", "data_source", "reward_model", "extra_info"):
            output.non_tensor_batch[key] = gen_batch.non_tensor_batch[key][:size].copy()
        with marked_timer("ean_reward", timing_raw, "yellow"):
            _, reward_extra_info = compute_reward(output, self.reward_fn)
        output.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_info.items()})
        return output

    def _append_ablated_negatives(self, batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        ablated_prompts = batch.non_tensor_batch.get("adv_raw_prompt")
        if ablated_prompts is None:
            return batch

        uids = batch.non_tensor_batch["uid"]
        groups = defaultdict(list)
        for index, uid in enumerate(uids):
            groups[uid].append(index)
        correct = is_correct(batch)
        with_image = [rows[0] for rows in groups.values() if ablated_prompts[rows[0]] is not None]
        prompt_indices = [index for index in with_image if correct[groups[uids[index]]].any()]
        metrics["ean/num_prompts_with_ablated_image"] = len(with_image)
        if not prompt_indices:
            return batch

        rollouts = self._generate_ablated_rollouts(batch, prompt_indices, timing_raw)
        rollout_groups = defaultdict(list)
        for index, uid in enumerate(rollouts.non_tensor_batch["uid"]):
            rollout_groups[uid].append(index)
        rollout_correct = is_correct(rollouts)
        picked = []
        for index in prompt_indices:
            wrong = [i for i in rollout_groups[uids[index]] if not rollout_correct[i]]
            if wrong:
                picked.append(int(np.random.choice(np.asarray(wrong, dtype=np.int64))))
        metrics["ean/num_ablated_rollouts"] = len(rollouts)
        metrics["ean/num_candidates"] = len(picked)
        if not picked:
            return batch

        picked_uids = rollouts.non_tensor_batch["uid"][picked]
        clean_rows = [groups[uid][0] for uid in picked_uids]
        prompt_length = batch.batch["input_ids"].shape[1] - rollouts.batch["responses"].shape[1]
        candidates = DataProto.from_dict(
            tensors={
                "input_ids": torch.cat(
                    [batch.batch["input_ids"][clean_rows, :prompt_length], rollouts.batch["responses"][picked]], dim=1
                ),
                "attention_mask": torch.cat(
                    [
                        batch.batch["attention_mask"][clean_rows, :prompt_length],
                        rollouts.batch["attention_mask"][picked, prompt_length:],
                    ],
                    dim=1,
                ),
                "position_ids": torch.cat(
                    [
                        batch.batch["position_ids"][clean_rows, ..., :prompt_length],
                        rollouts.batch["position_ids"][picked, ..., prompt_length:],
                    ],
                    dim=-1,
                ),
                "responses": rollouts.batch["responses"][picked],
            },
            non_tensors={"multi_modal_inputs": batch.non_tensor_batch["multi_modal_inputs"][clean_rows]},
        )
        candidates, _ = self._compute_log_probs(candidates, timing_raw, "ean_old_log_prob")

        clean_logp = token_mean(batch.batch["old_log_probs"], batch.batch["response_mask"])
        clean_length = batch.batch["response_mask"].sum(dim=-1)
        clean_advantage = token_mean(batch.batch["advantages"], batch.batch["response_mask"])
        candidate_logp = token_mean(candidates.batch["old_log_probs"], candidates.batch["response_mask"])
        candidate_length = candidates.batch["response_mask"].sum(dim=-1)

        keep, advantages, num_logp_pass, num_length_pass = [], [], 0, 0
        for position, uid in enumerate(picked_uids):
            nonempty = [row for row in groups[uid] if clean_length[row] > 0]
            if not nonempty or candidate_length[position] == 0:
                continue
            if not candidate_logp[position] > clean_logp[nonempty].min():
                continue
            num_logp_pass += 1
            if candidate_length[position] > clean_length[nonempty].max():
                continue
            num_length_pass += 1
            advantage = -clean_advantage[groups[uid]].abs().min() * self.config.ean.advantage_scale
            if not torch.isfinite(advantage) or advantage >= 0:
                continue
            keep.append(position)
            advantages.append(advantage.item())
        metrics["ean/logp_filter_pass_ratio"] = num_logp_pass / len(picked)
        metrics["ean/length_filter_pass_ratio"] = num_length_pass / max(num_logp_pass, 1)
        metrics["ean/num_appended"] = len(keep)
        if not keep:
            return batch

        negatives = candidates[keep]
        actor_keys = ACTOR_KEYS + ["advantages"] + (["ref_log_prob"] if "ref_log_prob" in batch.batch.keys() else [])
        actor_batch = batch.select(batch_keys=actor_keys, non_tensor_batch_keys=["multi_modal_inputs"])
        actor_batch.batch["is_adversarial_rollout"] = torch.zeros(len(actor_batch), dtype=torch.long)
        response_mask = negatives.batch["response_mask"]
        appended = DataProto.from_dict(
            tensors={
                **{key: negatives.batch[key] for key in actor_keys if key != "advantages"},
                "advantages": torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1) * response_mask,
                "is_adversarial_rollout": torch.ones(len(negatives), dtype=torch.long),
            },
            non_tensors={"multi_modal_inputs": negatives.non_tensor_batch["multi_modal_inputs"]},
        )
        global_token_num = actor_batch.meta_info["global_token_num"] + appended.batch["attention_mask"].sum(-1).tolist()
        actor_batch = DataProto.concat([actor_batch, appended])
        actor_batch.meta_info["global_token_num"] = global_token_num
        return actor_batch

    def _pad_for_dispatch(self, batch: DataProto) -> DataProto:
        padded, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
        if pad_size == 0:
            return batch
        for key in ("advantages", "returns", "response_mask", "is_adversarial_rollout"):
            if key in padded.batch.keys():
                padded.batch[key][len(batch) :] = 0
        return padded

    def _filter_groups(self, new_batch: DataProto) -> tuple[DataProto, int]:
        metric_values = defaultdict(list)
        for uid, value in zip(
            new_batch.non_tensor_batch["uid"],
            new_batch.non_tensor_batch[self.config.algorithm.filter_groups.metric],
            strict=True,
        ):
            metric_values[uid].append(value)
        kept_uids = {uid for uid, values in metric_values.items() if np.std(values) > 0 or len(values) == 1}
        kept = [index for index, uid in enumerate(new_batch.non_tensor_batch["uid"]) if uid in kept_uids]
        return new_batch[kept], len(kept_uids)

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        rollout_n = self.config.actor_rollout_ref.rollout.n
        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for _ in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                ablated_prompts = new_batch.non_tensor_batch.pop("adv_raw_prompt", None)
                gen_batch = self._get_gen_batch(new_batch)
                if ablated_prompts is not None:
                    new_batch.non_tensor_batch["adv_raw_prompt"] = ablated_prompts
                gen_batch = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.pop("timing"))

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                        new_batch.batch["token_level_scores"] = reward_tensor
                        new_batch.batch["token_level_rewards"] = reward_tensor
                        new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    if self.config.algorithm.filter_groups.enable:
                        new_batch, num_kept = self._filter_groups(new_batch)
                        num_prompt_in_batch += num_kept
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                        if num_prompt_in_batch < self.config.data.train_batch_size:
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_prompt_in_batch=} < {self.config.data.train_batch_size}. Keep generating...")
                                continue
                            raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many.")
                        batch = batch[: self.config.data.train_batch_size * rollout_n]
                    else:
                        batch = new_batch

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    batch, entropys = self._compute_log_probs(batch, timing_raw, "old_log_prob")
                    metrics["actor/entropy"] = agg_loss(
                        loss_mat=entropys,
                        loss_mask=batch.batch["response_mask"],
                        loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                    ).item()

                    with marked_timer("adv", timing_raw, "brown"):
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=rollout_n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        )

                    with marked_timer("ean", timing_raw, "cyan"):
                        actor_batch = self._append_ablated_negatives(batch, metrics, timing_raw)

                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(self._pad_for_dispatch(actor_batch))
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics["train/num_gen_batches"] = num_gen_batches
                timing_raw = defaultdict(float)
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
