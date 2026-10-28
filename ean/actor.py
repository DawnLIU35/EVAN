import logging
import os

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def negative_rows(response_mask, is_negative):
    if is_negative is None:
        return None
    rows = is_negative.bool() & (response_mask.sum(dim=-1) > 0)
    return rows if rows.any() else None


def add_negative_seq_mean(loss, subset_loss, rows, loss_agg_mode):
    """loss + seq-mean-token-mean(negatives) - token-mean(negatives); a no-op for a single negative."""
    if rows is None or loss_agg_mode != "token-mean":
        return loss
    token_mean = subset_loss(rows, "token-mean")
    return loss + (subset_loss(rows, "seq-mean-token-mean") - token_mean)


class EANDataParallelPPOActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if "is_adversarial_rollout" in data.batch.keys():
            select_keys.append("is_adversarial_rollout")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]
                    rows = negative_rows(response_mask, model_inputs.get("is_adversarial_rollout", None))

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]
                    policy_loss_fn = get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))

                    def subset_pg_loss(subset, mode):
                        return policy_loss_fn(
                            old_log_prob=old_log_prob[subset],
                            log_prob=log_prob[subset],
                            advantages=advantages[subset],
                            response_mask=response_mask[subset],
                            loss_agg_mode=mode,
                            config=self.config,
                        )[0]

                    def aggregate(loss_mat):
                        loss = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        return add_negative_seq_mean(
                            loss,
                            lambda subset, mode: agg_loss(
                                loss_mat=loss_mat[subset], loss_mask=response_mask[subset], loss_agg_mode=mode
                            ),
                            rows,
                            loss_agg_mode,
                        )

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )
                    pg_loss = add_negative_seq_mean(pg_loss, subset_pg_loss, rows, loss_agg_mode)
                    micro_batch_metrics.update(pg_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = aggregate(entropy)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=model_inputs["ref_log_prob"], kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = aggregate(kld)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics
