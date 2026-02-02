from collections import defaultdict
import numpy as np
import torch

from rllava.data.protocol import DataProto
from rllava.data.data_utils import collate_fn


class ExperienceMixer:
    def apply(self, batch: DataProto) -> DataProto:
        return batch


class HPTBatchReplacer(ExperienceMixer):
    def __init__(self, config, rollout, reward, train_dataloader):
        self.config = config
        self.rollout = rollout
        self.reward = reward
        self.train_dataloader = train_dataloader
        self._remove_sfted = getattr(self.config.algorithm, "remove_sfted_data", False)

    def apply(self, batch: DataProto) -> DataProto:
        strategy = getattr(self.config.algorithm, "unify_strategy", "none")
        if strategy == "none":
            return batch
        if getattr(self.config.data, "target_key", None) is None:
            return batch
        if "token_level_scores" not in batch.batch or "uid" not in batch.non_tensor_batch:
            return batch

        threshold = getattr(self.config.algorithm, "success_reward_threshold", 1.0)
        replace_num = getattr(self.config.algorithm, "off_policy_replace_num", 1)
        replace_num_mid = getattr(self.config.algorithm, "off_policy_replace_num_mid", 0)
        switch_gate = getattr(self.config.algorithm, "switch_gate", 0)
        switch_gate_off = getattr(self.config.algorithm, "switch_gate_off", 0)

        scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid_to_indices[uid].append(idx)

        replace_indices: list[int] = []
        replace_uids: list[str] = []
        for uid, idxs in uid_to_indices.items():
            uid_scores = scores[idxs]
            on_solve_num = int(np.sum(uid_scores >= threshold))
            if strategy == "switch":
                if on_solve_num <= switch_gate:
                    cur_replace = replace_num
                elif on_solve_num <= switch_gate_off:
                    cur_replace = replace_num_mid
                else:
                    cur_replace = 0
            else:
                cur_replace = replace_num
            if cur_replace <= 0:
                continue
            sorted_pairs = sorted(zip(idxs, uid_scores), key=lambda x: x[1])
            chosen = [i for i, _ in sorted_pairs[:cur_replace]]
            replace_indices.extend(chosen)
            replace_uids.extend([uid] * len(chosen))

        if not replace_indices:
            return batch

        if self._remove_sfted and "item" in batch.non_tensor_batch:
            remove_items = []
            for uid, idxs in uid_to_indices.items():
                if uid in replace_uids:
                    remove_items.append(int(batch.non_tensor_batch["item"][idxs[0]]))
            dataset = getattr(self.train_dataloader, "dataset", None)
            if dataset is not None and hasattr(dataset, "remove_data"):
                dataset.remove_data(remove_items)

        off_batch = self._sample_off_policy_dataproto(len(replace_indices), replace_uids)
        if off_batch is None:
            return batch

        off_batch = self.rollout.generate_off_batch(off_batch)
        reward_tensor, reward_metrics = self.reward.compute_rewards(off_batch)
        off_batch.batch["token_level_scores"] = reward_tensor
        for k, v in reward_metrics.items():
            off_batch.batch[f"reward_{k}"] = torch.tensor(v, dtype=torch.float32, device=reward_tensor.device)

        for key in off_batch.batch.keys():
            if key not in batch.batch:
                continue
            batch.batch[key][replace_indices] = off_batch.batch[key].to(batch.batch[key].device)
        for key in off_batch.non_tensor_batch.keys():
            if key not in batch.non_tensor_batch:
                continue
            batch.non_tensor_batch[key][replace_indices] = off_batch.non_tensor_batch[key]
        return batch

    def _sample_off_policy_dataproto(self, num_samples: int, uids: list[str]) -> DataProto | None:
        dataset = getattr(self.train_dataloader, "dataset", None)
        if dataset is None or num_samples <= 0:
            return None
        total = len(dataset)
        if total == 0:
            return None
        indices = np.random.randint(0, total, size=num_samples)
        items = [dataset[int(i)] for i in indices]
        batch_dict = collate_fn(items)
        off_batch = DataProto.from_single_dict(batch_dict)
        off_batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        return off_batch
