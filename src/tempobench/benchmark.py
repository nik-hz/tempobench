import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import TempoBench_Dataset


def collate_keep_dict(batch):
    prompts, labels, stats = zip(*batch)
    return list(prompts), list(labels), list(stats)


class TempobenchBenchmarker:
    def __init__(
        self,
        dataset_path,
        task,
        model_id="openai/gpt-4o-mini",
        batch_size=1,
        mode="normal",
        log_dir="runs/tempo_bench",
        max_workers=8,
        prebuilt=True,
        results_dir="benchmark_results",
        tqdm_position=0,
        console_prints=False,
    ):
        self.tqdm_position = tqdm_position
        self.console_prints = console_prints

        self.mode = mode
        self.task = task
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.dataset = TempoBench_Dataset(
            dataset_path, tokenizer=None, task=task, prebuilt=prebuilt
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_keep_dict,
        )

        # OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model_id = model_id
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.max_workers = max_workers

    def _query_openrouter(self, prompt: str) -> dict:
        """Send a single prompt to OpenRouter and return structured result."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                timeout=600,
            )
            text = response.choices[0].message.content.strip()

            # Accurate cost estimation
            time.sleep(0.5)
            url = f"https://openrouter.ai/api/v1/generation?id={response.id}"
            headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
            gen_resp = requests.get(url, headers=headers).json()

            return {
                "text": text,
                "cost": gen_resp["data"]["total_cost"],
                "native_prompt_tokens": gen_resp["data"]["native_tokens_prompt"],
                "native_completion_tokens": gen_resp["data"][
                    "native_tokens_completion"
                ],
                "generation_id": response.id,
            }
        except Exception as e:
            return {
                "text": "",
                "cost": 0.0,
                "native_prompt_tokens": 0,
                "native_completion_tokens": 0,
                "generation_id": None,
                "error": f"chat.completions failed: {str(e)}",
            }

    def extract_json(self, text: str):
        """Extract fenced ```json ...

        ``` block and parse it.
        """
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    def extract_aps_list(self, constraint_list):
        """Return a LIST of atomic propositions (e.g., ['s_0', '!s_1']).

        Handles strings like 's_0 AND !s_1' and ignores 'no constraints'.
        """
        if not constraint_list or constraint_list == ["no constraints"]:
            return []
        joined = " ".join(constraint_list)
        # split by AND / OR, case-insensitive
        parts = re.split(r"\s*(?:AND|OR)\s*", joined, flags=re.IGNORECASE)
        aps = []
        for p in parts:
            p = p.strip()
            if not p or p.lower() == "no constraints":
                continue
            # normalize spacing and remove parens, keep leading '!' if present
            p = re.sub(r"[()\s]+", "", p)
            if p:
                aps.append(p)
        return aps

    def _ap_overlap_stats(self, gold_json, pred_json, dedupe=True):
        """Return precision, recall, f1 for AP-level overlap."""
        if not gold_json or not pred_json:
            return 0.0, 0.0, 0.0

        try:
            cause_gt = list(gold_json.keys())[0]
            cause_pred = list(pred_json.keys())[0]
        except Exception:
            return 0.0, 0.0, 0.0

        TP = FP = FN = 0
        for time_step in gold_json[cause_gt]:
            gt_raw = gold_json[cause_gt][time_step]
            pred_raw = pred_json.get(cause_pred, {}).get(time_step, ["no constraints"])

            if gt_raw == ["no constraints"] and pred_raw == ["no constraints"]:
                TP += 1
                continue

            gt_list = self.extract_aps_list(gt_raw)
            pred_list = self.extract_aps_list(pred_raw)

            gt_aps = set(gt_list) if dedupe else gt_list
            pred_aps = set(pred_list) if dedupe else pred_list

            for ap in pred_aps:
                if ap in gt_aps:
                    TP += 1
                else:
                    FP += 1
            for ap in gt_aps:
                if ap not in pred_aps:
                    FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    def _time_step_overlap_stats(self, gold_json, pred_json):
        """Return precision, recall, f1 for exact time-step match."""
        if not gold_json or not pred_json:
            return 0.0, 0.0, 0.0

        try:
            cause_gt = list(gold_json.keys())[0]
            cause_pred = list(pred_json.keys())[0]
        except Exception:
            return 0.0, 0.0, 0.0

        TP = FP = FN = 0
        for time_step in gold_json[cause_gt]:
            gt_value = gold_json[cause_gt][time_step]
            pred_value = pred_json.get(cause_pred, {}).get(
                time_step, ["no constraints"]
            )

            if gt_value == pred_value:
                TP += 1
            else:
                if gt_value == ["no constraints"] and pred_value != ["no constraints"]:
                    FP += 1
                elif gt_value != ["no constraints"] and pred_value == [
                    "no constraints"
                ]:
                    FN += 1
                else:
                    FP += 1
                    FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    def _ap_overlap_stats_trace(self, gold_json, pred_json, dedupe=True):
        """Return precision, recall, f1 for AP-level overlap between defining inputs."""
        if not gold_json or not pred_json:
            return 0.0, 0.0, 0.0

        TP = FP = FN = 0
        steps = sorted(gold_json.keys())

        for step in steps:
            gt_entry = gold_json.get(step, {})
            pred_entry = pred_json.get(step, {})

            gt_raw = gt_entry.get("defining inputs", "no constraints")
            pred_raw = pred_entry.get("defining inputs", "no constraints")

            # normalize into lists of APs
            gt_list = self.extract_aps_list(
                gt_raw if isinstance(gt_raw, list) else [gt_raw]
            )
            pred_list = self.extract_aps_list(
                pred_raw if isinstance(pred_raw, list) else [pred_raw]
            )

            gt_aps = set(gt_list) if dedupe else gt_list
            pred_aps = set(pred_list) if dedupe else pred_list

            # special case: both are no constraints
            if not gt_aps and not pred_aps:
                TP += 1
                continue

            for ap in pred_aps:
                if ap in gt_aps:
                    TP += 1
                else:
                    FP += 1
            for ap in gt_aps:
                if ap not in pred_aps:
                    FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    def _time_step_overlap_stats_trace(self, gold_json, pred_json):
        """Return precision, recall, f1 for exact time-step match (all fields in step
        must match)."""
        if not gold_json or not pred_json:
            return 0.0, 0.0, 0.0

        TP = FP = FN = 0
        steps = sorted(gold_json.keys())

        for step in steps:
            gt_entry = gold_json.get(step)
            pred_entry = pred_json.get(step)

            if not gt_entry and not pred_entry:
                # both missing = skip
                continue
            elif gt_entry and not pred_entry:
                FN += 1
                continue
            elif not gt_entry and pred_entry:
                FP += 1
                continue

            # Both exist: check exact equality of the whole dict
            if gt_entry == pred_entry:
                TP += 1
            else:
                FP += 1
                FN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    def evaluate(self):
        total, correct, total_cost, sum_f1_ap, sum_f1_ts = 0, 0, 0.0, 0.0, 0.0
        results = []

        for batch in tqdm(
            self.dataloader,
            desc=f"Evaluating {self.task} | {self.model_id}",
            position=self.tqdm_position,
            leave=True,
            dynamic_ncols=True,
        ):
            prompts, labels, sample_stats = batch

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._query_openrouter, p) for p in prompts]
                responses = [f.result() for f in as_completed(futures)]
            responses = [f.result() for f in futures]  # preserve order

            for _, gold, st, resp in zip(prompts, labels, sample_stats, responses):
                if resp.get("error"):
                    continue

                total += 1
                pred = resp["text"]

                gold_json, pred_json = self.extract_json(gold), self.extract_json(pred)

                # TODO Issue on github, normalize the JSON format and simplify this
                if "causal" in self.task:
                    precision_ap, recall_ap, f1_ap = self._ap_overlap_stats(
                        gold_json, pred_json
                    )
                    precision_ts, recall_ts, f1_ts = self._time_step_overlap_stats(
                        gold_json, pred_json
                    )
                elif "trace" in self.task:
                    precision_ap, recall_ap, f1_ap = self._ap_overlap_stats_trace(
                        gold_json, pred_json
                    )
                    precision_ts, recall_ts, f1_ts = (
                        self._time_step_overlap_stats_trace(gold_json, pred_json)
                    )
                else:
                    precision_ap, recall_ap, f1_ap = 0.0, 0.0, 0.0
                    precision_ts, recall_ts, f1_ts = 0.0, 0.0, 0.0

                # mark as correct if both F1s are perfect
                is_correct = (f1_ap == 1.0) and (f1_ts == 1.0)
                sum_f1_ap += f1_ap
                sum_f1_ts += f1_ts

                if is_correct:
                    correct += 1

                if is_correct:
                    correct += 1

                total_cost += resp["cost"]

                flat_stats = st if st else {}
                result_row = {
                    "model": self.model_id,
                    "gold": gold,
                    "pred": pred,
                    "GT": gold_json,
                    "PRED": pred_json,
                    "correct": is_correct,
                    "precision_ap": precision_ap,
                    "recall_ap": recall_ap,
                    "F1_ap": f1_ap,
                    "precision_ap": precision_ap,
                    "recall_ap": recall_ap,
                    "F1_ap": f1_ap,
                    "precision_timestep": precision_ts,
                    "recall_timestep": recall_ts,
                    "F1_timestep": f1_ts,
                    "cost": resp["cost"],
                    "generation_id": resp["generation_id"],
                    "native_prompt_tokens": resp["native_prompt_tokens"],
                    "native_completion_tokens": resp["native_completion_tokens"],
                    **flat_stats,
                }
                results.append(result_row)

        # --- Aggregate metrics ---
        accuracy = correct / total if total > 0 else 0.0
        avg_f1_ap = sum_f1_ap / total if total > 0 else 0.0
        avg_f1_ts = sum_f1_ts / total if total > 0 else 0.0

        if self.console_prints:
            print(f"[✓] {self.task} benchmark done.")
            print(f"    Accuracy: {accuracy:.3f}")
            print(f"    Avg F1 AP: {avg_f1_ap:.3f}")
            print(f"    Avg F1 TS: {avg_f1_ts:.3f}")
            print(f"    Total cost: ${total_cost:.4f}")

        # --- Save results ---
        df = pd.DataFrame(results)
        base_name = self.model_id.replace("/", "_")
        out_dir = os.path.join(self.results_dir, self.task)
        os.makedirs(out_dir, exist_ok=True)
        csv_out = os.path.join(out_dir, f"results_{base_name}_{self.task}.csv")
        jsonl_out = os.path.join(out_dir, f"results_{base_name}_{self.task}.jsonl")

        df.to_csv(csv_out, index=False)
        df.to_json(jsonl_out, orient="records", lines=True)

        if self.console_prints:
            print(f"[✓] DataFrame saved to {csv_out} and {jsonl_out}")

        return df
