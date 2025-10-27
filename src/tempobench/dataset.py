import argparse
import ast
import json
import random
import re
from pathlib import Path

# import torch
from torch.utils.data import Dataset

from .causality_sample import one_shot as causality_one_shot
from .trace_acceptance_sample import one_shot as trace_one_shot

# from pathlib import Path


def replace_indices_with_APs(line: str, aps: list[str]) -> str:
    """Replace only AP indices inside [...] with their AP names.

    Leaves state numbers outside the brackets unchanged.
    """

    def repl(match):
        idx = int(match.group(0))
        if 0 <= idx < len(aps):
            return aps[idx]
        return match.group(0)  # leave as is if not a valid AP index

    # process only the [...] section
    return re.sub(
        r"\[(.*?)\]", lambda m: "[" + re.sub(r"\b\d+\b", repl, m.group(1)) + "]", line
    )


class TempoBench_Dataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer=None,
        task="trace_acceptance",
        prebuilt=False,
        subset_size=None,
        subset_mode="normal",
        seed=42,
    ):
        self.data = []
        self.task = task
        self.tokenizer = tokenizer
        self.prebuilt = prebuilt

        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                if item.get("error") is None:  # skip failed runs
                    self.data.append(item)

        # ---- Helper: ensure numeric value (None -> 0)
        def _nz(v):
            return 0 if v is None else v

        # ---- Helper: compute difficulty score from stats
        def _difficulty_score(stats: dict) -> float:
            # Tune these weights to match your intuition of “hard”
            w = {
                "effect_depth": 3.0,
                "hoa_states": 0.6,
                "transition_count": 0.25,
                "trace_length": 0.35,
                "causal_inputs_count": 0.6,
                "unique_inputs_in_trace": 0.25,
            }
            return (
                w["effect_depth"] * _nz(stats.get("effect_depth"))
                + w["hoa_states"] * _nz(stats.get("hoa_states"))
                + w["transition_count"] * _nz(stats.get("transition_count"))
                + w["trace_length"] * _nz(stats.get("trace_length"))
                + w["causal_inputs_count"] * _nz(stats.get("causal_inputs_count"))
                + w["unique_inputs_in_trace"] * _nz(stats.get("unique_inputs_in_trace"))
            )

        if subset_size is not None and subset_size < len(self.data):
            if subset_mode == "normal":
                random.seed(seed)  # ensures same subset each run
                self.data = random.sample(self.data, subset_size)
            elif subset_mode == "hard":
                stats_list = []
                for item in self.data:
                    # If the file is already prebuilt (prompt/label/stats)
                    # use stats directly
                    if self.prebuilt and "stats" in item:
                        s = item["stats"]
                    else:
                        # raw item contains  (hoa, trace, effects, etc.)
                        s = self.compute_summary_stats(item)
                    stats_list.append(s)

                # Stable tie-break using index so we can seed shuffle for
                # reproducibility if desired
                scored = [
                    (i, _difficulty_score(stats_list[i])) for i in range(len(self.data))
                ]

                # Optional: deterministic shuffle  to avoid systematic bias
                random.seed(seed)
                random.shuffle(scored)

                # Sort by score desc and pick top-K
                scored.sort(key=lambda t: t[1], reverse=True)
                keep_idx = [i for (i, _) in scored[:subset_size]]
                self.data = [self.data[i] for i in keep_idx]

            else:
                raise ValueError(f"Unknown subset_mode: {subset_mode}")

    def construct_acceptance_trace(self, result: dict, hoax_idx: int = -1):
        """Read in hoax and hoa to construct a NL version of the trace.

        hoax: "0: (0, {'s_0'}, 3)\n0: (3, {'g_1', 's_1'}, 12)\n..."
        hoa:  full HOA string

        also returns a json object of the hoax for exact labeling
        """
        # hoa = result["hoa"]
        aps = result["aps"]
        hoax = result["hoax"]

        # TODO extract the state rules for each state from the hoa
        nl_transitions = []
        json_transitions = {}
        for i, line in enumerate(hoax.strip().split("\n")):
            _, tup_str = line.split(":", 1)
            start, inputs, nxt = ast.literal_eval(tup_str.strip())
            inputs_list = sorted(list(inputs))  # stable order
            inputs_list_named = [
                aps[int(ap.split("_")[-1])] if ap.isdigit() else ap
                for ap in inputs_list
            ]
            if inputs_list_named:
                inputs_str = " and ".join(inputs_list)
            else:
                inputs_str = "no inputs"

            # NL prompt
            nl_transitions.append(
                f"From state {start}, "
                f"on inputs {inputs_str}, "
                f"the automaton moves to state {nxt}."
            )
            # JSON prompt
            json_transitions[f"step_{i}"] = {
                "current state": start,
                "defining inputs": inputs_str,
                "next state": nxt,
            }
            if hoax_idx > 0 and i >= hoax_idx:
                break

        prompt = (
            "These are the corresponding state transitions to the automaton:\n\n"
            + "\n".join(nl_transitions)
        )
        json_gt = (
            "\n\n### JSON Ground Truth ###\n"
            f"```json\n{json.dumps(json_transitions, indent=4)}```"
        )

        return prompt, json_gt

    def construct_causality_label(self, result: dict) -> str:
        """Construct a descriptive causality label with both NL explanation and the raw
        JSON causality mapping."""
        trace = result["trace"]
        effect = result["effects"][0]
        causality = result["causality"]
        # hoax = result["hoax"]

        # Count Xs in the effect name → max steps to show
        num_x = effect.count("X")
        max_steps = num_x + 1  # include step 0..num_x

        # edge case, cycle{1} is in first step, should never happen though
        trace_steps = [s for s in trace.split(";") if s and not s.startswith("cycle")]
        truncated_trace = ";".join(trace_steps[0:max_steps])

        nl_text = (
            "Causal explanations:\n"
            f"Effect: {effect} (showing first {max_steps} steps of trace)\n"
            f"The relevant portion of the trace is: {truncated_trace}\n"
            f"Reasoning over the transitions for the first {max_steps}: \n\n"
            f"{self.construct_acceptance_trace(result, num_x)[0]}"
            "\n\n### JSON Ground Truth ###:\n"
            f"```json\n{json.dumps(causality, indent=2)}```"
        )
        return nl_text

    def compute_summary_stats(self, result):
        stats = {}
        effect = result.get("effects", [""])[0] if result.get("effects") else ""

        # effect depth (number of Xs)
        stats["effect_depth"] = effect.count("X") if effect else None

        # HOA features
        hoa = result.get("hoa", "")
        if hoa:
            # States
            match = re.search(r"States:\s*(\d+)", hoa)
            stats["hoa_states"] = int(match.group(1)) if match else 0

            # Transitions
            stats["transition_count"] = len(re.findall(r"\[.*?\]\s+\d+", hoa))

            # acc-name
            match = re.search(r"acc-name:\s*(.+)", hoa)
            stats["acc_name"] = match.group(1).strip() if match else None

            # Acceptance (raw)
            match = re.search(r"Acceptance:\s*(.+)", hoa)
            stats["acceptance_condition"] = match.group(1).strip() if match else None
            if stats["acceptance_condition"]:
                parts = stats["acceptance_condition"].split()
                stats["acceptance_sets"] = (
                    int(parts[0]) if parts and parts[0].isdigit() else None
                )
                stats["acceptance_formula"] = parts[1:] if len(parts) > 1 else []

        # Trace features
        trace = result.get("trace", "")
        if trace:
            steps = [s for s in trace.split(";") if s and not s.startswith("cycle")]
            stats["trace_length"] = len(steps)
            stats["cycle_length"] = sum(
                1 for s in trace.split(";") if s.startswith("cycle")
            )
            # extract all APs seen in trace
            atoms = re.findall(r"[a-zA-Z]+", trace)
            stats["unique_inputs_in_trace"] = len(set(atoms))
        else:
            stats.update(
                {"trace_length": 0, "cycle_length": 0, "unique_inputs_in_trace": 0}
            )

        # Causality-specific
        if self.task == "causality":
            causality = result.get("causality", {})
            stats["causal_inputs_count"] = sum(len(v) for v in causality.values())
        else:
            stats["causal_inputs_count"] = None

        return stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.prebuilt:
            record = self.data[idx]
            prompt, label, stats = record["prompt"], record["label"], record["stats"]
            if self.tokenizer:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )
                labels = self.tokenizer(label, return_tensors="pt")["input_ids"]
                return {"inputs": inputs, "labels": labels}
            return prompt, label, stats

        result = self.data[idx]
        aps = result["aps"]
        hoa_pretty = "\n".join(
            replace_indices_with_APs(line, aps) for line in result["hoa"].splitlines()
        )

        if self.task == "trace_acceptance":
            prompt = (
                "Here is a Prompt and Label sample. Be sure to give your final answer "
                "in JSON format as shown in the example.\n"
                "======================================================================"
                "\nSample start:\n"
                f"{trace_one_shot}\n"
                "======================================================================"
                "\nProblem start:\n"
                "This is task that requires you to trace through a state machine.\n"
                "You must step through state by state, compare the inputs with "
                "all accepted inputs at each state, determine the transition and then "
                "after consuming the whole state, determine if this trace will be "
                "accepted by the state machine.\n\n"
                "Ignore the cycle{1} at the end of the trace."
                " You should just determine if the trace ends in an accept state after "
                "the first cycle."
                f"You are given an automaton (HOA format) with APs {result['aps']}.\n\n"
                f"Automaton:\n{hoa_pretty}\n\n"
                f"Trace:\n{result['trace']}\n\n"
                f"Question: Does the automaton accept this trace? "
                "Solve this by stepping trough the state machine."
            )
            label = (
                f"{self.construct_acceptance_trace(result=result)[0]}\n"
                f"{self.construct_acceptance_trace(result=result)[1]}\n\n"
                f"{'Accepted: Yes' if result['accepted'] else 'Accepted: No'}"
            )

            if self.tokenizer:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )
                labels = self.tokenizer(label, return_tensors="pt")["input_ids"]
                return {"inputs": inputs, "labels": labels}

            stats = self.compute_summary_stats(result)

            return prompt, label, stats

        elif self.task == "causality":
            """
            NOTE: Will's NL description of the task it is a credit assignment task over
            time the goal is to identify over time the minimumn set of inputs that were
            given which caused the effect that is given. Specifically your goal is to
            find the minimumn set of inputs over time such that if one did not occur
            the output observed would also not occur.
            """
            prompt = (
                "Here is a Prompt and Label sample. Be sure to give your final answer "
                "in JSON format as shown in the example.\n"
                "======================================================================"
                "\nSample start:\n"
                f"{causality_one_shot}\n"
                "======================================================================"
                "\nProblem start:\n"
                "This is a credit assignment task over time.\n"
                "Your goal is to identify the minimal set of inputs "
                "that caused a given effect in the automaton. "
                "If any one of these inputs were missing, the effect "
                "would not have occurred.\n\n"
                f"You are given an automaton (HOA format) with APs:\n"
                f"{result['aps']}\n\n"
                f"Automaton:\n{hoa_pretty}\n\n"
                f"Trace:\n{result['trace']}\n\n"
                f"Effects to analyze:\n{result['effects']}\n\n"
                "Explain the causal constraints step by step.\n"
            )

            label = self.construct_causality_label(result)
            stats = self.compute_summary_stats(result)
            return prompt, label, stats


# python -m dataset.dataset c --file data/raw-causal.jsonl --dump data/causal-done.jsonl
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["t", "c"], help="trace or causality mode")
    parser.add_argument("diff", choices=["normal", "hard"], help="hard or easy mode")
    parser.add_argument("--file", type=Path, help="Path to dataset JSONL")
    parser.add_argument("--dump", type=str, help="Path to dump static dataset JSONL")
    parser.add_argument("-s", type=int, default=None, help="subset size")
    args = parser.parse_args()

    task = "trace_acceptance" if args.mode == "t" else "causality"
    ds = TempoBench_Dataset(
        args.file,
        tokenizer=None,
        task=task,
        subset_size=args.s,
        subset_mode=args.diff,
        seed=1234,
    )

    print(f"Loaded {len(ds)} items")

    # If dumping, stream into a JSONL file
    if args.dump:
        with open(args.dump, "w") as f:
            for i in range(len(ds)):
                prompt, label, stats = ds[i]
                record = {"prompt": prompt, "label": label, "stats": stats}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[✓] Static dataset written to {args.dump}")

    else:
        # Otherwise just preview
        print("--- First item ---")
        prompt, label, stats = ds[0]
        print("Prompt:\n", prompt)
        print("Label:\n", label)
        print("Stats:\n", stats)
