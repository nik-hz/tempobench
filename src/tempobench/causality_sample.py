# flake8: noqa
one_shot = """Prompt:
This is a credit assignment task over time.
Your goal is to identify the minimal set of inputs that caused a given effect in the automaton. If any one of these inputs were missing, the effect would not have occurred.

You are given an automaton (HOA format) with APs:
['g', 'r']

Automaton:
HOA: v1
States: 6
Start: 0
AP: 2 "g" "r"
acc-name: all
Acceptance: 0 t
properties: trans-labels explicit-labels state-acc deterministic
controllable-AP: 0
--BODY--
State: 0
[!g] 1
State: 1
[!g] 2
State: 2
[!g] 3
State: 3
[!g&!r] 4
[g&r] 5
State: 4
[!g] 4
State: 5
[g&r] 5
[!g&!r] 5
--END--

Trace:
!g&!r;!g&r;!g&!r;g&r;g&r;!g&!r;g&r;g&r;g&r;g&r;cycle{1}

Effects to analyze:
['XXX g']

Explain the causal constraints step by step.

Label:
 Causal explanations:
Effect: XXX g (showing first 4 steps of trace)
The relevant portion of the trace is: !g&!r;!g&r;!g&!r;g&r
Reasoning over the transitions for the first 4:

These are the corresponding state transitions to the automaton:

From state 0, on inputs !g and !r, the automaton moves to state 1.
From state 1, on inputs r, the automaton moves to state 2.
From state 2, on inputs !g and !r, the automaton moves to state 3.
From state 3, on inputs g and r, the automaton moves to state 5.

(Add in this line below for grading to work properly before giving your answer)
### JSON Ground Truth ###:
```json
{
  "XXX g": {
    "0": [
      "no constraints"
    ],
    "1": [
      "no constraints"
    ],
    "2": [
      "no constraints"
    ],
    "3": [
      "r"
    ]
  }
}
```
"""
