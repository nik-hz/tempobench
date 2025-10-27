# flake8: noqa
one_shot = """Prompt:
This is task that requires you to trace through a state machine.
You must step through state by state, compare the inputs with all accepted inputs at each state, determine the transition and then after consuming the whole state, determine if this trace will be accepted by the state machine.

Ignore the cycle{1} at the end of the trace. You should just determine if the trace ends in an accept state after the first cycle.Here is a Prompt and Label sample. Be sure to give your final answer in JSON format or you will fail the question.

You are given an automaton (HOA format) with APs ['g', 'r'].

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

Question: Does the automaton accept this trace? Solve this by stepping trough the state machine.

======================================================================
Sample Solution start:
This is how you should format your answer.
Label:
These are the corresponding state transitions to the automaton:

From state 0, on inputs !g and !r, the automaton moves to state 1.
From state 1, on inputs r, the automaton moves to state 2.
From state 2, on inputs !g and !r, the automaton moves to state 3.
From state 3, on inputs g and r, the automaton moves to state 5.
From state 5, on inputs g and r, the automaton moves to state 5.
From state 5, on inputs !g and !r, the automaton moves to state 5.
From state 5, on inputs g and r, the automaton moves to state 5.
From state 5, on inputs g and r, the automaton moves to state 5.
From state 5, on inputs g and r, the automaton moves to state 5.
From state 5, on inputs g and r, the automaton moves to state 5.


(Add in this line below for grading to work properly before giving your answer)
### JSON Output ###
``json
{
    "step_0": {
        "current state": 0,
        "defining inputs": "!g and !r",
        "next state": 1
    },
    "step_1": {
        "current state": 1,
        "defining inputs": "r",
        "next state": 2
    },
    "step_2": {
        "current state": 2,
        "defining inputs": "!g and !r",
        "next state": 3
    },
    "step_3": {
        "current state": 3,
        "defining inputs": "g and r",
        "next state": 5
    },
    "step_4": {
        "current state": 5,
        "defining inputs": "g and r",
        "next state": 5
    },
    "step_5": {
        "current state": 5,
        "defining inputs": "!g and !r",
        "next state": 5
    },
    "step_6": {
        "current state": 5,
        "defining inputs": "g and r",
        "next state": 5
    },
    "step_7": {
        "current state": 5,
        "defining inputs": "g and r",
        "next state": 5
    },
    "step_8": {
        "current state": 5,
        "defining inputs": "g and r",
        "next state": 5
    },
    "step_9": {
        "current state": 5,
        "defining inputs": "g and r",
        "next state": 5
    }
}
```

Accepted: Yes
"""
