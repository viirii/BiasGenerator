# GlassBridgeEnv MVP Implementation Spec

## Goal

Implement a minimal multi-agent environment called `GlassBridgeEnv`, modeled after the existing training loop conventions in this repo.

The environment should be intentionally simple:

- 10 agents
- 20 bridge steps
- fixed order
- truthful-only information sharing
- no lying
- no order swapping
- no side payments
- goal is to study **multi-agent interaction via selective information withholding/sharing**

The implementation should prioritize:

- compatibility with the existing repo structure
- minimal working version first
- small action space
- easy evaluation

---

## Core game rules

### Agents and order

- Agents are `p0 ... p9`
- They cross in fixed order: `p0`, then `p1`, ..., then `p9`
- Order never changes

### Bridge

- Steps are `s0 ... s19`
- Each step has exactly one safe side:
  - `0 = left`
  - `1 = right`
- If an agent picks the wrong side on a step, they die immediately
- If they pick correctly, that step becomes publicly known for all later agents

### Private information

Use this exact hardcoded mapping:

```python
STEP_ASSIGNMENT = {
    0: [18, 19],
    1: [16, 17],
    2: [14, 15],
    3: [12, 13],
    4: [10, 11],
    5: [8, 9],
    6: [6, 7],
    7: [4, 5],
    8: [2, 3],
    9: [0, 1],
}
```

Each agent privately knows the correct safe side of its two assigned steps.

### Why this mapping exists

- early agents need immediate help from later agents
- later agents need early agents’ future information
- there are 20 steps and only 10 people, so physical traversal alone cannot reveal everything
- information trade is necessary

### Communication

Truthful only.

Before the current acting agent attempts their next unknown step:

- each other alive agent may choose to reveal:
  - none
  - first owned step
  - second owned step
  - both owned steps

Any revealed info becomes public for the rest of the episode.

Agents cannot lie.

### Rewards

Per-agent final reward:

```python
reward_i = (1.0 if finished_i else 0.0) - alpha * num_finished_agents_ahead_of_i
```

Use:

```python
alpha = 0.05
```

Where:

- “ahead” means lower player index
- only agents who finished the entire bridge count as survivors

### Main evaluation metric

- total number of survivors at the end

---

## Suggested files to add

Adjust names if this repo has a strong existing convention, but aim for something like:

```text
envs/
  glass_bridge_env/
    __init__.py
    glass_bridge_env.py
    policies.py
    run_glass_bridge_rollout.py
```

If the repo already has an env registration pattern, follow that instead.

---

## File 1: `glass_bridge_env.py`

Implement the main env.

### Suggested class

```python
class GlassBridgeEnv:
    ...
```

### Constants

```python
NUM_AGENTS = 10
NUM_STEPS = 20
UNKNOWN = -1
LEFT = 0
RIGHT = 1
ALPHA = 0.05

PHASE_COMMUNICATION = "communication"
PHASE_MOVEMENT = "movement"
PHASE_TERMINAL = "terminal"
```

### Hardcoded step assignment

```python
STEP_ASSIGNMENT = {
    0: [18, 19],
    1: [16, 17],
    2: [14, 15],
    3: [12, 13],
    4: [10, 11],
    5: [8, 9],
    6: [6, 7],
    7: [4, 5],
    8: [2, 3],
    9: [0, 1],
}
```

### Recommended internal state

```python
self.safe_sides: list[int]              # len 20
self.alive: list[bool]                  # len 10
self.finished: list[bool]               # len 10
self.progress: list[int]                # len 10, next step index to cross
self.public_known: list[int]            # len 20, -1 unknown else 0/1
self.private_knowledge: dict[int, dict[int, int]]
self.current_actor: int | None
self.phase: str
self.rng
```

### Recommended methods

```python
def reset(self, seed: int | None = None):
    ...

def step(self, action_dict: dict):
    ...

def get_observation(self, agent_id: int) -> dict:
    ...

def legal_actions(self, agent_id: int) -> list[str]:
    ...

def _apply_comm_actions(self, action_dict: dict) -> None:
    ...

def _apply_move_action(self, actor: int, action: str) -> None:
    ...

def _advance_actor(self) -> None:
    ...

def _find_next_alive_unfinished(self, start_idx: int) -> int | None:
    ...

def _final_rewards(self) -> dict[str, float]:
    ...

def _obs_all(self) -> dict[str, dict]:
    ...

def _info(self) -> dict:
    ...
```

---

## Observation format

Keep it simple and explicit.

### `get_observation(agent_id)` should return something like:

```python
{
    "self_id": 3,
    "current_actor": 2,
    "phase": "communication",
    "alive": [1,1,1,0,...],
    "finished": [0,0,0,0,...],
    "progress": [3,3,1,0,...],
    "public_known": [-1,1,0,-1,...],   # len 20
    "owned_steps": [12, 13],
    "owned_sides": [0, 1],
    "owned_is_public": [False, True],
}
```

Use ints/bools/lists only. No fancy wrappers.

Do not expose other agents’ private knowledge.

---

## Action format

Use small string action vocab for readability first.

### Communication actions

Only valid during `phase == "communication"`.

For non-acting alive agents:

```python
"SHARE_NONE"
"SHARE_FIRST"
"SHARE_SECOND"
"SHARE_BOTH"
```

For dead agents or current actor:

```python
"NOOP"
```

### Movement actions

Only valid during `phase == "movement"`.

For current actor:

```python
"LEFT"
"RIGHT"
```

For everyone else:

```python
"NOOP"
```

If the existing framework prefers ints over strings, map these to ints internally, but keep the above semantic names in comments/constants.

---

## Step logic

### `reset()`

1. Seed RNG
2. Sample `safe_sides` randomly for all 20 steps
3. Set all players alive, unfinished, progress=0
4. Set `public_known = [-1] * 20`
5. Populate `private_knowledge` from `STEP_ASSIGNMENT`
6. Set `current_actor = 0`
7. Set `phase = "communication"`
8. Return observations for all agents

### `step(action_dict)`

#### If phase is `"communication"`:

- for each alive non-acting agent:
  - inspect its communication action
  - reveal the appropriate owned step(s), truthfully
  - revealing writes into `public_known[step]`
- switch phase to `"movement"`
- return zero rewards, `done=False`

#### If phase is `"movement"`:

- only current actor’s move matters
- let `step_idx = progress[current_actor]`
- compare chosen side against `safe_sides[step_idx]`

If correct:

- set `public_known[step_idx] = safe_side`
- increment `progress[current_actor]`
- if `progress[current_actor] == NUM_STEPS`:
  - mark finished
  - call `_advance_actor()`
- else:
  - stay on same actor
- set phase back to `"communication"` unless terminal

If incorrect:

- mark `alive[current_actor] = False`
- call `_advance_actor()`
- set phase back to `"communication"` unless terminal

If no alive unfinished actor remains:

- set phase = `"terminal"`
- compute final rewards
- `done=True`

Else:

- zero rewards
- `done=False`

---

## Actor advancement logic

### `_advance_actor()`

Set current actor to the next player with:

- `alive[i] == True`
- `finished[i] == False`

If none exists:

- set `current_actor = None`

---

## Survival semantics

Be careful here.

Use:

- `alive[i] == True and finished[i] == False` → still in play
- `finished[i] == True` → fully survived the bridge
- `alive[i] == False` → dead

Final survival for reward/eval should mean:

```python
finished[i] == True
```

Not merely “not dead yet.”

---

## Reward function

Implement exactly this:

```python
def _final_rewards(self) -> dict[str, float]:
    rewards = {}
    for i in range(self.NUM_AGENTS):
        survived = 1.0 if self.finished[i] else 0.0
        survivors_ahead = sum(1 for j in range(i) if self.finished[j])
        rewards[f"p{i}"] = survived - self.ALPHA * survivors_ahead
    return rewards
```

---

## Legal actions

Implement masking helper.

### During communication

For `agent_id`:

- if dead: `["NOOP"]`
- if current actor: `["NOOP"]`
- else:
  - always allow `"SHARE_NONE"`
  - allow `"SHARE_FIRST"` only if first owned step is not public
  - allow `"SHARE_SECOND"` only if second owned step is not public
  - allow `"SHARE_BOTH"` only if at least one of the two is not public
  - otherwise just `["SHARE_NONE"]`

### During movement

- if `agent_id == current_actor`: `["LEFT", "RIGHT"]`
- else: `["NOOP"]`

---

## File 2: `policies.py`

Implement a few baseline scripted policies.

### Base interface

```python
class Policy:
    def act(self, obs: dict, legal_actions: list[str]) -> str:
        raise NotImplementedError
```

### 1) `RandomPolicy`

- choose uniformly from legal actions

### 2) `NeverSharePolicy`

- in communication: choose `"SHARE_NONE"`
- in movement:
  - if current step is public, choose that side
  - else random left/right

### 3) `AlwaysSharePolicy`

- in communication:
  - prefer `"SHARE_BOTH"` if legal
  - else `"SHARE_FIRST"` / `"SHARE_SECOND"` / `"SHARE_NONE"`
- in movement:
  - if current step is public, choose that side
  - else random left/right

### 4) `PublicInfoGreedyMoveMixin` or helper

Useful helper:

- parse current actor’s next step from obs
- if it is public, follow it deterministically

---

## File 3: `run_glass_bridge_rollout.py`

Implement a tiny rollout script for sanity checking.

### What it should do

- create env
- create one policy per agent
- reset env
- run until done
- print:
  - final survivors
  - final rewards
  - number of publicly known steps
  - progress of each player
  - maybe a compact event trace

### Example scenarios

Run at least:

1. all `NeverSharePolicy`
2. all `AlwaysSharePolicy`
3. mixed population

This is just for validation, not benchmarking.

---

## Suggested compact event trace

During rollout, print lines like:

```text
[COMM] actor=p0 reveals: p8->s2, p9->s0,s1
[MOVE] p0 step=s0 chose=RIGHT correct=RIGHT result=success
[MOVE] p0 step=s1 chose=LEFT correct=RIGHT result=death
...
[END] survivors=3 finished=['p4','p7','p9']
```

This will make debugging much faster.

---

## Useful implementation helpers

### Helper: owned steps

```python
def _owned_steps(self, agent_id: int) -> list[int]:
    return self.STEP_ASSIGNMENT[agent_id]
```

### Helper: current step for actor

```python
def _current_step_idx(self, actor: int) -> int:
    return self.progress[actor]
```

### Helper: public side or unknown

```python
def _public_side(self, step_idx: int) -> int:
    return self.public_known[step_idx]
```

---

## Suggested minimal info dict

`_info()` can return:

```python
{
    "phase": self.phase,
    "current_actor": self.current_actor,
    "num_survivors": sum(self.finished),
    "finished": self.finished[:],
    "alive": self.alive[:],
    "progress": self.progress[:],
    "public_known_count": sum(1 for x in self.public_known if x != self.UNKNOWN),
}
```

At terminal, also include:

```python
"survivor_ids": [f"p{i}" for i in range(self.NUM_AGENTS) if self.finished[i]]
```

---

## MVP acceptance criteria

Consider the task done when all of these are true:

1. `GlassBridgeEnv.reset()` and `step()` work end-to-end
2. environment terminates correctly
3. public knowledge updates correctly from both communication and traversal
4. dead agents no longer act or share
5. final rewards match the “survival minus survivors-ahead penalty” formula
6. scripted rollout script runs without crashes
7. always-share population produces more survivors on average than never-share population over a few random seeds

That last one is a useful sanity check.

---

## Cursor prompt

Please implement a minimal `GlassBridgeEnv` in this repo, following this spec:

- 10 agents `p0..p9`, 20 steps `s0..s19`
- fixed turn order
- each step has a random safe side
- hardcoded reversed private knowledge mapping:
  - p0 knows s18,s19
  - p1 knows s16,s17
  - ...
  - p9 knows s0,s1
- before the current actor moves, all other alive agents can truthfully reveal none / one / both of their owned steps
- revealed info becomes public for the rest of the episode
- current actor then chooses LEFT/RIGHT for their next step
- success reveals the step publicly and advances progress
- failure kills the agent immediately
- final reward for agent i is:
  - `1.0 if finished else 0.0`
  - minus `0.05 * number_of_finished_agents_ahead_of_i`
- add legal action masking
- add baseline scripted policies: RandomPolicy, NeverSharePolicy, AlwaysSharePolicy
- add a small rollout script that prints an event trace and final results

Prefer clear readable code over abstraction. Start with a single working env implementation, then wire up the policies and rollout script.

---

## Notes on intent

This environment is meant to model **multi-agent interaction**, not just many agents acting in parallel.

The key interaction is:

- agents hold private information
- agents choose how much of it to reveal
- agents want to survive
- agents also prefer fewer agents ahead of them to survive
- public revelation through traversal is incomplete because there are 20 steps and only 10 people
- therefore information trade matters

This means the benchmark is mainly about:

- selective cooperation
- withholding
- asymmetric dependence
- preserving useful future information carried by other agents

It is intentionally **not** a deception benchmark in the MVP version.
