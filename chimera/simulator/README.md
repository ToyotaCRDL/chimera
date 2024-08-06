# <b>Simulator</b>

Simulator modules for `chimera`.

# Modules
## Simulators

- [Habitat](./habitat/) : [https://github.com/facebookresearch/habitat-lab/](https://github.com/facebookresearch/habitat-lab)
- [MetaWorld](./metaworld/) : [https://meta-world.github.io/](https://meta-world.github.io/)

## Real Robots

- [Vizbot](./vizbot/)

# Get Started

Construct Simulator using `create_simulator` function.

The simulation loop is performed as follows:

```python
import chimera
import torch

sim = chimera.create_simulator(name="Habitat")

for episode in range(sim.num_episodes()):
    obs, info = sim.reset()

    while not sim.is_episode_over():
        inputs = { "action": torch.randint(1, 3, (1, 1)).to(device) }
        obs, info = sim.step(inputs)
```

# Data Format

## Observations

Observations from simulators.
Available observations depend on the task and simulator.

<pre>
<details><summary>observations</summary> | (Basic)
 |-- rgb: Tensor (batch, channel, height, width)
 |-- depth: Tensor (batch, 1, height, width)
 |-- prev_action: Tensor (batch, 1) # discrete previous action (0=move forward, 1=turn left, 2=turn right)
 |-- instruction: str[batch] # instruction
 |-- question: str[batch] # for eqa
 |
 | (for Habitat Tasks)
 |-- position: Tensor (batch, 3) # 3d position (x, y, z) [m]
 |-- rotation: Tensor (batch, 4) # quaternion (w, x, y, z)
 |-- position2d: Tensor (batch, 2) # 2d position (x, y) [m]
 |-- position2d: Tensor (batch, 1) # 2d rotation (theta) [rad]
 |-- goal2d: Tensor (batch, 2) # 2d relative position of goal (r, theta)
 |-- objectgoal: str[batch] # goal object category name
 |-- semantic: Tensor (batch, 1, height, width)
 |
 | (for Rearrange)
 |-- head_depth: Tensor (batch, channel, height, width)
 |-- obj_start_sensor: Tensor (batch, 3)
 |-- obj_goal_sensor: Tensor (batch, 3)
 |-- obj_start_gps_compass: Tensor (batch, 2)
 |-- obj_goal_gps_compass: Tensor (batch, 2)
 |-- ee_pos: Tensor (batch, 3)
 |-- relative_resting_position: Tensor (batch, 3)
 |
 | (for MetaWorld)
 |-- ee_pos: Tensor(batch, 3) # 3d position (x, y, z) [m]
 |-- gripper: Tnesor(batch, 1)
 |-- obj1_pos: Tensor(batch, 3) # 3d position (x, y, z) [m]
 |-- obj1_rot: Tensor(batch, 4) # quaternion (w, x, y, z)
 |-- obj2_pos: Tensor(batch, 3) # 3d position (x, y, z) [m]
 |-- obj2_rot: Tensor(batch, 4) # quaternion (w, x, y, z)
 |-- prev_ee_pos: Tensor(batch, 3)
 |-- prev_gripper: Tnesor(batch, 1)
 |-- prev_obj1_pos: Tensor(batch, 3) # 3d position (x, y, z) [m]
 |-- prev_obj1_rot: Tensor(batch, 4) # quaternion (w, x, y, z)
 |-- prev_obj2_pos: Tensor(batch, 3) # 3d position (x, y, z) [m]
 |-- prev_obj2_rot: Tensor(batch, 4) # quaternion (w, x, y, z)
 |-- goal_pos: Tensor(batch, 3) # 3d position (x, y, z) [m]
</details></pre>

## Info

Additional information from simulators.
Available observations depend on the task and simulator.

<pre>
<details><summary>info</summary> |
 |-- position: Tensor (batch, 3) # gt position (x, y, z) [m]
 |-- rotation: Tensor (batch, 4) # gt rotation (w, x, y, z)
 |-- distance_to_goal: Tensor (batch, 1)
 |-- success: Tensor (batch, 1) # 1 (success) or 0 (failed)
 |-- spl: Tensor (batch, 1)
 |-- softspl: Tensor (batch, 1)
 |-- map: Tensor (batch, channel, height, width) # oracle map from sim
 |-- goal_position: Tensor(batch, num, 3) # goal position (x, y, z) [m]
 |
 | and other specific infos (depend on the task and simulator) 
</details></pre>

## Inputs

Inputs to simulators.

<pre>
<details><summary>inputs</summary> |
 | (for Habitat)
 |-- action: Tensor (batch, 1) # discrete action (0=stop, 1=move forward, 2=turn left, 3=turn right)
 |
 | (for MetaWorld)
 |-- cmd_ee_pos: Tensor (batch, 3)
 |-- cmd_gripper: Tensor (batch, 1)
 |
 | (for Vizbot)
 |-- cmd_vel: Tensor (batch, 2) # velocity [m/s] and angle velocity [rad/s]
</details></pre>


# Task and Datasets

Available combinations of `name`, `task`, `dataset`, and `split`.

| name      |   task          | dataset    | split            |
|-----------|-----------------|------------|------------------|
| Habitat   | pointnav        | mp3d       | val_mini         |
|           |                 |            | train            |
|           |                 |            | val              |
|           |                 |            | test             |
|           |                 | hm3d       | train            |
|           |                 |            | train_10_percent |
|           |                 |            | train_50_percent |
|           |                 |            | train            |
|           |                 |            | test             |
|           | objectnav       | mp3d       | val_mini         |
|           |                 |            | train            |
|           |                 |            | val              |
|           |                 | hm3d       | val_mini         |
|           |                 |            | train            |
|           |                 |            | val              |
|           | vln_r2r         | mp3d       | train            |
|           |                 |            | val_seen         |
|           |                 |            | val_unseen       |
| MetaWorld | assembly-v2     | MT1        | train            |
|           |                 |            | test             |
|           | basketball-v2   | MT1        | train            |
|           |                 |            | test             |
|           | bin-picking-v2  | MT1        | train            |
|           |                 |            | test             |
|           | box-close-v2    | MT1        | train            |
|           |                 |            | test             |
|           | button-press-topdown-v2  | MT1        | train            |
|           |                          |            | test             |
|           | button-press-topdown-wall-v2  | MT1        | train            |
|           |                               |            | test             |
|           | button-press-v2  | MT1        | train            |
|           |                  |            | test             |
|           | button-press-wall-v2  | MT1        | train            |
|           |                       |            | test             |
|           | coffee-button-v2  | MT1        | train            |
|           |                   |            | test             |
|           | coffee-pull-v2  | MT1        | train            |
|           |                 |            | test             |
|           | coffee-push-v2  | MT1        | train            |
|           |                 |            | test             |
|           | dial-turn-v2  | MT1        | train            |
|           |               |            | test             |
|           | disassemble-v2  | MT1        | train            |
|           |                 |            | test             |
|           | door-close-v2  | MT1        | train            |
|           |                |            | test             |
|           | door-lock-v2  | MT1        | train            |
|           |               |            | test             |
|           | door-open-v2  | MT1        | train            |
|           |               |            | test             |
|           | door-unlock-v2  | MT1        | train            |
|           |                 |            | test             |
|           | hand-insert-v2  | MT1        | train            |
|           |                 |            | test             |
|           | drawer-close-v2  | MT1        | train            |
|           |                  |            | test             |
|           | drawer-open-v2  | MT1        | train            |
|           |                 |            | test             |
|           | faucet-open-v2  | MT1        | train            |
|           |                 |            | test             |
|           | faucet-close-v2  | MT1        | train            |
|           |                  |            | test             |
|           | hammer-v2    | MT1        | train            |
|           |              |            | test             |
|           | handle-press-side-v2  | MT1        | train            |
|           |                       |            | test             |
|           | handle-press-v2  | MT1        | train            |
|           |                  |            | test             |
|           | handle-pull-side-v2  | MT1        | train            |
|           |                      |            | test             |
|           | handle-pull-v2  | MT1        | train            |
|           |                 |            | test             |
|           | lever-pull-v2  | MT1        | train            |
|           |                |            | test             |
|           | peg-insert-side-v2  | MT1        | train            |
|           |                     |            | test             |
|           | pick-place-wall-v2  | MT1        | train            |
|           |                     |            | test             |
|           | pick-out-of-hole-v2  | MT1        | train            |
|           |                      |            | test             |
|           | reach-v2         | MT1        | train            |
|           |                  |            | test             |
|           | push-back-v2     | MT1        | train            |
|           |                  |            | test             |
|           | push-v2          | MT1        | train            |
|           |                  |            | test             |
|           | pick-place-v2    | MT1        | train            |
|           |                  |            | test             |
|           | plate-slide-v2   | MT1        | train            |
|           |                  |            | test             |
|           | plate-slide-side-v2  | MT1        | train            |
|           |                      |            | test             |
|           | plate-slide-back-v2  | MT1        | train            |
|           |                      |            | test             |
|           | plate-slide-back-side-v2  | MT1        | train            |
|           |                           |            | test             |
|           | peg-insert-side-v2  | MT1        | train            |
|           |                     |            | test             |
|           | peg-unplug-side-v2  | MT1        | train            |
|           |                     |            | test             |
|           | soccer-v2        | MT1        | train            |
|           |                  |            | test             |
|           | stick-push-v2  | MT1        | train            |
|           |                |            | test             |
|           | stick-pull-v2  | MT1        | train            |
|           |                |            | test             |
|           | push-wall-v2  | MT1        | train            |
|           |               |            | test             |
|           | push-v2       | MT1        | train            |
|           |               |            | test             |
|           | reach-wall-v2  | MT1        | train            |
|           |                |            | test             |
|           | reach-v2       | MT1        | train            |
|           |                |            | test             |
|           | shelf-place-v2  | MT1        | train            |
|           |                 |            | test             |
|           | sweep-into-v2  | MT1        | train            |
|           |                |            | test             |
|           | sweep-v2     | MT1        | train            |
|           |              |            | test             |
|           | window-open-v2  | MT1        | train            |
|           |                 |            | test             |
|           | window-close-v2  | MT1        | train            |
|           |                  |            | test             |
|           | ML10         | ML10       | train            |
|           |              |            | test             |
|           | ML45         | ML45       | train            |
|           |              |            | test             |
|           | MT10         | MT10       | train            |
|           |              |            | test             |
|           | MT50         | MT50       | train            |
|           |              |            | test             |
|           |              |            |                  |

