# <b>Chimera</b>

## Modules

- [Simulator](./simulator/)
- [Detector](./detector/)
- [Mapper](./mapper/)
- [Localizer](./localizer/)
- [Navigator](./navigator/)
- [Generator](./generator/)

## Config

We use `dict` for our config data.
The data structure of config is shown as follows:

<pre>
<details><summary>config</summary> |
 |-- sampling_interval
 |-- agent
 |    |-- radius
 |    |-- height
 |    |-- action_type: "discrete" or "continuous"
 |    |-- foward_step_size
 |    |-- turn_angle
 |
 |-- rgb
 |    |-- width
 |    |-- height
 |    |-- hfov
 |    |-- position
 |
 |-- depth
 |    |-- width
 |    |-- height
 |    |-- hfov
 |    |-- position
 |    |-- min_depth
 |    |-- max_depth
 |
 |-- map2d
 |    |-- scale
 |    |-- size
 |
 |-- objects
 |    |-- names
 |    |-- conf_thresh
</details></pre>

This config treat the parameter information of the agent.
When you use multiple agents, please prepare a config for each agent.

Our system focuses agent with a RGB-D camera.
Therefore it is not suit for multiple cameras.

## Data Format

In this library, data is managed using `dict`.
The data format is defined as follows:

### Inputs / Outputs

Inputs and outputs from/to modules.

<pre>
<details><summary>inputs/outputs</summary> |
 |-- rgb: Tensor (batch, channel, height, width)
 |-- depth: Tensor (batch, 1, height, width)
 |-- position: Tensor (batch, 3) # 3d position (x, y, z) [m]
 |-- rotation: Tensor (batch, 4) # quaternion (w, x, y, z)
 |-- position2d: Tensor (batch, 2) # 2d position (x, y) [m]
 |-- rotation2d: Tensor (batch, 1) # 2d rotation (theta) [rad]
 |-- map2d: Tensor (batch, 1, height, width)
 |-- mapsem2d: Tensor (batch, channel, height, width) # channel is object category
 |-- pointcloud: Tensor (batch, num, 3) # points (x, y, z) [m]
 |-- action: Tensor (batch, 1) # discrete action (-1=fail, 0=done, 1=move forward, 2=turn left, 3=turn right, ...)
 |-- prev_action: Tensor (batch, 1) # discrete previous action
 |-- cmd_vel: Tensor (batch, 2) # velocity [m/s] and angle velocity [rad/s]
 |-- goal2d: List[Tensor (2)] # range [m] and direction [rad]
 |-- goal2d_xy: List[Tensor (2)] # 2d position of goal [m]
 |-- objects: dict
 |    |-- names: str[batch][names] # the name of object for each class
 |    |-- boxes: Tensor (batch, num, 6) # box on image (x, y, w, h, conf, class) [pix]
 |    |-- masks: Tensor (batch, num, height, width) # masks for each objects
 |    |-- position: Tensor(batch, num, 5) # 3d position (x, y, z, conf, class) [m]     
 |    |-- position2d: Tensor (batch, num, 4) # 2d position (x, y, conf, class) [m]
</details></pre>

## Utilities

`Chiemra` provides several utility functions to facilitate smooth use of the modules.

For more details, please see [here](./util/).
