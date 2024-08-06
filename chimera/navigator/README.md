# <b>Navigator</b>

Navigator modules for `chimera`.

# Modules

- [AStar](./astar_pycpp/)

# Get Started

Construct Navigator using `create_navigator` function.
Navigation is performed by `act` method.

```python
import chimera

config = chimera.create_config()
sim = chimera.create_simulator(name="Habitat", task="pointnav", dataset="habitat-test-scenes", split="val")
config.update(sim.get_config())
loc = chimera.create_localizer(config)
mapper = chimera.create_mapper(config)
nav = chimera.create_navigator(config, name="Astar")
obs, info = sim.reset()
obs["timestamp"] = [0.0]
obs.update(loc.track(obs))
obs = chimera.expand_inputs(obs)
obs.update(mapper.add(obs))
inputs = nav.act(obs)
```