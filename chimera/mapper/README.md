# <b>Mapper</b>

Mapper modules for `chimera`.

# Modules

- [Mapper2D](./mapper2d/)
- [CLIPMapper](./clip_mapper/)
- [L2M](./l2m/)
- [VLMap](./vlmaps/)

# Get Started

Construct Mapper using `create_mapper` function.
Mapping is performed by `add` method.

```python
import chimera

config = chimera.create_config()
sim = chimera.create_simulator(config, name="Habitat")
config.update(sim.get_config())
mapper = chimera.create_mapper(config, name="DepthAndTrajectoryMapper")
obs, info = sim.reset()
outputs = mapper.add(obs)
```