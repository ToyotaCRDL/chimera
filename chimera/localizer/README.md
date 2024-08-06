# <b>Localizer</b>

Localizer modules for `chimera`.


# Modules

- [DroidSLAM](./droidslam/)
- [ORBSLAM3](./orbslam3/)

# Get Started

Construct Localizer using `create_localizer` function.
Localization is performed by `track` method.

```python
import chimera

sim = chimera.create_simulator(name="Habitat")
config = sim.get_config()
loc = chimera.create_localizer(config, name="DroidSLAM")
obs, info = sim.reset()
outputs = loc.track(obs)
```