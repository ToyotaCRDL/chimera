# <b>Detector</b>

Detector modules for `chimera`.

# Modules

- [YOLOv8](./yolov8/) 
- [Detic](./detic/) 
- [SegmentAnything](./segment_anything/) 

# Get Started

Construct Detector using `create_detector` function.
Detection is performed by `detect` method.

```python
import chimera

sim = chimera.create_simulator(name="Habitat")
config = sim.get_config()
det = chimera.create_detector(config, name="YOLOv8")
obs, info = sim.reset()
outputs = det.detect(obs)
```
