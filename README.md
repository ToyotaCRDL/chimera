# <b>Chimera</b>

Chimera is a modular library to develop robotics system.
This library provides wrappers to third-party software and original software for building robotics systems.
Click [here](./chimera/) to learn more about `chimera`.

- [Installation](#installation)
- [Preparation](#preparation)
- [Examples](#examples)
- [Citing Chimera](#citing-chimera)
- [Our Recent Research](#our-recent-research)
- [License](#license)

# Installation

1. Assuming you have `conda` installed, let's prepare a conda env:
    ```
    conda create -n chimera python=3.9 cmake=3.14.0
    conda activate chimera
    ```

2. Follow the instractions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to install `pytorch` according to your `cuda` environment. 
    ```
    conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    (this command is for cuda=11.8)
    where we recommend to install `pytorch=2.1.0` for compatibility.

3. Git clone this repository.

4. Install modules via `install.sh`.
    ```
    bash install.sh
    ```

    If you want to install all modules, run with `all` option instead:
    ```
    bash install.sh all
    ```

    If you want to install specific module, run with module class and name as follows:
    ```
    bash install.sh Simulator Habitat
    ```
    Note that this command does not install `chimera`.

# Preparation

## Habitat Test Scenes

For running examples, let's download Habitat test scenes using download script:
  ```bash
  bash script/download_habitat_test_scenes.sh
  ```

## HM3D

If you want to run examples about Object-goal Navigation tasks, please use HM3D datasets to follow the link:

[Access HM3D Datasets](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d)

After getting access to the dataset following above link, you can download HM3D datasets using download script:
  ```bash
  bash script/download_habitat_hm3d.sh --username <api-token-id> --password <api-token-secret>
  ```

## OpenAI API

If you want to run examples using OpenAI API, please get an OpenAI API Key [here](https://openai.com/index/openai-api/) and set the `OPENAI_API_KEY` in the environment variables:
  ```bash
  export OPENAI_API_KEY=<your-openai-api-key>
  ```

# Examples

Run demo scripts in `example`.
  ```
  python example/demo_pointnav.py
  ```
To learn more about `example`, please click [here](./example/).


# Citing Chimera

If you use `Chimera` in your research, please use the following BibTeX entry.

```
@misc{taguchi2024chimera,
  title={Chimera},
  author={Shun Taguchi and Hideki Deguchi},
  howpublished={\url{https://github.com/ToyotaCRDL/chimera}},
  year={2024}
}
```

# Our Recent Research

Our recent researches in this repository is as follows:

## [<b>Online Embedding Multi-Scale CLIP Features into 3D Maps</b>](./chimera/mapper/clip_mapper/)

<b>Shun Taguchi and Hideki Deguchi</b>

<b>[Paper](https://arxiv.org/pdf/2403.18178.pdf) | [Code](./chimera/mapper/clip_mapper/)</b>

`CLIPMapper` is online embedding method of multi-scale CLIP features into 3D maps. 
By harnessing CLIP, this method surpasses the constraints of conventional vocabulary-limited methods and enables the incorporation of semantic information into the resultant maps.

## [<b>Language to Map:  Topological map generation from natural language path instructions</b>](./chimera/mapper/l2m/) <b>(ICRA2024)</b>

<b>Hideki Deguchi, Kazuki Shibata and Shun Taguchi</b>

<b>[Paper](https://arxiv.org/pdf/2403.10008) | [Code](./chimera/mapper/l2m/)</b>

`L2M` creates a topological map with actions (forward, turn left, turn right) at each node based on natural language path instructions. 'L2M' then generates a path instruction in response to user queries about a destination.

# License

Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.