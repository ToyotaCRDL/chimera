# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

##### Dowload Script for Habitat Test Scenes dataset #####

# Find Path
CHIMERA_PATH=$(pip show chimera | grep Location | awk '{print $2}')
HABITAT_PATH=$CHIMERA_PATH/chimera/simulator/habitat/habitat-lab
cd $HABITAT_PATH

# Dowload habitat_test_scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/


