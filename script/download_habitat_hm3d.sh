# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

##### Dowload Script for HM3D dataset #####
# required options
# --username <USERNAME>
# --password <PASSWORD>

USERNAME=""
PASSWORD=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --username)
            USERNAME="$2"
            shift 2
            ;;
        --password)
            PASSWORD="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$USERNAME" ]; then
    echo "Error: --username is required"
    exit 1
fi

if [ -z "$PASSWORD" ]; then
    echo "Error: --password is required"
    exit 1
fi

# Find Path
CHIMERA_PATH=$(pip show chimera | grep Location | awk '{print $2}')
HABITAT_PATH=$CHIMERA_PATH/chimera/simulator/habitat/habitat-lab
cd $HABITAT_PATH

# Dowload HM3D dataset
python -m habitat_sim.utils.datasets_download --username $USERNAME --password $PASSWORD --uids hm3d_minival_v0.2 --data-path data/
python -m habitat_sim.utils.datasets_download --username $USERNAME --password $PASSWORD --uids hm3d_train_v0.2 --data-path data/
python -m habitat_sim.utils.datasets_download --username $USERNAME --password $PASSWORD --uids hm3d_val_v0.2 --data-path data/
DATA_PATH=$HABITAT_PATH/data
cd $DATA_PATH
mkdir -p datasets

# Download pointnav_hm3d_v1
cd $DATA_PATH/datasets
mkdir -p pointnav/hm3d/v1
cd $DATA_PATH/datasets/pointnav/hm3d/v1
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/hm3d/v1/pointnav_hm3d_v1.zip
unzip pointnav_hm3d_v1.zip
rm pointnav_hm3d_v1.zip

# Download objectnav_hm3d_v1
cd $DATA_PATH/datasets
mkdir -p objectnav/hm3d/v1
cd $DATA_PATH/datasets/objectnav/hm3d/v1
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip objectnav_hm3d_v1.zip
rm objectnav_hm3d_v1.zip

# Download_objectnav_hm3d_v2
cd $DATA_PATH/datasets
mkdir -p objectnav/hm3d/v2
cd $DATA_PATH/datasets/objectnav/hm3d/v2
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip objectnav_hm3d_v2.zip
rm objectnav_hm3d_v2.zip

# Download_imagenav_hm3d_v1
cd $DATA_PATH/datasets
mkdir -p instance_imagenav/hm3d/v1
cd $DATA_PATH/datasets/instance_imagenav/hm3d/v1
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v1/instance_imagenav_hm3d_v1.zip
unzip instance_imagenav_hm3d_v1.zip
rm instance_imagenav_hm3d_v1.zip

# Download_imagenav_hm3d_v2
cd $DATA_PATH/datasets
mkdir -p instance_imagenav/hm3d/v2
cd $DATA_PATH/datasets/instance_imagenav/hm3d/v2
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v2/instance_imagenav_hm3d_v2.zip
unzip instance_imagenav_hm3d_v2.zip
rm instance_imagenav_hm3d_v2.zip

# Download_imagenav_hm3d_v3
cd $DATA_PATH/datasets
mkdir -p instance_imagenav/hm3d/v3
cd $DATA_PATH/datasets/instance_imagenav/hm3d/v3
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip
unzip instance_imagenav_hm3d_v3.zip
mv instance_imagenav_hm3d_v3/* .
rm -r instance_imagenav_hm3d_v3
rm instance_imagenav_hm3d_v3.zip

