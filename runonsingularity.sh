#!/bin/sh

module load singularity

singularity exec --nv \
                    -B \
                    /data/:/data/ \
                    /data/mril/users/dhinesh/ubuntu.simg/ \
                    python3 /home/dhinesh/Desktop/generative-adverserial-networks/Fetal-Brain-Segmentation-3D-UNET/trainSeGAN.py