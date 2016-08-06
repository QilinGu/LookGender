#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/gender
eval ./gender_detect model/cascades/facefinder \
     -g /home/ajax/Dropbox/work/LookGender/lib/gender/test/pretrainSVM.xml \
     -i data/test/test2.mp4