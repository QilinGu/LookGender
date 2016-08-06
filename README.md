## Face dataset

I use MultiPIE to train the SVM classifier.
MultiPIE  contains:
1. 7 face angles (front, left30, left45, left60, right30, right60, right45);
2. 4 sessions (session1, session2, session3, session4)


### To generate the training list

1. Get into the image directory, then run the command below

    ```python
     find . -print | grep -i foo > session1.txt
    ```

2. After generate all image list, select some female and male imges for trainging process

    Here, we use ipython.
    ```python
    ## LookGender/data/MultiPIE/genlist.ipynb
    ipython notebook
    ```
    The generated training list is as follows, please modify the  PATH in "genlist.ipynb"

    ```bash
    ...
    /home/ajax/work/MultiPIE/front/session1/045_01_01_051_03.png 0                                                                                                                                                                                                                  /home/ajax/work/MultiPIE/front/session1/235_01_01_051_16.png 0
    /home/ajax/work/MultiPIE/front/session1/174_01_01_051_07.png 0
    ...
    ```

    14000 Female, 14000 Male.

3. After get the training list, shuffle it!
   ```bash
   python largefile_shuffle.py  train.txt shuffle_train.txt
   ```

   ```
   ...
   /home/ajax/work/MultiPIE/left60/session3/193_03_01_080_09.png 1                                                                                                                                                                                                                 /home/ajax/work/MultiPIE/front/session4/172_04_01_051_17.png 0
   /home/ajax/work/MultiPIE/front/session3/028_03_01_051_00.png 1
   /home/ajax/work/MultiPIE/right30/session2/223_02_01_050_10.png 0
   ...

   ```


## SVM classifier training

1. Generate the dynamic link library

    ```bash
    cd LookGender/lib/gender
    cmake .
    make
    ```

    Then, we will have "libGenderClassify.so"

2. Start train SVM model

    ```
    cd LookGender/lib/gender/test
    make
    ./run.sh
    ```

    This will takes some time, including feature extraction and model training.
    Genreate model : pretrainSVM.xml

## Testing

1. Go back to home directory

    ```bash
    cd ../../../
    ```
2. Compile the project
    ```
    make
    ```
3. Run test

    1. run mp4/avi video
    ``` bash

    #!/bin/bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/gender
    eval ./gender_detect model/cascades/facefinder \
         -g /home/ajax/Dropbox/work/LookGender/lib/gender/test/pretrainSVM.xml \
         -i data/test/test2.mp4

    ```

    2. run camera
    ``` bash

    #!/bin/bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/gender
    eval ./gender_detect model/cascades/facefinder \
         -g /home/ajax/Dropbox/work/LookGender/lib/gender/test/pretrainSVM.xml

    ```
