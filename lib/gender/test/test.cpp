#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/core/core.hpp"
#include "../includes/lbp.hpp"
#include "../includes/histogram.hpp"
#include "opencv2/ml/ml.hpp"
#include "../includes/gender_classification.h"

using namespace FaceGender;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{  
    int test = 0; 
    FaceGender::gender_classification gender;
    string modelname = "pretrainSVM.xml";
    string imagetrainpath = "/home/ajax/work/MultiPIE/shuffle_train.txt";
    string train_ftpath = "/home/ajax/work/MultiPIE/shuffle_train_ft.txt";
    cout<<"Start feature extraction process"<<endl;
    gender.lbp_feature_gentxt(imagetrainpath, train_ftpath);
    cout<<"Start training process"<<endl;
    gender.svm_train(train_ftpath,modelname, 28000, 531);
    
    if(test == 1)
    {
        string imagetestpath = "/home/ajax/Dropbox/work/LookGender/lib/gender/test/data/train/nottingham/test.txt";
        string test_ftpath = "test_ft.txt";
        gender.lbp_feature_gentxt(imagetestpath, test_ftpath);
        gender.svm_test(test_ftpath,modelname, 100, 531);
    }

    return 0;
}

