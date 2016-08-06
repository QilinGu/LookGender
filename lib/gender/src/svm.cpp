#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

const int NUM_TRAINING_EXAMPLES = 80;
const int NUM_VALIDATION_EXAMPLES =  20;
const int NUM_FEATURES =  531;
bool trainnewsvm = 0;

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "USAGE: ./svm [TRAINING] [VALIDATION]\n";
        return 1;
    }
    const char* training_file = argv[1];
    const char* validation_file = argv[2];

    ifstream fin_t, fin_v;
    fin_t.open(training_file);
    fin_v.open(validation_file);

    Mat training_data(NUM_TRAINING_EXAMPLES, NUM_FEATURES, CV_32FC1);
    Mat class_labels(NUM_TRAINING_EXAMPLES, 1, CV_32SC1);
    Mat validation_data(NUM_VALIDATION_EXAMPLES, NUM_FEATURES, CV_32FC1);
    Mat prediction_results(NUM_VALIDATION_EXAMPLES, 1, CV_32FC1);

    //for(int i = 0; i < NUM_TRAINING_EXAMPLES; ++i)
    //{
    //    for(int j = 0; j < (NUM_FEATURES+1); ++j)
    //    {
    //        vector<float> example((NUM_FEATURES+1), 0.0f);
    //        fin_t >> example[j];

    //        if(j < NUM_FEATURES)
    //            training_data.at<float>(i, j) = example[j];
    //        else
    //            class_labels.at<int>(i, 0) = example[j];
    //    }
    //}

    vector<float> correct_results(NUM_VALIDATION_EXAMPLES, 0.0f);
    for(int i = 0; i < NUM_VALIDATION_EXAMPLES; ++i)
    {
        for(int j = 0; j < (NUM_FEATURES+1); ++j)
        {
            if(j < NUM_FEATURES)
                fin_v >> validation_data.at<float>(i, j);
            else
                fin_v >> correct_results[i];
        }
    }

    //CvSVMParams params;
    //params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::POLY;
    //params.coef0 = 1;
    //params.degree = 2;
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

    //CvSVM SVM;
    //SVM.train(training_data, class_labels, Mat(), Mat(), params);
    //SVM.predict(validation_data, prediction_results);

	if (trainnewsvm){

	Ptr<ml::SVM> svm = ml::SVM::create();
   // edit: the params struct got removed,
   // we use setter/getter now:
   svm->setType(ml::SVM::C_SVC);
   svm->setKernel(ml::SVM::POLY);
   svm->setCoef0(1);
   svm->setDegree(2); 
   svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, FLT_EPSILON));

//Mat trainData; // one row per feature
//Mat labels;    
  // cout<<class_labels<<endl;
   //cout<<training_data<<endl;
   Ptr<ml::TrainData> tData = ml::TrainData::create(training_data, ml::SampleTypes::ROW_SAMPLE, class_labels);
svm->train(tData);
svm->save("pretrainSVM.xml");
svm->predict(validation_data, prediction_results);
//cout<<"new SVM model is trained"<<endl;
	}
	else{
string svmFile = "pretrainSVM.xml";
Ptr<ml::SVM> genderrecg = ml::SVM::load<ml::SVM>(svmFile);
 genderrecg->predict(validation_data,prediction_results);
// cout<<"old SVM model is used"<<endl;
	}

//svm->train(training_data, ml::ROW_SAMPLE, class_labels);
//// ...
//Mat query; // input, 1channel, 1 row (apply reshape(1,1) if nessecary)
//Mat res;   // output

//svm->predict(validation_data, prediction_results);

cout<<prediction_results<<endl;

    /*for(int i = 0; i < NUM_VALIDATION_EXAMPLES; ++i)
    {
        float prediction = prediction_results.at<float>(i, 0);
        cout << response << " " << correct_results[i] << "\n";
    }*/
	//getchar();
    return 0;
}
