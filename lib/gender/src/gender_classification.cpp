#include "gender_classification.h"

using namespace FaceGender;
using namespace cv;

vector<int> gender_classification::video_feature_extractor2(cv::Mat frame)
{
    Mat dst; // image after preprocessing
    Mat lbp_img; // lbp image
    int resize_set = 0;
    int GaussianBlur_opt = 0;
    int lbp_opt = 1;
    int normalize_set = 0;
    cvtColor(frame, dst, CV_BGR2GRAY);
    if(GaussianBlur_opt==1)
    {
        GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    }
        
    if(resize_set)
    {
        resize(frame, frame, Size(), 0.5, 0.5);
        resize(dst,dst,Size(), 0.5, 0.5);
    }
    // initial values
    int radius = 1;
    int neighbors = 8;
    switch(lbp_opt)
    {
        case 1:
            lbp::OLBP(dst, lbp_img); // use the original operator
            break;
        case 2:
            lbp::ELBP(dst,lbp_img,radius,neighbors);
            break;
        case 3:
            lbp::VARLBP(dst,lbp_img,radius,neighbors);
            break;
        default:
            lbp::OLBP(dst, lbp_img); // use the original operator
            break;
    }

    // now to show the patterns a normalization is necessary, a simple min-max norm will do the job...
    if(normalize_set==1)
    {
        normalize(lbp_img, lbp_img, 0, 255, NORM_MINMAX, CV_8UC1);
    }
    
    Mat spatial_hist = lbp::uniformPatternSpatialHistogram(lbp_img, 256, 3, 3, 0);
    vector<int> feature_vector;
    for(int j = 0; j < spatial_hist.cols; ++j)
    {
        if(spatial_hist.at<int>(0, j) != -1)
        feature_vector.push_back(spatial_hist.at<int>(0, j));
    }
    return feature_vector;
}

vector<int> gender_classification::video_feature_extractor(cv::Mat frame)
{
    Mat dst; // image after preprocessing
    Mat lbp_img; // lbp image
    int resize_set = 0;
    int GaussianBlur_opt = 0;
    int lbp_opt = 1;
    int normalize_set = 1;
    cvtColor(frame, dst, CV_BGR2GRAY);
    if(GaussianBlur_opt==1)
    {
        GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    }
        
    if(resize_set)
    {
        resize(frame, frame, Size(), 0.5, 0.5);
        resize(dst,dst,Size(), 0.5, 0.5);
    }
    // initial values
    int radius = 1;
    int neighbors = 8;
    switch(lbp_opt)
    {
        case 1:
            lbp::OLBP(dst, lbp_img); // use the original operator
            break;
        case 2:
            lbp::ELBP(dst,lbp_img,radius,neighbors);
            break;
        case 3:
            lbp::VARLBP(dst,lbp_img,radius,neighbors);
            break;
        default:
            lbp::OLBP(dst, lbp_img); // use the original operator
            break;
    }

    // now to show the patterns a normalization is necessary, a simple min-max norm will do the job...
    if(normalize_set==1)
    {
        normalize(lbp_img, lbp_img, 0, 255, NORM_MINMAX, CV_8UC1);
    }
    
    Mat spatial_hist = lbp::uniformPatternSpatialHistogram(lbp_img, 256, 3, 3, 0);
    vector<int> feature_vector;
    for(int j = 0; j < spatial_hist.cols; ++j)
    {
        if(spatial_hist.at<int>(0, j) != -1)
        feature_vector.push_back(spatial_hist.at<int>(0, j));
    }
    return feature_vector;
}

void gender_classification::lbp_feature_gentxt(string imlist, string ftname)
{
    cout<<"Feature extraction process starts"<<endl;
    ifstream fin(imlist);
    ofstream ftout;
    ftout.clear();
    ftout.open(ftname,ios::out);
    Mat lbp_img;
    string name;
    int var1;
    int var2;
    int i=0;
    while (fin >> name >> var1 )
    {
        /* do something with name, var1 etc. */
        //cout << name << "tag "<< var1 << "\n";
        Mat dst;
        Mat src = imread(name, CV_LOAD_IMAGE_COLOR);
        
        vector<int> feature_vector = gender_classification::video_feature_extractor(src);

        for(int i = 0; i < feature_vector.size(); ++i)
        {
            if(i != (feature_vector.size() - 1))
                ftout << feature_vector[i] << " ";
            else
                ftout << feature_vector[i];
        }
        ftout << " "<<var1;
        ftout << "\n";
    }
    cout<<"Feature extraction process is finished"<<endl;
}

void gender_classification::svm_train(string training_file, 
                                        string modelname = "pretrainSVM.xml", 
                                        int NUM_TRAINING_EXAMPLES=100, 
                                        int NUM_FEATURES=531)
{
    cout<<"Trainging process starts"<<endl;
    bool trainnewsvm = 0;
    ifstream fin_t;
    fin_t.open(training_file);
    Mat training_data(NUM_TRAINING_EXAMPLES, NUM_FEATURES, CV_32FC1);
    Mat class_labels(NUM_TRAINING_EXAMPLES, 1, CV_32SC1);

    for(int i = 0; i < NUM_TRAINING_EXAMPLES; ++i)
    {
        for(int j = 0; j < (NUM_FEATURES+1); ++j)
        {
            vector<float> example((NUM_FEATURES+1), 0.0f);
            fin_t >> example[j];

            if(j < NUM_FEATURES)
                training_data.at<float>(i, j) = example[j];
            else
                class_labels.at<int>(i, 0) = example[j];
        }
    }
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::POLY);
    svm->setCoef0(1);
    svm->setDegree(2); 
    svm->setGamma(3); 
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, FLT_EPSILON));
    Ptr<ml::TrainData> tData = ml::TrainData::create(training_data, ml::SampleTypes::ROW_SAMPLE, class_labels);
    svm->train(tData);
    svm->save(modelname);
    cout<<"Trainging process is finished"<<endl;

}

int gender_classification::gender_recog_video(vector<int> feature_vector, string gmodel)
{
    Mat validation_data(1, 531, CV_32FC1);
    for(int i = 0; i < 531; ++i)
    {
        validation_data.at<float>(0, i) = (float)feature_vector[i];
    }
    Mat prediction_results(1, 1, CV_32FC1);

    Ptr<ml::SVM> genderrecg = ml::SVM::load<ml::SVM>(gmodel);
    genderrecg->predict(validation_data, prediction_results);
    //cout<<"Predict result"<<endl; 
    //cout<<prediction_results<<endl;

	return prediction_results.at<float>(0,0);
}
	
