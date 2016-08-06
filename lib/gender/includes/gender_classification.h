#ifndef  __GENDER_HPP_
#define  __GENDER_HPP_

#include <iostream>
#include <time.h>
#include <fstream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iterator>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include "lbp.hpp"
#include "histogram.hpp"

namespace FaceGender{
	class gender_classification
	{
        private:
            int radius;
            int neighbors;
            int resize_set;
            int lbp_operator;
            int normalize_set;
            int GaussianBlur_set;
		public:
            gender_classification(void)
            {
                 radius = 1;
                 neighbors = 8;
                 resize_set = 0;
                 lbp_operator = 2;
                 normalize_set = 0;
                 GaussianBlur_set = 0;
            } 
       
			int gender_classify(cv::Mat &face);
			vector<int> feature_extractor(cv::Mat src);
			bool gender_recog(vector<int> validation_data);
            vector<int>  video_feature_extractor(cv::Mat frame);
            vector<int>  video_feature_extractor2(cv::Mat frame);
            int  gender_recog_video(vector<int> ,string gmodel);
            void lbp_feature_gentxt(string imlist, string ftname);
            void svm_train(string training_file, 
                                        string modelname,
                                        int NUM_TRAINING_EXAMPLES, 
                                        int NUM_FEATURES);
            void svm_test(string validation_file, 
                                        string modelname,
                                        int NUM_VALIDATION_EXAMPLES,
                                        int NUM_FEATURES);
	};
}
#endif
