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

#include "../LBP/LBP.hpp"
#include "histogram.hpp"

namespace FaceGender{
	class gender_classification
	{
		public:
			int gender_classify(cv::Mat &face);
			vector<double> feature_extractor(cv::Mat src);
			bool gender_recog(vector<int> validation_data);
            Mat video_feature_extractor(cv::Mat frame, int lbp_operator, bool original_szie);
            int gender_recog_video(Mat lbp_ft,string gmodel);
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
