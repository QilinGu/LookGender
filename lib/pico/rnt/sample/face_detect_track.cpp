/*
 *	Copyright (c) 2013, Nenad Markus
 *	All rights reserved.
 *
 *	This is an implementation of the algorithm described in the following paper:
 *		N. Markus, M. Frljak, I. S. Pandzic, J. Ahlberg and R. Forchheimer,
 *		A method for object detection based on pixel intensity comparisons,
 *		http://arxiv.org/abs/1305.4537
 *
 *	Redistribution and use of this program as source code or in binary form, with or without modifications, are permitted provided that the following conditions are met:
 *		1. Redistributions may not be sold, nor may they be used in a commercial product or activity without prior permission from the copyright holder (contact him at nenad.markus@fer.hr).
 *		2. Redistributions may not be used for military purposes.
 *		3. Any published work which utilizes this program shall include the reference to the paper available at http://arxiv.org/abs/1305.4537
 *		4. Redistributions must retain the above copyright notice and the reference to the algorithm on which the implementation is based on, this list of conditions and the following disclaimer.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *	IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//
#include "lib/pico/picornt.h"
#include "lib/cf_libs/kcf/kcf_tracker.hpp"

void* cascade = 0;

int minsize;
int maxsize;

float angle;

float scalefactor;
float stridefactor;

float qthreshold;

int usepyr;
int noclustering;
int verbose;

using namespace cv;
using namespace std;

std::vector<cv::Rect> process_image(cv::Mat frame, int draw)
{
	int i, j;

	uint8_t* pixels;
	int nrows, ncols, ldim;

	#define MAXNDETECTIONS 2048
	int ndetections;
	float qs[MAXNDETECTIONS], rs[MAXNDETECTIONS], cs[MAXNDETECTIONS], ss[MAXNDETECTIONS];

	static cv::Mat gray;
	static cv::Mat pyr[5];

	/*
		...
	*/

	//
	if(pyr[0].empty())
	{
		//
		gray = cv::Mat(cv::Size(frame.cols, frame.rows), frame.depth(), 1);

		//
		pyr[0] = gray;
		pyr[1] = cv::Mat(cv::Size(frame.cols/2, frame.rows/2), frame.depth(), 1);
		pyr[2] = cv::Mat(cv::Size(frame.cols/4, frame.rows/4), frame.depth(), 1);
		pyr[3] = cv::Mat(cv::Size(frame.cols/8, frame.rows/8), frame.depth(), 1);
		pyr[4] = cv::Mat(cv::Size(frame.cols/16, frame.rows/16), frame.depth(), 1);
	}

	// get grayscale image
	if(frame.channels() == 3)
		cv::cvtColor(frame, gray, CV_BGR2GRAY);
	else
		gray = frame.clone();

	// perform detection with the pico library
	int64 t0 = cv::getTickCount();

	if(usepyr)
	{
		int nd;

		//
		pyr[0] = gray;

		pixels = (uint8_t*)pyr[0].data;
		nrows = pyr[0].rows;
		ncols = pyr[0].cols;
		ldim = pyr[0].step;

		ndetections = find_objects(rs, cs, ss, qs, MAXNDETECTIONS, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, MAX(16, minsize), MIN(128, maxsize));

		for(i=1; i<5; ++i)
		{
			cv::resize(pyr[i-1], pyr[i], pyr[i].size());

			pixels = (uint8_t*)pyr[i].data;
			nrows = pyr[i].rows;
			ncols = pyr[i].cols;
			ldim = pyr[i].step;

			nd = find_objects(&rs[ndetections], &cs[ndetections], &ss[ndetections], &qs[ndetections], MAXNDETECTIONS-ndetections, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, MAX(64, minsize>>i), MIN(128, maxsize>>i));

			for(j=ndetections; j<ndetections+nd; ++j)
			{
				rs[j] = (1<<i)*rs[j];
				cs[j] = (1<<i)*cs[j];
				ss[j] = (1<<i)*ss[j];
			}

			ndetections = ndetections + nd;
		}
	}
	else
	{
		//
		pixels = (uint8_t*)gray.data;
		nrows = gray.rows;
		ncols = gray.cols;
		ldim = gray.step;

		//
		ndetections = find_objects(rs, cs, ss, qs, MAXNDETECTIONS, cascade, angle, pixels, nrows, ncols, ldim, scalefactor, stridefactor, minsize, MIN(nrows, ncols));
	}

	if(!noclustering)
		ndetections = cluster_detections(rs, cs, ss, qs, ndetections);

	double t1 = (double(cv::getTickCount()-t0)/cv::getTickFrequency());

	// if the flag is set, draw each detection
	std::vector<cv::Rect> detected_faces;
	if(draw)
		for(i=0; i<ndetections; ++i)
			if(qs[i]>=qthreshold) { // check the confidence threshold
				//cv::circle(frame, cv::Point(cs[i], rs[i]), ss[i]/2, cv::Scalar(0, 0, 255), 4, 8, 0); // we draw circles here since height-to-width ratio of the detected face regions is 1.0f
				cv::rectangle(frame, cv::Point(cs[i]-ss[i]/2, rs[i]-ss[i]/2), cv::Point(cs[i]+ss[i]/2, rs[i]+ss[i]/2), cv::Scalar(0, 0, 255), 4, 8, 0);
				// store the detection results to vector<Rect>
				detected_faces.push_back(cv::Rect(cs[i]-ss[i]/2, rs[i]-ss[i]/2, ss[i], ss[i]));
			}

	// if the `verbose` flag is set, print the results to standard output
	if(verbose)
	{
		//
		for(i=0; i<ndetections; ++i)
			if(qs[i]>=qthreshold) // check the confidence threshold
				std::cout << (int)rs[i] << " " << (int)cs[i] << " " << (int)ss[i] << " " << qs[i] << std::endl;

		//
		std::cout << "# time elapsed (ms) = " << t1*1000 << std::endl;
	}


	return detected_faces;
}

struct face_bbs {
  std::vector<cv::Rect> bbs;
	int count_time;
};

std::vector<cv::Rect> confident_faces(std::vector<face_bbs>& acc_faces, std::vector<cv::Rect> new_faces, std::vector<cv::Rect>& track_bbs) {

	int time_max = 5;
	int num_detection_thresh = 3;

	std::vector<cv::Rect> faces_to_track;

	for(unsigned int i = 0; i < new_faces.size(); i++) {

		bool found = false, tracked = false;

		for (unsigned int j = 0; j < track_bbs.size(); j++) {
			cv::Rect overlap = new_faces[i] & track_bbs[j];
			int overlap_area = overlap.width*overlap.height;
			float overlap_ratio = (overlap_area)/float((new_faces[i].width*new_faces[i].height)+
														(track_bbs[j].width*track_bbs[j].height)-2*overlap_area);
			if (overlap_ratio > 0.5) {
				tracked = true;
				break;
			}
		}

		if (tracked)
			continue;

		for (unsigned int j = 0; j < acc_faces.size(); j++) {
			cv::Rect overlap = new_faces[i] & acc_faces[j].bbs[acc_faces[j].bbs.size()-1];
			int overlap_area = overlap.width*overlap.height;
			float overlap_ratio = (overlap_area)/float((new_faces[i].width*new_faces[i].height)+
														(acc_faces[j].bbs[acc_faces[j].bbs.size()-1].width*acc_faces[j].bbs[acc_faces[j].bbs.size()-1].height)-2*overlap_area);
			if (overlap_ratio > 0.5) {
				acc_faces[j].bbs.push_back(new_faces[i]);
				if (acc_faces[j].bbs.size() >= num_detection_thresh)
					faces_to_track.push_back(new_faces[i]);
				found = true;
				break;
			}
		}

		if (!found) {
			face_bbs temp;
			temp.bbs.push_back(new_faces[i]);
			temp.count_time = 1;
			acc_faces.push_back(temp);
		}

	}

	for (std::vector<face_bbs>::iterator it = acc_faces.begin(); it!=acc_faces.end();) {
		it->count_time += 1;
		if(it->count_time > time_max) {
			if (it->bbs.size() == 1)
				it = acc_faces.erase(it);
			else {
				it->bbs.erase(it->bbs.begin());
				++it;
			}
		} else
			++it;
	}

	return faces_to_track;
}

void rot90(cv::Mat &matImage, int rotflag){
  //1=CW, 2=CCW, 3=180
  if (rotflag == 1){
    transpose(matImage, matImage);  
    flip(matImage, matImage,1); //transpose+flip(1)=CW
  } else if (rotflag == 2) {
    transpose(matImage, matImage);  
    flip(matImage, matImage,0); //transpose+flip(0)=CCW     
  } else if (rotflag ==3){
    flip(matImage, matImage,-1);    //flip(-1)=180          
  } else if (rotflag != 0){ //if not 0,1,2,3:
    printf("Unknown rotation flag");
  }
}
/**
 *  \brief Automatic brightness and contrast optimization with optional histogram clipping
 *  \param [in]src Input image GRAY or BGR or BGRA
 *  \param [out]dst Destination image 
 *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
 *  \note In case of BGRA image, we won't touch the transparency
*/
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0)
{

    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source 
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    return;
}
void process_video(std::string vid)
{
	cv::VideoCapture capture;
	if (vid.compare("") == 0)
		capture.open(0);
	else
		capture.open(vid);

	cv::Mat frame;
	cv::Mat framecopy;

	int stop;
	bool start_track = false;

	// tracking stuffs
	cf_tracking::KcfParameters paras;
	paras.enableTrackingLossDetection = true;
	std::vector<face_bbs> acc_faces;
	std::vector<cf_tracking::KcfTracker*> trackers;
	std::vector<cv::Rect> track_bbs, new_track_bbs;

	std::string windowname = "--------------------";

	if(!capture.isOpened())
	{
		std::cout << "* cannot initialize video capture ..." << std::endl;
		return;
	}

	// the main loop
	framecopy = 0;
	stop = 0;

	while(!stop)
	{
		// wait 5 miliseconds
		int key = cv::waitKey(5);

		// get the frame from webcam
		if(!capture.grab())
		{
			stop = 1;
			frame = 0;
		}
		else
		capture.retrieve(frame,1);
        //BrightnessAndContrastAuto(frame, frame, 0);
        
        rot90(frame,1);
        cv::resize( frame, frame, cv::Size(frame.cols / 3, frame.rows / 3) );
        //cv::Mat img_hist_equalized;
        //vector<cv::Mat> channels; 
        //cvtColor(frame, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
        //split(frame,channels); //split the image into channels
        //equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
        //merge(channels,frame); //merge 3 channels including the modified 1st channel into one image
        //cvtColor(frame, frame, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)
        
		// we terminate the loop if the user has pressed 'q'
		if(frame.empty() || key=='q')
			stop = 1;
		else
		{
			// we mustn't tamper with internal OpenCV buffers
			//if(framecopy.empty())
			framecopy = frame.clone();
			cv::Mat framecopy2  = frame.clone();

			// webcam outputs mirrored frames (at least on my machines)
			// you can safely comment out this line if you find it unnecessary
			//cv::flip(framecopy, framecopy, 1);
			int64 t0 = cv::getTickCount();
			new_track_bbs.clear();
			int i = 0;
			for (std::vector<cf_tracking::KcfTracker*>::iterator it = trackers.begin(); it!=trackers.end(); i++) {
				bool track_success = (*it)->updateAt(frame, track_bbs[i]);
				if (track_success) {
					cv::rectangle(framecopy2, track_bbs[i], cv::Scalar(255, 0, 0), 4, 8, 0);
					new_track_bbs.push_back(track_bbs[i]);
					++it;
				} else {
					delete (*it);
					(*it) = NULL;
					it = trackers.erase(it);
				}
			}

			track_bbs = new_track_bbs;

			// ...
			std::vector<cv::Rect> detected_faces = process_image(framecopy, 1);
			std::vector<cv::Rect> faces_to_track = confident_faces(acc_faces, detected_faces, track_bbs);

			for (unsigned int i = 0; i < faces_to_track.size(); i++) {
				cf_tracking::KcfTracker* temp = new cf_tracking::KcfTracker(paras);
				temp->reinit(frame, faces_to_track[i]);
				trackers.push_back(temp);
				track_bbs.push_back(faces_to_track[i]);
			}
			double t1 = (double(cv::getTickCount()-t0)/cv::getTickFrequency());
			std::cout << "# time elapsed (ms) = " << t1*1000 << std::endl;

			// ...
			cv::imshow(windowname, framecopy);
			cv::imshow("lol", framecopy2);
		}
	}

	// cleanup
	framecopy.release();
	capture.release();
	cv::destroyWindow(windowname);
}

int main(int argc, char* argv[])
{
	//
	int arg;
	//char input[1024], output[1024];
	std::string input, output;

	//
	if(argc<2 || 0==strcmp("-h", argv[1]) || 0==strcmp("--help", argv[1]))
	{
		std::cout << "Usage: pico <path/to/cascade> <options>..." << std::endl;
		std::cout << "Detect objects in images." << std::endl;
		std::cout << "" << std::endl;

		// command line options
		std::cout << "Mandatory arguments to long options are mandatory for short options too." << std::endl;
		std::cout << "  -i,  --input=PATH          set the path to the input image/video" << std::endl;
		std::cout << "                               (*.jpg, *.png, *.avi, *.mp4, etc.)" << std::endl;
		std::cout << "  -o,  --output=PATH         set the path to the output image/video" << std::endl;
		std::cout << "                               (*.jpg, *.png, *.avi, *.mp4, etc.)" << std::endl;
		std::cout << "  -m,  --minsize=SIZE        sets the minimum size (in pixels) of an" << std::endl;
		std::cout << "                               object (default is 128)" << std::endl;
		std::cout << "  -M,  --maxsize=SIZE        sets the maximum size (in pixels) of an" << std::endl;
		std::cout << "                               object (default is 1024)" << std::endl;
		std::cout << "  -a,  --angle=ANGLE         cascade rotation angle:" << std::endl;
		std::cout << "                               0.0 is 0 radians and 1.0 is 2*pi radians" << std::endl;
		std::cout << "                               (default is 0.0)" << std::endl;
		std::cout << "  -q,  --qthreshold=THRESH   detection quality threshold (>=0.0):" << std::endl;
		std::cout << "                               all detections with estimated quality" << std::endl;
		std::cout << "                               below this threshold will be discarded" << std::endl;
		std::cout << "                               (default is 5.0)" << std::endl;
		std::cout << "  -c,  --scalefactor=SCALE   how much to rescale the window during the" << std::endl;
		std::cout << "                               multiscale detection process (default is 1.1)" << std::endl;
		std::cout << "  -t,  --stridefactor=STRIDE how much to move the window between neighboring" << std::endl;
		std::cout << "                               detections (default is 0.1, i.e., 10%%)" << std::endl;
		std::cout << "  -u,  --usepyr              turns on the coarse image pyramid support" << std::endl;
		std::cout << "  -n,  --noclustering        turns off detection clustering" << std::endl;
		std::cout << "  -v,  --verbose             print details of the detection process" << std::endl;
		std::cout << "                               to `stdout`" << std::endl;

		//
		std::cout << "Exit status:" << std::endl;
		std::cout << " 0 if OK," << std::endl;
		std::cout << " 1 if trouble (e.g., invalid path to input image)." << std::endl;

		//
		return 0;
	}
	else
	{
		int size;
		FILE* file;

		//
		file = fopen(argv[1], "rb");

		if(!file)
		{
			std::cout << "# cannot read cascade from " << argv[1] << std::endl;
			return 1;
		}

		//
		fseek(file, 0L, SEEK_END);
		size = ftell(file);
		fseek(file, 0L, SEEK_SET);

		//
		cascade = malloc(size);

		if(!cascade || size!=fread(cascade, 1, size, file))
			return 1;

		//
		fclose(file);
	}

	// set default parameters
	minsize = 128;
	maxsize = 1024;

	angle = 0.0f;

	scalefactor = 1.1f;
	stridefactor = 0.1f;

	qthreshold = 5.0f;

	usepyr = 1;
	noclustering = 0;
	verbose = 0;

	//
	input = "";
	output = "";

	// parse command line arguments
	arg = 2;

	while(arg < argc)
	{
		//
		if(0==strcmp("-u", argv[arg]) || 0==strcmp("--usepyr", argv[arg]))
		{
			usepyr = 1;
			++arg;
		}
		else if(0==strcmp("-i", argv[arg]) || 0==strcmp("--input", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> input;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-o", argv[arg]) || 0==strcmp("--output", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> output;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-m", argv[arg]) || 0==strcmp("--minsize", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> minsize;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-M", argv[arg]) || 0==strcmp("--maxsize", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> maxsize;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-a", argv[arg]) || 0==strcmp("--angle", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> angle;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-c", argv[arg]) || 0==strcmp("--scalefactor", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> scalefactor;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-t", argv[arg]) || 0==strcmp("--stridefactor", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> stridefactor;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-q", argv[arg]) || 0==strcmp("--qthreshold", argv[arg]))
		{
			if(arg+1 < argc)
			{
				//
				std::stringstream ss(argv[arg+1]);
				ss >> qthreshold;
				arg = arg + 2;
			}
			else
			{
				std::cout << "# missing argument after " << argv[arg] << std::endl;
				return 1;
			}
		}
		else if(0==strcmp("-n", argv[arg]) || 0==strcmp("--noclustering", argv[arg]))
		{
			noclustering = 1;
			++arg;
		}
		else if(0==strcmp("-v", argv[arg]) || 0==strcmp("--verbose", argv[arg]))
		{
			verbose = 1;
			++arg;
		}
		else
		{
			std::cout << "# invalid command line argument " << argv[arg] << std::endl;
			return 1;
		}
	}

	if(verbose)
	{
		//
		std::cout << "# Copyright (c) 2013, Nenad Markus" << std::endl;
		std::cout << "# All rights reserved." << std::endl << std::endl;

		std::cout << "# cascade parameters:" << std::endl;
		std::cout << "#	tsr = " << ((float*)cascade)[0] << std::endl;
		std::cout << "#	tsc = " << ((float*)cascade)[1] << std::endl;
		std::cout << "#	tdepth = " << ((int*)cascade)[2] << std::endl;
		std::cout << "#	ntrees = " << ((int*)cascade)[3] << std::endl;
		std::cout << "# detection parameters:" << std::endl;
		std::cout << "#	minsize = " << minsize << std::endl;
		std::cout << "#	maxsize = " << maxsize << std::endl;
		std::cout << "#	scalefactor = " << scalefactor << std::endl;
		std::cout << "#	stridefactor = " << stridefactor << std::endl;
		std::cout << "#	qthreshold = " << qthreshold << std::endl;
		std::cout << "#	usepyr = " << usepyr << std::endl;
	}

	//
	if (input.compare("") == 0)
		process_video(input);
	else {
  	if (input.find(".jpg") != std::string::npos || input.find(".jpeg") != std::string::npos || input.find(".png") != std::string::npos) {
			cv::Mat img;

			//
			img = cv::imread(input, CV_LOAD_IMAGE_COLOR);
			if(img.empty())
			{
				std::cout << "# cannot load image from " << input << std::endl;
				return 1;
			}

			process_image(img, 1);

			//
			if(0!=output[0])
				cv::imwrite(output, img);
			else if(!verbose)
			{
				cv::imshow(input, img);
				cv::waitKey(0);
			}

			//
			img.release();
		} else
			process_video(input);

	}

	return 0;
}
