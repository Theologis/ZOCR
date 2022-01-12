#ifndef __MYLIB__
#define __MYLIB__

#define POS 1
#define NEG -1
#define THRESHOLD_STEP 8
#define MIN_ER_AREA 120
#define MAX_ER_AREA 900000
#define NMS_STABILITY_T 2
#define NMS_OVERLAP_COEF 0.7
#define MIN_OCR_PROBABILITY 0.15
#define OCR_IMG_L 30
#define OCR_FEATURE_L 15
#define MAX_WIDTH 15000
#define MAX_HEIGHT 8000
#define MAX_FILE_PATH 100
#define MAX_FILE_NUMBER 50000

/* for libsvm */
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <queue>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
//#include <filesystem>


#include <time.h>
#include "ER.h"
#include "OCR_DPU.h"




using namespace std;
using namespace std::chrono;
using namespace cv;


// info function
void usage();
void print_result(int img_count, vector<double> avg_time);

// getopt function
int image_mode(ERFilter* er_filter, OCR_DPU* OCRdpu, char filename[]);
void show_result(Mat& src,double times[], vector<Text> &text);

class Profiler
{
public:
	Profiler();
	void Start();
	int Count();
	double Stop();
	void Log(std::string name);
	void Message(std::string msg, float value);
	void Report();

protected:
	int count;
	std::chrono::time_point<std::chrono::steady_clock> time;
	struct record
	{
		record(std::string _name, long long _duration);
		record(std::string _name, float _value, bool _is_msg);
		std::string name;
		long long duration;
		float value;
		bool is_msg;
	};
	std::queue<record> logs;
};

#endif
