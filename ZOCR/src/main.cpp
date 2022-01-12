/*
  main.cpp - Main Function for OCR
  Copyright (c) 2020-2021 UTH CAS LAB Team.
  The executable file OCR must ultimately exist in the project root directory.
  Run code with the following arguments:
  ./OCR -v:            take default webcam as input  
  ./OCR -v [video]:    take a video as input  
  ./OCR -i [image]:    take an image as input 
  ./OCR -i [path]:     take folder with images as input
*/
#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "../inc/ER.h"

#include "../inc/utils.h"

#include "../inc/OCR_DPU.h"

#ifdef _WIN32
#include "../inc/getopt.h"
#elif __linux__
#include <getopt.h>
#endif


using namespace std;

int main(int argc, char* argv[])
{
	//class for Lineat MSER
	ERFilter* er_filter = new ERFilter(THRESHOLD_STEP, MIN_ER_AREA, MAX_ER_AREA, NMS_STABILITY_T, NMS_OVERLAP_COEF);
	// class to Run OCR on DPU
	OCR_DPU* OCRdpu = new OCR_DPU();

	char *filename = nullptr;
	char c = 0;
	c = getopt (argc, argv, "v:i:o:l:t:");
	
	switch (c)
	{
	case 'i':
		filename = optarg;
		image_mode(er_filter, OCRdpu, filename);
		break;
	default:
		usage();
		abort();
	}

	return 0;
}
