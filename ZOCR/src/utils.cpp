#include "../inc/utils.h"

// info funciton
void usage()
{
	cout << "Usage: " << endl;
	cout << "./OCR -v:            take default webcam as input  " << endl;
	cout << "./OCR -v [video]:    take a video as input  " << endl;
	cout << "./OCR -i [image]:    take an image as input  " << endl;
	cout << "./OCR -i [path]:     take folder with images as input,  " << endl;
	cout << endl ;
}


void print_result(int img_count, vector<double> avg_time)
{
	cout << "Total frame number: " << img_count << "\n"
		<< "ER extraction = " << avg_time[0] * 1000 / img_count << "ms\n"
		<< "Non-maximum suppression = " << avg_time[1] * 1000 / img_count << "ms\n"
		<< "Classification = " << avg_time[2] * 1000 / img_count << "ms\n"
		<< "Character tracking = " << avg_time[3] * 1000 / img_count << "ms\n"
		<< "Character grouping = " << avg_time[4] * 1000 / img_count << "ms\n"
		<< "Total execution time = " << avg_time[5] * 1000 / img_count << "ms\n\n";
}

int image_mode(ERFilter* er_filter,OCR_DPU* OCRdpu, char filename[])
{
	double times[2];
	Mat src = imread(filename); // RG image is the input for Linear MSER

	Mat src_gray_scale = cv::imread(filename,CV_LOAD_IMAGE_GRAYSCALE); // Grayscale image is the input for our ANN


	if (src.empty())
	{
		cerr << "ERROR! Unable to open the image file\n";
		usage();
		return -1;
	}


	ERs root; //Tracked ERs from  linear MSER
	vector<ERs> pool; // Tracked ERs after Non-maximum supresion
	ERs strong; //Ers with strong posibility to be Chars
	ERs tracked; // Tracked Ers with chars
	vector<Text> result_text; // Result of OCR

	// Ran linear MSER traking algorithm //
	times[0] = er_filter->Ers_detect(src, root, pool, strong); 
	// Ran OCR on ZedBoard DPU //
	times[1] = OCRdpu->OCR(src_gray_scale,strong,tracked,result_text);
	// Save image and show results from OCR //
	show_result(src, times, result_text);

	return 0;
}

void show_result(Mat& src, double times[], vector<Text> &text)
{	
	/*
	input: src "The input image for the OCR"
	input: result_text "Tracked Letters after runing DPU OCR"
	-Show results for the bounding boxes in steps(pool,tracked,OCR) and the times for each step.
	 */
	Mat result_img = src.clone();

	
	#if defined(WEBCAM_MODE) || defined(VIDEO_MODE)
		//draw_FPS(result_img, times.back());
	#endif
	cout << "ER extraction = " << times[0] * 1000 << "ms\n"
		<< "OCR = " << times[1] * 1000 << "ms\n"

		<< "Total execution time = " << (times[0] + times[1]) * 1000 << "ms\n\n";

	//Save OCR image with the tracked results
	for (auto it : text) { rectangle(result_img, it.box, Scalar(0, 255, 255), 2); }

	for (auto it : text) {
		Size text_size = getTextSize(it.word, FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0);
		rectangle(result_img, Rect(it.box.tl().x, it.box.tl().y - 20, text_size.width, text_size.height + 5), Scalar(30, 30, 200, 0), cv::FILLED);
		putText(result_img, it.word, Point(it.box.tl().x, it.box.tl().y - 4), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0xff, 0xff, 0xff), 1);
	}
	imwrite("result.png", result_img);

	//print results
	cout << "Detected: " << endl;
	for (auto it : text){ cout << it.word << "\n";}
	
}