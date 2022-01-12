#ifndef __ZedBoard_DPU__
#define __ZedBoard_DPU__

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <math.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>
#include <forward_list>
#include <sstream>
#include <iterator>


/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>


#include "ER.h"


/* header file for DNNDK APIs */
#include <dnndk/dnndk.h> 



using namespace std;
using namespace std::chrono;
using namespace cv;

/* 7.71 GOP MAdds for OCR */
#define OCR_WORKLOAD (0.11f)
/* DPU Kernel name for OCR */              
#define KERNEL_OCR "chars74k_0"		     
/* Input Node for Kernel OCR */
#define INPUT_NODE  "discriminator_conv2d_Conv2D"
/* Output Node for Kernel OCR */
#define OUTPUT_NODE  "discriminator_dense_MatMul"
//All classes from Chars74k dataset
#define classes "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
//Min threshold propability
#define MIN_OCR_PROB (.9f)
//Inpute size
#define  input_size 32

struct Text
{
	Text(){};
	Text(ER *x, ER *y, ER *z)
	{
		ers.push_back(x);
		ers.push_back(y);
		ers.push_back(z);
	};
	ERs ers;
	double slope;
	Rect box;
	string word;
};

struct GraphNode
{
	GraphNode(ER *v, const int i) : vertex(v), index(i){};
	ER* vertex;
	int index;
	vector<GraphNode> adj_list;
	vector<double> edge_prob;
};

typedef vector<GraphNode> Graph;

class OCR_DPU
{
public:

	//OCR_DPU::OCR_DPU(/* args */);
	//OCR_DPU::~OCR_DPU(){}

	//OCR Functions
    double OCR(Mat src, ERs &strong, ERs &tracked, vector<Text> &text);
    void Run_OCR(Mat src, ERs &strong , ERs &tracked);
    void er_grouping(ERs &all_er, vector<Text> &text, bool overlap_sup = false, bool inner_sup = false);
	void er_ocr(ERs &all_er, vector<Text> &text);
    void runNN(Mat src,DPUTask *task_OCR, ERs &strong, ERs &tracked);
    void CPUCalcSoftmax(const float *data, size_t size, float *result);
    char Calc_prediction(const float *d, int size);
    float Calc_P_is_real(const float *d, int size);
    Mat Load_ANN_INPUT(Rect boundary, Mat src);

private:
    // Gouping operation functions
	inline bool is_overlapping(ER *a, ER *b);
	void inner_suppression(ERs &pool);
	void overlap_suppression(ERs &pool);

    // OCR operation functions
	void build_graph(Text &text, Graph &graph);
	void solve_graph(Text &text, Graph &graph);

	double tp[62][62]; // The propubility that i,j letters can form a word e.g. The propublity to have to AA is 20%.

};



enum category
{
	big = 2,
	small = 1,
	change = 0
};

double fitline_avgslope(const vector<Point> &p);

//OCR
void feedback_verify(Text &text); // twin character corection 
int index_mapping(char c);

#endif