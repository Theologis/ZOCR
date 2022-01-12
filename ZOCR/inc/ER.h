#ifndef __ERTREMAL_REGION__
#define __ERTREMAL_REGION__

#include <stdio.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <forward_list>
#include <chrono>
//#include <omp.h>
#include <memory>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

struct plist 
{
	plist() :p(0), next(nullptr) {};
	plist(int _p) :p(_p), next(nullptr) {};
	plist(plist *_next) :p(0), next(_next) {};
	plist(int _p, plist *_next) :p(_p), next(_next) {};

	int p;
	plist *next;
};

struct ER
{
public:
	/*
	Struct ER is a bounding box with some extra information stored needed for trecking the letters and for OCR
	*/
	//! Constructor
	ER() {};
	ER(const int level_, const int pixel_, const int x_, const int y_);

	//! seed point and threshold (max grey-level value)
	int level;
	int pixel;
	int x;
	int y;
	plist *p;
	
	//! feature
	int area;     //Area of the Bounding box
	Rect bound;   //https://docs.opencv.org/3.4/d2/d44/classcv_1_1Rect__.html
	Point center; //Center of the bounding box
	double color1;//color domains
	double color2;
	double color3;
	double stkw;

	// for non maximal supression
	bool done;
	double stability;

	//! pointers preserving the tree structure of the component tree
	ER* parent;
	ER* child;
	ER* next;
	ER *sibling_L;
	ER *sibling_R;

	//! for OCR
	int ch;
	char letter; //character of the OCR A-z,a-z,0-9
	double prob; //propability to be a letter
};

typedef vector<ER *> ERs;





class ERFilter
{
public:
	ERFilter(int thresh_step = 2, int min_area = 100, int max_area = 100000, int stability_t = 2, double overlap_coef = 0.7);
	~ERFilter()	{}
	

	
	//! functions
	double Ers_detect(Mat src, ERs &root, vector<ERs> &pool, ERs &strong);
	void compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels);
	ER* er_tree_extract(Mat input);
	void non_maximum_supression(ER *er, ERs &all, ERs &pool, Mat input);
	void er_track(Mat src,vector<ERs> &weak, ERs &all_er, vector<Mat> &channel, Mat Ycrcb);
	void set_thresh_step(int t);
	void set_min_area(int m);
	

private:
	//! Parameters
	int THRESH_STEP;
	int MIN_AREA;
	int MAX_AREA;
	int STABILITY_T;
	double OVERLAP_COEF;
	enum { right, bottom, left, top };

	//! ER operation functions
	inline void er_accumulate(ER *er, const int &current_pixel, const int &x, const int &y);
	void er_merge(ER *parent, ER *child);
	void process_stack(const int new_pixel_grey_level, ERs &er_stack);

	// Gouping operation functions
	void inner_suppression(ERs &pool);
	void overlap_suppression(ERs &pool);

};


class StrokeWidth
{
public:
	double SWT(Mat input);

private:
	struct SWTPoint2d {
		SWTPoint2d(int _x, int _y) : x(_x), y(_y) {};
		int x;
		int y;
		float SWT;
	};
	struct Ray {
		Ray(SWTPoint2d _p, SWTPoint2d _q, vector<SWTPoint2d> _points) : p(_p), q(_q), points(_points){};
		SWTPoint2d p;
		SWTPoint2d q;
		vector<SWTPoint2d> points;
	};
};


void calc_color(ER* er, Mat mask_channel, Mat color_img);


#endif
