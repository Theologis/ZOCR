#include "../inc/ER.h"

// ====================================================
// ======================== ER ========================
// ====================================================
ER::ER(const int level_, const int pixel_, const int x_, const int y_) : level(level_), pixel(pixel_), x(x_), y(y_), area(1), stkw(0), done(false), stability(.0), 
																		parent(nullptr), child(nullptr), next(nullptr), sibling_L(nullptr), sibling_R(nullptr)
{
	bound = Rect(x_, y_, 1, 1);
}

// ====================================================
// ===================== ER_filter ====================
// ====================================================
ERFilter::ERFilter(int thresh_step, int min_area, int max_area, int stability_t, double overlap_coef) : THRESH_STEP(thresh_step), MIN_AREA(min_area), MAX_AREA(max_area),
																													STABILITY_T(stability_t), OVERLAP_COEF(overlap_coef)
																													
{

}

void ERFilter::set_thresh_step(int t)
//Set threshold step for the AR tree alorithm
{
	THRESH_STEP = t;
}


void ERFilter::set_min_area(int m)
//Min threshold area of the Bounding box
{
	MIN_AREA = m;
}

double ERFilter::Ers_detect(Mat src,ERs &root, vector<ERs> &pool, ERs &strong)
{
	/*MAIN Track letters function
	  Iput: src "The input image for the OCR"
	  Output: root "All the detected bouding boxes after runing Characher candidates extraction"
	  Output: all ""
	  Output: pool  pool "The pooled bounding boxes after the non_maximum_supression step"
	  Output: tracked "Tracked Letters after runing DPU OCR"
	  Output: text "The text result of the OCR"
	  -Take as input the source image and run linear MSER algorithm and Er traking.
	  */
	//initialization and declaration
	chrono::high_resolution_clock::time_point start, end;
	start = chrono::high_resolution_clock::now();

	Mat Ycrcb;
	vector<ERs> all;
	vector<Mat> channel;
	compute_channels(src, Ycrcb, channel);

	root.resize(channel.size());
	all.resize(channel.size());
	pool.resize(channel.size());

	//Run the 5 steps.
	for (int i =0 ; i < int (channel.size()); i++)
	{
		//Linear MSER algorithm
		root[i] = er_tree_extract(channel[i]);
		non_maximum_supression(root[i], all[i], pool[i], channel[i]);
	}

	end = chrono::high_resolution_clock::now();
	double time = chrono::duration<double>(end - start).count();

	er_track(src,pool, strong, channel, Ycrcb);

	return time;
}


void ERFilter::compute_channels(Mat &src, Mat &YCrcb, vector<Mat> &channels)
{
	/*
	Iput: src "The input image for the OCR"
	Outputs: channel "The channel of the pool ERs"
	Outputs: Ycrcb "Converted rgb to Ycrcb img"
	-Take as input the src image and covert image from RGB to Ycrcb.Then calculate the channels of the image because
	er_tree_extract algorthim need it(er_tree_extract step 1).
	 */
	vector<Mat> splited;
	channels.clear();

	cv::cvtColor(src, YCrcb, COLOR_BGR2YCrCb);
	split(YCrcb, splited);

	channels.push_back(splited[0]);
	channels.push_back(splited[1]);
	channels.push_back(splited[2]);
	channels.push_back(255 - splited[0]);
	channels.push_back(255 - splited[1]);
	channels.push_back(255 - splited[2]);
}


inline void ERFilter::er_accumulate(ER *er, const int &current_pixel, const int &x, const int &y)
{
	/*
	  Input : er "Struct of the pointed Bounding box"		
	  Input : er "Struct of the pointed Bounding box"		
	  Input : current_pixel not USED
	  Input : x "x point to be accumulate"
	  Input : y "y point to be accumulate"
	  -Accumulate x,y to the er.Er is a bounding box with the appropriate info needed 
	 */
	er->area++;

	const int x1 = min(er->bound.x, x);		      //er->bound.x is the x start coordinates of the bounding box
	const int x2 = max(er->bound.br().x - 1, x); //er->bound.br().x - 1 is the y end coordinates of the bounding box
	const int y1 = min(er->bound.y, y);         //er->bound.y is the y start coordinates of the bounding box so (x1,y1) is the down left corner of box "start_point "
	const int y2 = max(er->bound.br().y - 1, y); //er->bound.br().y - 1 is the y end coordinates of the bounding box (x2,y2) is the up right corner of box "end_point"  more info https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/

	er->bound.x = x1;
	er->bound.y = y1;
	er->bound.width = x2 - x1 + 1;
	er->bound.height = y2 - y1 + 1;
}

void ERFilter::er_merge(ER *parent, ER *child)
{
	/*
	Input: parent "Struct of the pointed Bounding box"
	Input: child "Struct of the pointed Bounding box"
	-Merge child struct pointer (bounding box) to parent
	*/
	parent->area += child->area;

	const int x1 = min(parent->bound.x, child->bound.x);
	const int x2 = max(parent->bound.br().x - 1, child->bound.br().x - 1);
	const int y1 = min(parent->bound.y, child->bound.y);
	const int y2 = max(parent->bound.br().y - 1, child->bound.br().y - 1);

	parent->bound.x = x1;
	parent->bound.y = y1;
	parent->bound.width = x2 - x1 + 1;
	parent->bound.height = y2 - y1 + 1;

	if (child->area <= MIN_AREA) //if child's area is too small dellete it
	{
		ER *new_child = child->child;

		if (new_child)
		{
			while (new_child->next)
				new_child = new_child->next;
			new_child->next = parent->child;
			parent->child = child->child;
			child->child->parent = parent;
		}
		delete child;
	}
	else //merge to parent
	{
		child->next = parent->child;
		parent->child = child;
		child->parent = parent;
	}

}



// extract the component tree and store all the ER regions
// base on OpenCV source code, see https://github.com/Itseez/opencv_contrib/tree/master/modules/text for more info
// uses the algorithm described in 
// Linear time maximally stable extremal regions, D Nistér, H Stewénius – ECCV 2008
ER* ERFilter::er_tree_extract(Mat input)
{
	CV_Assert(input.type() == CV_8UC1);

	Mat input_clone = input.clone();
	const int width = input_clone.cols;
	const int height = input_clone.rows;
	const int highest_level = (255 / THRESH_STEP) + 1;
	const uchar *imgData = input_clone.data;

	input_clone /= THRESH_STEP;

	//!< 1. Clear the accessible pixel mask, the heap of boundary pixels and the component
	bool *pixel_accessible = new bool[height*width]();
	vector<int> boundary_pixel[256];
	vector<int> boundary_edge[256];
	vector<ER *>er_stack;
	
	int priority = highest_level;


	//!< 1-2. push a dummy-component onto the stack, 
	//!<	  with grey-level heigher than any allowed in the image
	er_stack.push_back(new ER(256, 0, 0, 0));


	//!< 2. make the top-right corner the source pixel, get its gray level and mark it accessible
	int current_pixel = 0;
	int current_edge = 0;
	int current_level = imgData[current_pixel];
	pixel_accessible[current_pixel] = true;

	
step_3:
	int x = current_pixel % width;
	int y = current_pixel / width;

	//!< 3. push an empty component with current_level onto the component stack
	er_stack.push_back(new ER(current_level, current_pixel, x, y));


	for (;;)
	{
		//!< 4. Explore the remaining edges to the neighbors of the current pixel, in order, as follows : 
		//!<	For each neighbor, check if the neighbor is already accessible.If it
		//!<	is not, mark it as accessible and retrieve its grey - level.If the grey - level is not
		//!<	lower than the current one, push it onto the heap of boundary pixels.If on
		//!<	the other hand the grey - level is lower than the current one, enter the current
		//!<	pixel back into the queue of boundary pixels for later processing(with the
		//!<	next edge number), consider the new pixel and its grey - level and go to 3.
		int neighbor_pixel;
		int neighbor_level;
		

		for (; current_edge < 4; current_edge++)
		{
			switch (current_edge)
			{
				case right	: neighbor_pixel = (x + 1 < width)	? current_pixel + 1		: current_pixel;	break;
				case bottom	: neighbor_pixel = (y + 1 < height) ? current_pixel + width : current_pixel;	break;
				case left	: neighbor_pixel = (x > 0)			? current_pixel - 1		: current_pixel;	break;
				case top	: neighbor_pixel = (y > 0)			? current_pixel - width : current_pixel;	break;
				default: break;
			}
						
			if (!pixel_accessible[neighbor_pixel] && neighbor_pixel != current_pixel)
			{
				pixel_accessible[neighbor_pixel] = true;
				neighbor_level = imgData[neighbor_pixel];

				if (neighbor_level >= current_level)
				{
					boundary_pixel[neighbor_level].push_back(neighbor_pixel);
					boundary_edge[neighbor_level].push_back(0);

					if (neighbor_level < priority)
						priority = neighbor_level;
				}
				else
				{
					boundary_pixel[current_level].push_back(current_pixel);
					boundary_edge[current_level].push_back(current_edge + 1);

					if (current_level < priority)
						priority = current_level;

					current_pixel = neighbor_pixel;
					current_level = neighbor_level;
					current_edge = 0;
					goto step_3;
				}
			}
		}




		//!< 5. Accumulate the current pixel to the component at the top of the stack 
		//!<	(water saturates the current pixel).
		er_accumulate(er_stack.back(), current_pixel, x, y);

		//!< 6. Pop the heap of boundary pixels. If the heap is empty, we are done. If the
		//!<	returned pixel is at the same grey - level as the previous, go to 4	
		if (priority == highest_level)
		{
			delete[] pixel_accessible;
			return er_stack.back();
		}
			
			
		int new_pixel = boundary_pixel[priority].back();
		int new_edge = boundary_edge[priority].back();
		int new_pixel_grey_level = imgData[new_pixel];

		boundary_pixel[priority].pop_back();
		boundary_edge[priority].pop_back();

		while (boundary_pixel[priority].empty() && priority < highest_level)
			priority++;

		current_pixel =  new_pixel;
		current_edge = new_edge;
		x = current_pixel % width;
		y = current_pixel / width;

		if (new_pixel_grey_level != current_level)
		{
			//!< 7. The returned pixel is at a higher grey-level, so we must now process all
			//!<	components on the component stack until we reach the higher grey - level.
			//!<	This is done with the ProcessStack sub - routine, see below.Then go to 4.
			current_level = new_pixel_grey_level;
			process_stack(new_pixel_grey_level, er_stack);
		}
	}
}


void ERFilter::process_stack(const int new_pixel_grey_level, ERs &er_stack)
{
	do
	{
		//!< 1. Process component on the top of the stack. The next grey-level is the minimum
		//!<	of new_pixel_grey_level and the grey - level for the second component on
		//!<	the stack.
		ER *top = er_stack.back();
		ER *second_top = er_stack.end()[-2];
		er_stack.pop_back();

		//!< 2. If new_pixel_grey_level is smaller than the grey-level on the second component
		//!<	on the stack, set the top of stack grey-level to new_pixel_grey_level and return
		//!<	from sub - routine(This occurs when the new pixel is at a grey-level for which
		//!<	there is not yet a component instantiated, so we let the top of stack be that
		//!<	level by just changing its grey - level.
		if (new_pixel_grey_level < second_top->level)
		{
			er_stack.push_back(new ER(new_pixel_grey_level, top->pixel, top->x, top->y));
			er_merge(er_stack.back(), top);
			return;
		}

		//!< 3. Remove the top of stack and merge it into the second component on stack
		//!<	as follows : Add the first and second moment accumulators together and / or
		//!<	join the pixel lists.Either merge the histories of the components, or take the
		//!<	history from the winner.Note here that the top of stack should be considered
		//!<	one ’time-step’ back, so its current size is part of the history.Therefore the
		//!<	top of stack would be the winner if its current size is larger than the previous
		//!<	size of second on stack.
		//er_stack.pop_back();
		er_merge(second_top, top);
		
	}
	//!< 4. If(new_pixel_grey_level>top_of_stack_grey_level) go to 1.
	while (new_pixel_grey_level > er_stack.back()->level);
}


void ERFilter::non_maximum_supression(ER *er, ERs &all, ERs &pool, Mat input)
{
	// Non Recursive Preorder Tree Traversal
	// See http://algorithms.tutorialhorizon.com/binary-tree-preorder-traversal-non-recursive-approach/ and 
	//https://github.com/martinkersner/non-maximum-suppression-cpp for more info.

	// 1. Create a Stack
	vector<ER *> tree_stack;
	ER *root = er;
	root->parent = root;

save_step_2:
	// 2. Print the root and push it to Stack and go left, i.e root=root.left and till it hits the nullptr.
	for (; root != nullptr; root = root->child)
	{
		tree_stack.push_back(root);
	#ifdef GET_ALL_ER
		all.push_back(root);
	#endif
	}
	

	// 3. If root is null and Stack is empty Then
	//		return, we are done.
	if (root == nullptr && tree_stack.empty())
	{
		//cout << "Before NMS: " << n << "    After NMS: " << pool.size() << endl;
		return;
	}

	// 4. Else
	//		Pop the top Node from the Stack and set it as, root = popped_Node.
	//		Go right, root = root.right.
	//		Go to step 2.

	root = tree_stack.back();
	tree_stack.pop_back();
	
	if (!root->done)
	{
		ERs overlapped;
		ER *parent = root;
		while ((root->bound&parent->bound).area() / (double)parent->bound.area() > OVERLAP_COEF && (!parent->done))
		{
			parent->done = true;
			overlapped.push_back(parent);
			parent = parent->parent;
		}
		
		// Core part of NMS
		// Rt-k is the parent of Rt in component tree
		// Remove ERs such that number of overlap < 3, select the one with highest stability
		// If there exist 2 or more overlapping ER with same stability, choose the one having smallest area
		// overlap		O(Rt-k, Rt) = |Rt| / |Rt-k|
		// stability	S(Rt) = (|Rt-t'| - |Rt|) / |Rt|
		if (int(overlapped.size()) >= 1 + STABILITY_T)
		{
			for (int i = 0; i < int (overlapped.size() - STABILITY_T); i++)
			{
				overlapped[i]->stability =  (double)overlapped[i]->bound.area() / (double)(overlapped[i + STABILITY_T]->bound.area() - overlapped[i]->bound.area());
			}

			int max = 0;
			for (int i = 1; i < int(overlapped.size() - STABILITY_T); i++)
			{
				if (overlapped[i]->stability > overlapped[max]->stability )
					max = i;
				else if (overlapped[i]->stability == overlapped[max]->stability)
					max = (overlapped[i]->bound.area() < overlapped[max]->bound.area()) ? i : max;
			}

			double aspect_ratio = (double)overlapped[max]->bound.width / (double)overlapped[max]->bound.height;
			if (aspect_ratio < 2.0 && aspect_ratio > 0.10 && 
				overlapped[max]->area < MAX_AREA && 
				overlapped[max]->area > MIN_AREA &&
				overlapped[max]->bound.height < input.rows*0.8 &&
				overlapped[max]->bound.width < input.cols*0.8)
			{
				pool.push_back(overlapped[max]);
			}
		}
	}

	root = root->next;
	goto save_step_2;

	// 5. End If
}


void ERFilter::er_track(Mat src,vector<ERs> &pool,ERs &strong, vector<Mat> &channel, Mat Ycrcb)
{
	/*Input: src : source image for the OCR 
	  Input: pool "The pooled bounding boxes"
	  Input: channel "The channel of the pool ERs"
	  Input: Ycrcb "Converted rgb to Ycrcb img"
	  Output: strong "Bounding Box with strong possibility to be letter "
	  -Take the pooled bounding boxes detected runing OpenCV an returns the Bounding Box with strong possibility to be letter .
	  */

#ifdef USE_STROKE_WIDTH
	StrokeWidth SWT;
#endif

	//Calculate the center of each pool bounding box
	ERs all_er;
	for (int i = 0; i < int(pool.size()); i++)
	{
		for (auto it : pool[i])
		{
			calc_color(it, channel[i], Ycrcb);
#ifdef USE_STROKE_WIDTH //NOT USED
			it->stkw = SWT.SWT(channel[i](it->bound));
#endif
			it->center = Point(it->bound.x + it->bound.width / 2, it->bound.y + it->bound.height / 2);
			it->ch = i;
		}
	}
	
	//Store all pool ERs without needing the use of channel.
	for (int i = 0; i < int(pool.size()); i++)
	{
		all_er.insert(all_er.end(), pool[i].begin(), pool[i].end());
	}


	vector<vector<bool>> strong_log(pool.size());//strong logistics:keeps the logistics for the strong ERs
	for (int i = 0; i < int(strong_log.size()); i++)
	{
		strong_log[i].resize(pool[i].size());
	}

	//calculate threshold WIDTH and HEIGHT of the strong ERs.
	int MAX_WIDTH = (all_er.size() > 60) ? int(src.cols / 4) : int(src.cols); //min 4 letters in a col if they are too big.
	int MAX_HEIGHT = (all_er.size() > 20) ? int(src.rows / 3) : int(src.rows); //min 3 letters in a row if they are too big.

	//for loops for keeping the bounding boxes that have strong posibility to be letters.
	//One bounding box to bee leeter mast have:
	// -State1: A bounding box very close to it(1st line) ,having close hights(2st line) and widths(3st line) and colors.
	// -State2: Must be in the Threshold WIDTH and HEIGHT 
	// -State3: Must be in the Threshold area
	for (int i = 0; i < int(all_er.size()); i++)
	{
		ER *s = all_er[i];
		for (int m = 0; m < int(pool.size()); m++)
		{
			for (int n = 0; n < int(pool[m].size()); n++)
			{
				if (strong_log[m][n] == true) continue;

				ER* w = pool[m][n];
				if (abs(s->center.x - w->center.x) + abs(s->center.y - w->center.y) < max(s->bound.width, s->bound.height) << 1 && //1st line
					abs(s->bound.height - w->bound.height) < min(s->bound.height, w->bound.height) &&							  //2st line
					abs(s->bound.width - w->bound.width) < (s->bound.width + w->bound.width) >> 1 &&                             //3st line
					abs(s->color1 - w->color1) < 25 &&                                                                          //
					abs(s->color2 - w->color2) < 25 &&                                                                         //
					abs(s->color3 - w->color3) < 25 &&                                                                      //->state1
					w->bound.width < MAX_WIDTH &&   
					w->bound.height < MAX_HEIGHT &&                                                //state2                      
					//!((s->bound.x <= w->bound.x) && (s->bound.y <= w->bound.y) &&
					//(s->bound.x + s->bound.width >= w->bound.x + w->bound.width) && (s->bound.y + s->bound.height >= w->bound.y + w->bound.height)) &&


#ifdef USE_STROKE_WIDTH //NOT USED
					(s->stkw / w->stkw) < 4 &&
					(s->stkw / w->stkw) > 0.25 &&
#endif
					abs(s->area - w->area) < min(s->area, w->area) * 3)//state3
				{
					strong_log[m][n] = true;
					strong.push_back(w);
				}
			}
		}
	}
	overlap_suppression(strong);
	inner_suppression(strong);
}


void calc_color(ER* er, Mat mask_channel, Mat color_img)
{
	// calculate the color of each ER
	Mat img = mask_channel(er->bound).clone();
	threshold(255-img, img, 128, 255, THRESH_OTSU);

	int count = 0;
	double color1 = 0;
	double color2 = 0;
	double color3 = 0;
	for (int i = 0; i < img.rows; i++)
	{
		uchar* ptr = img.ptr(i);
		uchar* color_ptr = color_img.ptr(i);
		for (int j = 0, k = 0; j < img.cols; j++, k += 3)
		{
			if (ptr[j] != 0)
			{
				++count;
				color1 += color_ptr[k];
				color2 += color_ptr[k + 1];
				color3 += color_ptr[k + 2];
			}
		}
	}
	er->color1 = color1 / count;
	er->color2 = color2 / count;
	er->color3 = color3 / count;


}


void ERFilter::overlap_suppression(ERs &pool)
{
	//#Merge the bounding boxes that are overlaped.
	vector<bool> merged(pool.size(), false);

	for (int i = 0; i < int(pool.size()); i++)
	{
		for (int j = i + 1; j < int(pool.size()); j++)
		{
			if (merged[j])	continue;

			Rect overlap = pool[i]->bound & pool[j]->bound;
			Rect union_box = pool[i]->bound | pool[j]->bound;
			
			if ((double)overlap.area() / (double)union_box.area() > 0.5)
			{
				merged[j] = true;

				int x = (int)((pool[i]->bound.x + pool[j]->bound.x) * 0.5);
				int y = (int)((pool[i]->bound.y + pool[j]->bound.y) * 0.5);
				int width = (int)((pool[i]->bound.width + pool[j]->bound.width) * 0.5);
				int height = (int)((pool[i]->bound.height + pool[j]->bound.height) * 0.5);

				pool[i]->bound.x = x;
				pool[i]->bound.y = y;
				pool[i]->bound.height = height;
				pool[i]->bound.width = width;			
				pool[i]->center.x = (int)(x + pool[i]->bound.width * 0.5);
				pool[i]->center.y = (int)(y + pool[i]->bound.height * 0.5);
			}
		}
	}

	for (int i = (int)pool.size()-1; i >= 0; i--)
	{
		if (merged[i])
		{
			pool.erase(pool.begin() + i);
		}
	}
}


void ERFilter::inner_suppression(ERs &pool)
{
	//Detele the bounding boxes that are inner other bounding boxes
 	vector<bool> to_delete(pool.size(), false);
	const double T1 = 2.0;
	const double T2 = 0.2;

	for (int i = 0; i < int(pool.size()); i++)
	{
		for (int j = 0; j < int(pool.size()); j++)
		{
			if (norm(pool[i]->center - pool[j]->center) < T2 * max(pool[i]->bound.width, pool[i]->bound.height))
			{
				if (pool[i]->bound.x <= pool[j]->bound.x &&
					pool[i]->bound.y <= pool[j]->bound.y &&
					pool[i]->bound.br().x >= pool[j]->bound.br().x &&
					pool[i]->bound.br().y >= pool[j]->bound.br().y &&
					(double)pool[i]->bound.area() / (double)pool[j]->bound.area() > T1)
					to_delete[j] = true;
			}
		}
	}



	for (int i = (int)pool.size() - 1; i >= 0; i--)
	{
		if (to_delete[i])
			pool.erase(pool.begin() + i);
	}
}

double StrokeWidth::SWT(Mat input)
{
	//Stroke Width Transform for letters detection check for more info (NOT USED in this project): http://www.math.tau.ac.il/~turkel/imagepapers/text_detection.pdf
	Mat thresh;
	Mat canny;
	Mat blur;
	Mat grad_x;
	Mat grad_y;
	threshold(input, thresh, 128, 255, THRESH_OTSU);
	cv::Canny(thresh, canny, 150, 300, 3);
	cv::GaussianBlur(thresh, blur, Size(5, 5), 0);
	cv::Sobel(blur, grad_x, CV_32F, 1, 0, 3);
	cv::Sobel(blur, grad_y, CV_32F, 0, 1, 3);


	// Stroke Width Transform 1st pass
	Mat SWT_img(input.rows, input.cols, CV_32F, FLT_MAX);

	vector<Ray> rays;

	for (int i = 0; i < canny.rows; i++)
	{
		uchar *ptr = canny.ptr(i);
		float *xptr = grad_x.ptr<float>(i);
		float *yptr = grad_y.ptr<float>(i);
		for (int j = 0; j < canny.cols; j++)
		{
			if (ptr[j] != 0)
			{
				int x = j;
				int y = i;
				double dir_x = xptr[j] / sqrt(xptr[j] * xptr[j] + yptr[j] * yptr[j]);
				double dir_y = yptr[j] / sqrt(xptr[j] * xptr[j] + yptr[j] * yptr[j]);
				double cur_x = x;
				double cur_y = y;
				int cur_pixel_x = x;
				int cur_pixel_y = y;
				vector<SWTPoint2d> point;
				point.push_back(SWTPoint2d(x, y));
				for (;;)
				{
					cur_x += dir_x;
					cur_y += dir_y;

					if (round(cur_x) == cur_pixel_x && round(cur_y) == cur_pixel_y)
						continue;
					else
						cur_pixel_x = (int)round(cur_x), cur_pixel_y = (int)round(cur_y);

					if (cur_pixel_x < 0 || (cur_pixel_x >= canny.cols) || cur_pixel_y < 0 || (cur_pixel_y >= canny.rows))
						break;

					point.push_back(SWTPoint2d(cur_pixel_x, cur_pixel_y));
					double gx = grad_x.at<float>(cur_pixel_y, cur_pixel_x);
					double gy = grad_y.at<float>(cur_pixel_y, cur_pixel_x);
					double mag = sqrt(gx*gx + gy*gy);;
					double q_x = grad_x.at<float>(cur_pixel_y, cur_pixel_x) / mag;
					double q_y = grad_y.at<float>(cur_pixel_y, cur_pixel_x) / mag;
					if (acos(dir_x * -q_x + dir_y * -q_y) < CV_PI / 2.0)
					{
						double length = sqrt((cur_pixel_x - x)*(cur_pixel_x - x) + (cur_pixel_y - y)*(cur_pixel_y - y));
						for (auto it : point)
						{
							if (length < SWT_img.at<float>(it.y, it.x))
							{
								SWT_img.at<float>(it.y, it.x) = (float)length;
							}
						}
						rays.push_back(Ray(SWTPoint2d(j, i), SWTPoint2d(cur_pixel_x, cur_pixel_y), point));
						break;
					}
				}
			}
		}
	}

	// Stroke Width Transform 2nd pass
	for (auto& rit : rays) {
		for (auto& pit : rit.points)
			pit.SWT = SWT_img.at<float>(pit.y, pit.x);

		std::sort(rit.points.begin(), rit.points.end(), [](SWTPoint2d lhs, SWTPoint2d rhs){return lhs.SWT < rhs.SWT; });
		float median = (rit.points[rit.points.size() / 2]).SWT;
		for (auto& pit : rit.points)
			SWT_img.at<float>(pit.y, pit.x) = std::min(pit.SWT, median);
	}


	// return mean stroke width
	double stkw = 0;
	int count = 0;
	for (int i = 0; i < SWT_img.rows; i++)
	{
		float* ptr = SWT_img.ptr<float>(i);
		for (int j = 0; j < SWT_img.cols; j++)
		{
			if (ptr[j] != FLT_MAX)
			{
				stkw += ptr[j];
				count++;
			}
		}
	}
	
	stkw /= count;
	return stkw;
}
