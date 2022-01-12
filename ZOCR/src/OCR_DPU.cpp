#include "../inc/OCR_DPU.h"

// ==================================================================
// ===================== RUN ANN on ZedBoard DPU ====================
// ==================================================================

double OCR_DPU::OCR(Mat src, ERs &strong, ERs &tracked, vector<Text> &text)
{
    chrono::high_resolution_clock::time_point start, end;
	start = chrono::high_resolution_clock::now();

	Run_OCR(src, strong, tracked);
	er_grouping(tracked, text, false, false);
	er_ocr(tracked, text);

	end = chrono::high_resolution_clock::now();
	double time = chrono::duration<double>(end - start).count();

    return time;
}
/**
 * @brief Entry for runing OCR neural network
 *
 * @note DNNDK APIs prefixed with "dpu" are used to easily program &
 *       deploy OCR on DPU platform.
 *
 */
void OCR_DPU::Run_OCR(Mat src, ERs &strong, ERs &tracked)
{
	/*
	Input: ERs for each bounding box traced by the ER Filter"
	Output: 
    -P_is_real : the propability of the ER to be character
    -character : the classify chatacter "
	-Run_NN_model Unit. Main code to Run OCR Neural Network on xilinx DPU. */

	/* DPU Kernel/Task for running OCR */
    DPUKernel *kernel_OCR;
    DPUTask *task_OCR;

    /* Attach to DPU driver and prepare for running */
    dpuOpen();

    /* Load DPU Kernel for OCR */
    kernel_OCR = dpuLoadKernel(KERNEL_OCR);

    /* Create DPU Task for OCR */
    task_OCR = dpuCreateTask(kernel_OCR, 0);

    /* Run OCR Task */
    runNN(src,task_OCR,strong,tracked);

    /* Destroy DPU Task & free resources */
    dpuDestroyTask(task_OCR);

    /* Destroy DPU Kernel & free resources */
    dpuDestroyKernel(kernel_OCR);

    /* Dettach from DPU driver & free resources */
    dpuClose();
		
		
	}


/**
 * @brief Run DPU Task for OCR
 *
 * @param task_OCR - pointer to OCR Task
 *
 * @return none 
 */
void OCR_DPU::runNN(Mat src, DPUTask *task_OCR,ERs &strong, ERs &tracked) {
	
	/* Run OCR ANN and calculate softmax and P is real*/
	assert(task_OCR);

    char character;    //The predicted char
	float P_is_real;   //Posibility that the inut_img is char
	Rect boundary;   //Boundaries of the bounding box
    Mat input_img;  //Input of NN

    /* Get channel count of the output Tensor for OCR Task  */
    int channel = dpuGetOutputTensorChannel(task_OCR, OUTPUT_NODE);
    float *softmax = new float[channel];
    float *FCResult = new float[channel];
	

	for (int i = 0; i < int(strong.size()); i++)
	{
		boundary = strong[i]->bound;
		//Load Bounding Box as image to input ANN
		input_img = Load_ANN_INPUT(boundary, src);
        
        /*Set image into DPU Task for OCR */
        dpuSetInputImage2(task_OCR, INPUT_NODE, input_img);

        /* Launch Chars74k model Task */
        dpuRunTask(task_OCR);

        /* Get DPU execution time (in us) of DPU Task */
        //long long timeProf = dpuGetTaskProfile(task_OCR);
        //float prof = (OCR_WORKLOAD / timeProf) * 1000000.0f;

        /* Get FC result and convert from INT8 to FP32 format */
        dpuGetOutputTensorInHWCFP32(task_OCR, OUTPUT_NODE, FCResult, channel);

        /* Calculate softmax on CPU*/
        CPUCalcSoftmax(FCResult, channel, softmax);


        /* Calculate P_is_real and characher predictions */
        character = Calc_prediction(softmax, channel);
        P_is_real = Calc_P_is_real(FCResult, channel);

        //Threshold the P value  
		if(P_is_real > MIN_OCR_PROB) {
			strong[i]->letter = character;
			strong[i]->prob = 1.0;
			tracked.push_back(strong[i]);
    	}
	}
{
    delete[] softmax;
    delete[] FCResult;
}
}

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void OCR_DPU::CPUCalcSoftmax(const float *data, size_t size, float *result) {
    assert(data && result);
    double sum = 0.0f;

    for (size_t i = 0; i < size; i++) {
        result[i] = exp(data[i]);
        sum += result[i];
    }

    for (size_t i = 0; i < size; i++) {
        result[i] /= sum;
    }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 *
 * @return none
 */

char OCR_DPU::Calc_prediction(const float *d, int size) {
    /*
	-input d : The output of the NN
	-input size : 62 for model's ouput vecture
    -output character : The predected character
    	*/
    assert(d && size > 0 );
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    pair<float, int> ki = q.top();
	char character = classes[ki.second - 1];
    return character;

}

float OCR_DPU::Calc_P_is_real(const float *d, int size){
    //Real vs fake class calculation with Numerical stability proposed by OpanAI
    float mx,gan_logits,stable_real_class_logits;

    mx = d[0]; //Max of d
    for (auto i = 0; i < size; ++i) {if(mx < d[i]){ mx = d[i]; }   
    }

    //max stability
    stable_real_class_logits = exp(d[0] - mx);
    for (auto i = 1; i < size; ++i) { stable_real_class_logits = stable_real_class_logits + exp(d[i] - mx);}
    gan_logits = log(exp(stable_real_class_logits)) + mx;
    float P_is_real ;
    P_is_real =  1/(1 + exp(-gan_logits));     // sigmoid gives you the propability that is a real char

    return P_is_real;
    }

Mat OCR_DPU::Load_ANN_INPUT(Rect boundary, Mat src) {
	/*Load the img from some boundaries as input of the ANN
	-input boundary : The boundaries of the Bounding box
	-input src : The source image to perform OCR
	-Return input_img : The converted image to Run ANN on DPU
	*/

    int Input_size[3] = {32, 32 , 1};
    cv::Mat input_img(3, Input_size, CV_32FC1);
	cv::Mat img_Blur(3, Input_size, CV_32FC1);

    // Because The boundaries of the bounding box is exactly in boundaries of the char, an we don't want that,We add some exctra in order to look like the Chars74k dataset. 
	// 1. Take the correspond bounding box
	int x_plus = int (boundary.width * .15); 
	int y_plus = int (boundary.height * .15);

    //2 .make soure that I always access the src image
    int x_start = (boundary.x - x_plus > 0) ? (boundary.x - x_plus) : 0;
    int x_end =  (boundary.x + boundary.width + x_plus > src.cols) ? (src.cols) : boundary.x + boundary.width + x_plus;
    int y_start = (boundary.y - y_plus > 0) ? (boundary.y - y_plus) : 0;
    int y_end = (boundary.y + boundary.height + y_plus > src.rows) ? (src.rows) : boundary.y + boundary.height + y_plus;
    
	// Take the corespond img
    cv::Mat image = src(cv::Range( y_start, y_end), cv::Range(x_start, x_end));

	//remove noise
	//cv::GaussianBlur(image,img_Blur,Size(3,3),0);

	//Normalization 
	//img_Blur = img_Blur/255.0;
	image = image/255.0;

	// Resize image to 32x32X1 (the fixed input size of the NN)
    cv::resize(image, input_img, input_img.size());

	//we crop image from center
	//int short_edge = Input_size[1];
    //int yy = int((Input_size[0] - short_edge) / 2);	    
    //int xx = int((Input_size[1] - short_edge) / 2);
	//input_img = input_img(cv::Range( yy, yy + short_edge), cv::Range(xx, xx + short_edge));
	
	//cv::resize(input_img, input_img, input_img.size());

    return input_img;


}


void OCR_DPU::er_grouping(ERs &all_er, vector<Text> &text, bool overlap_sup, bool inner_sup)
{
	//Take as input all detected letters and form them into words.
	sort(all_er.begin(), all_er.end(), [](ER *a, ER *b) { return a->center.x < b->center.x; });
	
	if(overlap_sup)
		overlap_suppression(all_er);
	if(inner_sup)
		inner_suppression(all_er);
	
	vector<int> group_index(all_er.size(), -1);
	int index = 0;
	for (int i = 0; i < int(all_er.size()); i++)
	{
		ER *a = all_er[i];
		for (int j = i+1; j < int(all_er.size()); j++)
		{
			ER *b = all_er[j];
			if (abs(a->center.x - b->center.x) < max(a->bound.width,b->bound.width)*3.0 &&
				abs(a->center.y - b->center.y) < (a->bound.height + b->bound.height)*0.25 &&			// 0.5*0.5
				abs(a->bound.height - b->bound.height) < min(a->bound.height, b->bound.height) &&       //The same if statement as in er_track.That means that b and a letters are close enough 
				abs(a->bound.width - b->bound.width) < min(a->bound.height, b->bound.height * 2) &&		//to be in the same word
				abs(a->color1 - b->color1) < 25 &&
				abs(a->color2 - b->color2) < 25 &&
				abs(a->color3 - b->color3) < 25 &&
				abs(a->area - b->area) < min(a->area, b->area)*4)
			{
				if (group_index[i] == -1 && group_index[j] == -1) //all_er[i] and all_er[j] letters not stored in the text so store them both
				{
					group_index[i] = index;
					group_index[j] = index;
					text.push_back(Text());
					text[index].ers.push_back(a);
					text[index].ers.push_back(b);
					index++;
				}

				else if (group_index[j] != -1)  // all_er[j] letters not stored in the text 
				{
					group_index[i] = group_index[j];
					text[group_index[i]].ers.push_back(a);
				}

				else
				{
					group_index[j] = group_index[i]; // all_er[i] letters not stored in the text 
					text[group_index[j]].ers.push_back(b);
				}
			}
		}
	}

	for (int i = 0; i < int(text.size()); i++)
	{
		sort(text[i].ers.begin(), text[i].ers.end(), [](ER *a, ER *b) { return a->center.x < b->center.x; });

		ERs tmp_ers;
		tmp_ers.assign(text[i].ers.begin(), text[i].ers.end());
		overlap_suppression(tmp_ers);
		inner_suppression(tmp_ers);

		vector<Point> points;
		for (int j = 0; j < int(tmp_ers.size()); j++)
		{
			points.push_back(tmp_ers[j]->bound.br());
		}

		text[i].slope = fitline_avgslope(points);
		text[i].box = text[i].ers.front()->bound;
		for (int j = 0; j < int (text[i].ers.size()); j++)
		{
			text[i].box |= text[i].ers[j]->bound;
		}
	}
}


void OCR_DPU::overlap_suppression(ERs &pool)
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

void OCR_DPU::inner_suppression(ERs &pool)
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


double fitline_avgslope(const vector<Point> &p)
{
	if (p.size() <= 2)
		return 0;

	const double epsilon = 0.07;
	double slope = .0;

	for (int i = 0; i < int(p.size()) - 2; i++)
	{
		double slope12 = (double)(p[i + 0].y - p[i + 1].y) / (p[i + 0].x - p[i + 1].x);
		double slope23 = (double)(p[i + 1].y - p[i + 2].y) / (p[i + 1].x - p[i + 2].x);
		double slope13 = (double)(p[i + 0].y - p[i + 2].y) / (p[i + 0].x - p[i + 2].x);

		if (abs(slope12 - slope23) < epsilon && abs(slope23 - slope13) < epsilon && abs(slope12 - slope13) < epsilon)
			slope += (slope12 + slope23 + slope13) / 3;
		else if (abs(slope12) < abs(slope23) && abs(slope12) < abs(slope13))
			slope += slope12;
		else if (abs(slope23) < abs(slope12) && abs(slope23) < abs(slope13))
			slope += slope23;
		else if (abs(slope13) < abs(slope12) && abs(slope13) < abs(slope23))
			slope += slope13;
	}
	
	slope /= (p.size() - 2);
	
	return slope;
}


void OCR_DPU::er_ocr(ERs &all_er,vector<Text> &text)
{
	//const unsigned min_er = 6;
	const unsigned min_pass_ocr = 2;
	
	for (int i = (int)text.size()-1; i >= 0; i--)
	{
		// delete ERs that are in the same channel and are highly overlap
		vector<bool> to_delete(text[i].ers.size(), false);
		for (int m = 0; m < int(text[i].ers.size()); m++)
		{
			for (int n = m + 1; n < int(text[i].ers.size()); n++)
			{
				double overlap_area = (text[i].ers[m]->bound & text[i].ers[n]->bound).area();
				double union_area = (text[i].ers[m]->bound | text[i].ers[n]->bound).area();
				if (overlap_area / union_area > 0.95)
				{
					if (text[i].ers[m]->bound.area() > text[i].ers[n]->bound.area())
						to_delete[n] = true;
					else
						to_delete[m] = true;
				}
			}
		}
		for (int j = (int)text[i].ers.size() - 1; j >= 0; j--)
		{
			if (to_delete[j])
				text[i].ers.erase(text[i].ers.begin() + j);
		}

		
		if (text[i].ers.size() < min_pass_ocr)
		{
			text.erase(text.begin() + i);
			continue;
		}

		Graph graph;
		build_graph(text[i], graph);
		solve_graph(text[i], graph);
		feedback_verify(text[i]);

		text[i].box = text[i].ers.front()->bound;
		for (int j = 0; j < int(text[i].ers.size()); j++)
		{
			text[i].box |= text[i].ers[j]->bound;
		}
	}
}


inline bool OCR_DPU::is_overlapping(ER *a, ER *b)
{
	const double T1 = 0.7;	// area overlap
	const double T2 = 0.5;	// distance

	Rect intersect = a->bound & b->bound;

	if (intersect.area() > T1 * min(a->bound.area(), b->bound.area()) &&
		norm(a->center - b->center) < T2 * max(a->bound.height, b->bound.height))
		return true;

	else
		return false;
}



// model as a graph problem
void OCR_DPU::build_graph(Text &text, Graph &graph)
{
	for (int j = 0; j < int(text.ers.size()); j++)
		graph.push_back(GraphNode(text.ers[j], j));

	for (int j = 0; j < int(text.ers.size()); j++)
	{
		bool found_next = false;
		int cmp_idx = -1;
		for (int k = j + 1; k < int(text.ers.size()); k++)
		{
			// encounter an ER that is overlapping to j
			if (is_overlapping(text.ers[j], text.ers[k]))
				continue;

			// encounter an ER that is the first one different from j
			else if (!found_next)
			{
				found_next = true;
				cmp_idx = k;

				//const int a = index_mapping(graph[j].vertex->letter);
				//const int b = index_mapping(graph[k].vertex->letter);
				//graph[j].edge_prob.push_back(tp[a][b]); // adge prop is the propubility that a a,b letters can form a word
				graph[j].edge_prob.push_back(1.0);
				graph[j].adj_list.push_back(graph[k]);
			}

			// encounter an ER that is overlapping to cmp_idx
			else if (is_overlapping(text.ers[cmp_idx], text.ers[k]))
			{
				cmp_idx = k;

				//const int a = index_mapping(graph[j].vertex->letter);
				//const int b = index_mapping(graph[k].vertex->letter);
				//graph[j].edge_prob.push_back(tp[a][b]);
				graph[j].edge_prob.push_back(1.0);
				graph[j].adj_list.push_back(graph[k]);
			}

			// encounter an ER that is different from cmp_idx, the stream is ended
			else
				break;
		}
	}
}


// solve the graph problem by Dynamic Programming
void OCR_DPU::solve_graph(Text &text, Graph &graph)
{
	vector<double> DP_score(graph.size(), 0);
	vector<int> DP_path(graph.size(), -1);
	const double char_weight = 100;
	const double edge_weight = 50;

	for (int j = 0; j < int(graph.size()); j++)
	{
		if (DP_path[j] == -1)
			DP_score[j] = graph[j].vertex->prob * char_weight;

		for (int k = 0; k < int(graph[j].adj_list.size()); k++)
		{
			const int &adj = graph[j].adj_list[k].index;
			const double score = DP_score[j] + graph[j].edge_prob[k] * edge_weight + text.ers[adj]->prob * char_weight;
			
			if (score > DP_score[adj])
			{
				DP_score[adj] = score;
				DP_path[adj] = j;
			}
		}
	}

	// construct the optimal path
	double max = 0;
	int arg_max = 0;
	for (int j = 0; j < int(DP_score.size()); j++)
	{
		if (DP_score[j] > max)
		{
			max = DP_score[j];
			arg_max = j;
		}
	}

	int node_idx = arg_max;

	text.ers.clear();
	while (node_idx != -1)
	{
		text.ers.push_back(graph[node_idx].vertex);
		node_idx = DP_path[node_idx];
	}

	reverse(text.ers.begin(), text.ers.end());

	for (auto it : text.ers)
		text.word.append(string(1, it->letter));
}





void feedback_verify(Text &text)
{
	// 1. See how many unchangeable upper-case letter and lower-case letter
	ERs big_letter;
	ERs small_letter;

	const int cat[65] = { big, big, big, big, big, big, big, big, big, big, 
	big, big, change, big, big, big, big, big, change, change, big, big, big, big, change, change, big, big, change, big, change, change, change, change, big, change,
	small, big, change, big, small, big, small, big, change, change, big, change, small, small, change, change, small, small, change, big, change, change, change, change, small, change,
	big, big, big};

	for (auto it : text.ers)
	{
		int idx = index_mapping(it->letter);
		if (cat[idx] == category::big)
			big_letter.push_back(it);
		else if (cat[idx] == category::small)
			small_letter.push_back(it);
	}

	// 2. Correct interchangeable letter
	for (auto it : text.ers)
	{
		const double T = 0.20;
		unsigned vote_big = 0;
		unsigned vote_little = 0;
		switch (it->letter)
		{
		case 'C': case 'J': case 'O': case 'P': case 'S': case 'U': case 'V': case 'W': case 'X': case 'Z':
			for (auto it2 : big_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it->bound.y - it->bound.height*T > it2->bound.y - offset)
					vote_little++;
				else
					vote_big++;
			}
			for (auto it2 : small_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it2->bound.y - it2->bound.height*T < it->bound.y + offset)
					vote_little++;
				else
					vote_big++;
			}
			it->letter = (vote_big > vote_little) ? it->letter : it->letter + 0x20;
			break;

		case 'c': case 'j': case 'o': case 'p': case 's': case 'u': case 'v': case 'w': case 'x': case 'z':
			if (!big_letter.empty())
			{
				for (auto it2 : big_letter)
				{
					double offset = (it2->bound.x - it->bound.x) * text.slope;
					if (it->bound.y - it->bound.height*T < it2->bound.y - offset)
						vote_big++;
					else
						vote_little++;
				}
			}
			else
			{
				for (auto it2 : small_letter)
				{
					double offset = (it2->bound.x - it->bound.x) * text.slope;
					if (it2->bound.y - it2->bound.height*T > it->bound.y + offset)
						vote_big++;
					else
						vote_little++;
				}
			}
			it->letter = (vote_big > vote_little) ? it->letter - 0x20 : it->letter;
			break;


		case '1': case 'i': case 'l':
			for (auto it2 : big_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it->bound.y - it->bound.height*T > it2->bound.y - offset)
					vote_little++;
				else
					vote_big++;
			}
			for (auto it2 : small_letter)
			{
				double offset = (it2->bound.x - it->bound.x) * text.slope;
				if (it2->bound.y - it2->bound.height*T < it->bound.y + offset)
					vote_little++;
				else
					vote_big++;
			}
			if (vote_little > vote_big)
				it->letter = 'i';
			else
				it->letter = 'l';

			break;

		default:
			break;
		}
	}

}
int index_mapping(char c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	else if (c >= 'A' && c <= 'Z')
		return c - 'A' + 10;
	else if (c >= 'a' && c <= 'z')
		return c - 'a' + 26 + 10;
	else if (c == '&')
		return 62;
	else if (c == '(')
		return 63;
	else if (c == ')')
		return 64;
	else
		return -1;
}


