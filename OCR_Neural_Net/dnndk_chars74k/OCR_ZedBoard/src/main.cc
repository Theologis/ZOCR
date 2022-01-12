/*

*/

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

using namespace std;
using namespace cv;

/* 7.71 GOP MAdds for OCR */
#define OCR_WORKLOAD (0.11f)
//LOOK!! Figure 10 page 24 in user guid
/* DPU Kernel name for OCR */              
#define KERNEL_OCR "chars74k_0"		     
/* Input Node for Kernel OCR */
#define INPUT_NODE      "discriminator_conv2d_Conv2D"
/* Output Node for Kernel OCR */
#define OUTPUT_NODE     "discriminator_dense_MatMul"

const string baseImagePath = "./test_data/";

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
                (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
    sort(images.begin(), images.end());
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const &path, vector<string> &kinds) {
    kinds.clear();
    fstream fkinds(path);
    if (fkinds.fail()) {
        fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
        exit(1);
    }
    string kind;
    while (getline(fkinds, kind)) {
        kinds.push_back(kind);
    }

    fkinds.close();
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
void CPUCalcSoftmax(const float *data, size_t size, float *result) {
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
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float *d, int size, int k) {
    assert(d && size > 0 && k > 0);
    priority_queue<pair<float, int>> q;

    for (auto i = 0; i < size; ++i) {
        q.push(pair<float, int>(d[i], i));
    }

    for (auto i = 0; i < k; ++i) {
        pair<float, int> ki = q.top();
        printf("top[%d] prob = %-8f  name = %d\n", i, d[ki.second],ki.second);
        q.pop();
    }
}

/**
 * @brief Run DPU Task for OCR
 *
 * @param task_OCR - pointer to OCR Task
 *
 * @return none
 */
void runOCR(DPUTask *task_OCR) {
    assert(task_OCR);

    /* Mean value for OCR specified in Caffe prototxt */
    vector<string> kinds, images;

    /* Load all image names.*/
    ListImages(baseImagePath, images);
    if (images.size() == 0) {
        cerr << "\nError: No images existing under " << baseImagePath << endl;
        return;
    }
    
    /* 
    //Load all kinds words.
    LoadWords(baseImagePath + "words.txt", kinds);
    if (kinds.size() == 0) {
        cerr << "\nError: No words exist in file words.txt." << endl;
        return;
    }
    */
    /* Get channel count of the output Tensor for OCR Task  */
    int channel = dpuGetOutputTensorChannel(task_OCR, OUTPUT_NODE);
    float *softmax = new float[channel];
    float *FCResult = new float[channel];
    int counter = 0;

    for (auto &imageName : images) {
        cout << "\nLoad image : " << imageName << endl;
        /* Load image and Set image into DPU Task for OCR */
        cv::Mat image = cv::imread(baseImagePath + imageName,CV_LOAD_IMAGE_GRAYSCALE);
        image = image/255.0;
        cv::Mat input_img(32,32,1);
        cv::resize(image, input_img,input_img.size() );
        
//        cout << "Image size : " << input_img.size() << endl;
        
        dpuSetInputImage2(task_OCR, INPUT_NODE, input_img);

        /* Launch RetNet50 Task */
        cout << "\nRun DPU Task for OCR ..." << endl;
        dpuRunTask(task_OCR);

        /* Get DPU execution time (in us) of DPU Task */
        long long timeProf = dpuGetTaskProfile(task_OCR);
        cout << "  DPU Task Execution time: " << (timeProf * 1.0f) << "us\n";
        float prof = (OCR_WORKLOAD / timeProf) * 1000000.0f;
        cout << "  DPU Task Performance: " << prof << "GOPS\n";

        /* Get FC result and convert from INT8 to FP32 format */
        dpuGetOutputTensorInHWCFP32(task_OCR, OUTPUT_NODE, FCResult, channel);

        /* Calculate softmax on CPU and display TOP-5 classification results */
        CPUCalcSoftmax(FCResult, channel, softmax);
	

        /* Display the impage */
        //cv::imshow("Classification of OCR", image);
        TopK(softmax, channel, 1);
        counter++;
        if( counter == 7){break;}
        cout << "_____________________________________________________________________________" << endl;
{
	/* code */
}

    }

    delete[] softmax;
    delete[] FCResult;
}

/**
 * @brief Entry for runing OCR neural network
 *
 * @note DNNDK APIs prefixed with "dpu" are used to easily program &
 *       deploy OCR on DPU platform.
 *
 */
int main(void) {
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
    runOCR(task_OCR);

    /* Destroy DPU Task & free resources */
    dpuDestroyTask(task_OCR);

    /* Destroy DPU Kernel & free resources */
    dpuDestroyKernel(kernel_OCR);

    /* Dettach from DPU driver & free resources */
    dpuClose();

    return 0;
}
