/*
 * implementHarris.hpp
 *
 *  Created on: Feb 27, 2020
 *      Author: Mafijul Bhuiyan
 */

#ifndef SRC_IMPLEMENTHARRIS_HPP_
#define SRC_IMPLEMENTHARRIS_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <vector>
#include <cmath>

#include "harrisFilterParams.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

class implementHarris
{

public:
	vector< pair<int,int> > harrisCornerDetect(Mat img, harrisFilterParams* hp, gradientTypes derType);

private:

	vector< vector <float> > bgrToGrayScale(Mat& img);
	pair< vector< vector <float> >, float > gaussKernel(int kernelSize, float std);
	vector< vector <float> > gaussConvolve(vector< vector <float> > &img, pair< vector< vector <float> >, float> &kernel, int kernelSize);
	vector< vector <float> > gaussFilteredImg(vector<vector <float>> &grayScaleImg, int kernelSize, float std);
	pair<vector< vector <float> >, vector< vector <float> > > computeSobelGradients( vector< vector <float> > &filteredImg);
	pair<vector< vector <float> >, vector< vector <float> > > computeRobertsGradients(vector< vector <float> > &filteredImg);
	pair<vector< vector <float> >, float> computeResponseMat(float k, pair<vector< vector <float> >, vector< vector <float> > > &gradients, int windowSize);
	vector<pair<int,int> > findCornerPoints(pair<vector< vector <float> >, float> &responseMat, float thresHoldNMS, int nmsWindowSize, int nmsWinSep);
	vector<pair<int,int> > mapCornerPointsToImage(vector< pair<int,int> > &cornerPoints, gradientTypes derType);

};

#endif /* SRC_IMPLEMENTHARRIS_HPP_ */
