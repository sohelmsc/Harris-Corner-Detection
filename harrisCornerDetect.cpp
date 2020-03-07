/*
 * harrisCornerDetect.cpp
 *
 * The function in this file create an object of the Harris corner detection algorithm class.
 * This also initialize the class containing required parameters for this algorithm.
 *
 * Input: Color or Grayscale image,
 *		  Standard deviation of Gaussian filter [optional]
 *		  Non-maximum suppression (NMS) threshold [optional]
 *        NMS window size  [optional]
 *        NMS window separation  [optional]
 *        Gradient sum window size  [optional]
 *        Gradient type [optional]
 * Output:
 * 		  Image with all detected corner coordinates
 *
 *  Created on: Feb 26, 2020
 *      Author: Mafijul Bhuiyan
 */

#include "implementHarris.hpp"

int main(int argc, char** argv){

	Mat image;
	vector< pair<int,int> > cornerCoordinates;
	gradientTypes derType = gradientTypes::Sobel;
	double elapsedTime = 0.0;

	harrisFilterParams hp;
	implementHarris imHr;

	/*
	 * Read image from user's given filename as well as other filtering parameters.
	 * Image read from openCV is in "BGR" format.
	 */
    if(argc > 1){

    	if(image = imread( argv[1], IMREAD_COLOR ); !image.empty())
			for(int i=2; i<argc; i++){

				switch(i){
					case(2):
						hp.gaussSigma = std::stof(argv[i]);
						break;
					case(3):
						hp.thresHoldNMS = std::stof(argv[i]);
						break;
					case(4):
						hp.nmsWindowSize = std::stoi(argv[i]);
						break;
					case(5):
						hp.nmsWindowSeparation = std::stoi(argv[i]);
						break;
					case(6):
						hp.gradSumWindowSize = std::stoi(argv[i]);
						break;
					default:
						if(std::stoi(argv[i]) <= 1)
							derType = gradientTypes::Sobel;
						else if(std::stoi(argv[i]) >= 2)
							derType = gradientTypes::Roberts;
				}
			}
    	else{
    		std::cout<<  "Could not open or find the image" << endl;
    		return -1;
    	}
    }
    else{
    	cout << "Error: Please provide an image file (color or grayscale)." << endl;
    	return 0;
    }

	auto start = high_resolution_clock::now();

	// Few edge cases for different parameters of the algorithm where the input parameters are out of range.
	if(hp.gaussSigma <= 0.0 ){
		cout << "Error: Standard deviation of the Gaussian filter must be greater than zero." << endl;
		return 0;
	}
	if(hp.thresHoldNMS >= 1.0 || hp.thresHoldNMS <= 0.0){
		cout << "Error: Non-maximum suppression threshold fraction must be between 0 and 1." << endl;
		return 0;
	}
	if(hp.gradSumWindowSize%2 == 0  || hp.gradSumWindowSize <= 1 || hp.gradSumWindowSize >= ((image.rows<=image.cols)?image.rows:image.cols) ){
		cout << "Error: Grad sum window size must be an odd and positive number and must be smaller than the smaller dimension of the input image." << endl;
		return 0;
	}
	if(hp.nmsWindowSize <= 0 || hp.nmsWindowSize >= ((image.rows<=image.cols)?image.rows:image.cols) ){
		cout << "Error: Non-maximum suppression window size must be greater than zero and must be smaller than the smaller dimension of the input image." << endl;
		return 0;
	}
	if(hp.nmsWindowSeparation < 1 || hp.nmsWindowSeparation > hp.nmsWindowSize){
		cout << "Error: Non-maximum suppression window separation must be greater than zero and less than or equal to non-maximum suppression window size." << endl;
		return 0;
	}

	/*
	 * Call the primary function of the implementHarris class.
	 * This function returns the detected corner coordinates
	 * of the given color or grayscale image.
	 */
	cornerCoordinates = imHr.harrisCornerDetect(image, &hp, derType);

	cout << cornerCoordinates.size() << endl;

	/*
	 * Set points in the corresponding corner pixels of the output image.
	 */
	for (auto const& points : cornerCoordinates)
		circle(image, Point(points.second ,points.first), 3, Scalar(0,255,0), 1 );

	imwrite("out.jpg", image);

	auto stop = high_resolution_clock::now();

	elapsedTime += duration<double, std::milli>(stop-start).count();

    std::cout << "Total Execution time: " << elapsedTime << " milliseconds" << endl;

	return 0;
}
