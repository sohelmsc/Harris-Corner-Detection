/*
 * implementHarris.cpp
 *
 * This file consists of all the necessary functions required to detect the corners of a given image
 * using Harris corner detection algorithm.
 *
 *  Created on: Feb 27, 2020
 *      Author: Mafijul Bhuiyan
 */

#include "implementHarris.hpp"

/*
 * Detect corners of an image using the method proposed by Harris and Stephens.
 */
vector< pair<int,int> > implementHarris::harrisCornerDetect(Mat img, harrisFilterParams* hp, gradientTypes derType){

	Mat image;
	pair<vector< vector <float> >, vector< vector <float> > > gradients;
	vector< vector <float> > grayScaleImg, filteredGrayImg;
	vector< pair<int,int> > cornerPoints, imageCorners;
	pair<vector< vector <float> >, float> responseMat;

    /*
     * Convert BGR image from openCV to gray scale image.
     */
	if(img.channels() == 3)
		grayScaleImg = bgrToGrayScale(img);
	else if(img.channels() == 1)
		grayScaleImg = img;

    /*
     * Apply Gaussian filter to the grayscale image. Gaussian filtering basically reduce the noise
     * and make the image smoother.
     */
    filteredGrayImg = gaussFilteredImg(grayScaleImg, hp->gaussKernelSize, hp->gaussSigma);

    /*
     * Compute Gradients of the Gaussian filtered image.
     * Two gradient methods have been implemented: Sobel and Roberts cross
     */
    if(derType == gradientTypes::Sobel)
    	gradients = computeSobelGradients(filteredGrayImg);
    else
    	gradients = computeRobertsGradients(filteredGrayImg);

    /*
     * Compute Harris response matrix from the image gradients.
     * The summation of the products of horizontal and vertical
     * gradients are computed within the "harrisWindowSize".
     */
    responseMat = computeResponseMat(hp->k, gradients, hp->gradSumWindowSize);

    /*
     * Get the coordinates of local maximum Harris responses by
     * applying non-maximum suppression in a window.
     */
    cornerPoints = findCornerPoints(responseMat, hp->thresHoldNMS, hp->nmsWindowSize, hp->nmsWindowSeparation);

    // Map those corner points into the actual image coordinates.
    imageCorners = mapCornerPointsToImage(cornerPoints, derType);

    return imageCorners;
}

/*
 * Implement Gaussian filter to the gray scale image.
 */
vector< vector <float> > implementHarris::gaussFilteredImg(vector<vector <float> > &grayScaleImg, int kernelSize, float std){

	vector< vector <float> > gaussFilteredImage;
	pair< vector< vector <float> >, float > kernelMat;

	// If the kernel size or standard deviation is zero, same image will be returned.
    if(kernelSize <= 0 || std <= 0)
        return grayScaleImg;

    // Compute the kernel matrix based on the kernel size and standard deviation.
    kernelMat = gaussKernel(kernelSize, std);

    // Compute the convolution with gray scale image and the Gaussian kernel.
    gaussFilteredImage = gaussConvolve(grayScaleImg, kernelMat, kernelSize);

    return gaussFilteredImage;
}

/*
 * Compute the 2D Gaussian kernel for a specific std and kernelSize. This kernel will be used
 * for the convolution with the gray scale image.
 */
pair<vector< vector <float> >, float > implementHarris::gaussKernel(int kernelSize, float std){

	int halfKernel = kernelSize/2;
	vector < vector <float> > kernelMat;
	float normalizationFactor = 0.0;

	for(int i = -halfKernel; i<=halfKernel; i++){
		kernelMat.push_back(vector<float>());
		for(int j = -halfKernel; j<=halfKernel; j++){
			kernelMat[i+halfKernel].push_back((1.0/(2.0*M_PI*std*std))*exp(-(0.5*(i*i*1.0 + j*j*1.0))/(std*std)));
			normalizationFactor += kernelMat[i+halfKernel][j+halfKernel];
		}
	}

	return make_pair(kernelMat, normalizationFactor);
}

/*
 * Convolve the image with the Gaussian kernel computed earlier.
 * The output is a Gaussian filtered image.
 */
vector< vector <float> >implementHarris::gaussConvolve(vector< vector <float> > &img, pair< vector< vector <float> >, float> &kernel, int kernelSize){

    int halfKernel = kernelSize/2;
    size_t height = img.size();
    size_t width = img[0].size();
    float gaussNormalization = kernel.second, res;

    vector< vector <float> > convolvedImage;
    vector< vector<float> > kernelMat = kernel.first;

    for(size_t imgRow=0; imgRow<height; imgRow++){
		convolvedImage.push_back(vector<float>());
		for(size_t imgCol=0; imgCol<width; imgCol++){
			res = 0.0;
			// Convolution of an image patch with the Gaussian kernel
			for(int kerRow = -halfKernel; kerRow<=halfKernel; kerRow++)
				for(int kerCol = -halfKernel; kerCol<=halfKernel; kerCol++)
					// Check boundary in the horizontal and vertical directions
					if( (imgRow + kerRow) < 0 || (imgRow + kerRow) >= height || (imgCol + kerCol) < 0 || (imgCol + kerCol) >= width )
						res += 0.0;
					else
						res += kernelMat[kerRow+halfKernel][kerCol+halfKernel] * img[imgRow+kerRow][imgCol+kerCol];

			convolvedImage[imgRow].push_back(res/gaussNormalization);
		}
	}
    return convolvedImage;
}

/*
 * Sobel operators to compute vertical and horizontal gradients of the Gaussian filtered image
 */
pair<vector< vector <float> >, vector< vector <float> > > implementHarris::computeSobelGradients( vector< vector <float> > &filteredImg){

	float Gx, Gy;
	size_t r = 0, c = 0;

	vector< vector <float> >  gradX;
	vector< vector <float> >  gradY;

    for (auto const& row : filteredImg ){

    	if(r>=filteredImg.size()-2)
    	    break;

    	gradX.push_back(vector<float>());
    	gradY.push_back(vector<float>());

    	c = 0;
        for(auto const& elem : row) {
        	if(c>=row.size()-2)
        		break;

        	// Derivative in the horizontal direction of the image
            Gx = filteredImg[r][c+2] - elem + filteredImg[r+2][c+2] - filteredImg[r+2][c]
				+ 2*(filteredImg[r+1][c+2] - filteredImg[r+1][c]);

            gradX[r].push_back(Gx);

            // Derivative in the vertical direction of the image
            Gy = elem - filteredImg[r+2][c] + filteredImg[r][c+2] - filteredImg[r+2][c+2]
				+ 2*(filteredImg[r][c+1] - filteredImg[r+2][c+1]);

            gradY[r].push_back(Gy);

            c++;
        }

        r++;
    }

    return make_pair(gradX, gradY);
}

/*
 * Roberts cross operators to compute vertical and horizontal gradients of the Gaussian filtered image.
 */
pair<vector< vector <float> >, vector< vector <float> > > implementHarris::computeRobertsGradients(vector< vector <float> > &filteredImg){

	size_t r = 0, c = 0;
	vector< vector <float> >  gradX;
	vector< vector <float> >  gradY;

	for(auto const& row : filteredImg ) {
		if(r>=filteredImg.size()-1)
		    break;

		gradX.push_back(vector<float>());
		gradY.push_back(vector<float>());

		c = 0;
		for(auto const& elem : row) {
			if(c>=row.size()-1)
			    break;

			// Derivative in the horizontal direction of the image
			gradX[r].push_back(elem - filteredImg[r+1][c+1]);

			// Derivative in the vertical direction of the image
			gradY[r].push_back(-filteredImg[r+1][c] + filteredImg[r][c+1]);

			c++;
		}
		r++;
	}

	return make_pair(gradX, gradY);
}

/*
 * Compute Harris response matrix based on the horizontal and vertical gradients.
 */
pair<vector< vector <float> >, float> implementHarris::computeResponseMat(float k, pair<vector< vector <float> >, vector< vector <float> > > &gradients, int windowSize){

	vector< vector <float> > gradX = gradients.first;
	vector< vector <float> > gradY = gradients.second;
	vector< vector <float> > responseMat;

	float maxResponse = FLT_MIN;

	float sumXX, sumYY, sumXY, det, trace;
	int halfWindow = windowSize/2;

	for(size_t row=0; row<gradX.size() ; row++) {
		responseMat.push_back(vector<float>());
		for(size_t col=0; col<gradX[0].size(); col++) {
			sumXX=0.0; sumYY=0.0; sumXY= 0.0;

			for(int winRow = -halfWindow; winRow<=halfWindow; winRow++)
				for(int winCol = -halfWindow; winCol<=halfWindow; winCol++)

					// Checking the boundary conditions of the gradient matrix in horizontal and vertical directions.
					if( (row + winRow) < 0 || (row + winRow) >= gradX.size() || (col + winCol) < 0 || (col + winCol) >= gradX[0].size() ){

						sumXX += 0.0;
						sumYY += 0.0;
						sumXY += 0.0;

					}
					else{

						sumXX += gradX[row + winRow][col+winCol] * gradX[row + winRow][col+winCol];
						sumYY += gradY[row + winRow][col+winCol] * gradY[row + winRow][col+winCol];
						sumXY += gradX[row + winRow][col+winCol] * gradY[row + winRow][col+winCol];
					}

			// Compute the determinant of the Hessian matrix.
			det = sumXX*sumYY - sumXY*sumXY;

			// Compute the trace of the Hessian matrix.
			trace = sumXX + sumYY;

			responseMat[row].push_back((det - k*trace*trace));

			if(maxResponse < responseMat[row][col])
				maxResponse = responseMat[row][col];
		}
	}

    return make_pair(responseMat, maxResponse);

}

/*
 * Get all corner coordinates from the Harris response matrix based on the local maxima
 * and the threshold response value.
 */
vector<pair<int,int> > implementHarris::findCornerPoints(pair<vector< vector <float> >, float> &responseMat, float thresHoldNMS, int nmsWindowSize, int nmsWinSep){

	// The NMS threshold value based on the threshold fraction set by the user.
	float thresholdVal = responseMat.second * thresHoldNMS;
	float localMaxResponse;
	int startRow, startCol;
	bool flagOverlapRow=false, flagOverlapCol=false;

	vector<vector <float> > responseVals = responseMat.first;
	pair<int, int> responseCoordinates;
	vector<pair<int,int> > cornerCoordinates;

    for (size_t row = 0; row < responseVals.size(); row+=nmsWinSep){

    	// If already the window is overlapped in the previous iteration
    	if(flagOverlapRow)
    	    break;

    	// Check the vertical boundary after applying NMS window separation
    	if((row + nmsWindowSize) >= responseVals.size()){

    		startRow = responseVals.size() - nmsWindowSize;
    		flagOverlapRow = true;

    	}
    	else
    		startRow = row;

    	flagOverlapCol = false;

        for(size_t col = 0; col < responseVals[0].size(); col+=nmsWinSep){

        	// If already the window is overlapped in the previous iteration
        	if(flagOverlapCol)
        		break;

        	// Check horizontal boundary after applying NMS window separation
        	if((col + nmsWindowSize) >= responseVals[0].size()){

				startCol = responseVals[0].size() - nmsWindowSize;
        		flagOverlapCol = true;
        	}
			else
				startCol = col;

        	localMaxResponse = FLT_MIN;

        	// Get the local maxima of responses and corresponding coordinates within the NMS window
        	for(int r=0; r<nmsWindowSize; r++)
        		for(int c=0; c<nmsWindowSize; c++)
        			if(localMaxResponse < responseVals[startRow+r][startCol+c]){

        				localMaxResponse = responseVals[startRow+r][startCol+c];
        				responseCoordinates = pair(startRow+r, startCol+c);

        			}

        	// Consider the local maxima if it is greater or equal to the threshold response value
            if(localMaxResponse >= thresholdVal)
            	cornerCoordinates.push_back(responseCoordinates);
        }
    }

    return cornerCoordinates;
}

/*
 * Map the coordinates of the local maximum of Harris response matrix to the provided color image.
 */
vector<pair<int,int> > implementHarris::mapCornerPointsToImage(vector< pair<int,int> > &cornerPoints, gradientTypes derType){

	vector<pair<int,int> > imageCoord;

	for(auto coordinates : cornerPoints)

		// Consider the shifting of the coordinates for different gradient types
		if(derType == gradientTypes::Sobel)
			imageCoord.push_back(pair(coordinates.first+2, coordinates.second+2));
		else
			imageCoord.push_back(pair(coordinates.first+1, coordinates.second+1));

	return imageCoord;
}

/*
* Convert the BGR color image to Gray Scale Image.
* [R,G,B] = [0.0722, 0.7152, 0.2126] coefficients are used
* during conversion. Pixel values are in the range of [0..1].
*/
vector<vector <float> > implementHarris::bgrToGrayScale(Mat& img){

    vector<vector <float> > grayScaleImg;

	for (int imgRow = 0; imgRow < img.rows; imgRow++){
		grayScaleImg.push_back(vector<float>());
		for (int imgCol = 0; imgCol < img.cols; imgCol++)
			grayScaleImg[imgRow].push_back(0.0722 * img.at<cv::Vec3b>(imgRow,imgCol)[0] + 0.7152 * img.at<cv::Vec3b>(imgRow,imgCol)[1] +
				0.2126 * img.at<cv::Vec3b>(imgRow,imgCol)[2]);
	}
    return grayScaleImg;
}
