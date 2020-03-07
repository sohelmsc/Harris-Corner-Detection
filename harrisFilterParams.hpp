/*
 * This class holds different parameters required for Harris corner detector
 * Algorithm.
 *
 * Author: Mafijul, 28 Feb 2020.
 */

enum class gradientTypes { Sobel, Roberts }; // Different gradient types.

class harrisFilterParams{

public:

	float k = 0.04;               // Specifies the empirical constant for computing Harris response (0.04 <= k <=0.06 )

	int gradSumWindowSize = 3;    // Size of the window for summing grad product. Window size should be an odd number.

	int gaussKernelSize = 3;      // Size of the window for Gaussian filtering, an odd number is preferred.

	float thresHoldNMS = 0.02;    // Fraction for computing the threshold for non-maximum suppression.
								  // Th = thresHoldNMS * max(Harris Response)

	int nmsWindowSize = 3;        // Non-maximum suppression window size. Generally compute local maximum within this window.

	int nmsWindowSeparation = 1;  // Non-maximum suppression window separation. Difference between two NMS window.

	float gaussSigma = 1.0;       // Standard deviation of the Gaussian filter.

};
