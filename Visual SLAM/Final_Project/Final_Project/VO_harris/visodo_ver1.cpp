/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include <filesystem>
#include <string>
#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 9000
#define MIN_NUM_FEAT 2000

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

/*
double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)
{
	string line;
	int i = 0;
	ifstream myfile("/home/ukyeon/Desktop/KITTI_dataSet/00/times.txt");
	double x = 0, y = 0, z = 0;
	double x_prev, y_prev, z_prev;
	if (myfile.is_open())
	{
		while ((getline(myfile, line)) && (i <= frame_id))
		{
			z_prev = z;
			x_prev = x;
			y_prev = y;
			std::istringstream in(line);
			// cout << line << '\n';
			for (int j = 0; j < 12; j++)
			{
				in >> z;
				if (j == 7)
					y = z;
				if (j == 3)
					x = z;
			}
			i++;
		}
		myfile.close();
	}
	else
	{
		cout << "Unable to open file";
		return 0;
	}
	return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));
}
*/

int main(int argc, char **argv)
{
	Mat img_1, img_2;
	Mat R_f, t_f; // the final rotation and tranlation vectors containing the

	ofstream myfile;
	myfile.open("results1_1.txt");

	double scale = 1.0; //0.2;
	char filename1[200];
	char filename2[200];
	sprintf(filename1, "/home/ukyeon/Desktop/test_NU/frame%04d_s.jpg", 0);
	sprintf(filename2, "/home/ukyeon/Desktop/test_NU/frame%04d_s.jpg", 1);

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	int rotate = 0;
	int redetection = 0;
	cv::Point textOrg(10, 50);

	// read the first two frames from the dataset
	Mat img_1_c = imread(filename1);
	Mat img_2_c = imread(filename2);

	// Mat img_1_c = img_1_ori(Range(0, 500), Range(0, 1224));
	// Mat img_2_c = img_2_ori(Range(0, 500), Range(0, 1224));

	if (!img_1_c.data || !img_2_c.data)
	{
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}

	// we work with grayscale images
	cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
	cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

	// feature detection, tracking
	vector<Point2f> points1, points2; // vectors to store the coordinates of the feature points
	int cornerCnt = featureDetection(img_1, points1); // detect features in img_1
	cout << "detect features in img_1: " << cornerCnt << ", " << points1.size() << endl;

	vector<uchar> status;
	featureTracking(img_1, img_2, points1, points2, status, 1); // track those features to img_2

	// TODO: add a fucntion to load these values directly from KITTI's calib files
	//  WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
	double focal = 1888.4451 / 2.5; //1888.4000;
	cv::Point2d pp(613.1897 / 2.5, 482.1189 / 2.5);//pp(607.1928, 185.2157);
	// vector<Point2f> cameraMatrix = {{1888.44515582, 0, 613.1897},
	// 								{0, 1888.4451, 482.1189},
	// 								{0, 0, 1}};

// 	"Calibration results 
// ====================
// Camera-system parameters:
// 	cam0 (/selected/cam0/image_raw):
// 	 type: <class 'aslam_cv.libaslam_cv_python.EquidistantDistortedPinholeCameraGeometry'>
// 	 distortion: [-0.03116674  0.50057031 -7.69105705 41.71286545] +- [0.01138717 0.23891735 1.38386047 0.32767867]
// 	 projection: [1888.44515582 1888.40009491  613.18976514  482.11894092] +- [1.2474561  1.27109206 0.04597261 0.17133183]
// 	 reprojection error: [0.000033, 0.000066] +- [0.238872, 0.224465]
// 	 "
	 
	// recovering the pose and the essential matrix
	Mat E, R, t, mask;
	//E = findEssentialMat(points2, points1, cameraMatrix, RANSAC, 0.999, 1.0, mask);

	E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points2, points1, R, t, focal, pp, mask);

	Mat prevImage = img_2;
	Mat currImage;
	vector<Point2f> prevFeatures = points2;
	vector<Point2f> currFeatures;

	char filename[100];

	R_f = R.clone();
	t_f = t.clone();

	clock_t begin = clock();

	namedWindow("Road facing camera", WINDOW_AUTOSIZE); // Create a window for display.
	namedWindow("Trajectory", WINDOW_AUTOSIZE);			// Create a window for display.

	Mat traj = Mat::zeros(900, 900, CV_8UC3);

	for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++)
	{
		sprintf(filename, "/home/ukyeon/Desktop/test_NU/frame%04d_s.jpg", numFrame);
		//cout << numFrame << endl;
		Mat currImage_c = imread(filename);
		//Mat currImage_c = img_2_ori(Range(0, 500), Range(0, 1224));
		cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
		vector<uchar> status;

		//cornerCnt = featureDetection(prevImage, prevFeatures);
		int correctionCnt = featureTracking(prevImage, currImage, prevFeatures, currFeatures, status, numFrame);

		cout << numFrame << "-> cornerCnt: "<< cornerCnt << ", correctionCnt: " << correctionCnt << ", prevFeatures: " << prevFeatures.size() << endl;

		E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
		recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

		Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

		for (int i = 0; i < prevFeatures.size(); i++)
		{ // this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
			prevPts.at<double>(0, i) = prevFeatures.at(i).x;
			prevPts.at<double>(1, i) = prevFeatures.at(i).y;

			currPts.at<double>(0, i) = currFeatures.at(i).x;
			currPts.at<double>(1, i) = currFeatures.at(i).y;
		}

		//scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
		//cout << "Scale is " << scale << endl;
		myfile << numFrame << ":(t) " << t.at<double>(0) << ", " << t.at<double>(1) << ", " << t.at<double>(2) << endl;
		myfile << "R: " << R << endl;

		if ((t.at<double>(2) < t.at<double>(0)) && (t.at<double>(1) < t.at<double>(0)) && (t.at<double>(0) > 0.9))
		{
			rotate++;
			if(rotate > 3)
			{
				cout << "******* R otating Update t_f *************" << rotate << endl;
				t.at<double>(0) /= 1.8;
				t.at<double>(2) *= 2.1;
				t_f = t_f + scale * (R_f * t);
				R_f = R * R_f;
			}
		}
		else if (10 <= correctionCnt && t.at<double>(2) > 0 ) //(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
		{
			if (redetection > 0 )
			{
				cout << "!!!!!!! No Update (Stable)!! " << redetection << endl;
				redetection = 0;
			}
			else{
				rotate = 0;
				cout << "====== OK Update t_f ======" << t.at<double>(2) << endl;
				t_f = t_f + scale * (R_f * t);
				R_f = R * R_f;
			}
		}
		else{
			rotate = 0;
			cout << "!!!!!!! No Update !!!!!!!!" << endl;
		}

		myfile << "R_f: " << R_f << endl;
		myfile << "t_f: " << t_f << endl;

		// lines for printing results
		myfile << numFrame << " end update" << endl;

		// a redetection is triggered in case the number of feautres being trakced go below a particular threshold
		if (prevFeatures.size() < MIN_NUM_FEAT)
		{
			redetection = 1;
			cout << numFrame  << ": " << "Number of tracked features reduced to " << prevFeatures.size() << endl;
			
			cornerCnt = featureDetection(prevImage, prevFeatures);
			cout << numFrame  << ": " << "redetection! " << cornerCnt << ", " << prevFeatures.size() << ", " << status.size() << endl;

			featureTracking(prevImage, currImage, prevFeatures, currFeatures, status, numFrame);
			
			cout << numFrame  << ": " << "featureTracking result! " << cornerCnt << ", " << prevFeatures.size() << ", " << status.size() << endl;
		}

		prevImage = currImage.clone();
		prevFeatures = currFeatures;

		int x = int(t_f.at<double>(0)) + 400;
		int y = int(t_f.at<double>(2)) + 100;  // WHY 2?????????????????
		circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

		imshow("Road facing camera", currImage_c);
		imshow("Trajectory", traj);

		waitKey(1); // 10ms
	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;

	myfile.close();

	return 0;
}
