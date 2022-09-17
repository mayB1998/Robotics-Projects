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

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;

#define MAX_CORNER 4000
typedef struct _CornerPoints
{
    int num;
    int x[MAX_CORNER];
    int y[MAX_CORNER];
}CornerPoints;

int featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, int numFrame)	{ 

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;				
    Size winSize = Size(21,21);																								
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;

    for( int i=0; i<status.size(); i++)
    {  
        Point2f pt = points2.at(i- indexCorrection);

        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
            if((pt.x<0)||(pt.y<0))	{
                status.at(i) = 0;
            }
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }

    }
    cout << numFrame << " indexCorrection: " << indexCorrection << " status.size(): " << status.size() << endl;
    return indexCorrection;
}

double getGraybuf(Mat img, int y, int x)
{
    unsigned int b = img.at<Vec3b>(y, x)[0];
    unsigned int g = img.at<Vec3b>(y, x)[1];
    unsigned int r = img.at<Vec3b>(y, x)[2];
    return (r + g + b) / 3.0;
}

int featureDetection(Mat img, vector<Point2f>& points)	{   //uses FAST as of now, modify parameters as necessary
    vector<KeyPoint> keypoints;
    //int fast_threshold = 20;
    //bool nonmaxSuppression = true;
	
	// FAST feature detection ===> ORB feature detection
    // Ptr<Feature2D> feature = ORB::create();

    // feature->detect(img, keypoints);

    // Mat desc;
    // feature->compute(img, keypoints, desc);

    // cout << keypoints.size() << desc.size() << endl;

    // FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
	//cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures=2000, fast_threshold=fast_threshold);

    // Mat harris;
    // cornerHarris(img_1, keypoints_1, 3, 3, 0.04);
    
    // Mat harris_norm;
    // normalize(harris, harris_norm, 0, 255, NORM_MINMAX, CV_8U);

    register int i, j, x, y;
    double threshold = 20000.0;

    int w = img.cols;
    int h = img.rows;

    //cout << "width:" << w << "height:" << h << endl;

    // E[^x, ^y] = [^x ^y] * M * [^x / ^y]
    // R = det(M)-k*tr(M)^2

    // 1. Calculate lx*lx, ly*ly, lx*ly for M
    double** dxdx = new double*[h];
    double** dydy = new double*[h];
    double** dxdy = new double*[h];

    for(i = 0; i < h; i++)
    {
        dxdx[i] = new double[w];
        dydy[i] = new double[w];
        dxdy[i] = new double[w];
        memset(dxdx[i], 0, sizeof(int)*w);
        memset(dydy[i], 0, sizeof(int)*w);
        memset(dxdy[i], 0, sizeof(int)*w);
    }

    double tx, ty;
    for(j = 1; j < h-1; j++)
    {
        for(i = 1; i < w-1; i++)
        {
            tx = (getGraybuf(img, j-1, i+1) + getGraybuf(img, j, i+1) + getGraybuf(img, j+1, i+1) 
             - getGraybuf(img, j-1, i-1) - getGraybuf(img, j, i-1) - getGraybuf(img, j+1, i-1)) / 6.0;
            ty = (getGraybuf(img, j+1, i-1) + getGraybuf(img, j+1, i) + getGraybuf(img, j+1, i+1) 
             - getGraybuf(img, j-1, i-1) - getGraybuf(img, j-1, i) - getGraybuf(img, j-1, i+1)) / 6.0;

            dxdx[j][i] = tx * tx;
            dydy[j][i] = ty * ty;
            dxdy[j][i] = tx * ty;
            //cout << i << j << getGraybuf(img, j, i) << endl;
        }
    }
    
    // 2. Gausian filtering for effective corner detection
    double** gdx = new double*[h];
    double** gdy = new double*[h];
    double** gdxy = new double*[h];

    for(i = 0; i < h; i++)
    {
        gdx[i] = new double[w];
        gdy[i] = new double[w];
        gdxy[i] = new double[w];
        memset(gdx[i], 0, sizeof(int)*w);
        memset(gdy[i], 0, sizeof(int)*w);
        memset(gdxy[i], 0, sizeof(int)*w);
    }

    double g[5][5] = {{1, 4, 6, 4, 1}, 
                      {4, 16, 24, 16, 4},
                      {6, 24, 36, 24, 6},
                      {4, 16, 24, 16, 4},
                      {1, 4, 6, 4, 1}};

    for (y = 0; y < 5; y++)
    for (x = 0; x < 5; x++)
    {
        g[y][x] /= 256.;
    }

    double tx2, ty2, txy;
    for (j = 2; j < h - 2; j++)
    for (i = 2; i < w - 2; i++)
    {
        tx2 = ty2 = txy = 0;
        for (y = 0; y < 5; y++)
        for (x = 0; x < 5; x++)
        {
            if (j > h * 1 / 3 && j < h * 2 / 3)
            {
                tx2 += (dxdx[j + y - 2][i + x - 2]);
                ty2 += (dydy[j + y - 2][i + x - 2]);
                txy += (dxdy[j + y - 2][i + x - 2]);
            }
            else{
                tx2 += (dxdx[j + y - 2][i + x - 2] * g[y][x]);
                ty2 += (dydy[j + y - 2][i + x - 2] * g[y][x]);
                txy += (dxdy[j + y - 2][i + x - 2] * g[y][x]);
            }
        }

        gdx[j][i] = tx2;
        gdy[j][i] = ty2;
        gdxy[j][i] = txy;
    }
    // cout << "=================22222222222222222=====================" << endl;
    // 3. Corner response fuction

    double** crf = new double*[h];
    for(i = 0; i < h; i++)
    {
        crf[i] = new double[w];
        memset(crf[i], 0, sizeof(double)*w);
    }
    double k = 0.04; // const k value 0.04~0.06

    for (j = 2; j < h - 2; j++)
    for (i = 2; i < w - 2; i++)
    {
        crf[j][i] = (gdx[j][i] * gdy[j][i] - gdxy[j][i] * gdxy[j][i])
            - k*(gdx[j][i] + gdy[j][i])*(gdx[j][i] + gdy[j][i]);
    }
    // cout << "=================33333333=====================" << endl;

    // 4. Set corner point that is larger than the threshold
    CornerPoints cp;
    cp.num = 0;

    for (j = 2; j < h - 2; j++)
    for (i = 2; i < w - 2; i++)
    {
        double cvf_value = crf[j][i];
        
        if (cvf_value > threshold)
        {
            //cout << cvf_value << endl;
            if (cvf_value > crf[j - 1][i] && cvf_value > crf[j - 1][i + 1] &&
                cvf_value > crf[j][i + 1] && cvf_value > crf[j + 1][i + 1] &&
                cvf_value > crf[j + 1][i] && cvf_value > crf[j + 1][i - 1] &&
                cvf_value > crf[j][i - 1] && cvf_value > crf[j - 1][i - 1])
            {
                if(cp.num < MAX_CORNER)
                {
                    //points[cp.num] = Point2f(i, j);
                    cp.x[cp.num] = i;
                    cp.y[cp.num] = j;
                    cp.num++;
                }
                //corners.push_back(IppPoint(i, j));
            }
        }
    }
    
    //memset(points->x, 0, sizeof(Point2f)*cp.num);

    // KeyPoint::convert(keypoints, points, vector<int>());
    //cout << points << endl;

    for(i = 0; i < cp.num ; i++)
    {
        points.push_back(Point2f(cp.y[i], cp.x[i]));
    }

    for(i = 0; i < h; i++)
    {
        delete [] dxdx[i];
        delete [] dydy[i];
        delete [] dxdy[i];
        delete [] gdx[i];
        delete [] gdy[i];
        delete [] gdxy[i];
        delete [] crf[i];
    }
    delete [] dxdx;
    delete [] dydy;
    delete [] dxdy;
    delete [] gdx;
    delete [] gdy;
    delete [] gdxy;
    delete [] crf;

    return cp.num;
}
