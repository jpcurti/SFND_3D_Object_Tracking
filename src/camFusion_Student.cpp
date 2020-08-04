
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    bool printDebugMsg = false;
    bool bWait = false;
    std::map<std::pair<int, int>, int> possibleBBMatches; 
    std::multimap<int, std::pair<int, int>>rv_possibleBBMatches;
    std::set<int> k1s, k2s;
    // for all keypoint matches...
    for(auto &it: matches)
    {   
        
        if(printDebugMsg) cout << endl << "Loop for Match with prev keypoint: " << it.queryIdx << " and current keypoint " << it.trainIdx <<endl; 

        //Check if the prev keypoint (queryIdx) is within one or more boundary box(es) in the previous Frame
        for (auto &bb: prevFrame.boundingBoxes)
        {   
           

            if( bb.roi.contains(prevFrame.keypoints[it.queryIdx].pt))
            {   
                if(printDebugMsg)
                {
                    
                    cout <<" PrevFrame: Keypoint  "  << it.queryIdx <<  " found on BB " << bb.boxID <<  endl;
                    
                    //cout << "Keypoint  with ID " << it.queryIdx << " and position x: " << prevFrame.keypoints[it.queryIdx].pt.x << ", y: " << prevFrame.keypoints[it.queryIdx].pt.y << endl;
                    //cout << "Is inside BoundaryBox with ID " << bb.boxID <<  "and position x: " << bb.roi.x  << ", y: " << bb.roi.x << ", width :" << bb.roi.width<< ", height: "<< bb.roi.height<< endl;
                }
                        //Check if the current keypoint (trainIdx) is within one or more boundary box(es) in the previous Frame
                for (auto &bb2: currFrame.boundingBoxes)
                {   
                    if( bb2.roi.contains(currFrame.keypoints[it.trainIdx].pt))
                    {   
                        if(printDebugMsg)
                        {
                        
                            cout <<"currFrame: Keypoint  "  << it.trainIdx <<  " found on BB " << bb2.boxID <<  endl;
                            //cout << "Keypoint (curr) with ID " << it.trainIdx << " and position x: " << currFrame.keypoints[it.trainIdx].pt.x << ", y: " << currFrame.keypoints[it.trainIdx].pt.y << endl;
                            //cout << "Is inside BoundaryBox with ID " << bb2.boxID <<  "and position x: " << bb2.roi.x  << ", y: " << bb2.roi.x << ", width :" << bb2.roi.width<< ", height: "<< bb2.roi.height<< endl;
                            cout <<"Possible BB match. Prev Frame BB: "  << bb.boxID <<  " Curr Frame BB: " << bb2.boxID <<  endl;
                            if(bWait) cv::waitKey(0);
                        }

                        //Check if first occurence of this pair B1-B2
                        if(possibleBBMatches.count(pair<int,int>(bb.boxID,bb2.boxID))==0)
                        {   
                            //If yes, insert {{B1,B2},0} on possibleBBmatches
                            possibleBBMatches.insert(pair<pair<int,int>,int>(make_pair(bb.boxID,bb2.boxID),1));
                        }
                        else
                        {   
                            //if already inserted in list, increment the counter of {{B1,B2},counter}
                            possibleBBMatches.find(pair<int,int>(bb.boxID,bb2.boxID))->second++;
                        }
                        
                         

                    }
                    
                }

            }
               
        }

    }
    if(printDebugMsg)
        {   
            cout << endl << endl<< "Possible matches report" << endl;
            for (auto &it : possibleBBMatches) cout << "{" << it.first.first << ", "<< it.first.second << "}" << " Counter: " << it.second << endl;

        }
    
    //
    for (auto &it : possibleBBMatches)
    {   

        rv_possibleBBMatches.insert(pair<int, pair<int,int>>(it.second,make_pair(it.first.first, it.first.second)));

    }
    if(printDebugMsg)
        {   
            cout << endl << endl<< "Possible BB matches in order" << endl;
            for (auto &it : rv_possibleBBMatches) cout << "{" << it.second.first << ", "<< it.second.second << "}" << " Counter: " << it.first << endl;

        }

    for (auto rit=rv_possibleBBMatches.rbegin(); rit!=rv_possibleBBMatches.rend(); ++rit)
    {   
        //This does not check if second BB is also not being repeated
        //bbBestMatches.insert(make_pair(rit->second.first,rit->second.second));

        //This does:
        const auto insertion_k1 = k1s.insert(rit->second.first);
        const auto insertion_k2 = k2s.insert(rit->second.second);
        if (insertion_k1.second && insertion_k2.second) {
            bbBestMatches.insert(make_pair(rit->second.first,rit->second.second));
        }
    }
    
    if(printDebugMsg)
        {   
            cout << endl << endl<< "Best matches return value" << endl;
            for (auto it:bbBestMatches) cout << "{" << it.first << ", "<< it.second << "}"  << endl;

        }
    
}
