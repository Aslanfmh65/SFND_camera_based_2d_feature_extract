#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        descRef.convertTo(descRef,CV_32F);
        descSource.convertTo(descSource,CV_32F);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matcher;
        matcher->knnMatch(descSource, descRef, knn_matcher, 2);

        double minDistRatio = 0.8;

        for (int i=0; i<knn_matcher.size(); i++){
            if (knn_matcher[i][0].distance / knn_matcher[i][1].distance < minDistRatio)
            {
                matches.push_back(knn_matcher[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {   
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        cout << "No Method found" << endl;
    }
    

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time = t*1000/1.0;
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    // cout << descriptorType << "descriptor extrac " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    time = t*1000/1.0;
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time, bool bVis){
    int blockSize = 6;
    int apertureSize = 3;
    int minResponse = 50;
    double k = 0.04;
    double maxOverlap = 0.0;
    int response = 0;
    bool bOverlap;

    cv::KeyPoint newKeyPoint;
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    double t = (double)cv::getTickCount();

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (size_t j = 0; j < dst_norm.rows; j++){
        for (size_t i =0; i < dst_norm.cols; i++){
            response = (int)dst_norm.at<float>(j,i);

            if (response > minResponse){
                newKeyPoint.pt = cv::Point2f(i,j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bOverlap = false;

                for (auto it = keypoints.begin(); it != keypoints.end(); ++it){
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap){
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response){
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap){
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }

    t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
    time = t * 1000 / 1.0;
    // cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    if (bVis){
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorName, double &time, bool bVis)
{
    double t = 0;
    if (detectorName.compare("FAST")==0){
        t = (double)cv::getTickCount();
        cv::FAST(img, keypoints, true);
        t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (detectorName.compare("ORB")==0){
        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> orbDetector = cv::ORB::create();
        orbDetector->detect(img,keypoints);
        t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout << "ORB with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;        
    }
    else if (detectorName.compare("BRISK")==0){
        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> briskDetector = cv::BRISK::create();
        briskDetector->detect(img,keypoints);
        t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout << "BRISK with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;        
    }
    else if (detectorName.compare("AKAZE")==0){
        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> akazeDetector = cv::AKAZE::create();
        akazeDetector->detect(img,keypoints);
        t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout << "AKAZE with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;        
    }
    else if (detectorName.compare("SIFT")==0){
        t = (double)cv::getTickCount();
        cv::Ptr<cv::FeatureDetector> siftDetector = cv::xfeatures2d::SIFT::create();
        siftDetector->detect(img,keypoints);
        t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
        cout << "SIFT with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;        
    }
    else{
        cout << "No method found" << endl;
    }
    time = t*1000/1.0;
}