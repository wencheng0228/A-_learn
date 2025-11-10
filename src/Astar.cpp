//
// Created by lihao on 19-7-9.
//

#include "Astar.h"

namespace pathplanning {

void Astar::InitAstar(Mat& _Map, AstarConfig _config) {
    Mat Mask;
    InitAstar(_Map, Mask, _config);
}

void Astar::InitAstar(Mat& _Map, Mat& Mask, AstarConfig _config) {
    // 8行2列的字符数组，代表某个点的8个邻接方向
    char neighbor8[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    Map = _Map;
    config = _config;
    // CV_8S 8位有符号整数
    neighbor = Mat(8, 2, CV_8S, neighbor8).clone();

    MapProcess(Mask);
}

void Astar::PathPlanning(Point _startPoint, Point _targetPoint, vector<Point>& path) {
    // Get variables
    startPoint = _startPoint;
    targetPoint = _targetPoint;

    // Path Planning
    Node* TailNode = FindPath();
    GetPath(TailNode, path);
}

void Astar::DrawPath(Mat& _Map, vector<Point>& path, InputArray Mask, Scalar color, int thickness, Scalar maskcolor) {
    if (path.empty()) {
        cout << "Path is empty!" << endl;
        return;
    }
    _Map.setTo(maskcolor, Mask);
    for (auto it : path) {
        rectangle(_Map, it, it, color, thickness);
    }
}

void Astar::MapProcess(Mat& Mask) {
    int width = Map.cols;
    int height = Map.rows;
    Mat _Map = Map.clone();

    // Transform RGB to gray image
    if (_Map.channels() == 3) {
        cvtColor(_Map.clone(), _Map, cv::COLOR_BGR2GRAY);
    }

    // Binarize
    // 二值化
    if (config.OccupyThresh < 0) {
        // THRESH_OTSU 自动计算最佳阈值
        threshold(_Map.clone(), _Map, 0, 255, cv::THRESH_OTSU);
    } else {
        // 标准二值化
        threshold(_Map.clone(), _Map, config.OccupyThresh, 255, cv::THRESH_BINARY);
    }

    // Inflate
    Mat src = _Map.clone();
    if (config.InflateRadius > 0) {
        // 创建一个结构元素（类似于不同形状的画笔（矩形、十字形、椭圆形），用于对障碍物做膨胀）
        Mat se = getStructuringElement(MORPH_ELLIPSE, Size(2 * config.InflateRadius, 2 * config.InflateRadius));
        /**
         * 基于创建的结构元素对原图做腐蚀。原理示例：假设有一个像素点的像素值是0(障碍物)，周边都是1(非障碍物)，
         * 然后用一个3*3的结构元素覆盖这个像素点，然后再调用erode函数做腐蚀，由于erode函数会返回区域内的最小值，
         * 所以经过腐蚀以后，这个3*3的区域内所有像素点就都被腐蚀成0了，因此障碍物区域变大了，膨胀了
         */
        erode(src, _Map, se);
    }

    // Get mask
    /**
     * 将腐蚀前和腐蚀后的两幅地图逐元素按位做异或，例如，腐蚀前的[0][2]像素的是0(二进制是00000000),腐蚀后的[0][2]像素的是255(二进制是11111111)，
     * 那么做异或以后的结果是11111111，转成十进制是255，所以这个像素点在腐蚀前后是不一样的，Mask中的[0][2]元素就被标注为255
     */
    bitwise_xor(src, _Map, Mask);

    // Initial LabelMap
    // 基于膨胀后的栅格地图构建障碍物标记地图
    LabelMap = Mat::zeros(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (_Map.at<uchar>(y, x) == 0) {
                LabelMap.at<uchar>(y, x) = obstacle;
            } else {
                LabelMap.at<uchar>(y, x) = free;
            }
        }
    }
}

Node* Astar::FindPath() {
    int width = Map.cols;
    int height = Map.rows;
    Mat _LabelMap = LabelMap.clone();

    // Add startPoint to OpenList
    Node* startPointNode = new Node(startPoint);
    OpenList.push(pair<int, Point>(startPointNode->F, startPointNode->point));
    int index = point2index(startPointNode->point);
    OpenDict[index] = startPointNode;
    _LabelMap.at<uchar>(startPoint.y, startPoint.x) = inOpenList;

    while (!OpenList.empty()) {
        // Find the node with least F value
        Point CurPoint = OpenList.top().second;
        OpenList.pop();
        int index = point2index(CurPoint);
        Node* CurNode = OpenDict[index];
        OpenDict.erase(index);

        int curX = CurPoint.x;
        int curY = CurPoint.y;
        _LabelMap.at<uchar>(curY, curX) = inCloseList;

        // Determine whether arrive the target point
        if (curX == targetPoint.x && curY == targetPoint.y) {
            return CurNode;  // Find a valid path
        }

        // Traversal the neighborhood
        for (int k = 0; k < neighbor.rows; k++) {
            int y = curY + neighbor.at<char>(k, 0);
            int x = curX + neighbor.at<char>(k, 1);
            if (x < 0 || x >= width || y < 0 || y >= height) {
                continue;
            }
            if (_LabelMap.at<uchar>(y, x) == free || _LabelMap.at<uchar>(y, x) == inOpenList) {
                // Determine whether a diagonal line can pass
                int dist1 = abs(neighbor.at<char>(k, 0)) + abs(neighbor.at<char>(k, 1));
                if (dist1 == 2 && _LabelMap.at<uchar>(y, curX) == obstacle && _LabelMap.at<uchar>(curY, x) == obstacle)
                    continue;

                // Calculate G, H, F value
                int addG, G, H, F;
                if (dist1 == 2) {
                    addG = 14;
                } else {
                    addG = 10;
                }
                G = CurNode->G + addG;
                if (config.Euclidean) {
                    int dist2 = (x - targetPoint.x) * (x - targetPoint.x) + (y - targetPoint.y) * (y - targetPoint.y);
                    H = round(10 * sqrt(dist2));
                } else {
                    H = 10 * (abs(x - targetPoint.x) + abs(y - targetPoint.y));
                }
                F = G + H;

                // Update the G, H, F value of node
                if (_LabelMap.at<uchar>(y, x) == free) {
                    Node* node = new Node();
                    node->point = Point(x, y);
                    node->parent = CurNode;
                    node->G = G;
                    node->H = H;
                    node->F = F;
                    OpenList.push(pair<int, Point>(node->F, node->point));
                    int index = point2index(node->point);
                    OpenDict[index] = node;
                    _LabelMap.at<uchar>(y, x) = inOpenList;
                } else  // _LabelMap.at<uchar>(y, x) == inOpenList
                {
                    // Find the node
                    int index = point2index(Point(x, y));
                    Node* node = OpenDict[index];
                    if (G < node->G) {
                        node->G = G;
                        node->F = F;
                        node->parent = CurNode;
                    }
                }
            }
        }
    }

    return NULL;  // Can not find a valid path
}

void Astar::GetPath(Node* TailNode, vector<Point>& path) {
    PathList.clear();
    path.clear();

    // Save path to PathList
    Node* CurNode = TailNode;
    while (CurNode != NULL) {
        PathList.push_back(CurNode);
        CurNode = CurNode->parent;
    }

    // Save path to vector<Point>
    int length = PathList.size();
    for (int i = 0; i < length; i++) {
        path.push_back(PathList.back()->point);
        PathList.pop_back();
    }

    // Release memory
    while (OpenList.size()) {
        Point CurPoint = OpenList.top().second;
        OpenList.pop();
        int index = point2index(CurPoint);
        Node* CurNode = OpenDict[index];
        delete CurNode;
    }
    OpenDict.clear();
}

}  // namespace pathplanning