//
// Created by lihao on 19-7-9.
//

#ifndef ASTAR_H
#define ASTAR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <unordered_map>

using namespace std;
using namespace cv;

namespace pathplanning {

enum NodeType { obstacle = 0, free, inOpenList, inCloseList };

struct Node {
    Point point;   // node coordinate
    int F, G, H;   // cost
    Node* parent;  // parent node

    Node(Point _point = Point(0, 0)) : point(_point), F(0), G(0), H(0), parent(NULL) {}
};

// 这是函数对象，重载了()。
// 目的是让对象表现的像函数，例如，vector<cmp>这样是允许的，然后调用()就可以调用cmp中重载的()，但是如果把cmp定义为普通的函数的话，那么vector<cmp>就是不允许的了
struct cmp {
    bool operator()(pair<int, Point> a, pair<int, Point> b)  // Comparison function for priority queue
    {
        return a.first > b.first;  // min heap
    }
};

struct AstarConfig {
    bool Euclidean;     // true/false
    int OccupyThresh;   // 0~255
    int InflateRadius;  // integer

    AstarConfig(bool _Euclidean = true, int _OccupyThresh = -1, int _InflateRadius = -1)
        : Euclidean(_Euclidean), OccupyThresh(_OccupyThresh), InflateRadius(_InflateRadius) {}
};

class Astar {
public:
    // Interface function
    void InitAstar(Mat& _Map, AstarConfig _config = AstarConfig());
    void InitAstar(Mat& _Map, Mat& Mask, AstarConfig _config = AstarConfig());
    void PathPlanning(Point _startPoint, Point _targetPoint, vector<Point>& path);
    void DrawPath(Mat& _Map,
                  vector<Point>& path,
                  InputArray Mask = noArray(),
                  Scalar color = Scalar(0, 0, 255),
                  int thickness = 1,
                  Scalar maskcolor = Scalar(255, 255, 255));

    // 2D坐标 → 1D索引，是一个映射函数
    inline int point2index(Point point) { return point.y * Map.cols + point.x; }
    // 1D索引 → 2D坐标，是一个反映射函数
    inline Point index2point(int index) { return Point(int(index / Map.cols), index % Map.cols); }

private:
    void MapProcess(Mat& Mask);
    Node* FindPath();
    void GetPath(Node* TailNode, vector<Point>& path);

private:
    // Object
    Mat Map;
    Point startPoint, targetPoint;  // 栅格坐标系下的点
    Mat neighbor;                   // 8行2列的矩阵，代表某个点的8个邻接方向

    Mat LabelMap;  // 障碍物标记地图
    AstarConfig config;

    // 开放列表中存放的是已经发现但是还没有去探索的那些节点
    priority_queue<pair<int, Point>, vector<pair<int, Point>>, cmp>
        OpenList;                        // open list，优先队列，队列中的元素是pair<int,
                                         // Point>,cmp是优先级排列的自定义函数，vector不用管，这是优先队列中的容器
    unordered_map<int, Node*> OpenDict;  // open dict,这里边存放了完整的node,最重要的是有父节点信息
    vector<Node*> PathList;              // path list
};

}  // namespace pathplanning

#endif  // ASTAR_H