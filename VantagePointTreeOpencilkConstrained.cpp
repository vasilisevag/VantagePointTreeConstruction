#include <cilk/cilk.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
using namespace std;

using Points = vector<vector<double>>;


struct Node{
    Node(const vector<double>& vantagePoint, double medianDistance, unsigned int vantagePointIndex) : vantagePoint(vantagePoint), medianDistance(medianDistance), vantagePointIndex(vantagePointIndex), inner(nullptr), outer(nullptr) {}

    vector<double> vantagePoint;
    double medianDistance;
    unsigned int vantagePointIndex;
    Node *inner, *outer;
};

struct Tree{
    Node* root;
};

vector<double> GetDistances(const Points& points, const vector<double>& vantagePoint){
    unsigned int pointsTotal = points.size();
    unsigned int dimensionsTotal = vantagePoint.size();
    
    vector<double> distances(pointsTotal - 1);

    for(int i = 0; i < pointsTotal - 1; i++){
        double squareDistance;squareDistance = 0;
        for(int j = 0; j < dimensionsTotal; j++)
            squareDistance += pow(vantagePoint[j] - points[i][j], 2);
        double distance = sqrt(squareDistance);
        distances[i] = distance;
    }

    return distances;
}

double GetMedianDistance(vector<double> distances){
    unsigned int distancesTotal = distances.size();
    if(distancesTotal % 2 == 0){
        nth_element(distances.begin(), distances.begin() + distancesTotal/2, distances.end());
        nth_element(distances.begin(), distances.begin() + (distancesTotal-1)/2, distances.end());
        return (distances[(distancesTotal-1)/2] + distances[distancesTotal/2]) / 2;
    }
    else{
        nth_element(distances.begin(), distances.begin() + distancesTotal/2, distances.end());
        return distances[distancesTotal/2];
    }
}

void VantageTreeConstructionHelper(Node*& node, const Points& points, const vector<unsigned int> indices, unsigned int neighborsTotal, unsigned int neighborsCurrent = 0){
    unsigned int pointsTotal = points.size();
    
    if(neighborsCurrent == neighborsTotal || pointsTotal <= 1) 
        return;

    vector<double> vantagePoint = points.back();
    unsigned int vantagePointIndex = indices[pointsTotal - 1];

    vector<double> distances = GetDistances(points, vantagePoint);
    double medianDistance = GetMedianDistance(distances);

    node = new Node(vantagePoint, medianDistance, vantagePointIndex);

    Points innerPoints, outerPoints;
    vector<unsigned int> innerPointsIndices, outerPointsIndices;
    for(int i = 0; i < pointsTotal - 1; i++){
        if(distances[i] < medianDistance){
            innerPoints.push_back(points[i]);
            innerPointsIndices.push_back(indices[i]);
        }
        else{
            outerPoints.push_back(points[i]);
            outerPointsIndices.push_back(indices[i]);
        }
    } 

    if(innerPoints.size() > 32)
        cilk_spawn VantageTreeConstructionHelper(node->inner, innerPoints, innerPointsIndices, neighborsTotal, neighborsCurrent + 1);
    else VantageTreeConstructionHelper(node->inner, innerPoints, innerPointsIndices, neighborsTotal, neighborsCurrent + 1);

    VantageTreeConstructionHelper(node->outer, outerPoints, outerPointsIndices, neighborsTotal, neighborsCurrent + 1);

    cilk_sync;
}

vector<unsigned int> GetInitialIndices(const Points& points){
    unsigned int pointsTotal = points.size();
    vector<unsigned int> indices(pointsTotal);

    for(int i = 0; i < pointsTotal; i++){
        indices[i] = i;
    }

    return indices;
}

Tree VantageTreeConstruction(const Points& points, unsigned int neighborsTotal){
    Tree vantagePointTree;

    vector<unsigned int> indices = GetInitialIndices(points);
    VantageTreeConstructionHelper(vantagePointTree.root, points, indices, neighborsTotal);

    return vantagePointTree;
}

Points generateRandomPoints(unsigned int pointsTotal, unsigned int dimensionsTotal){
    Points points(pointsTotal);
    vector<double> point(dimensionsTotal);
    
    for(int i = 0; i < pointsTotal; i++){
        for(double& elem : point)
            elem = rand();
        points[i] = point;
    }

    return points;
}

int main(){
    srand(time_t(time(NULL)));

    unsigned int pointsTotal = 2000000;
    unsigned int dimensionsTotal = 3;

    Points points = generateRandomPoints(pointsTotal, dimensionsTotal);

    unsigned int neighborsTotal = 8;

    chrono::high_resolution_clock::time_point start, end;

    start = chrono::high_resolution_clock::now();
    Tree vantagePointTree = VantageTreeConstruction(points, neighborsTotal);
    end = chrono::high_resolution_clock::now();

    cout << chrono::duration_cast<chrono::duration<double>>(end - start).count() << endl;
}