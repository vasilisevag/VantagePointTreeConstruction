#include <cilk/cilk.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <tbb/mutex.h> //mutex library
using namespace std;

using Points = vector<double>;

const int THREADS_PER_BLOCK = 256;

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

vector<double> GetDistancesCPU(const Points& points, const vector<double>& vantagePoint){
    unsigned int dimensionsTotal = vantagePoint.size();
    unsigned int pointsTotal = points.size() / dimensionsTotal;

    vector<double> distances(pointsTotal - 1);

    for(int i = 0; i < pointsTotal - 1; i++){
        double squareDistance;squareDistance = 0;
        for(int j = i * dimensionsTotal; j < (i + 1) * dimensionsTotal; j++)
            squareDistance += pow(vantagePoint[j - (i * dimensionsTotal)] - points[j], 2);
        double distance = sqrt(squareDistance);
        distances[i] = distance;
    }

    return distances;
}

__global__ void GetDistancesKernel(double* points, double* vantagePoint, double* distances, int pointsTotal, int dimensionsTotal){
    int pointIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if(pointIndex < pointsTotal){
        double squareDistance = 0;
        int startingIndex = pointIndex * dimensionsTotal;
        for(int i = startingIndex; i < startingIndex + dimensionsTotal; i++)
            squareDistance += pow(vantagePoint[i - startingIndex] - points[i], 2);
        double distance = sqrt(squareDistance);
        distances[pointIndex] = distance;
    }
}

tbb::mutex gpuMutex;

vector<double> GetDistancesGPU(const Points& points, const vector<double>& vantagePoint){
    static unsigned int kernelsCalled = 0;

    static double* vantagePointDevicePointer; // device pointers
    static double* distancesDevicePointer; 
    static double* pointsDevicePointer;

    unsigned int dimensionsTotal = vantagePoint.size();
    unsigned int pointsTotal = points.size() / dimensionsTotal;

    vector<double> distances(pointsTotal - 1);

    const double* pointsHostPointer = points.data(); // host pointers
    const double* vantagePointHostPointer = vantagePoint.data(); 
    double* distancesHostPointer = distances.data();

    if(kernelsCalled == 0){
        kernelsCalled = 1; // when this line will be executed we will still have just one thread in the program so no race conditions can occur
        cudaMalloc(&pointsDevicePointer, ((pointsTotal - 1) * dimensionsTotal) * sizeof(double)); // allocate device space
        cudaMalloc(&distancesDevicePointer, (pointsTotal - 1) * sizeof(double));
        cudaMalloc(&vantagePointDevicePointer, dimensionsTotal * sizeof(double));
    }

    
    gpuMutex.lock(); // the next 4 instructions should be executed by only one cpu thread at a time so that no gpu memory is overwritten 

    cudaMemcpy(pointsDevicePointer, pointsHostPointer, ((pointsTotal - 1) * dimensionsTotal) * sizeof(double), cudaMemcpyHostToDevice); // copy data from host to device memory 
    cudaMemcpy(vantagePointDevicePointer, vantagePointHostPointer, dimensionsTotal * sizeof(double), cudaMemcpyHostToDevice);

    
    GetDistancesKernel<<<(pointsTotal + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(pointsDevicePointer, vantagePointDevicePointer, distancesDevicePointer, pointsTotal - 1, dimensionsTotal); // kernel call


    cudaMemcpy(distancesHostPointer, distancesDevicePointer, (pointsTotal - 1) * sizeof(double), cudaMemcpyDeviceToHost); // copy data from device to host memory

    gpuMutex.unlock();

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

void VantageTreeConstructionHelper(Node*& node, const Points& points, const vector<unsigned int> indices, unsigned int dimensionsTotal, unsigned int neighborsTotal, unsigned int neighborsCurrent = 0){
    unsigned int pointsTotal = points.size() / dimensionsTotal;
    
    if(neighborsCurrent == neighborsTotal || pointsTotal <= 1) 
        return;
    
    vector<double> vantagePoint(points.end() - dimensionsTotal, points.end());
    unsigned int vantagePointIndex = indices[pointsTotal - 1];

    vector<double> distances;
    if(pointsTotal > 1000)
        distances = GetDistancesGPU(points, vantagePoint);    
    else 
        distances = GetDistancesCPU(points, vantagePoint);
        
    double medianDistance = GetMedianDistance(distances);

    node = new Node(vantagePoint, medianDistance, vantagePointIndex);

    Points innerPoints, outerPoints;
    vector<unsigned int> innerPointsIndices, outerPointsIndices;
    for(int i = 0; i < pointsTotal - 1; i++){
        if(distances[i] < medianDistance){
            for(int j = i * dimensionsTotal; j < (i + 1) * dimensionsTotal; j++)
                innerPoints.push_back(points[j]);
            innerPointsIndices.push_back(indices[i]);
        }
        else{
            for(int j = i * dimensionsTotal; j < (i + 1) * dimensionsTotal; j++)
                outerPoints.push_back(points[j]);
            outerPointsIndices.push_back(indices[i]);
        }
    } 

    if(innerPoints.size() > 32)
        cilk_spawn VantageTreeConstructionHelper(node->inner, innerPoints, innerPointsIndices, dimensionsTotal, neighborsTotal, neighborsCurrent + 1);
    else VantageTreeConstructionHelper(node->inner, innerPoints, innerPointsIndices, dimensionsTotal, neighborsTotal, neighborsCurrent + 1);

    VantageTreeConstructionHelper(node->outer, outerPoints, outerPointsIndices, dimensionsTotal, neighborsTotal, neighborsCurrent + 1);

    cilk_sync;
}

vector<unsigned int> GetInitialIndices(const Points& points, unsigned int dimensionsTotal){
    unsigned int pointsTotal = points.size() / dimensionsTotal;
    vector<unsigned int> indices(pointsTotal);

    for(int i = 0; i < pointsTotal; i++){
        indices[i] = i;
    }

    return indices;
}

Tree VantageTreeConstruction(const Points& points, unsigned int neighborsTotal, unsigned int dimensionsTotal){
    Tree vantagePointTree;

    vector<unsigned int> indices = GetInitialIndices(points, dimensionsTotal);
    VantageTreeConstructionHelper(vantagePointTree.root, points, indices, dimensionsTotal, neighborsTotal);

    return vantagePointTree;
}

Points generateRandomPoints(unsigned int pointsTotal, unsigned int dimensionsTotal){
    Points points(pointsTotal*dimensionsTotal);
    
    for(int i = 0; i < pointsTotal*dimensionsTotal; i++)
        points[i] = rand();

    return points;
}

int main(){
    srand(time_t(time(NULL)));

    unsigned int pointsTotal = 1000000;
    unsigned int dimensionsTotal = 3;

    Points points = generateRandomPoints(pointsTotal, dimensionsTotal);

    unsigned int neighborsTotal = 7;

    chrono::high_resolution_clock::time_point start, end;

    start = chrono::high_resolution_clock::now();
    Tree vantagePointTree = VantageTreeConstruction(points, neighborsTotal, dimensionsTotal);
    end = chrono::high_resolution_clock::now();

    cout << chrono::duration_cast<chrono::duration<double>>(end - start).count() << endl;
}