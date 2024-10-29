
#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <optional>
#include <random>
#include <cfloat>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>




//Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas.h"
#include "hipsolver.h"
#include "hipblas-export.h"

#include <roctx.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/hip/vector.h>
#include <thrust/partition.h>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


#include <gmsh.h>

#include <stack>

// NOTA: The goal is to create a BVH algorithm using the features of hip without using rocThrust and see the performance depending on the case.


#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error in %s at line %d: %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


struct Vec3 {
	float x, y, z;
	__host__ __device__ Vec3() : x(0), y(0), z(0) {}
	__host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	__host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
	__host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
	__host__ __device__ Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }

	__host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

	__host__ __device__ Vec3 operator/(const Vec3& other) const { return Vec3(x / other.x, y / other.y, z / other.z); }
	__host__ __device__ Vec3 operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }

	__host__ __device__ float& operator[](int i) { return (&x)[i]; }
	__host__ __device__ const float& operator[](int i) const { return (&x)[i]; }
};


__host__ __device__ Vec3 min(const Vec3& a, const Vec3& b) {
	return Vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ Vec3 max(const Vec3& a, const Vec3& b) {
	return Vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}


__host__ __device__ Vec3 cross(const Vec3& a, const Vec3& b) {
	return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ float dot(const Vec3& a, const Vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}


__device__ void atomicMinVec3(Vec3* addr, Vec3 val) {
	atomicMin((int*)&addr->x, __float_as_int(val.x));
	atomicMin((int*)&addr->y, __float_as_int(val.y));
	atomicMin((int*)&addr->z, __float_as_int(val.z));
}

__device__ void atomicMaxVec3(Vec3* addr, Vec3 val) {
	atomicMax((int*)&addr->x, __float_as_int(val.x));
	atomicMax((int*)&addr->y, __float_as_int(val.y));
	atomicMax((int*)&addr->z, __float_as_int(val.z));
}


struct Ray {
	Vec3 origin, direction;
};

struct Triangle {
	Vec3 v0, v1, v2;
	int id;
};

struct AABB {
	Vec3 min, max;
};

struct BVHNode {
	AABB bounds;
	int leftChild;
	int rightChild;
	int firstTriangleIndex;
	int triangleCount;

	int firstPrimitive;
	int primitiveCount;

	// addendum
	int triangleIndex;
};


struct Intersection {
	bool hit;
	float t;
	int triangleIndex;
};



void writeBVHNodes(const std::vector<BVHNode>& hostNodes)
{
	std::cout << "BVH Nodes:" << std::endl;
	for (size_t i = 0; i < hostNodes.size(); ++i) {
		const BVHNode& node = hostNodes[i];
		std::cout << "Node " << i << ":" << std::endl;
		std::cout << "  left : " << node.leftChild << "\n";
		std::cout << "  right: " << node.rightChild << "\n";
		std::cout << "  firstTriangleIndex : " << node.firstTriangleIndex << "\n";
		std::cout << "  triangleCount : " << node.triangleCount << "\n";
		std::cout << "  Bounds:\n";
		std::cout << "    Min: ";
		std::cout << "(" << node.bounds.min.x << ", " << node.bounds.min.y << ", " << node.bounds.min.z << ")" << "\n";
		std::cout << "    Max: ";
		std::cout << "(" << node.bounds.max.x << ", " << node.bounds.max.y << ", " << node.bounds.max.z << ")" << "\n";
		std::cout << std::endl;
	}
}




//*******************************************************************************************************************************
// BEGIN::RAY TRACING

__device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& tri, float& t, Vec3& intersectionPoint) {
	Vec3 edge1 = tri.v1 - tri.v0;
	Vec3 edge2 = tri.v2 - tri.v0;
	Vec3 h = cross(ray.direction, edge2);
	float a = dot(edge1, h);

	if (a > -1e-6f && a < 1e-6f) return false;

	float f = 1.0f / a;
	Vec3 s = ray.origin - tri.v0;
	float u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) return false;

	Vec3 q = cross(s, edge1);
	float v = f * dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f) return false;

	t = f * dot(edge2, q);

	if (t > 1e-6) {
		intersectionPoint.x = ray.origin.x + t * ray.direction.x;
		intersectionPoint.y = ray.origin.y + t * ray.direction.y;
		intersectionPoint.z = ray.origin.z + t * ray.direction.z;
	}
	else
	{
		intersectionPoint.x = INFINITY;
		intersectionPoint.y = INFINITY;
		intersectionPoint.z = INFINITY;
	}
	return (t > 1e-6f);
}

__device__ bool rayAABBIntersect(const Ray& ray, const AABB& aabb) {
	Vec3 invDir = Vec3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
	Vec3 tMin = (aabb.min - ray.origin) * invDir;
	Vec3 tMax = (aabb.max - ray.origin) * invDir;
	Vec3 t1 = Vec3(fminf(tMin.x, tMax.x), fminf(tMin.y, tMax.y), fminf(tMin.z, tMax.z));
	Vec3 t2 = Vec3(fmaxf(tMin.x, tMax.x), fmaxf(tMin.y, tMax.y), fmaxf(tMin.z, tMax.z));
	float tNear = fmaxf(fmaxf(t1.x, t1.y), t1.z);
	float tFar = fminf(fminf(t2.x, t2.y), t2.z);
	return tNear <= tFar;
}


__global__ void raytraceKernel(
	Ray* rays,
	int numRays,
	BVHNode* bvhNodes,
	Triangle* triangles,
	int* hitTriangles,
	float* distance,
	Vec3* intersectionPoint,
	int* hitId
)

{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numRays) return;

	Ray ray = rays[idx];
	int stack[64];
	int stackPtr = 0;
	stack[stackPtr++] = 0;
	//stack[stackPtr++] = 13;

	float closestT = INFINITY;
	int closestTriangle = -1;
	int closesIntersectionId = -1;

	Vec3 intersectionPointT;
	intersectionPointT.x = INFINITY;
	intersectionPointT.y = INFINITY;
	intersectionPointT.z = INFINITY;
	Vec3 closestIntersectionPoint;
	closestIntersectionPoint.x = INFINITY;
	closestIntersectionPoint.y = INFINITY;
	closestIntersectionPoint.z = INFINITY;

	bool isView = false; //isView = true;

	while (stackPtr > 0) {
		int nodeIdx = stack[--stackPtr];
		BVHNode& node = bvhNodes[nodeIdx];

		//printf("node[%i]\n",nodeIdx);

		//if (nodeIdx>0) printf("node[%i] %i %i\n",nodeIdx,node.triangleCount,node.firstTriangleIndex);

		if (!rayAABBIntersect(ray, node.bounds)) continue;

		if (node.triangleCount > 0) {
			for (int i = 0; i < node.triangleCount; ++i) {
				Triangle& tri = triangles[node.firstTriangleIndex + i];
				float t;
				if (rayTriangleIntersect(ray, tri, t, intersectionPointT)) {

					if (isView) printf("      Num Ray[%i] <%f %f %f>\n", idx, intersectionPointT.x, intersectionPointT.y, intersectionPointT.z);
					if (t < closestT) {
						closestT = t;
						closestTriangle = node.firstTriangleIndex + i;
						closestIntersectionPoint = intersectionPointT;
						closesIntersectionId = triangles[closestTriangle].id;
					}
				}
			}
		}
		else {
			stack[stackPtr++] = node.leftChild;
			stack[stackPtr++] = node.rightChild;
		}
	}

	//if (closesIntersectionId>0) printf("      Num Ray[%i] dist=%f <%f %f %f>\n", idx, closestT,intersectionPointT.x, intersectionPointT.y, intersectionPointT.z);

	if (closesIntersectionId > 0) printf("      Num Ray[%i] dist=%f\n", idx, closestT);

	hitTriangles[idx] = closestTriangle;
	distance[idx] = closestT;
	intersectionPoint[idx] = closestIntersectionPoint;
	hitId[idx] = closesIntersectionId;
}

// END::RAY TRACING
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************



//*******************************************************************************************************************************
// BEGIN::BVH CPU

void buildBVHRecursive(std::vector<Triangle>& triangles, std::vector<BVHNode>& bvhNodes, int start, int end, int depth) {
	BVHNode node;
	node.firstTriangleIndex = start;
	node.triangleCount = end - start;
	node.leftChild = node.rightChild = -1;

	// Calculer les limites du nœud
	node.bounds.min = node.bounds.max = triangles[start].v0;
	for (int i = start; i < end; i++) {
		const auto& tri = triangles[i];
		node.bounds.min = min(node.bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
		node.bounds.max = max(node.bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
	}

	// Si le nœud contient peu de triangles ou si nous sommes trop profonds, arrêter la division
	if (node.triangleCount <= 4 || depth > 20) {
		bvhNodes.push_back(node);
		return;
	}

	// Trouver l'axe le plus long pour diviser
	Vec3 extent = node.bounds.max - node.bounds.min;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;

	// Trier les triangles selon l'axe choisi
	int mid = (start + end) / 2;
	std::nth_element(triangles.begin() + start, triangles.begin() + mid, triangles.begin() + end,
		[axis](const Triangle& a, const Triangle& b) {
			return (a.v0[axis] + a.v1[axis] + a.v2[axis]) < (b.v0[axis] + b.v1[axis] + b.v2[axis]);
		});

	// Créer les enfants
	int currentIndex = bvhNodes.size();
	bvhNodes.push_back(node);

	buildBVHRecursive(triangles, bvhNodes, start, mid, depth + 1);
	bvhNodes[currentIndex].leftChild = bvhNodes.size() - 1;

	buildBVHRecursive(triangles, bvhNodes, mid, end, depth + 1);
	bvhNodes[currentIndex].rightChild = bvhNodes.size() - 1;
}

void buildBVH_CPU_Recursive(std::vector<Triangle>& triangles, std::vector<BVHNode>& bvhNodes) {
	bvhNodes.clear();
	buildBVHRecursive(triangles, bvhNodes, 0, triangles.size(), 0);
}

//...

void buildBVH_CPU_Iterative(
	std::vector<Triangle>& triangles,
	std::vector<BVHNode>& bvhNodes
)
{
	bvhNodes.clear();

	struct StackEntry {
		int start, end, depth;
		int parentIndex;
		bool isLeftChild;
	};

	std::stack<StackEntry> stack;
	stack.push({ 0, static_cast<int>(triangles.size()), 0, -1, false });

	while (!stack.empty()) {
		auto [start, end, depth, parentIndex, isLeftChild] = stack.top();
		stack.pop();

		BVHNode node;
		node.firstTriangleIndex = start;
		node.triangleCount = end - start;
		node.leftChild = node.rightChild = -1;

		// Calculer les limites du nœud
		node.bounds.min = node.bounds.max = triangles[start].v0;
		for (int i = start; i < end; i++) {
			const auto& tri = triangles[i];
			node.bounds.min = min(node.bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
			node.bounds.max = max(node.bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
		}

		int currentIndex = bvhNodes.size();
		bvhNodes.push_back(node);

		if (parentIndex != -1) {
			if (isLeftChild) {
				bvhNodes[parentIndex].leftChild = currentIndex;
			}
			else {
				bvhNodes[parentIndex].rightChild = currentIndex;
			}
		}

		// Si le nœud contient peu de triangles ou si nous sommes trop profonds, passer au suivant
		if (node.triangleCount <= 4 || depth > 20) {
			continue;
		}

		// Trouver l'axe le plus long pour diviser
		Vec3 extent = node.bounds.max - node.bounds.min;
		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > extent[axis]) axis = 2;

		// Trier les triangles selon l'axe choisi
		int mid = (start + end) / 2;
		std::nth_element(triangles.begin() + start, triangles.begin() + mid, triangles.begin() + end,
			[axis](const Triangle& a, const Triangle& b) {
				return (a.v0[axis] + a.v1[axis] + a.v2[axis]) < (b.v0[axis] + b.v1[axis] + b.v2[axis]);
			});

		// Ajouter les enfants à la pile
		stack.push({ mid, end, depth + 1, currentIndex, false });
		stack.push({ start, mid, depth + 1, currentIndex, true });
	}
}

//...

void buildBVH_CPU_Iterative_Memory_Unified(
	Triangle* triangles,
	int numTriangles,
	BVHNode* bvhNodes,
	int& numNodes)
{
	numNodes = 0;

	struct StackEntry {
		int start, end, depth;
		int parentIndex;
		bool isLeftChild;
	};

	std::vector<StackEntry> stack;
	stack.push_back({ 0, numTriangles, 0, -1, false });

	while (!stack.empty()) {
		auto entry = stack.back();
		stack.pop_back();

		int start = entry.start;
		int end = entry.end;
		int depth = entry.depth;
		int parentIndex = entry.parentIndex;
		bool isLeftChild = entry.isLeftChild;

		BVHNode node;
		node.firstTriangleIndex = start;
		node.triangleCount = end - start;
		node.leftChild = node.rightChild = -1;


		node.bounds.min = node.bounds.max = triangles[start].v0;
		for (int i = start; i < end; i++) {
			const auto& tri = triangles[i];
			node.bounds.min = min(node.bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
			node.bounds.max = max(node.bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
		}

		int currentIndex = numNodes;
		bvhNodes[numNodes++] = node;

		if (parentIndex != -1) {
			if (isLeftChild) {
				bvhNodes[parentIndex].leftChild = currentIndex;
			}
			else {
				bvhNodes[parentIndex].rightChild = currentIndex;
			}
		}


		if (node.triangleCount <= 4 || depth > 20) {
			continue;
		}


		Vec3 extent = node.bounds.max - node.bounds.min;
		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > extent[axis]) axis = 2;


		int mid = (start + end) / 2;
		std::nth_element(triangles + start, triangles + mid, triangles + end,
			[axis](const Triangle& a, const Triangle& b) {
				return (a.v0[axis] + a.v1[axis] + a.v2[axis]) < (b.v0[axis] + b.v1[axis] + b.v2[axis]);
			});


		stack.push_back({ mid, end, depth + 1, currentIndex, false });
		stack.push_back({ start, mid, depth + 1, currentIndex, true });


        
	}

    printf("numNodes=%i\n",numNodes);

    for (int i=0; i<numNodes; i++)
    {
        printf("Node %i \n",i);
        printf("    left  : %i\n",bvhNodes[i].leftChild);
        printf("    right : %i\n",bvhNodes[i].rightChild);
        printf("    firstTriangleIndex : %i\n",bvhNodes[i].firstTriangleIndex);
        printf("    triangleCount : %i\n",bvhNodes[i].triangleCount);

        printf("    Bounds : \n");
        printf("      Min: : %f %f %f\n",bvhNodes[i].bounds.min.x,bvhNodes[i].bounds.min.y,bvhNodes[i].bounds.min.z);
        printf("      Max: : %f %f %f\n",bvhNodes[i].bounds.max.x,bvhNodes[i].bounds.max.y,bvhNodes[i].bounds.max.z);
    }
    
}


// END::BVH CPU
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************



//*******************************************************************************************************************************
// BEGIN::BVH GPU 1

void verifyBVH(BVHNode* h_nodes, int numNodes) {
	for (int i = 0; i < numNodes; i++) {
		BVHNode& node = h_nodes[i];
		if (node.leftChild != -1) {

			std::cout << "[" << i << "] Internal Node: Left Child = " << node.leftChild << ", Right Child = " << node.rightChild;
			assert(node.leftChild < i);
			assert(node.rightChild < i);
			assert(node.triangleCount == h_nodes[node.leftChild].triangleCount + h_nodes[node.rightChild].triangleCount);

			std::cout << " triangleCount = " << node.triangleCount << std::endl;
			assert(node.leftChild < i);
		}
		else {
			assert(node.triangleCount == 1);
		}
	}
	printf("BVH verification passed.\n");
}


__global__ void buildBVHLevel(BVHNode* nodes, int levelSize, int levelOffset, int totalNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= levelSize / 2) return;

    int parentIdx = totalNodes - levelOffset - 1 - idx;
    int leftChildIdx = totalNodes - levelOffset * 2 - 1 - idx * 2;
    int rightChildIdx = leftChildIdx - 1;

    if (leftChildIdx < 0 || rightChildIdx < 0) return;

    BVHNode& parent = nodes[parentIdx];
    BVHNode& leftChild = nodes[leftChildIdx];
    BVHNode& rightChild = nodes[rightChildIdx];

    parent.bounds.min = min(leftChild.bounds.min, rightChild.bounds.min);
    parent.bounds.max = max(leftChild.bounds.max, rightChild.bounds.max);
    parent.leftChild = leftChildIdx;
    parent.rightChild = rightChildIdx;
    parent.firstPrimitive = leftChild.firstPrimitive;
    parent.primitiveCount = leftChild.primitiveCount + rightChild.primitiveCount;
    parent.triangleCount = leftChild.triangleCount + rightChild.triangleCount;
    parent.firstTriangleIndex = leftChild.firstTriangleIndex;
}

__global__ void buildBVHInitNodes(BVHNode* nodes, Triangle* triangles, int numTriangles, int totalNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodes) return;

    BVHNode& node = nodes[idx];
    if (idx < numTriangles) {
        Triangle& tri = triangles[idx];
        node.bounds.min = min(tri.v0, min(tri.v1, tri.v2));
        node.bounds.max = max(tri.v0, max(tri.v1, tri.v2));
        node.firstPrimitive = idx;
        node.primitiveCount = 1;
        node.triangleCount = 1;
        node.firstTriangleIndex = idx;
    } else {
        node.bounds.min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        node.bounds.max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        node.firstPrimitive = -1;
        node.primitiveCount = 0;
        node.triangleCount = 0;
        node.firstTriangleIndex = -1;
    }
    node.leftChild = -1;
    node.rightChild = -1;
}

void buildBVH_GPU_Version1(Triangle* d_triangles, BVHNode* d_nodes, int numTriangles) {
    int numThreads = 256;
    int totalNodes = 2 * numTriangles - 1;
    int numBlocks = (totalNodes + numThreads - 1) / numThreads;

    // Initialiser les nœuds feuilles
    buildBVHInitNodes<<<numBlocks, numThreads>>>(d_nodes, d_triangles, numTriangles, totalNodes);
    hipDeviceSynchronize();

    // Construire les niveaux supérieurs
    int levelSize = numTriangles;
    int levelOffset = 0;

    while (levelSize > 1) {
        int numParents = (levelSize + 1) / 2;
        numBlocks = (numParents + numThreads - 1) / numThreads;
        buildBVHLevel<<<numBlocks, numThreads>>>(d_nodes, levelSize, levelOffset, totalNodes);
        hipDeviceSynchronize();
        
        levelSize = numParents;
        levelOffset += numParents;
    }

	// Verification 2
	BVHNode* h_nodes = new BVHNode[2 * numTriangles - 1];
	hipMemcpy(h_nodes, d_nodes, (2 * numTriangles - 1) * sizeof(BVHNode), hipMemcpyDeviceToHost);
	verifyBVH(h_nodes, 2 * numTriangles - 1);
	delete[] h_nodes;
}

// END::GPU 1
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

//*******************************************************************************************************************************
// BEGIN::BVH GPU 2


__host__ __device__
void calculateBoundingBox(const Triangle& triangle, Vec3& min_values, Vec3& max_values)
{
	min_values = min(triangle.v0, min(triangle.v1, triangle.v2));
	max_values = max(triangle.v0, max(triangle.v1, triangle.v2));
}


__global__ void initializeLeaves(Triangle* triangles, BVHNode* nodes, int numTriangles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numTriangles) {
		BVHNode& node = nodes[numTriangles - 1 + idx];
		calculateBoundingBox(triangles[idx], node.bounds.min, node.bounds.max);
		node.triangleIndex = idx;
		node.leftChild = node.rightChild = -1;
        node.firstTriangleIndex = idx;
        node.triangleCount = 1;
	}
}


void buildBVH_GPU_Version2(Triangle* d_triangles, BVHNode* d_nodes, int numTriangles)
{
    int totalNodes = 2 * numTriangles - 1;
	int blockSize = 256;
	int numBlocks = (numTriangles + blockSize - 1) / blockSize;
	hipLaunchKernelGGL(initializeLeaves, dim3(numBlocks), dim3(blockSize), 0, 0, d_triangles, d_nodes, numTriangles);

	BVHNode* h_nodes = new BVHNode[2 * numTriangles - 1];
	hipMemcpy(h_nodes, d_nodes, (2 * numTriangles - 1) * sizeof(BVHNode), hipMemcpyDeviceToHost);

	for (int i = numTriangles - 2; i >= 0; --i) {
		BVHNode& node = h_nodes[i];
		int leftChild    = 2 * i + 1;
		int rightChild   = 2 * i + 2;
		node.leftChild   = leftChild;
		node.rightChild  = rightChild;
		node.triangleIndex = -1;
        
		BVHNode& leftNode = h_nodes[leftChild];
		BVHNode& rightNode = h_nodes[rightChild];
		node.bounds.min = min(leftNode.bounds.min, rightNode.bounds.min);
		node.bounds.max = max(leftNode.bounds.max, rightNode.bounds.max);
	}
	hipMemcpy(d_nodes, h_nodes, (2 * numTriangles - 1) * sizeof(BVHNode), hipMemcpyHostToDevice);
	delete[] h_nodes;
}



__global__ void buildEvaluationNodes(BVHNode* nodes, int numTriangles)
{
	for (int i = numTriangles - 2; i >= 0; --i) {
		BVHNode& node = nodes[i];
		int leftChild    = 2 * i + 1;
		int rightChild   = 2 * i + 2;
		node.leftChild   = leftChild;
		node.rightChild  = rightChild;
		node.triangleIndex = -1;
		BVHNode& leftNode = nodes[leftChild];
		BVHNode& rightNode = nodes[rightChild];
		node.bounds.min = min(leftNode.bounds.min, rightNode.bounds.min);
		node.bounds.max = max(leftNode.bounds.max, rightNode.bounds.max);
	}
    //__syncthreads();
}


void buildBVH_GPU_Version3(Triangle* d_triangles, BVHNode* d_nodes, int numTriangles)
{
    int blockSize = 256;
    int numBlocks = (numTriangles + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(initializeLeaves, dim3(numBlocks), dim3(blockSize), 0, 0, d_triangles, d_nodes, numTriangles);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(buildEvaluationNodes, dim3(1), dim3(1), 0, 0, d_nodes, numTriangles);
    hipDeviceSynchronize();
}


// END::GPU 2
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

// END::BVH GPU



//*******************************************************************************************************************************

// Load OBJ
void loadTrianglesFromOBJ(const std::string& filename, std::vector<Triangle>& triangles, int id) {
	std::ifstream file(filename);

	std::vector<Vec3> vertices;
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string type;
		iss >> type;

		if (type == "v") {
			float x, y, z;
			iss >> x >> y >> z;
			vertices.emplace_back(x, y, z);
		}
		else if (type == "f") {
			int v1, v2, v3;
			iss >> v1 >> v2 >> v3;
			triangles.push_back({ vertices[v1 - 1], vertices[v2 - 1], vertices[v3 - 1] });
			triangles.back().id = id;
		}
	}

	//std::cout << "Chargé " << triangles.size() << " triangles depuis " << filename << std::endl;
}

// Generate Rays
void generateRays(std::vector<Ray>& rays, int numRays, const Vec3& origin, float fov, int imageWidth, int imageHeight) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	float aspectRatio = static_cast<float>(imageWidth) / imageHeight;
	float tanHalfFov = std::tan(fov * 0.5f * M_PI / 180.0f);

	for (int i = 0; i < numRays; ++i) {
		float x = (2.0f * (i % imageWidth) + 1.0f) / (2.0f * imageWidth) - 1.0f;
		float y = (1.0f - (2.0f * (i / imageWidth) + 1.0f) / (2.0f * imageHeight)) * (1.0f / aspectRatio);

		Vec3 direction(
			x * tanHalfFov * aspectRatio,
			y * tanHalfFov,
			-1.0f
		);

		float length = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
		direction.x /= length;
		direction.y /= length;
		direction.z /= length;

		rays.push_back({ origin, direction });
	}

	//std::cout << "Généré " << rays.size() << " rayons" << std::endl;
}



void Test001(int mode)
{
	std::chrono::steady_clock::time_point t_begin_0, t_begin_1, t_begin_2;
	std::chrono::steady_clock::time_point t_end_0, t_end_1, t_end_2;
	long int t_laps;

	std::vector<Triangle> triangles;
	std::vector<Ray> rays;
	std::vector<BVHNode> bvhNodes;

	//loadTrianglesFromOBJ("Triangle2Cube.obj", triangles, 12321);
	loadTrianglesFromOBJ("Test.obj", triangles,12321);

	Vec3 cameraOrigin(0.75, 0.75, 3.0);
	float fov = 60.0f;
	int imageWidth = 5;
	int imageHeight = 5;
	int numRays = imageWidth * imageHeight;
	generateRays(rays, numRays, cameraOrigin, fov, imageWidth, imageHeight);


	Triangle* deviceTriangles;
	Ray* deviceRays;
	BVHNode* devicebvhNodes;
	int* deviceHitTriangles;
	Vec3* deviceIntersectionPoint;
	float* deviceDistanceResults;
	int* deviceIdResults;

	hipMalloc(&deviceTriangles, triangles.size() * sizeof(Triangle));
	hipMemcpy(deviceTriangles, triangles.data(), triangles.size() * sizeof(Triangle), hipMemcpyHostToDevice);

	bool isUnifiedMemory = false;
	bool isGPU = false; isGPU = true;

	if (mode==0) isGPU = false;

	if (!isGPU)
	{
		t_begin_0 = std::chrono::steady_clock::now();
		buildBVH_CPU_Recursive(triangles, bvhNodes);
		//buildBVH_CPU_Iterative(triangles, bvhNodes);
		t_end_0 = std::chrono::steady_clock::now();
		if (1==0) writeBVHNodes(bvhNodes);
		hipMalloc(&devicebvhNodes, bvhNodes.size() * sizeof(BVHNode));
		hipMemcpy(devicebvhNodes, bvhNodes.data(), bvhNodes.size() * sizeof(BVHNode), hipMemcpyHostToDevice);
	}

	if (isGPU)
	{
		int numTriangles = triangles.size();
		hipMalloc(&devicebvhNodes, (2 * numTriangles - 1) * sizeof(BVHNode));
		t_begin_0 = std::chrono::steady_clock::now();
		//buildBVH_GPU_Version1(deviceTriangles, devicebvhNodes, numTriangles);
        if (mode==2) buildBVH_GPU_Version2(deviceTriangles, devicebvhNodes, numTriangles);
        if (mode==3) buildBVH_GPU_Version3(deviceTriangles, devicebvhNodes, numTriangles);
		t_end_0 = std::chrono::steady_clock::now();

        //hipMemcpy(nodes, d_nodes, (2 * numTriangles - 1) * sizeof(BVHNode), hipMemcpyDeviceToHost);

        // CTRL::BEGIN
		if (1==0)
		{
			printf("**********************************\n");
				int totalNodes=(2 * numTriangles - 1);
				for (int i=0; i<totalNodes; i++)
				{
					printf("Node %i \n",i);
					printf("    left  : %i\n",devicebvhNodes[i].leftChild);
					printf("    right : %i\n",devicebvhNodes[i].rightChild);

					printf("    Bounds : \n");
					printf("      Min: : %f %f %f\n",devicebvhNodes[i].bounds.min.x,devicebvhNodes[i].bounds.min.y,devicebvhNodes[i].bounds.min.z);
					printf("      Max: : %f %f %f\n",devicebvhNodes[i].bounds.max.x,devicebvhNodes[i].bounds.max.y,devicebvhNodes[i].bounds.max.z);
				}
		}
        // CTRL::END
    }
        

	std::cout << "[Ray Tracing]\n";
	hipMalloc(&deviceRays, rays.size() * sizeof(Ray));
	hipMemcpy(deviceRays, rays.data(), rays.size() * sizeof(Ray), hipMemcpyHostToDevice);


	hipMalloc(&deviceHitTriangles, rays.size() * sizeof(int));
	hipMalloc(&deviceIntersectionPoint, rays.size() * sizeof(Vec3));
	hipMalloc(&deviceDistanceResults, rays.size() * sizeof(float));
	hipMalloc(&deviceIdResults, rays.size() * sizeof(int));


	t_begin_1 = std::chrono::steady_clock::now();
	int blockSize = 256;
	int numBlocks = (rays.size() + blockSize - 1) / blockSize;
	hipLaunchKernelGGL(raytraceKernel, dim3(numBlocks), dim3(blockSize), 0, 0,
		deviceRays,
		rays.size(),
		devicebvhNodes,
		deviceTriangles,
		deviceHitTriangles,
		deviceDistanceResults,
		deviceIntersectionPoint,
		deviceHitTriangles
	);
	t_end_1 = std::chrono::steady_clock::now();

	std::vector<int> hitTriangles(rays.size());
	hipMemcpy(hitTriangles.data(), deviceHitTriangles, rays.size() * sizeof(int), hipMemcpyDeviceToHost);

	hipFree(deviceTriangles);
	hipFree(deviceRays);
	hipFree(devicebvhNodes);
	hipFree(deviceHitTriangles);

	t_laps = std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0).count();
	std::cout << "[INFO]: Elapsed microseconds inside BVH : " << t_laps << " us\n";

	t_laps = std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1).count();
	std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : " << t_laps << " us\n";
}




int main() {
	Test001(0);
	Test001(2);
	Test001(3);


	return 0;
}


