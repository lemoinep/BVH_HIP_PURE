
//#pragma GCC diagnostic warning "-Wunused-result"
//#pragma clang diagnostic ignored "-Wunused-result"

//#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunused-result"

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// Link HIP
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hipblas-export.h"
#include "hipblas.h"
#include "hipsolver.h"

#include <roctx.h>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/system/hip/vector.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <gmsh.h>

#include <stack>

// NOTA: The goal is to create a BVH algorithm using the features of hip without
// using rocThrust and see the performance depending on the case.

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    hipError_t status = command;                                               \
    if (status != hipSuccess) {                                                \
      std::cerr << "Error: HIP reports " << hipGetErrorString(status)          \
                << std::endl;                                                  \
      std::abort();                                                            \
    }                                                                          \
  }

// Conditional assertion based on debug mode for HIP operations
#ifdef NDEBUG
#define HIP_ASSERT(x) x // No assertion in release mode
#else
#define HIP_ASSERT(x) (assert((x) == hipSuccess)) // Assert in debug mode
#endif

struct Vec3 {
  float x, y, z;
  __host__ __device__ __inline__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ __inline__ Vec3(float a) : x(a), y(a), z(a) {}
  __host__ __device__ __inline__ Vec3(float x, float y, float z)
      : x(x), y(y), z(z) {}

  __host__ __device__ __inline__ Vec3 operator+(const Vec3 &v) const {
    return Vec3(x + v.x, y + v.y, z + v.z);
  }
  __host__ __device__ __inline__ Vec3 operator-(const Vec3 &v) const {
    return Vec3(x - v.x, y - v.y, z - v.z);
  }
  __host__ __device__ __inline__ Vec3 operator*(float f) const {
    return Vec3(x * f, y * f, z * f);
  }

  __host__ __device__ __inline__ Vec3 operator*(const Vec3 &v) const {
    return Vec3(x * v.x, y * v.y, z * v.z);
  }

  __host__ __device__ __inline__ Vec3 operator/(const Vec3 &other) const {
    return Vec3(x / other.x, y / other.y, z / other.z);
  }
  __host__ __device__ __inline__ Vec3 operator/(float scalar) const {
    return Vec3(x / scalar, y / scalar, z / scalar);
  }

  __host__ __device__ __inline__ float &operator[](int i) { return (&x)[i]; }
  __host__ __device__ __inline__ const float &operator[](int i) const {
    return (&x)[i];
  }
};

__host__ __device__ __inline__ Vec3 min(const Vec3 &a, const Vec3 &b) {
  return Vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ __inline__ Vec3 max(const Vec3 &a, const Vec3 &b) {
  return Vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ __inline__ Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}

__host__ __device__ __inline__ float dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __inline__ void atomicMinVec3(Vec3 *addr, Vec3 val) {
  atomicMin((int *)&addr->x, __float_as_int(val.x));
  atomicMin((int *)&addr->y, __float_as_int(val.y));
  atomicMin((int *)&addr->z, __float_as_int(val.z));
}

__device__ __inline__ void atomicMaxVec3(Vec3 *addr, Vec3 val) {
  atomicMax((int *)&addr->x, __float_as_int(val.x));
  atomicMax((int *)&addr->y, __float_as_int(val.y));
  atomicMax((int *)&addr->z, __float_as_int(val.z));
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

struct BVHNode4 {
  AABB bounds;
  int triangleIndex;
  int triangleCount;
  int leftChild;
  int rightChild;
};

struct Intersection {
  bool hit;
  float t;
  int triangleIndex;
};

struct TriangleInfo {
  Vec3 centroid;
  int index;
};

__host__ __device__ __inline__ void normalizeRayDirection(Ray &ray) {
  float len = sqrtf(ray.direction.x * ray.direction.x +
                    ray.direction.y * ray.direction.y +
                    ray.direction.z * ray.direction.z);
  if (len > 0) {
    float invLen = 1.0f / len;
    ray.direction.x *= invLen;
    ray.direction.y *= invLen;
    ray.direction.z *= invLen;
  }
}

__host__ __device__ __inline__ float
calculateCircumradius(const Triangle &triangle) {
  float a = sqrt(pow(triangle.v1.x - triangle.v0.x, 2) +
                 pow(triangle.v1.y - triangle.v0.y, 2) +
                 pow(triangle.v1.z - triangle.v0.z, 2));

  float b = sqrt(pow(triangle.v2.x - triangle.v1.x, 2) +
                 pow(triangle.v2.y - triangle.v1.y, 2) +
                 pow(triangle.v2.z - triangle.v1.z, 2));

  float c = sqrt(pow(triangle.v0.x - triangle.v2.x, 2) +
                 pow(triangle.v0.y - triangle.v2.y, 2) +
                 pow(triangle.v0.z - triangle.v2.z, 2));
  float s = (a + b + c) / 2.0f;
  float area = sqrt(s * (s - a) * (s - b) * (s - c));
  float radius = (a * b * c) / (4.0f * area);
  return radius;
}

__host__ __device__ __inline__ float
calculateHalfOpeningAngle(const Triangle &triangle, const Vec3 &origin) {
  // This function will be used to speed up the calculations and will adapt the
  // limit angle of the sameDirection function
  Vec3 barycenter = {(triangle.v0.x + triangle.v1.x + triangle.v2.x) / 3.0f,
                     (triangle.v0.y + triangle.v1.y + triangle.v2.y) / 3.0f,
                     (triangle.v0.z + triangle.v1.z + triangle.v2.z) / 3.0f};
  float distance =
      sqrt(pow(barycenter.x - origin.x, 2) + pow(barycenter.y - origin.y, 2) +
           pow(barycenter.z - origin.z, 2));
  Vec3 edge1 = {triangle.v1.x - triangle.v0.x, triangle.v1.y - triangle.v0.y,
                triangle.v1.z - triangle.v0.z};
  Vec3 edge2 = {triangle.v2.x - triangle.v0.x, triangle.v2.y - triangle.v0.y,
                triangle.v2.z - triangle.v0.z};
  Vec3 cross = {edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x};
  float area =
      0.5f * sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
  float solidAngle = area / (distance * distance);
  float halfOpeningAngle = asin(sqrt(solidAngle / (4 * M_PI)));
  return halfOpeningAngle;
}

__host__ __device__ __inline__ float angleScalar(const Vec3 v1, const Vec3 v2) {
  float p = (v1.x) * (v2.x) + (v1.y) * (v2.y) + (v1.z) * (v2.z);
  float n1 = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  float n2 = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
  float d = n1 * n2;
  float res = 0.0f;
  if (d > 0.0f) {
    float r = p / d;
    if (r > 1.0f)
      r = 1.0f;
    res = acos(r);
  }
  return (res); // in radian
}

__host__ __device__ __inline__ bool sameDirection(Triangle &tri, Ray &ray,
                                                  const float &angleLim) {
  Vec3 dT;
  dT.x = (tri.v0.x + tri.v1.x + tri.v2.x) / 3.0f - ray.origin.x;
  dT.y = (tri.v0.y + tri.v1.y + tri.v2.y) / 3.0f - ray.origin.y;
  dT.z = (tri.v0.z + tri.v1.z + tri.v2.z) / 3.0f - ray.origin.z;
  float angle1 = angleScalar(dT, ray.direction);
  float angle2 = calculateHalfOpeningAngle(tri, ray.origin);
  // printf("angle =%f %f\n", angle, angleLim); |

  // return ( (angle1 <= angleLim) || (angleLim<=angle2 ) );
  // return ( (angle1 <= angleLim));
  return (angle1 <= angleLim) && (angle1 <= angle2);
}

__host__ __device__ __inline__ bool sameDirectionTest(Triangle &tri, Ray &ray,
                                                      const float &angleLim) {
  Vec3 dT0 = tri.v0 - ray.origin;
  Vec3 dT1 = tri.v1 - ray.origin;
  Vec3 dT2 = tri.v2 - ray.origin;
  Vec3 dT3;
  dT3.x = (tri.v0.x + tri.v1.x + tri.v2.x) / 3.0f - ray.origin.x;
  dT3.y = (tri.v0.y + tri.v1.y + tri.v2.y) / 3.0f - ray.origin.y;
  dT3.z = (tri.v0.z + tri.v1.z + tri.v2.z) / 3.0f - ray.origin.z;

  bool b0 = (fabs(angleScalar(dT0, ray.direction)) <= angleLim);
  bool b1 = (fabs(angleScalar(dT1, ray.direction)) <= angleLim);
  bool b2 = (fabs(angleScalar(dT2, ray.direction)) <= angleLim);
  bool b3 = (fabs(angleScalar(dT3, ray.direction)) <= angleLim);

  return (b0 || b1 || b2 || b3);
}

void writeBVHNodes(const std::vector<BVHNode> &hostNodes) {
  std::cout << "BVH Nodes:" << std::endl;
  for (size_t i = 0; i < hostNodes.size(); ++i) {
    const BVHNode &node = hostNodes[i];
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "  left : " << node.leftChild << "\n";
    std::cout << "  right: " << node.rightChild << "\n";
    std::cout << "  firstTriangleIndex : " << node.firstTriangleIndex << "\n";
    std::cout << "  triangleCount : " << node.triangleCount << "\n";
    std::cout << "  Bounds:\n";
    std::cout << "    Min: ";
    std::cout << "(" << node.bounds.min.x << ", " << node.bounds.min.y << ", "
              << node.bounds.min.z << ")"
              << "\n";
    std::cout << "    Max: ";
    std::cout << "(" << node.bounds.max.x << ", " << node.bounds.max.y << ", "
              << node.bounds.max.z << ")"
              << "\n";
    std::cout << std::endl;
  }
}

//*******************************************************************************************************************************
// BEGIN::RAY TRACING

__device__ __inline__ bool rayTriangleIntersect0(const Ray &ray,
                                                 const Triangle &tri, float &t,
                                                 Vec3 &intersectionPoint) {
  Vec3 edge1 = tri.v1 - tri.v0;
  Vec3 edge2 = tri.v2 - tri.v0;
  Vec3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);

  if (a > -1e-6f && a < 1e-6f)
    return false;

  float f = 1.0f / a;
  Vec3 s = ray.origin - tri.v0;
  float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  Vec3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  t = f * dot(edge2, q);

  if (t > 1e-6) {
    intersectionPoint.x = ray.origin.x + t * ray.direction.x;
    intersectionPoint.y = ray.origin.y + t * ray.direction.y;
    intersectionPoint.z = ray.origin.z + t * ray.direction.z;
  } else {
    intersectionPoint.x = INFINITY;
    intersectionPoint.y = INFINITY;
    intersectionPoint.z = INFINITY;
  }
  return (t > 1e-6f);
}

__device__ __inline__ bool rayTriangleIntersect(const Ray &ray,
                                                const Triangle &tri, float &t,
                                                Vec3 &intersectionPoint) {
  Vec3 edge1 = tri.v1 - tri.v0;
  Vec3 edge2 = tri.v2 - tri.v0;
  Vec3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);
  const float EPSILON = 1e-8f;
  if (fabs(a) < EPSILON)
    return false; // Rayon parallèle au triangle

  float f = 1.0f / a;
  Vec3 s = ray.origin - tri.v0;
  float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  Vec3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  t = f * dot(edge2, q);

  // If t is negative, the intersection is behind the origin of the ray
  // Which means that the origin is inside the triangle

  if (t < -EPSILON) {
    t = 0.0f;
    intersectionPoint = ray.origin;
    return true;
  }

  // If t is very close to zero, consider that the origin is on the triangle

  if (fabs(t) < EPSILON) {
    intersectionPoint = ray.origin;
    return true;
  }

  // Normal intersection in front of the ray origin
  if (t > EPSILON) {
    intersectionPoint.x = ray.origin.x + t * ray.direction.x;
    intersectionPoint.y = ray.origin.y + t * ray.direction.y;
    intersectionPoint.z = ray.origin.z + t * ray.direction.z;
    return true;
  }

  intersectionPoint.x = INFINITY;
  intersectionPoint.y = INFINITY;
  intersectionPoint.z = INFINITY;

  return false;
}





__device__ __inline__ bool pointInTriangle(const Vec3 &point, const Triangle &triangle /*, Vec3 &intersectionPoint*/) {
    Vec3 v0 = triangle.v2 - triangle.v0;
    Vec3 v1 = triangle.v1 - triangle.v0;
    Vec3 v2 = point - triangle.v0;

    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    bool isInside = (u >= 0) && (v >= 0) && (u + v <= 1);

/*
    if (isInside) {
        // Calculer le point d'intersection en utilisant les coordonnées barycentriques
        float w = 1.0f - u - v;
        intersectionPoint.x = w * triangle.v0.x + u * triangle.v1.x + v * triangle.v2.x;
        intersectionPoint.y = w * triangle.v0.y + u * triangle.v1.y + v * triangle.v2.y;
        intersectionPoint.z = w * triangle.v0.z + u * triangle.v1.z + v * triangle.v2.z;
    }
  */

    return isInside;
}




__device__ __inline__ bool rayTriangleIntersect4(const Ray &ray,
                                                 const Triangle &tri, float &t,
                                                 Vec3 &intersectionPoint) {
  Vec3 edge1 = tri.v1 - tri.v0;
  Vec3 edge2 = tri.v2 - tri.v0;
  Vec3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);
  const float EPSILON = 1e-8f;
  if (fabs(a) < EPSILON)
    return false; // Rayon parallèle au triangle

  float f = 1.0f / a;
  Vec3 s = ray.origin - tri.v0;
  float u = f * dot(s, h);

  if (u < 0.0f || u > 1.0f)
    return false;

  Vec3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);

  if (v < 0.0f || u + v > 1.0f)
    return false;

  t = f * dot(edge2, q);

  // If t is negative, the intersection is behind the origin of the ray
  // Which means that the origin is inside the triangle
  /*
   if (t < -EPSILON) {
     t = 0.0f;
     intersectionPoint = ray.origin;
     return true;
   }
   */

  // If t is very close to zero, consider that the origin is on the triangle

  /*
  if (fabs(t) < EPSILON) {
    intersectionPoint = ray.origin;
    return true;
  }
  */

  // Normal intersection in front of the ray origin
  // if (t > EPSILON) {
  if (t >= 0.0f) {
    intersectionPoint.x = ray.origin.x + t * ray.direction.x;
    intersectionPoint.y = ray.origin.y + t * ray.direction.y;
    intersectionPoint.z = ray.origin.z + t * ray.direction.z;
    return true;
  }

  intersectionPoint.x = INFINITY;
  intersectionPoint.y = INFINITY;
  intersectionPoint.z = INFINITY;

  return false;
}

/*
__device__ __inline__ bool rayTriangleIntersectBord(const Ray& ray, const
Triangle& tri, float& t, Vec3& intersectionPoint, bool* hitEdge, bool*
hitVertex) { Vec3 edge1 = tri.v1 - tri.v0; Vec3 edge2 = tri.v2 - tri.v0; Vec3 h
= cross(ray.direction, edge2); float a = dot(edge1, h);

        // Vérification si le rayon est parallèle au triangle
        if (a > -1e-6f && a < 1e-6f) return false;

        float f = 1.0f / a;
        Vec3 s = ray.origin - tri.v0;
        float u = f * dot(s, h);

        // Vérification des coordonnées barycentriques
        if (u < -1e-6f || u > 1.0f) return false;

        Vec3 q = cross(s, edge1);
        float v = f * dot(ray.direction, q);

        if (v < -1e-6f || u + v > 1.0f + 1e-6f) return false;

        // Calcul de t (distance à l'intersection)
        t = f * dot(edge2, q);

        // Vérification si l'intersection est devant le rayon
        if (t >= 0) {
                intersectionPoint.x = ray.origin.x + t * ray.direction.x;
                intersectionPoint.y = ray.origin.y + t * ray.direction.y;
                intersectionPoint.z = ray.origin.z + t * ray.direction.z;

                // Vérification si l'intersection se produit sur un bord
                *hitEdge = (u <= 1e-6f || v <= 1e-6f || u + v >= 1.0f - 1e-6f);

                // Vérification si l'intersection se produit sur un sommet
                float epsilon = 1e-6f; // Tolérance pour déterminer si on touche
un sommet *hitVertex = (length(intersectionPoint - tri.v0) < epsilon ||
                                          length(intersectionPoint - tri.v1) <
epsilon || length(intersectionPoint - tri.v2) < epsilon);

                return true; // Intersection valide
        } else {
                // L'intersection se produit derrière le rayon
                intersectionPoint.x = INFINITY;
                intersectionPoint.y = INFINITY;
                intersectionPoint.z = INFINITY;
                return false; // Pas d'intersection valide
        }
}
*/

__device__ __inline__ bool
rayTriangleIntersectSurfaceEdge(const Ray &ray, const Triangle &tri, float &t,
                                Vec3 &intersectionPoint, bool *hitEdge,
                                bool *hitVertex) {
  Vec3 edge1 = tri.v1 - tri.v0;
  Vec3 edge2 = tri.v2 - tri.v0;
  Vec3 h = cross(ray.direction, edge2);
  float a = dot(edge1, h);

  // Check if the ray is parallel to the triangle
  if (a > -1e-6f && a < 1e-6f)
    return false;

  float f = 1.0f / a;
  Vec3 s = ray.origin - tri.v0;
  float u = f * dot(s, h);

  // Checking barycentric coordinates
  if (u < -1e-6f || u > 1.0f)
    return false;

  Vec3 q = cross(s, edge1);
  float v = f * dot(ray.direction, q);

  if (v < -1e-6f || u + v > 1.0f + 1e-6f)
    return false;

  // Calculation of t (distance to intersection)
  t = f * dot(edge2, q);

  // Check if the intersection is in front of the ray
  if (t >= 0) {
    intersectionPoint.x = ray.origin.x + t * ray.direction.x;
    intersectionPoint.y = ray.origin.y + t * ray.direction.y;
    intersectionPoint.z = ray.origin.z + t * ray.direction.z;

    // Check if the intersection occurs on an edge
    *hitEdge = (u <= 1e-6f || v <= 1e-6f || u + v >= 1.0f - 1e-6f);

    // Check if the intersection occurs on a vertex
    float epsilon = 1e-6f; // Tolerance to determine if we touch a vertex

    auto distanceSquared = [](const Vec3 &a, const Vec3 &b) {
      float dx = a.x - b.x;
      float dy = a.y - b.y;
      float dz = a.z - b.z;
      return dx * dx + dy * dy + dz * dz;
    };

    *hitVertex =
        (distanceSquared(intersectionPoint, tri.v0) < epsilon * epsilon ||
         distanceSquared(intersectionPoint, tri.v1) < epsilon * epsilon ||
         distanceSquared(intersectionPoint, tri.v2) < epsilon * epsilon);

    return true;
  } else {
    intersectionPoint.x = INFINITY;
    intersectionPoint.y = INFINITY;
    intersectionPoint.z = INFINITY;
    return false;
  }
}

__device__ __inline__ bool rayAABBIntersect(const Ray &ray, const AABB &aabb) {
  Vec3 invDir = Vec3(1.0f / ray.direction.x, 1.0f / ray.direction.y,
                     1.0f / ray.direction.z);
  Vec3 tMin = (aabb.min - ray.origin) * invDir;
  Vec3 tMax = (aabb.max - ray.origin) * invDir;
  Vec3 t1 =
      Vec3(fminf(tMin.x, tMax.x), fminf(tMin.y, tMax.y), fminf(tMin.z, tMax.z));
  Vec3 t2 =
      Vec3(fmaxf(tMin.x, tMax.x), fmaxf(tMin.y, tMax.y), fmaxf(tMin.z, tMax.z));

  float tNear = fmaxf(fmaxf(t1.x, t1.y), t1.z);
  float tFar = fminf(fminf(t2.x, t2.y), t2.z);
  return tNear <= tFar;
}

__device__ __inline__ bool rayAABBIntersect4(const Ray &ray, const AABB &aabb) {
  const float EPSILON = 1e-8f;
  float3 invDir;
  invDir.x = 1.0f / ray.direction.x;
  invDir.y = 1.0f / ray.direction.y;
  invDir.z = 1.0f / ray.direction.z;

  float3 t1, t2;
  t1.x = (aabb.min.x - ray.origin.x) * invDir.x;
  t1.y = (aabb.min.y - ray.origin.y) * invDir.y;
  t1.z = (aabb.min.z - ray.origin.z) * invDir.z;
  t2.x = (aabb.max.x - ray.origin.x) * invDir.x;
  t2.y = (aabb.max.y - ray.origin.y) * invDir.y;
  t2.z = (aabb.max.z - ray.origin.z) * invDir.z;

  float tNear =
      fmaxf(fmaxf(fminf(t1.x, t2.x), fminf(t1.y, t2.y)), fminf(t1.z, t2.z));
  float tFar =
      fminf(fminf(fmaxf(t1.x, t2.x), fmaxf(t1.y, t2.y)), fmaxf(t1.z, t2.z));

  return tFar >= 0.0f && tNear <= tFar + EPSILON;
}

__global__ void raytraceKernelOld(Ray *rays, int numRays, BVHNode *bvhNodes,
                                  Triangle *triangles, int *hitTriangles,
                                  float *distance, Vec3 *intersectionPoint,
                                  int *hitId)

{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];
  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0;
  // stack[stackPtr++] = 13;

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

  const float angleLim = 0.6f;

  bool isView = false; // isView = true;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    BVHNode &node = bvhNodes[nodeIdx];

    // printf("node[%i]\n",nodeIdx);

    // if (nodeIdx>0) printf("node[%i] %i
    // %i\n",nodeIdx,node.triangleCount,node.firstTriangleIndex);

    if (!rayAABBIntersect(ray, node.bounds))
      continue;

    if (node.triangleCount > 0) {
      for (int i = 0; i < node.triangleCount; ++i) {
        Triangle &tri = triangles[node.firstTriangleIndex + i];
        if (sameDirectionTest(tri, ray, angleLim)) {
          float t;
          if (rayTriangleIntersect(ray, tri, t, intersectionPointT)) {

            if (isView)
              printf("      Num Ray[%i] <%f %f %f>\n", idx,
                     intersectionPointT.x, intersectionPointT.y,
                     intersectionPointT.z);
            if (t < closestT) {
              closestT = t;
              closestTriangle = node.firstTriangleIndex + i;
              closestIntersectionPoint = intersectionPointT;
              closesIntersectionId = triangles[closestTriangle].id;
            }
          }
        }
      }
    } else {
      stack[stackPtr++] = node.leftChild;
      stack[stackPtr++] = node.rightChild;
    }
  }

  // if (closesIntersectionId>0) printf("      Num Ray[%i] dist=%f <%f %f
  // %f>\n", idx, closestT,intersectionPointT.x, intersectionPointT.y,
  // intersectionPointT.z);

  // if (closesIntersectionId > 0) printf("      Num Ray[%i] dist=%f\n", idx,
  // closestT);

  hitTriangles[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closesIntersectionId;
}

__global__ void raytraceKernel(Ray *rays, int numRays, BVHNode *bvhNodes,
                               Triangle *triangles, int *hitTriangles,
                               float *distance, Vec3 *intersectionPoint,
                               int *hitId)

{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];
  int stack[64];
  int stackPtr = 0;
  stack[stackPtr++] = 0;
  // stack[stackPtr++] = 13;

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

  bool isView = false; // isView = true;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    BVHNode &node = bvhNodes[nodeIdx];

    // printf("node[%i]\n",nodeIdx);

    if (nodeIdx > 0)
      printf("node[%i] %i %i\n", nodeIdx, node.triangleCount,
             node.firstTriangleIndex);

    if (!rayAABBIntersect(ray, node.bounds))
      continue;

    

    if (node.triangleCount > 0) {
      for (int i = 0; i < node.triangleCount; ++i) {
        Triangle &tri = triangles[node.firstTriangleIndex + i];

        // if (isView) printf("*Num Ray[%i]\n",node.firstTriangleIndex + i);

        if (pointInTriangle(ray.origin,tri)) //test 2
        {
          float t;
          if (rayTriangleIntersect4(ray, tri, t, intersectionPointT)) {

            if (isView)
              printf("      Num Ray[%i] <%f %f %f> d=%f\n", idx,
                    intersectionPointT.x, intersectionPointT.y,
                    intersectionPointT.z, t);
            if (t < closestT) {
              closestT = t;
              closestTriangle = node.firstTriangleIndex + i;
              closestIntersectionPoint = intersectionPointT;
              closesIntersectionId = triangles[closestTriangle].id;
            }
          }
        }
      }
    } else {
      stack[stackPtr++] = node.leftChild;
      stack[stackPtr++] = node.rightChild;
    }
  }

  // if (closesIntersectionId>0) printf("      Num Ray[%i] dist=%f <%f %f
  // %f>\n", idx, closestT,intersectionPointT.x, intersectionPointT.y,
  // intersectionPointT.z);

  // if (closesIntersectionId > 0) printf("      Num Ray[%i] dist=%f\n", idx,
  // closestT);

  hitTriangles[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closesIntersectionId;
}

// FOR rayTracing parallele

__device__ __inline__ void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

__device__ __inline__ bool intersectAABB(const Ray &ray, const AABB &box,
                                         float &tMin, float &tMax) {
  // Inverser la direction du rayon pour éviter les divisions par zéro
  Vec3 invDir(1.0f / ray.direction.x, 1.0f / ray.direction.y,
              1.0f / ray.direction.z);

  float t0 = (box.min.x - ray.origin.x) * invDir.x;
  float t1 = (box.max.x - ray.origin.x) * invDir.x;

  if (invDir.x < 0.0f)
    swap(t0, t1);

  // Initialiser tMin et tMax
  tMin = t0;
  tMax = t1;

  // Répéter pour l'axe Y
  float ty0 = (box.min.y - ray.origin.y) * invDir.y;
  float ty1 = (box.max.y - ray.origin.y) * invDir.y;

  if (invDir.y < 0.0f)
    swap(ty0, ty1);

  // si les bornes se chevauchent
  if ((tMin > ty1) || (ty0 > tMax))
    return false;

  if (ty0 > tMin)
    tMin = ty0;

  if (ty1 < tMax)
    tMax = ty1;

  float tz0 = (box.min.z - ray.origin.z) * invDir.z;
  float tz1 = (box.max.z - ray.origin.z) * invDir.z;

  if (invDir.z < 0.0f)
    swap(tz0, tz1);

  if ((tMin > tz1) || (tz0 > tMax))
    return false;

  if (tz0 > tMin)
    tMin = tz0;

  if (tz1 < tMax)
    tMax = tz1;

  return true; // Le rayon intersecte AABB
}

__global__ void
raytraceKernel_Parallel004(Ray *rays, int numRays, BVHNode *bvhNodes,
                           Triangle *triangles, int *hitTriangles,
                           float *distance, Vec3 *intersectionPoint, int *hitId)

{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];
  int stack[64];
  int stackPtr = 0;
  // On commence par la racine du BVH
  stack[stackPtr++] = 0;

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

  bool isView = false; // isView = true;

  const float angleLim = 0.6f;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    BVHNode &node = bvhNodes[nodeIdx];

    // if (nodeIdx < 0 || nodeIdx >= numRays) continue;

    if (nodeIdx < 0)
      continue;

    if (!rayAABBIntersect4(ray, node.bounds))
      continue;

    if (node.triangleCount == 1) {
      Triangle &tri = triangles[node.triangleIndex];
      // if (sameDirectionTest(tri,ray,angleLim))
      {
        float t;
        if (rayTriangleIntersect4(ray, tri, t, intersectionPointT)) {

          if (isView)
            printf("      Num Ray[%i] <%f %f %f> dist=%f\n", idx,
                   intersectionPointT.x, intersectionPointT.y,
                   intersectionPointT.z, t);
          if (t < closestT) {
            closestT = t;
            closestTriangle = node.triangleIndex;
            closestIntersectionPoint = intersectionPointT;
            closesIntersectionId = triangles[closestTriangle].id;

            if (isView)
              printf("      Close Num Ray[%i] <%f %f %f> dist=%f\n", idx,
                     intersectionPointT.x, intersectionPointT.y,
                     intersectionPointT.z, t);
          }
        }
      }
    } else {
      stack[stackPtr++] = node.rightChild;
      stack[stackPtr++] = node.leftChild;
    }
  }

  // if (closesIntersectionId>0) printf("      Num Ray[%i] dist=%f <%f %f
  // %f>\n", idx, closestT,intersectionPointT.x, intersectionPointT.y,
  // intersectionPointT.z);

  // if (closesIntersectionId > 0) printf("      Num Ray[%i] dist=%f\n", idx,
  // closestT);

  hitTriangles[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closesIntersectionId;
}

__global__ void
raytraceKernel_Parallel005(Ray *rays, int numRays, BVHNode *bvhNodes,
                           int numNodes, Triangle *triangles, int numTriangles,
                           int *hitTriangles, float *distance,
                           Vec3 *intersectionPoint, int *hitId) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays || numNodes <= 0 || numTriangles <= 0)
    return;

  Ray ray = rays[idx];
  constexpr int MAX_STACK_SIZE = 64;
  int stack[MAX_STACK_SIZE];
  int stackPtr = 0;

  stack[stackPtr++] = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closestIntersectionId = -1;
  Vec3 intersectionPointT;
  intersectionPointT.x = INFINITY;
  intersectionPointT.y = INFINITY;
  intersectionPointT.z = INFINITY;

  Vec3 closestIntersectionPoint;
  closestIntersectionPoint.x = INFINITY;
  closestIntersectionPoint.y = INFINITY;
  closestIntersectionPoint.z = INFINITY;

  constexpr float angleLim = 0.6f;

  while (stackPtr > 0 && stackPtr < MAX_STACK_SIZE) {
    int nodeIdx = stack[--stackPtr];
    if (nodeIdx < 0 || nodeIdx >= numNodes)
      continue;

    BVHNode &node = bvhNodes[nodeIdx];
    if (node.triangleCount == 1 && node.triangleIndex >= 0 &&
        node.triangleIndex < numTriangles) {
      if (!rayAABBIntersect4(ray, node.bounds)) // add test 1
        continue;

      
      Triangle &tri = triangles[node.triangleIndex];
      if (! pointInTriangle(ray.origin,tri)) //add test2
        continue;

      // if (sameDirectionTest(tri, ray, angleLim)) //add test 3
      {
        float t;
        Vec3 intersectionPointT;
        if (rayTriangleIntersect4(ray, tri, t, intersectionPointT)) {
          if (t < closestT) {
            closestT = t;
            closestTriangle = node.triangleIndex;
            closestIntersectionPoint = intersectionPointT;
            closestIntersectionId = tri.id;
          }
        }
      }
    } else if (stackPtr < MAX_STACK_SIZE - 1) {
      if (node.leftChild >= 0 && node.leftChild < numNodes)
        stack[stackPtr++] = node.leftChild;
      if (node.rightChild >= 0 && node.rightChild < numNodes &&
          stackPtr < MAX_STACK_SIZE)
        stack[stackPtr++] = node.rightChild;
    }
  }
  hitTriangles[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closestIntersectionId;
}

__global__ void raytraceKernel_Parallel006(
    Ray *rays, int numRays, BVHNode *bvhNodes, Triangle *triangles,
    int *hitTriangles, float *distance, Vec3 *intersectionPoint, int *hitId) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];
  constexpr int MAX_STACK_SIZE = 64;
  // constexpr int MAX_STACK_SIZE = 32; // voir si on peut jouer là dessus pour
  // les perf constexpr int MAX_STACK_SIZE = 128;
  int stack[MAX_STACK_SIZE];
  int stackPtr = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closestIntersectionId = -1;
  Vec3 closestIntersectionPoint = Vec3(INFINITY, INFINITY, INFINITY);

  stack[stackPtr++] = 0;

  while (stackPtr > 0 && stackPtr < MAX_STACK_SIZE) {
    int nodeIdx = stack[--stackPtr];
    const BVHNode &node = bvhNodes[nodeIdx];

    if (!rayAABBIntersect(ray, node.bounds))
      continue;

    // if (nodeIdx < 0) continue;

    if (node.triangleCount > 0) {
      // Leaf node
      for (int i = 0; i < node.triangleCount; ++i) {
        int triIdx = node.firstTriangleIndex + i;
        Triangle &tri = triangles[triIdx];
        float t;
        Vec3 intersectionPointT;

        if (rayTriangleIntersect(ray, tri, t, intersectionPointT)) {
          if (t < closestT) {
            closestT = t;
            closestTriangle = triIdx;
            closestIntersectionPoint = intersectionPointT;
            closestIntersectionId = tri.id;
          }
        }
      }
    } else {
      // Internal node - traverse children
      if (node.leftChild >= 0)
        stack[stackPtr++] = node.leftChild;
      if (node.rightChild >= 0)
        stack[stackPtr++] = node.rightChild;
    }
  }

  // Store results
  hitTriangles[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closestIntersectionId;
}

// END::RAY TRACING
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

//*******************************************************************************************************************************
// BEGIN::BVH CPU

void buildBVHRecursive(std::vector<Triangle> &triangles,
                       std::vector<BVHNode> &bvhNodes, int start, int end,
                       int depth) {
  BVHNode node;
  node.firstTriangleIndex = start;
  node.triangleCount = end - start;
  node.leftChild = node.rightChild = -1;

  // Calculer les limites du nœud
  node.bounds.min = node.bounds.max = triangles[start].v0;
  for (int i = start; i < end; i++) {
    const auto &tri = triangles[i];
    node.bounds.min = min(node.bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
    node.bounds.max = max(node.bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
  }

  // Si le nœud contient peu de triangles ou si nous sommes trop profonds,
  // arrêter la division
  if (node.triangleCount <= 4 || depth > 20) {
    bvhNodes.push_back(node);
    return;
  }

  // Trouver l'axe le plus long pour diviser
  Vec3 extent = node.bounds.max - node.bounds.min;
  int axis = 0;
  if (extent.y > extent.x)
    axis = 1;
  if (extent.z > extent[axis])
    axis = 2;

  // Trier les triangles selon l'axe choisi
  int mid = (start + end) / 2;
  std::nth_element(triangles.begin() + start, triangles.begin() + mid,
                   triangles.begin() + end,
                   [axis](const Triangle &a, const Triangle &b) {
                     return (a.v0[axis] + a.v1[axis] + a.v2[axis]) <
                            (b.v0[axis] + b.v1[axis] + b.v2[axis]);
                   });

  // Créer les enfants
  int currentIndex = bvhNodes.size();
  bvhNodes.push_back(node);

  buildBVHRecursive(triangles, bvhNodes, start, mid, depth + 1);
  bvhNodes[currentIndex].leftChild = bvhNodes.size() - 1;

  buildBVHRecursive(triangles, bvhNodes, mid, end, depth + 1);
  bvhNodes[currentIndex].rightChild = bvhNodes.size() - 1;
}

void buildBVH_CPU_Recursive(std::vector<Triangle> &triangles,
                            std::vector<BVHNode> &bvhNodes) {
  bvhNodes.clear();
  buildBVHRecursive(triangles, bvhNodes, 0, triangles.size(), 0);
}

//...

void buildBVH_CPU_Iterative(std::vector<Triangle> &triangles,
                            std::vector<BVHNode> &bvhNodes) {
  bvhNodes.clear();

  struct StackEntry {
    int start, end, depth;
    int parentIndex;
    bool isLeftChild;
  };

  std::stack<StackEntry> stack;
  stack.push({0, static_cast<int>(triangles.size()), 0, -1, false});

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
      const auto &tri = triangles[i];
      node.bounds.min = min(node.bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
      node.bounds.max = max(node.bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
    }

    int currentIndex = bvhNodes.size();
    bvhNodes.push_back(node);

    if (parentIndex != -1) {
      if (isLeftChild) {
        bvhNodes[parentIndex].leftChild = currentIndex;
      } else {
        bvhNodes[parentIndex].rightChild = currentIndex;
      }
    }

    // Si le nœud contient peu de triangles ou si nous sommes trop profonds,
    // passer au suivant
    if (node.triangleCount <= 4 || depth > 20) {
      continue;
    }

    // Trouver l'axe le plus long pour diviser
    Vec3 extent = node.bounds.max - node.bounds.min;
    int axis = 0;
    if (extent.y > extent.x)
      axis = 1;
    if (extent.z > extent[axis])
      axis = 2;

    // Trier les triangles selon l'axe choisi
    int mid = (start + end) / 2;
    std::nth_element(triangles.begin() + start, triangles.begin() + mid,
                     triangles.begin() + end,
                     [axis](const Triangle &a, const Triangle &b) {
                       return (a.v0[axis] + a.v1[axis] + a.v2[axis]) <
                              (b.v0[axis] + b.v1[axis] + b.v2[axis]);
                     });

    // Ajouter les enfants à la pile
    stack.push({mid, end, depth + 1, currentIndex, false});
    stack.push({start, mid, depth + 1, currentIndex, true});
  }
}

//...

void buildBVH_CPU_Iterative_Memory_Unified(Triangle *triangles,
                                           int numTriangles, BVHNode *bvhNodes,
                                           int &numNodes) {
  numNodes = 0;

  struct StackEntry {
    int start, end, depth;
    int parentIndex;
    bool isLeftChild;
  };

  std::vector<StackEntry> stack;
  stack.push_back({0, numTriangles, 0, -1, false});

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
      const auto &tri = triangles[i];
      node.bounds.min = min(node.bounds.min, min(tri.v0, min(tri.v1, tri.v2)));
      node.bounds.max = max(node.bounds.max, max(tri.v0, max(tri.v1, tri.v2)));
    }

    int currentIndex = numNodes;
    bvhNodes[numNodes++] = node;

    if (parentIndex != -1) {
      if (isLeftChild) {
        bvhNodes[parentIndex].leftChild = currentIndex;
      } else {
        bvhNodes[parentIndex].rightChild = currentIndex;
      }
    }

    if (node.triangleCount <= 4 || depth > 20) {
      continue;
    }

    Vec3 extent = node.bounds.max - node.bounds.min;
    int axis = 0;
    if (extent.y > extent.x)
      axis = 1;
    if (extent.z > extent[axis])
      axis = 2;

    int mid = (start + end) / 2;
    std::nth_element(triangles + start, triangles + mid, triangles + end,
                     [axis](const Triangle &a, const Triangle &b) {
                       return (a.v0[axis] + a.v1[axis] + a.v2[axis]) <
                              (b.v0[axis] + b.v1[axis] + b.v2[axis]);
                     });

    stack.push_back({mid, end, depth + 1, currentIndex, false});
    stack.push_back({start, mid, depth + 1, currentIndex, true});
  }

  printf("numNodes=%i\n", numNodes);

  for (int i = 0; i < numNodes; i++) {
    printf("Node %i \n", i);
    printf("    left  : %i\n", bvhNodes[i].leftChild);
    printf("    right : %i\n", bvhNodes[i].rightChild);
    printf("    firstTriangleIndex : %i\n", bvhNodes[i].firstTriangleIndex);
    printf("    triangleCount : %i\n", bvhNodes[i].triangleCount);

    printf("    Bounds : \n");
    printf("      Min: : %f %f %f\n", bvhNodes[i].bounds.min.x,
           bvhNodes[i].bounds.min.y, bvhNodes[i].bounds.min.z);
    printf("      Max: : %f %f %f\n", bvhNodes[i].bounds.max.x,
           bvhNodes[i].bounds.max.y, bvhNodes[i].bounds.max.z);
  }
}

// END::BVH CPU
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

//*******************************************************************************************************************************
// BEGIN::BVH GPU 1

void verifyBVH(BVHNode *h_nodes, int numNodes) {
  for (int i = 0; i < numNodes; i++) {
    BVHNode &node = h_nodes[i];
    if (node.leftChild != -1) {

      std::cout << "[" << i
                << "] Internal Node: Left Child = " << node.leftChild
                << ", Right Child = " << node.rightChild;
      assert(node.leftChild < i);
      assert(node.rightChild < i);
      assert(node.triangleCount == h_nodes[node.leftChild].triangleCount +
                                       h_nodes[node.rightChild].triangleCount);

      std::cout << " triangleCount = " << node.triangleCount << std::endl;
      assert(node.leftChild < i);
    } else {
      assert(node.triangleCount == 1);
    }
  }
  printf("BVH verification passed.\n");
}

__global__ void buildBVHLevel(BVHNode *nodes, int levelSize, int levelOffset,
                              int totalNodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= levelSize / 2)
    return;

  int parentIdx = totalNodes - levelOffset - 1 - idx;
  int leftChildIdx = totalNodes - levelOffset * 2 - 1 - idx * 2;
  int rightChildIdx = leftChildIdx - 1;

  if (leftChildIdx < 0 || rightChildIdx < 0)
    return;

  BVHNode &parent = nodes[parentIdx];
  BVHNode &leftChild = nodes[leftChildIdx];
  BVHNode &rightChild = nodes[rightChildIdx];

  parent.bounds.min = min(leftChild.bounds.min, rightChild.bounds.min);
  parent.bounds.max = max(leftChild.bounds.max, rightChild.bounds.max);
  parent.leftChild = leftChildIdx;
  parent.rightChild = rightChildIdx;
  parent.firstPrimitive = leftChild.firstPrimitive;
  parent.primitiveCount = leftChild.primitiveCount + rightChild.primitiveCount;
  parent.triangleCount = leftChild.triangleCount + rightChild.triangleCount;
  parent.firstTriangleIndex = leftChild.firstTriangleIndex;
}

__global__ void buildBVHInitNodes(BVHNode *nodes, Triangle *triangles,
                                  int numTriangles, int totalNodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalNodes)
    return;

  BVHNode &node = nodes[idx];
  if (idx < numTriangles) {
    Triangle &tri = triangles[idx];
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

void buildBVH_GPU_Version1(Triangle *d_triangles, BVHNode *d_nodes,
                           int numTriangles) {
  int numThreads = 256;
  int totalNodes = 2 * numTriangles - 1;
  int numBlocks = (totalNodes + numThreads - 1) / numThreads;

  // Initialiser les nœuds feuilles
  buildBVHInitNodes<<<numBlocks, numThreads>>>(d_nodes, d_triangles,
                                               numTriangles, totalNodes);
  hipDeviceSynchronize();

  // Construire les niveaux supérieurs
  int levelSize = numTriangles;
  int levelOffset = 0;

  while (levelSize > 1) {
    int numParents = (levelSize + 1) / 2;
    numBlocks = (numParents + numThreads - 1) / numThreads;
    buildBVHLevel<<<numBlocks, numThreads>>>(d_nodes, levelSize, levelOffset,
                                             totalNodes);
    hipDeviceSynchronize();

    levelSize = numParents;
    levelOffset += numParents;
  }

  // Verification 2
  BVHNode *h_nodes = new BVHNode[2 * numTriangles - 1];
  HIP_ASSERT(hipMemcpy(h_nodes, d_nodes,
                       (2 * numTriangles - 1) * sizeof(BVHNode),
                       hipMemcpyDeviceToHost));
  verifyBVH(h_nodes, 2 * numTriangles - 1);
  delete[] h_nodes;
}

// END::GPU 1
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

//*******************************************************************************************************************************
// BEGIN::BVH GPU 2

__host__ __device__ __inline__ void
calculateBoundingBox(const Triangle &triangle, Vec3 &min_values,
                     Vec3 &max_values) {
  min_values = min(triangle.v0, min(triangle.v1, triangle.v2));
  max_values = max(triangle.v0, max(triangle.v1, triangle.v2));
}

__global__ void initializeLeaves(Triangle *triangles, BVHNode *nodes,
                                 int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    BVHNode &node = nodes[numTriangles - 1 + idx];
    calculateBoundingBox(triangles[idx], node.bounds.min, node.bounds.max);
    node.triangleIndex = idx;
    node.leftChild = node.rightChild = -1;
    node.firstTriangleIndex = idx;
    node.triangleCount = 1;
  }
}

void buildBVH_GPU_Version2(Triangle *d_triangles, BVHNode *d_nodes,
                           int numTriangles) {
  int totalNodes = 2 * numTriangles - 1;
  int blockSize = 256;
  int numBlocks = (numTriangles + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(initializeLeaves, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_triangles, d_nodes, numTriangles);

  BVHNode *h_nodes = new BVHNode[2 * numTriangles - 1];
  HIP_ASSERT(hipMemcpy(h_nodes, d_nodes,
                       (2 * numTriangles - 1) * sizeof(BVHNode),
                       hipMemcpyDeviceToHost));

  for (int i = numTriangles - 2; i >= 0; --i) {
    BVHNode &node = h_nodes[i];
    int leftChild = 2 * i + 1;
    int rightChild = 2 * i + 2;
    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;

    BVHNode &leftNode = h_nodes[leftChild];
    BVHNode &rightNode = h_nodes[rightChild];
    node.bounds.min = min(leftNode.bounds.min, rightNode.bounds.min);
    node.bounds.max = max(leftNode.bounds.max, rightNode.bounds.max);
  }
  HIP_ASSERT(hipMemcpy(d_nodes, h_nodes,
                       (2 * numTriangles - 1) * sizeof(BVHNode),
                       hipMemcpyHostToDevice));
  delete[] h_nodes;
}

__global__ void buildEvaluationNodes(BVHNode *nodes, int numTriangles) {
  for (int i = numTriangles - 2; i >= 0; --i) {
    BVHNode &node = nodes[i];
    int leftChild = 2 * i + 1;
    int rightChild = 2 * i + 2;
    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;
    BVHNode &leftNode = nodes[leftChild];
    BVHNode &rightNode = nodes[rightChild];
    node.bounds.min = min(leftNode.bounds.min, rightNode.bounds.min);
    node.bounds.max = max(leftNode.bounds.max, rightNode.bounds.max);
  }
  //__syncthreads();
}

void buildBVH_GPU_Version3(Triangle *d_triangles, BVHNode *d_nodes,
                           int numTriangles) {
  int blockSize = 256;
  int numBlocks = (numTriangles + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(initializeLeaves, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_triangles, d_nodes, numTriangles);
  hipDeviceSynchronize();
  hipLaunchKernelGGL(buildEvaluationNodes, dim3(1), dim3(1), 0, 0, d_nodes,
                     numTriangles);
  hipDeviceSynchronize();
}

// END::GPU 2
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

__device__ __inline__ bool rayAABBIntersectA(const Ray &ray, const AABB &aabb) {
  Vec3 invDir = Vec3(1.0f / ray.direction.x, 1.0f / ray.direction.y,
                     1.0f / ray.direction.z);
  Vec3 t0 = (aabb.min - ray.origin) * invDir;
  Vec3 t1 = (aabb.max - ray.origin) * invDir;
  Vec3 tmin = min(t0, t1);
  Vec3 tmax = max(t0, t1);
  float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
  float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);
  return tNear <= tFar && tFar > 0;
}

__global__ void raytraceKernel4(Ray *rays, int numRays, BVHNode *bvhNodes,
                                Triangle *triangles, int *hitTriangles,
                                float *distance, Vec3 *intersectionPoint,
                                int *hitId) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRays)
    return;

  Ray ray = rays[idx];
  constexpr int MAX_STACK_SIZE = 64;
  int stack[MAX_STACK_SIZE];
  int stackPtr = 0;

  float closestT = INFINITY;
  int closestTriangle = -1;
  int closestIntersectionId = -1;
  Vec3 closestIntersectionPoint = Vec3(INFINITY);

  stack[stackPtr++] = 0;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    const BVHNode &node = bvhNodes[nodeIdx];

    if (!rayAABBIntersectA(ray, node.bounds))
      continue;

    if (node.triangleCount > 0) {
      for (int i = 0; i < node.triangleCount; ++i) {
        Triangle &tri = triangles[node.triangleIndex + i];

        //if (pointInTriangle(ray.origin,tri)) // not ok.
        {
          float t;
          Vec3 intersectionPointT;

          if (rayTriangleIntersect4(ray, tri, t, intersectionPointT)) {
            // printf("BING\n");
            if (t < closestT) {
              closestT = t;
              closestTriangle = node.triangleIndex + i;
              closestIntersectionPoint = intersectionPointT;
              closestIntersectionId = tri.id;
            }
          }
        }
      }
    } else {
      if (node.leftChild >= 0)
        stack[stackPtr++] = node.leftChild;
      if (node.rightChild >= 0)
        stack[stackPtr++] = node.rightChild;
    }

    if (stackPtr >= MAX_STACK_SIZE)
      break;
  }

  hitTriangles[idx] = closestTriangle;
  distance[idx] = closestT;
  intersectionPoint[idx] = closestIntersectionPoint;
  hitId[idx] = closestIntersectionId;
}

__global__ void initializeLeaves4(Triangle *triangles, BVHNode *nodes,
                                  int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    BVHNode &node = nodes[numTriangles - 1 + idx];
    Triangle &tri = triangles[idx];

    node.bounds.min = min(tri.v0, min(tri.v1, tri.v2));
    node.bounds.max = max(tri.v0, max(tri.v1, tri.v2));
    node.triangleIndex = idx;
    node.triangleCount = 1;
    node.leftChild = -1;
    node.rightChild = -1;
  }
}

__global__ void buildInternalNodes4(BVHNode *nodes, int numInternalNodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInternalNodes) {
    int nodeIndex = numInternalNodes - 1 - idx;
    BVHNode &node = nodes[nodeIndex];

    int leftChild = 2 * nodeIndex + 1;
    int rightChild = 2 * nodeIndex + 2;

    if (leftChild < 2 * numInternalNodes - 1) {
      BVHNode &leftNode = nodes[leftChild];
      BVHNode &rightNode = nodes[rightChild];
      node.bounds.min = min(leftNode.bounds.min, rightNode.bounds.min);
      node.bounds.max = max(leftNode.bounds.max, rightNode.bounds.max);
      node.triangleCount = 0;
      node.leftChild = leftChild;
      node.rightChild = rightChild;
      node.triangleIndex = -1;
    }
  }
}

void buildBVH_GPU_Version4(Triangle *d_triangles, BVHNode *d_nodes,
                           int numTriangles) {
  int totalNodes = 2 * numTriangles - 1;
  int blockSize = 512;

  printf("buildBVH_GPU_Version4\n");

  // Initialize leaf nodes in parallel
  int numBlocksLeaves = (numTriangles + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(initializeLeaves4, dim3(numBlocksLeaves), dim3(blockSize),
                     0, 0, d_triangles, d_nodes, numTriangles);
  hipDeviceSynchronize();

  // Build internal nodes in parallel
  int numInternalNodes = numTriangles - 1;
  int numBlocksInternal = (numInternalNodes + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(buildInternalNodes4, dim3(numBlocksInternal),
                     dim3(blockSize), 0, 0, d_nodes, numInternalNodes);
  hipDeviceSynchronize();
}

// END::BVH GPU

//*******************************************************************************************************************************
// BEGIN::BVH GPU 3

__device__ __inline__ bool compareTriangles(const TriangleInfo &a,
                                            const TriangleInfo &b, int axis) {
  return a.centroid[axis] < b.centroid[axis];
}

__global__ void initTriangleInfo(Triangle *triangles, TriangleInfo *triInfo,
                                 int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    Triangle &tri = triangles[idx];
    triInfo[idx].centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    triInfo[idx].index = idx;
  }
}

__global__ void bitonicSort(TriangleInfo *triInfo, int j, int k,
                            int numTriangles, int axis) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int ixj = i ^ j;

  if ((ixj > i) && (i < numTriangles) && (ixj < numTriangles)) {
    bool ascending = ((i & k) == 0);
    if (compareTriangles(triInfo[i], triInfo[ixj], axis) == ascending) {
      TriangleInfo temp = triInfo[i];
      triInfo[i] = triInfo[ixj];
      triInfo[ixj] = temp;
    }
  }
}

__global__ void buildBVHNodes(BVHNode *nodes, TriangleInfo *triInfo,
                              Triangle *triangles, int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalNodes = 2 * numTriangles - 1;

  if (idx < totalNodes) {
    BVHNode &node = nodes[idx];

    if (idx >= numTriangles - 1) { // Leaf node
      int triIdx = triInfo[idx - (numTriangles - 1)].index;
      node.bounds.min = node.bounds.max =
          triangles[triIdx].v0; // Initialiser avec le premier sommet

      node.bounds.min =
          min(node.bounds.min, min(triangles[triIdx].v1, triangles[triIdx].v2));
      node.bounds.max =
          max(node.bounds.max, max(triangles[triIdx].v1, triangles[triIdx].v2));

      // A voir
      // node.bounds.min = Vec3::min(node.bounds.min,
      // Vec3::min(triangles[triIdx].v1, triangles[triIdx].v2)); node.bounds.max
      // = Vec3::max(node.bounds.max, Vec3::max(triangles[triIdx].v1,
      // triangles[triIdx].v2));
      //...

      node.triangleIndex = triIdx; // Index du triangle
      node.triangleCount = 1;      // Compte des triangles dans ce nœud
      node.leftChild = node.rightChild = -1; // Pas d'enfants pour les feuilles
    } else {                                 // Internal node
      int leftChild = 2 * idx + 1;
      int rightChild = 2 * idx + 2;
      node.leftChild = leftChild;
      node.rightChild = rightChild;
      node.triangleIndex = -1; // Pas de triangle ici

      node.bounds.min =
          min(nodes[leftChild].bounds.min, nodes[rightChild].bounds.min);
      node.bounds.max =
          max(nodes[leftChild].bounds.max, nodes[rightChild].bounds.max);

      // A voir
      // node.bounds.min = Vec3::min(nodes[leftChild].bounds.min,
      // nodes[rightChild].bounds.min); node.bounds.max =
      // Vec3::max(nodes[leftChild].bounds.max, nodes[rightChild].bounds.max);
      //...

      node.triangleCount =
          nodes[leftChild].triangleCount + nodes[rightChild].triangleCount;
    }
  }
}

__global__ void buildBVHNodesParallel(BVHNode *nodes, TriangleInfo *triInfo,
                                      Triangle *triangles, int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalNodes = 2 * numTriangles - 1;

  if (idx < totalNodes) {
    BVHNode &node = nodes[idx];

    if (idx >= numTriangles - 1) { // Leaf node
      int triIdx = triInfo[idx - (numTriangles - 1)].index;
      // Initialization of the node limits with the first top of the triangle
      node.bounds.min = node.bounds.max = triangles[triIdx].v0;

      // Calculation of AABB limits for the triangle
      node.bounds.min =
          min(node.bounds.min, min(triangles[triIdx].v1, triangles[triIdx].v2));
      node.bounds.max =
          max(node.bounds.max, max(triangles[triIdx].v1, triangles[triIdx].v2));
      node.triangleIndex = triIdx;
      node.triangleCount = 1;
      node.leftChild = -1;  // no children for a leaf knot
      node.rightChild = -1; // no children for a leaf knot
    } else {                // Internal node
      int leftChild = 2 * idx + 1;
      int rightChild = 2 * idx + 2;

      // Verification that children exist before accessing their limits
      if (leftChild < totalNodes && rightChild < totalNodes) {
        node.leftChild = leftChild;
        node.rightChild = rightChild;
        node.triangleIndex = -1; // no associated triangle

        // Calculation of AABB limits encompassing children
        node.bounds.min =
            min(nodes[leftChild].bounds.min, nodes[rightChild].bounds.min);
        node.bounds.max =
            max(nodes[leftChild].bounds.max, nodes[rightChild].bounds.max);
        node.triangleCount =
            nodes[leftChild].triangleCount + nodes[rightChild].triangleCount;
      } else {
        // If children do not exist (rare case), we can reset the node with
        // default values
        node.bounds.min = Vec3{0.0f, 0.0f, 0.0f}; // Default values
        node.bounds.max = Vec3{0.0f, 0.0f, 0.0f}; // Default values
        node.triangleIndex = -1;
        node.triangleCount = 0;
        node.leftChild = -1;
        node.rightChild = -1;
      }
    }
  }
}

__global__ void buildBVHNodesParallelBeta(BVHNode *nodes, TriangleInfo *triInfo,
                                          Triangle *triangles,
                                          int numTriangles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int totalNodes = 2 * numTriangles - 1;

  if (idx >= totalNodes)
    return;

  BVHNode &node = nodes[idx];

  if (idx >= numTriangles - 1) { // Leaf node
    int triIdx = idx - (numTriangles - 1);
    Triangle &tri = triangles[triInfo[triIdx].index];

    // Initialize node bounds with the first vertex of the triangle
    node.bounds.min = node.bounds.max = tri.v0;

    // Expand bounds to include all vertices
    node.bounds.min = min(node.bounds.min, min(tri.v1, tri.v2));
    node.bounds.max = max(node.bounds.max, max(tri.v1, tri.v2));

    node.triangleIndex = triInfo[triIdx].index;
    node.triangleCount = 1;
    node.leftChild = node.rightChild = -1;
  } else { // Internal node
    int leftChild = 2 * idx + 1;
    int rightChild = 2 * idx + 2;

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.triangleIndex = -1;
    node.triangleCount = 0;

    // Initialize with invalid bounds
    node.bounds.min = Vec3(INFINITY, INFINITY, INFINITY);
    node.bounds.max = Vec3(-INFINITY, -INFINITY, -INFINITY);

    if (leftChild < totalNodes) {
      node.bounds = nodes[leftChild].bounds;
      node.triangleCount = nodes[leftChild].triangleCount;

      if (rightChild < totalNodes) {
        // Expand bounds to include right child
        node.bounds.min = min(node.bounds.min, nodes[rightChild].bounds.min);
        node.bounds.max = max(node.bounds.max, nodes[rightChild].bounds.max);
        node.triangleCount += nodes[rightChild].triangleCount;
      }
    }
  }
}

__global__ void computeExtents(TriangleInfo *triInfo, int numTriangles,
                               float *minExtents, float *maxExtents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTriangles) {
    for (int axis = 0; axis < 3; ++axis) {
      atomicMin(&minExtents[axis], triInfo[idx].centroid[axis]);
      atomicMax(&maxExtents[axis], triInfo[idx].centroid[axis]);
    }
  }
}

void buildBVH_GPU_Parallel_Best_Axis(Triangle *d_triangles, BVHNode *d_nodes,
                                     int numTriangles) {
  std::cout << "[INFO]: buildBVH_GPU_Parallel\n";

  int totalNodes = 2 * numTriangles - 1;
  int blockSize = 512;
  int numBlocks = (numTriangles + blockSize - 1) / blockSize;

  TriangleInfo *d_triInfo;
  HIP_ASSERT(hipMalloc(&d_triInfo, numTriangles * sizeof(TriangleInfo)));
  hipLaunchKernelGGL(initTriangleInfo, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_triangles, d_triInfo, numTriangles);

  // Compute extents
  float *d_minExtents, *d_maxExtents;
  HIP_ASSERT(hipMalloc(&d_minExtents, 3 * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_maxExtents, 3 * sizeof(float)));

  // Initialize extents
  float initMin = std::numeric_limits<float>::max();
  float initMax = std::numeric_limits<float>::lowest();
  HIP_ASSERT(hipMemset(d_minExtents, *reinterpret_cast<int *>(&initMin),
                       3 * sizeof(float)));
  HIP_ASSERT(hipMemset(d_maxExtents, *reinterpret_cast<int *>(&initMax),
                       3 * sizeof(float)));

  hipLaunchKernelGGL(computeExtents, dim3(numBlocks), dim3(blockSize), 0, 0,
                     d_triInfo, numTriangles, d_minExtents, d_maxExtents);

  // Copy extents back to host
  float h_minExtents[3], h_maxExtents[3];
  HIP_ASSERT(hipMemcpy(h_minExtents, d_minExtents, 3 * sizeof(float),
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_maxExtents, d_maxExtents, 3 * sizeof(float),
                       hipMemcpyDeviceToHost));

  // Find axis with largest extent
  int bestAxis = 0;
  float maxExtent = h_maxExtents[0] - h_minExtents[0];
  for (int axis = 1; axis < 3; ++axis) {
    float extent = h_maxExtents[axis] - h_minExtents[axis];
    if (extent > maxExtent) {
      maxExtent = extent;
      bestAxis = axis;
    }
  }

  // Sort using the best axis
  for (int k = 2; k <= numTriangles; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      hipLaunchKernelGGL(bitonicSort, dim3(numBlocks), dim3(blockSize), 0, 0,
                         d_triInfo, j, k, numTriangles, bestAxis);
    }
  }

  numBlocks = (totalNodes + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(buildBVHNodesParallel, dim3(numBlocks), dim3(blockSize), 0,
                     0, d_nodes, d_triInfo, d_triangles, numTriangles);

  hipFree(d_triInfo);
  hipFree(d_minExtents);
  hipFree(d_maxExtents);
}

// END::GPU 3
//-------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************

//*******************************************************************************************************************************

// Load OBJ
void loadTrianglesFromOBJ(const std::string &filename,
                          std::vector<Triangle> &triangles, int id) {
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
    } else if (type == "f") {
      int v1, v2, v3;
      iss >> v1 >> v2 >> v3;
      triangles.push_back(
          {vertices[v1 - 1], vertices[v2 - 1], vertices[v3 - 1]});
      triangles.back().id = id;
    }
  }

  // std::cout << "Chargé " << triangles.size() << " triangles depuis " <<
  // filename << std::endl;
}

// Generate Rays
void generateRays(std::vector<Ray> &rays, int numRays, const Vec3 &origin,
                  float fov, int imageWidth, int imageHeight) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  float aspectRatio = static_cast<float>(imageWidth) / imageHeight;
  float tanHalfFov = std::tan(fov * 0.5f * M_PI / 180.0f);

  for (int i = 0; i < numRays; ++i) {
    float x = (2.0f * (i % imageWidth) + 1.0f) / (2.0f * imageWidth) - 1.0f;
    float y = (1.0f - (2.0f * (i / imageWidth) + 1.0f) / (2.0f * imageHeight)) *
              (1.0f / aspectRatio);

    Vec3 direction(x * tanHalfFov * aspectRatio, y * tanHalfFov, -1.0f);

    float length =
        std::sqrt(direction.x * direction.x + direction.y * direction.y +
                  direction.z * direction.z);
    direction.x /= length;
    direction.y /= length;
    direction.z /= length;

    rays.push_back({origin, direction});
  }

  // std::cout << "Généré " << rays.size() << " rayons" << std::endl;
}

void Test001(int mode) {
  std::chrono::steady_clock::time_point t_begin_0, t_begin_1, t_begin_2;
  std::chrono::steady_clock::time_point t_end_0, t_end_1, t_end_2;
  long int t_laps;

  std::vector<Triangle> triangles;
  std::vector<Ray> rays;
  std::vector<BVHNode> bvhNodes;

  // loadTrianglesFromOBJ("Triangle2Cube.obj", triangles, 12321);
  loadTrianglesFromOBJ("Test.obj", triangles, 12321);

  Vec3 cameraOrigin(0.75, 0.75, 5.0);
  float fov = 40.0f;
  int imageWidth = 10;
  int imageHeight = imageWidth;
  int numRays = imageWidth * imageHeight;
  generateRays(rays, numRays, cameraOrigin, fov, imageWidth, imageHeight);

  Triangle *deviceTriangles;
  Ray *deviceRays;
  BVHNode *devicebvhNodes;
  int *deviceHitTriangles;
  Vec3 *deviceIntersectionPoint;
  float *deviceDistanceResults;
  int *deviceIdResults;

  HIP_ASSERT(hipMalloc(&deviceTriangles, triangles.size() * sizeof(Triangle)));
  HIP_ASSERT(hipMemcpy(deviceTriangles, triangles.data(),
                       triangles.size() * sizeof(Triangle),
                       hipMemcpyHostToDevice));

  bool isUnifiedMemory = false;
  bool isGPU = false;
  isGPU = true;

  int nbTriangles = triangles.size();
  int nbNodes = 2 * nbTriangles - 1;

  if (mode == 0)
    isGPU = false;

  if (!isGPU) {
    t_begin_0 = std::chrono::steady_clock::now();
    buildBVH_CPU_Recursive(triangles, bvhNodes);
    // buildBVH_CPU_Iterative(triangles, bvhNodes);
    t_end_0 = std::chrono::steady_clock::now();
    if (1 == 0)
      writeBVHNodes(bvhNodes);
    HIP_ASSERT(hipMalloc(&devicebvhNodes, bvhNodes.size() * sizeof(BVHNode)));
    HIP_ASSERT(hipMemcpy(devicebvhNodes, bvhNodes.data(),
                         bvhNodes.size() * sizeof(BVHNode),
                         hipMemcpyHostToDevice));
  }

  if (isGPU) {
    int numTriangles = triangles.size();
    HIP_ASSERT(
        hipMalloc(&devicebvhNodes, (2 * numTriangles - 1) * sizeof(BVHNode)));
    t_begin_0 = std::chrono::steady_clock::now();
    // buildBVH_GPU_Version1(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 2)
      buildBVH_GPU_Version2(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 3)
      buildBVH_GPU_Version3(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 44)
      buildBVH_GPU_Version4(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 5)
      buildBVH_GPU_Parallel_Best_Axis(deviceTriangles, devicebvhNodes,
                                      numTriangles);
    t_end_0 = std::chrono::steady_clock::now();

    // hipMemcpy(nodes, d_nodes, (2 * numTriangles - 1) * sizeof(BVHNode),
    // hipMemcpyDeviceToHost);

    // CTRL::BEGIN
    if (1 == 0) {
      printf("**********************************\n");
      int totalNodes = (2 * numTriangles - 1);
      for (int i = 0; i < totalNodes; i++) {
        printf("Node %i \n", i);
        printf("    left  : %i\n", devicebvhNodes[i].leftChild);
        printf("    right : %i\n", devicebvhNodes[i].rightChild);

        printf("    Bounds : \n");
        printf("      Min: : %f %f %f\n", devicebvhNodes[i].bounds.min.x,
               devicebvhNodes[i].bounds.min.y, devicebvhNodes[i].bounds.min.z);
        printf("      Max: : %f %f %f\n", devicebvhNodes[i].bounds.max.x,
               devicebvhNodes[i].bounds.max.y, devicebvhNodes[i].bounds.max.z);
      }
    }
    // CTRL::END
  }

  std::cout << "[Ray Tracing]\n";
  HIP_ASSERT(hipMalloc(&deviceRays, rays.size() * sizeof(Ray)));
  HIP_ASSERT(hipMemcpy(deviceRays, rays.data(), rays.size() * sizeof(Ray),
                       hipMemcpyHostToDevice));

  HIP_ASSERT(hipMalloc(&deviceHitTriangles, rays.size() * sizeof(int)));
  HIP_ASSERT(hipMalloc(&deviceIntersectionPoint, rays.size() * sizeof(Vec3)));
  HIP_ASSERT(hipMalloc(&deviceDistanceResults, rays.size() * sizeof(float)));
  HIP_ASSERT(hipMalloc(&deviceIdResults, rays.size() * sizeof(int)));

  t_begin_1 = std::chrono::steady_clock::now();
  int blockSize = 512;
  int numBlocks = (rays.size() + blockSize - 1) / blockSize;

  if (mode == 44) {
    hipLaunchKernelGGL(raytraceKernel4, dim3(numBlocks), dim3(blockSize), 0, 0,
                       deviceRays, rays.size(), devicebvhNodes, deviceTriangles,
                       deviceHitTriangles, deviceDistanceResults,
                       deviceIntersectionPoint, deviceIdResults);
  }

  else if (mode == 4) {

  } else if (mode == 5) {
    hipLaunchKernelGGL(raytraceKernel_Parallel005, dim3(numBlocks),
                       dim3(blockSize), 0, 0, deviceRays, numRays,
                       devicebvhNodes, nbNodes, deviceTriangles, nbTriangles,
                       deviceHitTriangles, deviceDistanceResults,
                       deviceIntersectionPoint, deviceIdResults);
  } else {
    hipLaunchKernelGGL(raytraceKernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       deviceRays, rays.size(), devicebvhNodes, deviceTriangles,
                       deviceHitTriangles, deviceDistanceResults,
                       deviceIntersectionPoint, deviceIdResults);
  }

  t_end_1 = std::chrono::steady_clock::now();

  std::vector<int> hosthitTriangles(rays.size());
  HIP_ASSERT(hipMemcpy(hosthitTriangles.data(), deviceHitTriangles,
                       rays.size() * sizeof(int), hipMemcpyDeviceToHost));

  std::vector<Vec3> hostHipIntersectionPoint(rays.size());
  HIP_ASSERT(hipMemcpy(hostHipIntersectionPoint.data(), deviceIntersectionPoint,
                       rays.size() * sizeof(Vec3), hipMemcpyDeviceToHost));

  std::vector<float> hostHipDistanceResults(rays.size());
  HIP_ASSERT(hipMemcpy(hostHipDistanceResults.data(), deviceDistanceResults,
                       rays.size() * sizeof(float), hipMemcpyDeviceToHost));

  std::vector<int> hostHipIdResults(rays.size());
  HIP_ASSERT(hipMemcpy(hostHipIdResults.data(), deviceIdResults,
                       rays.size() * sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 0; i < rays.size(); ++i) {
    if (hostHipIdResults[i] != -1) {
      std::cout << " dist (min)=" << hostHipDistanceResults[i];
      std::cout << " IntersectionPoint= <" << hostHipIntersectionPoint[i].x
                << "," << hostHipIntersectionPoint[i].y << ","
                << hostHipIntersectionPoint[i].z << "> ";
      std::cout <<"\n";
    }
  }

  hipFree(deviceTriangles);
  hipFree(deviceRays);
  hipFree(devicebvhNodes);
  hipFree(deviceHitTriangles);

  t_laps =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0)
          .count();
  std::cout << "[INFO]: Elapsed microseconds inside BVH : " << t_laps
            << " us\n";

  t_laps =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1)
          .count();
  std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : " << t_laps
            << " us\n";
}

void Test002(int mode) {
  std::chrono::steady_clock::time_point t_begin_0, t_begin_1, t_begin_2;
  std::chrono::steady_clock::time_point t_end_0, t_end_1, t_end_2;
  long int t_laps;

  std::vector<Triangle> triangles;

  std::vector<BVHNode> bvhNodes;

  loadTrianglesFromOBJ("Test.obj", triangles, 12321);
  // loadTrianglesFromOBJ("Triangle2Cube.obj", triangles, 12321);

  int nbTriangles = triangles.size();
  int nbNodes = 2 * nbTriangles - 1;

  /*
    std::vector<Ray> rays(7);
    rays[0].origin = Vec3(0.0, 0.0, 1.0);
    rays[0].direction = Vec3(0.0f, 0.0f, -1.0f);
    normalizeRayDirection(rays[0]);

    rays[1].origin = Vec3(0.0, 0.0, 1.00001);
    rays[1].direction = Vec3(0.0f, 0.0f, -1.0f);
    normalizeRayDirection(rays[1]);

    rays[2].origin = Vec3(0.0, 0.0, 11.0);
    rays[2].direction = Vec3(0.0f, 0.0f, -1.0f);
    normalizeRayDirection(rays[2]);

    rays[3].origin = Vec3(0.5, 0.5, 21.0);
    rays[3].direction = Vec3(0.0f, 0.0f, -1.0f);
    normalizeRayDirection(rays[3]);

    rays[4].origin = Vec3(1.0, 1.0, 1.0);
    rays[4].direction = Vec3(0.0f, 0.0f, -1.0f);
    normalizeRayDirection(rays[4]);

    rays[5].origin = Vec3(1.0, 0.0, 1.0);
    rays[5].direction = Vec3(1.0f, 0.0f,0.0f);
    normalizeRayDirection(rays[5]);

    rays[6].origin = Vec3(0.9, 0.9, 0.9);
    rays[6].direction = Vec3(1.0f, 0.0f,0.0f);
    normalizeRayDirection(rays[6]);
    */

  std::vector<Ray> rays(1);
  rays[0].origin = Vec3(0.9, 0.9, 0.9);
  rays[0].direction = Vec3(1.0f, 0.0f, 0.0f);
  normalizeRayDirection(rays[0]);

  int numRays = rays.size();

  Triangle *deviceTriangles;
  Ray *deviceRays;
  BVHNode *devicebvhNodes;
  int *deviceHitTriangles;
  Vec3 *deviceIntersectionPoint;
  float *deviceDistanceResults;
  int *deviceIdResults;

  HIP_ASSERT(hipMalloc(&deviceTriangles, triangles.size() * sizeof(Triangle)));
  HIP_ASSERT(hipMemcpy(deviceTriangles, triangles.data(),
                       triangles.size() * sizeof(Triangle),
                       hipMemcpyHostToDevice));

  bool isUnifiedMemory = false;
  bool isGPU = false;
  isGPU = true;

  if (mode == 0)
    isGPU = false;

  if (!isGPU) {
    t_begin_0 = std::chrono::steady_clock::now();
    buildBVH_CPU_Recursive(triangles, bvhNodes);
    // buildBVH_CPU_Iterative(triangles, bvhNodes);
    t_end_0 = std::chrono::steady_clock::now();
    if (1 == 0)
      writeBVHNodes(bvhNodes);
    HIP_ASSERT(hipMalloc(&devicebvhNodes, bvhNodes.size() * sizeof(BVHNode)));
    HIP_ASSERT(hipMemcpy(devicebvhNodes, bvhNodes.data(),
                         bvhNodes.size() * sizeof(BVHNode),
                         hipMemcpyHostToDevice));
  }

  if (isGPU) {
    int numTriangles = triangles.size();
    HIP_ASSERT(
        hipMalloc(&devicebvhNodes, (2 * numTriangles - 1) * sizeof(BVHNode)));
    t_begin_0 = std::chrono::steady_clock::now();
    // buildBVH_GPU_Version1(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 2)
      buildBVH_GPU_Version2(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 3)
      buildBVH_GPU_Version3(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 44)
      buildBVH_GPU_Version4(deviceTriangles, devicebvhNodes, numTriangles);
    if (mode == 5)
      buildBVH_GPU_Parallel_Best_Axis(deviceTriangles, devicebvhNodes,
                                      numTriangles);
    hipDeviceSynchronize();
    t_end_0 = std::chrono::steady_clock::now();

    // hipMemcpy(nodes, d_nodes, (2 * numTriangles - 1) * sizeof(BVHNode),
    // hipMemcpyDeviceToHost);

    // CTRL::BEGIN
    if ((1 == 0) && (mode == 5)) {
      printf("**********************************\n");
      int totalNodes = (2 * numTriangles - 1);
      for (int i = 0; i < totalNodes; i++) {
        printf("Node %i \n", i);
        printf("    left   : %i\n", devicebvhNodes[i].leftChild);
        printf("    right  : %i\n", devicebvhNodes[i].rightChild);

        printf("    Bounds : \n");
        printf("      Min  : %f %f %f\n", devicebvhNodes[i].bounds.min.x,
               devicebvhNodes[i].bounds.min.y, devicebvhNodes[i].bounds.min.z);
        printf("      Max  : %f %f %f\n", devicebvhNodes[i].bounds.max.x,
               devicebvhNodes[i].bounds.max.y, devicebvhNodes[i].bounds.max.z);

        printf("    TriangleCount: %i\n", devicebvhNodes[i].triangleCount);
        // node.triangleCount
      }
    }
    // CTRL::END
  }

  std::cout << "[Ray Tracing]\n";
  HIP_ASSERT(hipMalloc(&deviceRays, rays.size() * sizeof(Ray)));
  HIP_ASSERT(hipMemcpy(deviceRays, rays.data(), rays.size() * sizeof(Ray),
                       hipMemcpyHostToDevice));

  HIP_ASSERT(hipMalloc(&deviceHitTriangles, rays.size() * sizeof(int)));
  HIP_ASSERT(hipMalloc(&deviceIntersectionPoint, rays.size() * sizeof(Vec3)));
  HIP_ASSERT(hipMalloc(&deviceDistanceResults, rays.size() * sizeof(float)));
  HIP_ASSERT(hipMalloc(&deviceIdResults, rays.size() * sizeof(int)));

  t_begin_1 = std::chrono::steady_clock::now();
  int blockSize = 512;
  int numBlocks = (rays.size() + blockSize - 1) / blockSize;

  if (mode == 44) {
    hipLaunchKernelGGL(raytraceKernel4, dim3(numBlocks), dim3(blockSize), 0, 0,
                       deviceRays, rays.size(), devicebvhNodes, deviceTriangles,
                       deviceHitTriangles, deviceDistanceResults,
                       deviceIntersectionPoint, deviceIdResults);
  }

  else if (mode == 4) {

  } else if (mode == 5) {

    hipLaunchKernelGGL(raytraceKernel_Parallel005, dim3(numBlocks),
                       dim3(blockSize), 0, 0, deviceRays, numRays,
                       devicebvhNodes, nbNodes, deviceTriangles, nbTriangles,
                       deviceHitTriangles, deviceDistanceResults,
                       deviceIntersectionPoint, deviceIdResults);
  } else {
    hipLaunchKernelGGL(raytraceKernel, dim3(numBlocks), dim3(blockSize), 0, 0,
                       deviceRays, rays.size(), devicebvhNodes, deviceTriangles,
                       deviceHitTriangles, deviceDistanceResults,
                       deviceIntersectionPoint, deviceIdResults);
  }

  hipDeviceSynchronize();

  t_end_1 = std::chrono::steady_clock::now();

  std::vector<int> hosthitTriangles(rays.size());
  HIP_ASSERT(hipMemcpy(hosthitTriangles.data(), deviceHitTriangles,
                       rays.size() * sizeof(int), hipMemcpyDeviceToHost));

  std::vector<Vec3> hostHipIntersectionPoint(rays.size());
  HIP_ASSERT(hipMemcpy(hostHipIntersectionPoint.data(), deviceIntersectionPoint,
                       rays.size() * sizeof(Vec3), hipMemcpyDeviceToHost));

  std::vector<float> hostHipDistanceResults(rays.size());
  HIP_ASSERT(hipMemcpy(hostHipDistanceResults.data(), deviceDistanceResults,
                       rays.size() * sizeof(float), hipMemcpyDeviceToHost));

  std::vector<int> hostHipIdResults(rays.size());
  HIP_ASSERT(hipMemcpy(hostHipIdResults.data(), deviceIdResults,
                       rays.size() * sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 0; i < rays.size(); ++i) {
    if (hostHipIdResults[i] != -1) {
      std::cout << " ray= <" << rays[i].origin.x << "," << rays[i].origin.y
                << "," << rays[i].origin.z << ">";
      std::cout << " dir= <" << rays[i].direction.x << ","
                << rays[i].direction.y << "," << rays[i].direction.z << ">";
      std::cout << " dist (min)=" << hostHipDistanceResults[i];
      std::cout << " IntersectionPoint= <" << hostHipIntersectionPoint[i].x
                << "," << hostHipIntersectionPoint[i].y << ","
                << hostHipIntersectionPoint[i].z << "> ";
      std::cout << "\n";
    }
  }

  hipFree(deviceTriangles);
  hipFree(deviceRays);
  hipFree(devicebvhNodes);
  hipFree(deviceHitTriangles);

  t_laps =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end_0 - t_begin_0)
          .count();
  std::cout << "[INFO]: Elapsed microseconds inside BVH : " << t_laps
            << " us\n";

  t_laps =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end_1 - t_begin_1)
          .count();
  std::cout << "[INFO]: Elapsed microseconds inside Ray Tracing : " << t_laps
            << " us\n";
}

int main() {

  if (1 == 1) {
    std::cout << "\n [INFO]: Test 0\n";
    Test001(0);
    std::cout << "\n [INFO]: Test 2\n";
    Test001(2);
    std::cout << "\n [INFO]: Test 3\n";
    Test001(3);
    std::cout << "\n [INFO]: Test 5\n";
    Test001(5);
    std::cout << "\n [INFO]: Test 44\n";
    Test001(44);
  }

  if (1 == 1) {
    std::cout << "\n [INFO]: Test 0\n";
    Test002(0);
    std::cout << "\n [INFO]: Test 2\n";
    Test002(2);
    std::cout << "\n [INFO]: Test 3\n";
    Test002(3);
    std::cout << "\n [INFO]: Test 5\n";
    Test002(5);
    std::cout << "\n [INFO]: Test 44\n";
    Test002(44);
  }

  return 0;
}