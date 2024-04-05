# K-Means Parallelizaion using CUDA and OpenMP

## Index

1. [Introduction to the Topic](#1-introduction-to-the-topic)
   1.1 [History and Background](#11-history-and-background)
   1.2 [What is K-Means Algorithm](#12-what-is-k-means-algorithm)
   1.3 [How K-Means Works](#13-how-k-means-works)
   1.4 [Applications of K-Means](#14-applications-of-k-means)
   1.5 [Importance of Parallelization](#15-importance-of-parallelization)

2. [Implementation](#2-implementation)
   2.1 [Sequential Code Complexity Analysis](#21-sequential-code-complexity-analysis)
   2.2 [OpenMP Implementation](#22-openmp-implementation)
       2.2.1 [Output](#221-output)
       2.2.2 [Observations](#222-observations)
   2.3 [Parallelization using CUDA](#23-parallelization-using-cuda)
       2.3.1 [Code Implementation](#231-code-implementation)
       2.3.2 [Outputs](#232-outputs)
       2.3.3 [Observations](#233-observations)
3. [System Specification](#24-system-specification)

4. [References](#3-references)

---

# 1. Introduction to the Topic

## 1.1 History and Background

The K-Means algorithm, one of the foundational clustering techniques in machine learning and data mining, was first introduced by Stuart Lloyd in 1957 as a method for pulse-code modulation (PCM) quantization. However, it was later independently rediscovered by James MacQueen in 1967, who introduced the term "K-Means" and formulated it as a clustering problem.

## 1.2 What is K-Means Algorithm

K-Means is an unsupervised learning algorithm used for partitioning a dataset into K distinct clusters. The goal of the algorithm is to minimize the within-cluster sum of squares, also known as inertia or distortion, by iteratively assigning data points to the nearest cluster centroid and updating the centroids to the mean of the data points assigned to each cluster. The process continues until convergence, typically when the centroids no longer change significantly or a specified number of iterations is reached.

## 1.3 How K-Means Works

The K-Means algorithm operates in the following steps:

- Initialization: Randomly initialize K cluster centroids. These centroids serve as the initial representatives of the clusters.
- Assignment: Assign each data point to the nearest cluster centroid based on a distance metric, commonly the Euclidean distance.
- Update: Recalculate the cluster centroids by taking the mean of all data points assigned to each cluster.
- Convergence: Repeat the assignment and update steps iteratively until convergence criteria are met, such as no change in centroids or a maximum number of iterations reached.
- Result: The final cluster centroids represent the centers of the clusters, and each data point belongs to the cluster associated with the nearest centroid.

## 1.4 Applications of K-Means

K-Means clustering finds applications in various fields, including:

- Customer Segmentation: Identifying distinct groups of customers based on their purchasing behavior.
- Image Segmentation: Partitioning images into meaningful regions for object recognition and analysis.
- Anomaly Detection: Identifying outliers or anomalies in datasets based on their deviation from normal behavior.
- Document Clustering: Grouping similar documents together for text mining and information retrieval.
- Genomic Data Analysis: Clustering gene expression data to discover patterns and relationships in biological datasets.

## 1.5 Importance of Parallelization

As datasets continue to grow in size and complexity, the need for efficient algorithms and computational techniques becomes paramount. Parallelization of the K-Means algorithm is essential for handling large datasets and improving performance by leveraging the computational resources of modern hardware architectures, including multicore CPUs and GPUs. In this project, we explore the parallelization of the K-Means algorithm using OpenMP and CUDA, two popular parallel programming paradigms, to achieve scalable and efficient clustering on parallel computing platforms.

# 2. Implementation

## 2.1 Sequential Code Complexity Analysis

The provided sequential K-means clustering algorithm has a complexity of O(I * K * N * D), where:
- I is the number of iterations,
- K is the number of clusters,
- N is the number of data points, and
- D is the dimensionality of each data point.

### Iteration Step Complexity (O(N * K * D))

Each iteration involves two major steps: data point reassignment and centroid update.

1. **Data Point Reassignment:**
   - Involves computing the distance between each data point and each cluster center, resulting in a complexity of O(N * K * D).
   - For N data points, each with D dimensions, and K clusters, the total number of distance computations is N * K * D.

2. **Centroid Update:**
   - Involves computing the mean for each cluster, which requires summing up the coordinates of all data points assigned to the cluster.
   - This operation has a complexity of O((N + K) * D), where (N + K) represents the number of data points and K represents the number of clusters.
   - Since each cluster has D dimensions, the total complexity for updating centroids is (N + K) * D.

### Parallelization Opportunity

Given that the data point reassignment step is more time-consuming, it presents an opportunity for parallelization. This step involves computing the distance between each data point and each cluster center independently, allowing for parallel execution.

In summary, the sequential K-means algorithm involves iterating over the data points and clusters, computing distances, reassigning points, and updating centroids. The complexity analysis helps in understanding the computational demands of the algorithm and identifies the step suitable for parallelization to improve performance.

## 2.2 OpenMP Implementation

Let's break down how the OpenMP directives in the provided code improve performance and how the code works:

1. **Parallelization of Centroid Initialization:**
   - The `#pragma omp parallel for` directive is used to parallelize the initialization of centroids.
   - In the original sequential code, the centroids were initialized in a loop, where each iteration could be executed independently. By adding the OpenMP directive, this loop is parallelized, allowing multiple threads to execute the initialization concurrently.
   - This parallelization leverages the available CPU cores, distributing the workload across threads and reducing the initialization time, especially when the number of clusters (`num_clusters`) is large.

2. **Parallelization of K-means Clustering:**
   - Within the main loop of the `k_means_clustering` function, the bulk of the computation occurs for assigning points to clusters and updating centroids.
   - The `#pragma omp parallel for reduction(+:converged)` directive is used to parallelize the loop responsible for assigning points to clusters.
     ```cpp
     #pragma omp parallel for reduction(+:converged)
     for (int i = 0; i < num_points; i++) {
         // ...
     }
     ```
   - In each iteration of this loop, multiple threads work concurrently to find the closest centroid for each point. The reduction clause ensures that the `converged` variable is updated safely across threads.
   - By parallelizing this computation, the workload is divided among threads, leading to faster execution, especially for datasets with a large number of points (`num_points`).

3. **Atomic Operations for Thread-Safe Updates:**
   - Inside the parallel loop for assigning points to clusters, atomic directives (`#pragma omp atomic`) are used to perform thread-safe updates to shared variables such as `cluster_counts` and `sum`.
   - Atomic operations ensure that concurrent updates by multiple threads do not result in race conditions, preserving the integrity of the data.
   - While atomic operations can introduce some overhead due to synchronization, they are necessary to maintain correctness in parallel execution.

4. **Convergence Check and Early Termination:**
   - After each iteration of assigning points to clusters, the code checks for convergence based on the change in centroids.
   - If the centroids have not changed significantly (`converged == 1`), the loop breaks early, terminating the K-means clustering process.
   - This convergence check helps reduce unnecessary iterations, improving the efficiency of the algorithm, especially when convergence is reached before the maximum number of iterations (`MAX_ITERATIONS`) is reached.

5. **Print Final Centroids:**
   - Once the clustering process completes, the code prints the final centroids for each cluster.
   - These centroids represent the center points of the clusters determined by the K-means algorithm.

Overall, the addition of OpenMP directives parallelizes the initialization of centroids and the main computation loop of the K-means clustering algorithm. This parallelization leverages multicore processors effectively, distributing the workload and reducing execution time, particularly for datasets with a large number of points and clusters. Additionally, the use of atomic operations ensures thread safety during concurrent updates to shared variables, maintaining the correctness of the algorithm's results.

### 2.2.1 Output

```plaintext
Final Centroids for DataPoints/points_250_000.txt:
Centroid 1: 95.6000 85.2348
Centroid 2: -25.5501 3.1604
Centroid 3: -20.9669 11.4897
Centroid 4: -71.5770 -13.2909
Centroid 5: -19.8030 2.0418
Centroid 6: 65.7581 -43.2929
Centroid 7: -26.2859 8.8829
Centroid 8: -21.8759 6.5813
Centroid 9: -16.9451 7.2151
Centroid 10: -37.4841 -68.4311
Execution Time for DataPoints/points_250_000.txt: 1.145000 seconds

Final Centroids for DataPoints/points_50_000.txt:
Centroid 1: -77.7882 -10.6182
Centroid 2: -73.8838 -13.3640
Centroid 3: -75.0547 -6.5627
Centroid 4: 94.5237 67.0213
Centroid 5: -44.2482 -85.2705
Centroid 6: -80.6164 -14.6192
Centroid 7: -7.8386 13.0080
Centroid 8: 76.4964 -41.8385
Centroid 9: -72.1684 77.0896
Centroid 10: -81.8154 -7.8539
Execution Time for DataPoints/points_50_000.txt: 0.196000 seconds

Final Centroids for DataPoints/points_500.txt:
Centroid 1: 75.7708 47.3781
Centroid 2: 94.3117 37.3416
Centroid 3: 90.2764 -39.1936
Centroid 4: 95.9438 43.0398
Centroid 5: 69.8918 50.7322
Centroid 6: -73.4475 49.2328
Centroid 7: 72.2408 47.2671
Centroid 8: 72.6873 42.8626
Centroid 9: 76.6375 54.1843
Centroid 10: -39.8720 -24.5480
Execution Time for DataPoints/points_500.txt: 0.002000 seconds

Final Centroids for DataPoints/points_100.txt:
Centroid 1: -13.5234 89.0133
Centroid 2: -8.2523 80.7285
Centroid 3: -19.8293 82.1487
Centroid 4: -26.2538 87.8962
Centroid 5: -23.1619 88.4389
Centroid 6: 86.3432 -86.0298
Centroid 7: -65.9601 -36.2510
Centroid 8: 81.5821 54.5611
Centroid 9: 34.8471 96.9314
Centroid 10: -28.4080 83.9109
Execution Time for DataPoints/points_100.txt: 0.002000 seconds

Final Centroids for DataPoints/points_1_000_000.txt:
Centroid 1: 78.4072 8.8014
Centroid 2: -62.0916 -7.7554
Centroid 3: -72.3199 72.4807
Centroid 4: -8.6740 81.9792
Centroid 5: 70.3589 -62.1909
Centroid 6: 21.8655 4.7733
Centroid 7: 62.1602 76.3337
Centroid 8: -30.3278 36.2345
Centroid 9: -2.8550 -60.2486
Centroid 10: -69.1091 -74.4174
Execution Time for DataPoints/points_1_000_000.txt: 2.056000 seconds

Final Centroids for DataPoints/points_100_000.txt:
Centroid 1: 79.9497 34.5753
Centroid 2: -14.2214 -87.4396
Centroid 3: -17.0166 -90.9211
Centroid 4: -19.9320 -94.4551
Centroid 5: -36.0088 -46.5655
Centroid 6: -13.6478 -93.8357
Centroid 7: -87.8236 -99.5472
Centroid 8: 55.5231 -89.0764
Centroid 9: 7.7132 -95.1920
Centroid 10: -20.4553 -88.0723
Execution Time for DataPoints/points_100_000.txt: 0.148000 seconds

Final Centroids for DataPoints/points_10_000.txt:
Centroid 1: -97.1763 55.0062
Centroid 2: 49.8569 -4.3192
Centroid 3: -99.1573 58.0298
Centroid 4: -86.8113 86.1628
Centroid 5: -98.5185 55.5605
Centroid 6: -64.5332 -63.6248
Centroid 7: -94.3945 53.7747
Centroid 8: -100.9397 54.3900
Centroid 9: -97.9094 52.1992
Centroid 10: -95.4566 57.1284
Execution Time for DataPoints/points_10_000.txt: 0.024000 seconds

Final Centroids for DataPoints/points_1_000.txt:
Centroid 1: 70.1878 -55.6718
Centroid 2: 81.7533 -19.7054
Centroid 3: 69.6239 -55.5760
Centroid 4: 69.1638 -55.1720
Centroid 5: -42.6703 -57.4265
Centroid 6: 34.1807 76.1991
Centroid 7: 69.5090 -56.1763
Centroid 8: 36.4509 -75.2303
Centroid 9: 69.7746 -54.9241
Centroid 10: 69.0216 -55.9074
Execution Time for DataPoints/points_1_000.txt: 0.002000 seconds

--------------------------------
Process exited after 4.932 seconds with return value 0
Press any key to continue . . .

```

### 2.2.2 Observations

| Points   | Time-Sequentail | Time-OpenMP | Speed Up |
|----------|-----------------|-------------|----------|
| 100      | 0               | 0.002       | 0        |
| 500      | 0.001           | 0.001       | 1        |
| 1000     | 0.002           | 0.002       | 1        |
| 10000    | 0.088           | 0.02        | 4.4      |
| 50000    | 0.801           | 0.148       | 5.412162 |
| 100000   | 0.752           | 0.14        | 5.371429 |
| 250000   | 5.165           | 1.096       | 4.712591 |
| 1000000  | 9.923           | 1.874       | 5.295091 |

**Table 1:** Execution Time Comparison between Sequential and OpenMP

### 2.2.3 Plot

![Number of Points vs Execution Time](https://github.com/kartabyakrishna/KartabyaKrishna/blob/main/Assets/KMeansParallel/openmp.png)
**Plot 1:** Number of Points vs Execution Time

### Interpretation of Observations

1. **Speed Up:**
   - The "Speed Up" column shows how much faster the OpenMP implementation performs compared to the sequential implementation. It's calculated as the ratio of sequential time to OpenMP time. For example, a speedup of 1 indicates no improvement (sequential and parallel implementations have similar execution times), while a speedup greater than 1 indicates improvement with parallel execution.

2. **Observations:**
   - As the number of data points increases, the execution time decreases significantly in both sequential and OpenMP implementations.
   - For a small number of data points (100, 500, 1000), the difference in execution time between the sequential and OpenMP implementations is minimal, resulting in a speedup close to 1.
   - However, as the number of data points grows larger (10,000, 50,000, 100,000, 250,000, 1,000,000), the speedup becomes more significant, indicating a substantial improvement in performance with OpenMP parallelization.
   - Notably, the speedup increases with the number of data points, suggesting that the parallel implementation becomes more advantageous as the computational workload grows.
   - The speedup values range from around 4 to over 5, indicating that the OpenMP implementation achieves approximately 4 to 5 times faster execution compared to the sequential implementation for larger datasets.

3. **Interpretation:**
   - The exponential decrease in execution time with an increase in the number of data points highlights the scalability of the K-means clustering algorithm, both in sequential and parallel contexts.
   - The significant speedup observed for larger datasets demonstrates the effectiveness of parallelization in distributing the computational workload across multiple threads, resulting in faster processing.
   - This observation underscores the importance of considering parallel computing techniques, such as OpenMP, for improving the performance of computationally intensive tasks, especially when dealing with large datasets.
   - Overall, the results indicate that the OpenMP implementation offers substantial performance benefits, particularly for large-scale data processing tasks, making it a valuable approach for accelerating K-means clustering and similar algorithms in practical applications.

## 2.3 Parallelization using CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It leverages the processing power of GPUs (Graphics Processing Units) to achieve significant performance improvements in computational tasks. In CUDA programming, the CPU serves as the host, while the GPU acts as the device. Each GPU can handle thousands of threads, with each thread processing a single piece of data. Threads are organized into blocks, and shared memory is limited to each block. Unlike CPU and GPU, host and device do not share memory directly, necessitating explicit communication between them.

The aim of parallelization in this context is to distribute the workload across multiple threads on the GPU, exploiting parallel processing capabilities to speed up computations. Below is the step-by-step explanation of how the parallel algorithm works, considering the communication between the host and the device:

1. **Initialization:** The host initializes cluster centres and copies the data coordinates to the device.
2. **Data Transfer:** Data membership and cluster centres are copied from the host to the device.
3. **Parallel Computation on Device:** Each thread on the device processes a single data point, computes the distance between each cluster centre, and updates data membership accordingly.
4. **Data Transfer Back to Host:** The updated data membership is copied back from the device to the host, and further computations are performed if needed.
5. **Memory Deallocation:** After computations are completed, memory allocated on both the host and the device is freed.

### Code Implementation

Before implementing the code, existing CUDA implementations for K-means parallelization were reviewed, providing insights into the implementation approach. The primary functions of both CUDA and C++ codes include:
- `distance()`: Computes the Euclidean distance between two 2D points.
- `centroid_init()`: Randomly selects one data point from the dataset for each cluster to initialize centroids.
- `kMeansClusterAssignment()`: Assigns each data point to the closest cluster using parallel computation.
- `kMeansCentroidUpdate()`: Updates centroids based on the mean value of assigned data points in each cluster.

Additional functions for file I/O operations and user interface were added to enhance code manageability and usability.

### Methodology

#### Data Preparation

Random data points were generated in varying sizes, ranging from 100 to 1,000,000, belonging to different clusters. Initial centroids for each cluster were randomly chosen among the data points. Coordinates of points and centroids were copied to the device for kernel execution.

#### Performance Analysis

Performance was evaluated by measuring the time spent in different regions of interest (ROIs) within the code. For the sequential C++ code, ROIs included duration of one iteration and cluster assignment. In the parallel CUDA code, ROIs encompassed data transfer from host to device (ROICP0), duration of one iteration, cluster assignment, data transfer from device to host (ROICP1), and data transfer for new centroids (ROICP2) after centroid update. This structured approach facilitated comprehensive performance analysis and comparison between CPU and GPU implementations across different data sizes.

### 2.3.2 Outputs

```plaintext
Number (int) of points you want to analyze (100, 1000, 10000, 100000, 1000000):
1000000
Please, insert number (int) of epochs for training (in the order of the hundreds is recommended):
1000
Initialization of 10 centroids:
(17.258211, 34.365688)
Initialization of 10 centroids:
(-98.480415, -98.089714)
Initialization of 10 centroids:
(-16.104473, 70.056351)
Initialization of 10 centroids:
(7.941577, -45.479034)
Initialization of 10 centroids:
(-64.990234, -19.727753)
Initialization of 10 centroids:
(9.752409, -23.636259)
Initialization of 10 centroids:
(44.591911, -67.964005)
Initialization of 10 centroids:
(-27.437700, -41.715809)
Initialization of 10 centroids:
(-6.620975, -55.921665)
Initialization of 10 centroids:
(-68.156624, 79.014420)
Time taken by transfering centroids, datapoints and cluster's sizes from host to device is : 1612 microseconds
Time taken by 1000 iterations is: 5.62452e+06 microseconds
Time taken by kMeansClusterAssignment: 28.437 microseconds
Time taken by transfering centroids and assignments from the device to the host: 1956.98 microseconds
Time taken by transfering centroids and assignments from the device to the host: 28.317 microseconds
N = 1000000,K = 10, MAX_ITER= 1000.
The centroids are:
centroid: 0: (58.0357, 78.1893)
centroid: 1: (-70.976, -77.8885)
centroid: 2: (-5.79254, 56.5004)
centroid: 3: (7.97468, -28.441)
centroid: 4: (-53.7974, 16.9508)
centroid: 5: (67.4143, 11.1885)
centroid: 6: (74.2779, -57.2018)
centroid: 7: (-49.6772, -38.595)
centroid: 8: (11.7544, -78.2558)
centroid: 9: (-70.6905, 74.0921)

C:\Users\Kartabya\Downloads\GitHub\ParallelProgramming - MiniProject\Sequential\cuda kmeans\x64\Debug\cuda kmeans.exe (process 26828) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .
```

### 2.3.3 Observations

| Points   | Time-Sequentail | Time-CUDA | Speed Up  |
|----------|-----------------|-----------|-----------|
| 100      | 0               | 0.13      | 0         |
| 500      | 0.001           | 0.13      | 0.007692  |
| 1000     | 0.002           | 0.15      | 0.013333  |
| 10000    | 0.088           | 0.15      | 0.586667  |
| 50000    | 0.801           | 0.373     | 2.147453  |
| 100000   | 0.752           | 0.591     | 1.272420  |
| 250000   | 5.165           | 1.3       | 3.973077  |
| 1000000  | 9.923           | 4.921     | 2.016460  |

**Table 2:** Execution Time Comparison between Sequential and CUDA

### 2.2.3 Plot

![Number of Points vs Exec. Time](https://github.com/kartabyakrishna/KartabyaKrishna/blob/main/Assets/KMeansParallel/cuda.png)
**Plot 2:** Number of Points vs Execution Time

From the obtained data, it's evident that CUDA tends to reach a plateau in performance improvement with smaller numbers of points compared to OpenMP. Additionally, OpenMP outperforms CUDA in certain scenarios, particularly with larger datasets. Let's delve into potential reasons for these observations:

**Plateau in CUDA Performance with Smaller Numbers of Points:**
1. **Overhead of Parallelization:** CUDA introduces overhead associated with launching and managing threads on the GPU. With smaller datasets, this overhead can become more pronounced relative to the actual computation time, leading to a plateau in performance improvement. The parallelization benefits may not outweigh the overhead for small datasets.
2. **Thread Block Size:** In CUDA, threads are organized into blocks, and choosing an optimal block size is crucial for performance. With smaller datasets, choosing an overly large block size can lead to underutilization of GPU resources, while too small a block size can increase overhead. Finding the right balance becomes challenging with smaller datasets, potentially limiting performance gains.
3. **Memory Transfer Latency:** In CUDA, data must be transferred between the host (CPU) and the device (GPU), incurring latency overhead. With smaller datasets, the proportion of time spent on data transfer relative to computation increases, limiting the overall performance improvement achievable through parallelization.

**Reasons for OpenMP Outperforming CUDA in Certain Scenarios:**
1. **Algorithmic Efficiency:** OpenMP may exhibit superior performance in scenarios where the algorithm is inherently better suited for parallelization using shared memory multiprocessing (OpenMP) compared to parallelization using GPUs (CUDA). Certain algorithms may not efficiently exploit the parallelism offered by GPUs.
2. **Data Dependency:** Some algorithms may have inherent data dependencies that are difficult to parallelize effectively on GPUs using CUDA. OpenMP's shared-memory model may better handle such scenarios, resulting in better performance.
3. **Hardware Considerations:** The hardware characteristics of the system, including the CPU and GPU specifications, can influence performance. In certain scenarios, the CPU may have advantages over the GPU, such as faster memory access or more efficient utilization of available resources.
4. **Implementation Complexity:** CUDA programming requires specialized knowledge of GPU architecture and parallelization techniques. In contrast, OpenMP offers a simpler parallelization model, making it easier to implement and potentially optimize for certain scenarios.
5. **Optimization and Tuning:** The performance of CUDA code heavily depends on optimization techniques such as memory access patterns, thread synchronization, and memory hierarchy utilization. In scenarios where CUDA code is not effectively optimized, OpenMP implementations may outperform CUDA counterparts.

In conclusion, the plateau in CUDA performance with smaller datasets can be attributed to overhead associated with parallelization and memory transfer latency, while OpenMP's superior performance in certain scenarios may stem from algorithmic efficiency, data dependency considerations, hardware characteristics, implementation complexity, and optimization. Understanding these factors is crucial for selecting the appropriate parallelization framework and optimizing performance based on the specific characteristics of the problem at hand.

### System Specification

Here's the device configuration of the Device used for the experiments:

- **Operating System:**
  - Name: Windows
  - Version: 23H2
- **CPU Information:**
  - Architecture: x64
  - Model Name: AMD Ryzen 7 6800H with 8 cores and 16 threads
- **GPU Information:**
  - Name: NVIDIA GeForce RTX 3070 Laptop GPU
  - Memory Size: 8 GB
  - Memory Type: GDDR6
  - TDP: 145 Watt

### References

1. A Clustering Method Based on K-Means Algorithm: Li, Y., Wu, H. (2012), Physics Procedia, 25, 1104–1109. doi:10.1016/j.phpro.2012.03.206
2. CUDA C++ Programming Guide: [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
3. K-Means algorithm for CUDA: [cuKMean](https://github.com/alexminnaar/cuKMean)
