#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <cstring>
#include <omp.h> // OpenMP for parallelism
#include "rand_helper.h"
#include "csr.h"
#include "debug.h"

#define NUM_POOLS 2

// recWalk class for the recWalk algorithm
template <typename VID_T>
class recWalk {
public:
    recWalk(CSR<VID_T>* _graph) : graph(_graph) {}

    void sample_into_pool(VID_T* sample_array, int max_queue_size, long long samples_per_thread, long long samples_per_section, int samples_per_segment, int tid, int walk_length, int augmentation_distance) {
        VID_T* private_sample_pool = new VID_T[max_queue_size];
        for (long long current = 0; current < samples_per_thread; current++) {
            VID_T source = graph->get_random_vertex();
            VID_T p_sample = source;
            for (int i = 0; i < walk_length; i++) {
                p_sample = graph->get_random_neighbor(p_sample);
                if (p_sample == UINT_MAX) break;
                if (i >= augmentation_distance) {
                    private_sample_pool[current] = source;
                    private_sample_pool[current + samples_per_thread] = p_sample;
                    break;
                }
            }
        }
        memcpy(sample_array + tid * samples_per_segment * 2, private_sample_pool, samples_per_thread * 2 * sizeof(VID_T));
        delete[] private_sample_pool;
    }

private:
    CSR<VID_T>* graph;
};

template <typename VID_T, typename E_T>
class random_walk_training {
public:
    random_walk_training(int _sample_pool_size, int _shuffle_base, int _sampling_threads, int _deviceID, int negative_samples)
        : max_sample_pool_size(_sample_pool_size), shuffle_base(_shuffle_base), sampling_threads(_sampling_threads), deviceID(_deviceID), negative_samples(negative_samples) {
        // Allocate memory for sample pools
        positive_sample_pools = new VID_T*[NUM_POOLS];
        negative_sample_pools = new VID_T*[NUM_POOLS];
        d_positive_sample_pools = new VID_T*[NUM_POOLS];
        d_negative_sample_pools = new VID_T*[NUM_POOLS];
        for (int i = 0; i < NUM_POOLS; i++) {
            positive_sample_pools[i] = new VID_T[max_sample_pool_size * 2];
            CUDA_CHECK(cudaHostRegister(positive_sample_pools[i], max_sample_pool_size * 2 * sizeof(VID_T), cudaHostRegisterPortable));
            CUDA_CHECK(cudaMalloc((void**)&(d_positive_sample_pools[i]), sizeof(VID_T) * max_sample_pool_size * 2));

            negative_sample_pools[i] = new VID_T[max_sample_pool_size * negative_samples];
            CUDA_CHECK(cudaHostRegister(negative_sample_pools[i], max_sample_pool_size * negative_samples * sizeof(VID_T), cudaHostRegisterPortable));
            CUDA_CHECK(cudaMalloc((void**)&(d_negative_sample_pools[i]), sizeof(VID_T) * max_sample_pool_size * negative_samples));

            device_pool_full[i] = 0;
            host_pool_full[i] = 0;
        }
        // Calculate parameters
        max_sample_pool_size = max_sample_pool_size / (shuffle_base * sampling_threads) * (shuffle_base * sampling_threads);
        max_samples_per_section = max_sample_pool_size / shuffle_base;
        max_samples_per_thread = max_sample_pool_size / sampling_threads;
        max_samples_per_segment = max_samples_per_section / sampling_threads;
        private_sample_pools = new VID_T*[sampling_threads];
        for (int i = 0; i < sampling_threads; i++) {
            private_sample_pools[i] = new VID_T[max_samples_per_thread * 2];
        }
        // Create CUDA streams
        sampling_stream = new cudaStream_t;
        kernel_stream = new cudaStream_t;
        CUDA_CHECK(cudaStreamCreateWithFlags(sampling_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(kernel_stream, cudaStreamNonBlocking));
        TM_PRINTF(true, "max_sample_pool_size %lu  max_samples_per_section %lu max_samples_per_segment %lu max_samples_per_thread %lu\n", max_sample_pool_size, max_samples_per_section, max_samples_per_segment, max_samples_per_thread);
    }

    template <typename KernelCallerLambda>
    void train_num_samples(size_t num_samples, int lrd_strategy, float starting_learning_rate, int walk_length, int augmentation_distance, CSR<VID_T>* graph, KernelCallerLambda kernel) {
        size_t sample_pool_size = min(max_sample_pool_size, num_samples);
        sample_pool_size = sample_pool_size / (shuffle_base * sampling_threads) * (shuffle_base * sampling_threads);
        size_t samples_per_section = sample_pool_size / shuffle_base;
        size_t samples_per_thread = sample_pool_size / sampling_threads;
        size_t samples_per_segment = samples_per_section / sampling_threads;
        size_t num_rounds = ((float)(num_samples / sample_pool_size));
        for (int p = 0; p < NUM_POOLS; p++) {
            host_pool_full[p] = 0;
            device_pool_full[p] = 0;
        }
        omp_set_nested(1);
#pragma omp parallel num_threads(4)
        {
#pragma omp single
            {
#pragma omp task
                {
                    TM_PRINTF(true, "COPIER - start\n");
                    copier_task(num_rounds, sample_pool_size);
                }
#pragma omp task
                {
                    TM_PRINTF(true, "SAMPLER - start\n");
                    sampler_task_recWalk(num_rounds, sample_pool_size, graph, walk_length, augmentation_distance, samples_per_section, samples_per_segment, samples_per_thread);
                }
#pragma omp task
                {
                    TM_PRINTF(true, "KERNEL - start\n");
                    kernel_dispatched_task(num_rounds, num_samples, sample_pool_size, graph, lrd_strategy, starting_learning_rate, kernel);
                }
            }
        }
        cudaStreamSynchronize(*kernel_stream);
        cudaStreamSynchronize(*sampling_stream);
        cudaDeviceSynchronize();
    }

private:
    int deviceID;
    int negative_samples;
    VID_T **positive_sample_pools, **negative_sample_pools;
    VID_T **d_positive_sample_pools, **d_negative_sample_pools;
    int sampling_threads;
    int device_pool_full[NUM_POOLS], host_pool_full[NUM_POOLS];
    mutex device_pool_mutex[NUM_POOLS], host_pool_mutex[NUM_POOLS];
    condition_variable device_pool_condition_variable_full[NUM_POOLS], device_pool_condition_variable_empty[NUM_POOLS], host_pool_condition_variable_full[NUM_POOLS], host_pool_condition_variable_empty[NUM_POOLS];
    const int shuffle_base = 5;
    size_t max_samples_per_section;
    size_t max_samples_per_thread;
    size_t max_samples_per_segment;
    size_t max_sample_pool_size;
    VID_T** private_sample_pools;

    cudaStream_t *sampling_stream, *kernel_stream;

    void enqueue(VID_T* queue, int max_queue_size, char& front, char &rear, char &size, unsigned int val) {
        if (front == -1) {
            front = rear = 0;
            queue[rear] = val;
        }
        else {
            rear++;
            rear %= max_queue_size;
            queue[rear] = val;
        }
        size++;
    }

    unsigned int dequeue(VID_T* queue, int max_queue_size, char &front, char &rear, char &size) {
        if (size == 0) return UINT_MAX;
        int old_front = front;
        front++;
        front %= max_queue_size;
        size--;
        return queue[old_front];
    }

    void copier_task(size_t num_rounds, size_t sample_pool_size) {
        for (int i = 0; i < num_rounds; i++) {
            int selected_pool = i % NUM_POOLS;
            unique_lock<mutex> host_lock(host_pool_mutex[selected_pool]);
            while (host_pool_full[selected_pool] != 1) {
                host_pool_condition_variable_full[selected_pool].wait(host_lock);
            }
            host_pool_full[selected_pool] = 2;
            unique_lock<mutex> device_lock(device_pool_mutex[selected_pool]);
            while (device_pool_full[selected_pool] != 0) {
                device_pool_condition_variable_empty[selected_pool].wait(device_lock);
            }
            device_pool_full[selected_pool] = 1;
            CUDA_CHECK(cudaMemcpyAsync(d_positive_sample_pools[selected_pool], positive_sample_pools[selected_pool], sample_pool_size * 2 * sizeof(VID_T), cudaMemcpyHostToDevice, *sampling_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_negative_sample_pools[selected_pool], negative_sample_pools[selected_pool], sample_pool_size * negative_samples * sizeof(VID_T), cudaMemcpyHostToDevice, *sampling_stream));
            device_pool_condition_variable_full[selected_pool].notify_all();
        }
    }

    static void sample_pool_used(cudaStream_t event, cudaError_t status, void* data) {
        int selected_pool = ((call_back_pool_id*)data)->selected_pool;
        random_walk_training<VID_T, E_T>* us = ((call_back_pool_id*)data)->us;
        unique_lock<mutex> device_lock(us->device_pool_mutex[selected_pool]);
        us->device_pool_full[selected_pool] = 0;
        us->device_pool_condition_variable_empty[selected_pool].notify_all();
    }

    void kernel_dispatched_task(size_t num_rounds, unsigned long long num_samples, size_t sample_pool_size, CSR<VID_T>* csr, int lrd_strategy, float starting_learning_rate, KernelCallerLambda kernel) {
        float learning_rate_ec;
        for (int i = 0; i < num_rounds; i++) {
            if (lrd_strategy < 2) {
                learning_rate_ec = max(float(1 - float(i) * 1.0 / (num_samples / sample_pool_size)), 1e-4f) * (starting_learning_rate);
            }
            else {
                learning_rate_ec = starting_learning_rate;
            }
            int selected_pool = i % NUM_POOLS;
            unique_lock<mutex> device_lock(device_pool_mutex[selected_pool]);
            while (device_pool_full[selected_pool] != 2) {
                device_pool_condition_variable_full[selected_pool].wait(device_lock);
            }
            device_pool_full[selected_pool] = 3;
            kernel(csr, d_positive_sample_pools[selected_pool], d_negative_sample_pools[selected_pool], learning_rate_ec, sample_pool_size, kernel_stream);
            call_back_pool_id* payload = new call_back_pool_id; payload->selected_pool = selected_pool;
            CUDA_CHECK(cudaStreamAddCallback(*kernel_stream, sample_pool_used, (void*)payload, 0));
        }
    }

    void sampler_task_recWalk(size_t num_rounds, size_t sample_pool_size, CSR<VID_T>* csr, int walk_length, int augmentation_distance, size_t samples_per_section, size_t samples_per_segment, size_t samples_per_thread) {
        for (int i = 0; i < num_rounds; i++) {
            int selected_pool = i % NUM_POOLS;
            unique_lock<mutex> lock(host_pool_mutex[selected_pool]);
            while (host_pool_full[selected_pool] != 0) {
                host_pool_condition_variable_empty[selected_pool].wait(lock);
            }
            recWalk<VID_T> walker(csr);
            walker.sample_into_pool(positive_sample_pools[selected_pool], 6, samples_per_thread, samples_per_section, samples_per_segment, i, walk_length, augmentation_distance);
            host_pool_full[selected_pool] = 1;
            host_pool_condition_variable_full[selected_pool].notify_all();
        }
    }
};

int main() {
    // Usage example
    // Initialize graph, etc.
    CSR<int>* graph = new CSR<int>();
    random_walk_training<int, int> trainer(1000, 5, 4, 0, 5); // Example parameters
    trainer.train_num_samples(10000, 1, 0.01f, 10, 3, graph, [] (CSR<int>*, int**, int**, float, size_t, cudaStream_t*) {
        // Kernel implementation
    });
    return 0;
}
