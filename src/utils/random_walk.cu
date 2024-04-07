#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <cstring>
#include "rand_helper.h"
#include "csr.h"
#include "debug.h"
#include <vector>

#define NUM_POOLS 2

template <typename VID_T, typename E_T>
class random_walk_training{
public:
    random_walk_training(int _sample_pool_size, int _shuffle_base, int _sampling_threads, int _deviceID, int negative_samples): max_sample_pool_size(_sample_pool_size), shuffle_base(_shuffle_base), sampling_threads(_sampling_threads), deviceID(_deviceID), negative_samples(negative_samples){

        cudaSetDevice(deviceID);
        positive_sample_pools = new VID_T*[NUM_POOLS];
        negative_sample_pools = new VID_T*[NUM_POOLS];
        d_positive_sample_pools = new VID_T*[NUM_POOLS];
        d_negative_sample_pools = new VID_T*[NUM_POOLS];
        for (int i =0; i<NUM_POOLS; i++){
            positive_sample_pools[i] = new VID_T[max_sample_pool_size*2];
            CUDA_CHECK(cudaHostRegister(positive_sample_pools[i], max_sample_pool_size*2*sizeof(VID_T), cudaHostRegisterPortable));

            CUDA_CHECK(cudaMalloc((void**)&(d_positive_sample_pools[i]), sizeof(VID_T)*max_sample_pool_size*2)); 

            negative_sample_pools[i] = new VID_T[max_sample_pool_size*negative_samples];
            CUDA_CHECK(cudaHostRegister(negative_sample_pools[i], max_sample_pool_size*negative_samples*sizeof(VID_T), cudaHostRegisterPortable));

            CUDA_CHECK(cudaMalloc((void**)&(d_negative_sample_pools[i]), sizeof(VID_T)*max_sample_pool_size*negative_samples)); 

            device_pool_full[i] = 0;
            host_pool_full[i] = 0;
        }
        max_sample_pool_size = max_sample_pool_size/(shuffle_base*sampling_threads)*(shuffle_base*sampling_threads);
        max_samples_per_section = max_sample_pool_size/shuffle_base;
        max_samples_per_thread = max_sample_pool_size/sampling_threads;
        max_samples_per_segment = max_samples_per_section/sampling_threads;
        private_sample_pools = new VID_T*[sampling_threads];
        for (int i =0; i<sampling_threads; i++){
            private_sample_pools[i] = new VID_T[max_samples_per_thread*2];
        }
        sampling_stream = new cudaStream_t;
        kernel_stream = new cudaStream_t;
        CUDA_CHECK(cudaStreamCreateWithFlags(sampling_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(kernel_stream, cudaStreamNonBlocking));
        TM_PRINTF(true, "max_sample_pool_size %lu  max_samples_per_section %lu max_samples_per_segment %lu max_samples_per_thread %lu\n", max_sample_pool_size, max_samples_per_section, max_samples_per_segment, max_samples_per_thread);
    }

    template <typename KernelCallerLambda>
    void train_num_samples(size_t num_samples, int lrd_strategy, float starting_learning_rate, int walk_length, int augmentation_distance, CSR<VID_T>* graph, KernelCallerLambda kernel){
        size_t sample_pool_size = min(max_sample_pool_size, num_samples);
        sample_pool_size = sample_pool_size/(shuffle_base*sampling_threads)*(shuffle_base*sampling_threads);
        size_t samples_per_section = sample_pool_size/shuffle_base;
        size_t samples_per_thread = sample_pool_size/sampling_threads;
        size_t samples_per_segment = samples_per_section/sampling_threads;
        size_t num_rounds = ((float)(num_samples/sample_pool_size));
        for (int p =0; p < NUM_POOLS; p++){
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
                    sampler_task(num_rounds, sample_pool_size, graph, walk_length, augmentation_distance, samples_per_section, samples_per_segment, samples_per_thread);
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
    std::mutex device_pool_mutex[NUM_POOLS], host_pool_mutex[NUM_POOLS];
    std::condition_variable device_pool_condition_variable_full[NUM_POOLS], device_pool_condition_variable_empty[NUM_POOLS], host_pool_condition_variable_full[NUM_POOLS], host_pool_condition_variable_empty[NUM_POOLS];
    const int shuffle_base = 5;
    size_t max_samples_per_section;
    size_t max_samples_per_thread;
    size_t max_samples_per_segment;
    size_t max_sample_pool_size;
    VID_T** private_sample_pools;
    struct call_back_pool_id {
        int selected_pool;
#ifdef _DEBUG
        int i;
#endif
        random_walk_training<VID_T, E_T>* us;
    };

    cudaStream_t *sampling_stream, *kernel_stream;

    void recursive_random_walk(VID_T start_vertex, std::vector<VID_T>& walk, int walk_length, CSR<VID_T>* graph, seed* sd) {
        VID_T current_vertex = start_vertex;
        walk.push_back(current_vertex);
        for (int step = 0; step < walk_length - 1; ++step) {
            std::vector<VID_T> neighbors = graph->neighbors(current_vertex);
            if (neighbors.empty()) break;
            VID_T next_vertex = neighbors[randn(sd) % neighbors.size()];
            walk.push_back(next_vertex);
            current_vertex = next_vertex;
        }
    }

    void sample_into_pool(VID_T* sample_array, int max_queue_size, int shuffle_base, long long samples_per_thread, long long samples_per_section, int samples_per_segment, int tid, int walk_length, int augmentation_distance, CSR<VID_T>* csr, seed* sd){
        std::vector<VID_T> walk;
        recursive_random_walk(csr->get_random_vertex(sd), walk, walk_length, csr, sd);
        for (VID_T v : walk) {
            // Store the walk into the sample array
            sample_array[tid * walk_length + v] = v;
        }
    }

    void sampler_task(size_t num_rounds, size_t sample_pool_size, CSR<VID_T>* csr, int walk_length, int augmentation_distance, size_t samples_per_section, size_t samples_per_segment, size_t samples_per_thread){
        cudaSetDevice(deviceID);
        printf("Execution %ld rounds\n", num_rounds);
        for (int i =0; i< num_rounds;i++){
            int selected_pool = i%NUM_POOLS;
            unique_lock<std::mutex> lock(host_pool_mutex[selected_pool]);
            while (host_pool_full[selected_pool] != 0){
                host_pool_condition_variable_empty[selected_pool].wait(lock);
            }

#pragma omp parallel num_threads(sampling_threads)
            {
#pragma omp single
                {
                    for (int y =0 ;y<sampling_threads; y++){
#pragma omp task
                        {
                            seed mySeed;
                            mySeed.x = (13*y)+(i*23)+123456789;
                            mySeed.y = (13*y)+(i*23)+362436069;
                            mySeed.z = (13*y)+(i*23)+521288629;
                            sample_into_pool(positive_sample_pools[selected_pool], 6, shuffle_base, samples_per_thread, samples_per_section, samples_per_segment, y, walk_length, augmentation_distance, csr, &mySeed);
                        }
                    }
                }
            }

            host_pool_full[selected_pool] = 1;
            host_pool_condition_variable_full[selected_pool].notify_all();
        }
    }

    static void CUDART_CB sample_pool_copied(cudaStream_t event, cudaError_t status,void * data){
        int selected_pool = ((call_back_pool_id*)data)->selected_pool;
#ifdef _DEBUG
        int i = ((call_back_pool_id*)data)->i;
#endif
        random_walk_training<VID_T, E_T>* us = ((call_back_pool_id*)data)->us;
        std::unique_lock<std::mutex> host_lock(us->host_pool_mutex[selected_pool]);
        std::unique_lock<std::mutex> device_lock(us->device_pool_mutex[selected_pool]);
        us->host_pool_full[selected_pool] = 0;
        us->device_pool_full[selected_pool] = 2;
        us->host_pool_condition_variable_empty[selected_pool].notify_all();
        us->device_pool_condition_variable_full[selected_pool].notify_all();
    }

    void copier_task(size_t num_rounds, size_t sample_pool_size){
        cudaSetDevice(deviceID);
        for (int i = 0; i < num_rounds; i++){
            int selected_pool = i%NUM_POOLS;
            std::unique_lock<std::mutex> host_lock(host_pool_mutex[selected_pool]);
            while (host_pool_full[selected_pool] != 1){
                host_pool_condition_variable_full[selected_pool].wait(host_lock);
            }
            host_pool_full[selected_pool] = 2;
            std::unique_lock<std::mutex> device_lock(device_pool_mutex[selected_pool]);
            while (device_pool_full[selected_pool] != 0){
                device_pool_condition_variable_empty[selected_pool].wait(device_lock);
            }
            device_pool_full[selected_pool] = 1;
            CUDA_CHECK(cudaMemcpyAsync(d_positive_sample_pools[selected_pool], positive_sample_pools[selected_pool], sample_pool_size*2*sizeof(VID_T), cudaMemcpyHostToDevice, *sampling_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_negative_sample_pools[selected_pool], negative_sample_pools[selected_pool], sample_pool_size*negative_samples*sizeof(VID_T), cudaMemcpyHostToDevice, *sampling_stream));
            call_back_pool_id *payload = new call_back_pool_id; 
#ifdef _DEBUG
            payload->i = i; 
#endif
            payload->selected_pool = selected_pool; payload->us = this;
            CUDA_CHECK(cudaStreamAddCallback(*sampling_stream, sample_pool_copied, (void*)payload, 0));
        }
    }

    static void CUDART_CB sample_pool_used(cudaStream_t event, cudaError_t status,void * data){
        int selected_pool = ((call_back_pool_id*)data)->selected_pool;
#ifdef _DEBUG
        int i = ((call_back_pool_id*)data)->i;
#endif
        random_walk_training<VID_T, E_T>* us = ((call_back_pool_id*)data)->us;
        std::unique_lock<std::mutex> device_lock(us->device_pool_mutex[selected_pool]);
        us->device_pool_full[selected_pool] = 0;
        us->device_pool_condition_variable_empty[selected_pool].notify_all();
    }

    template <typename KernelCallerLambda>
    void kernel_dispatched_task(size_t num_rounds, unsigned long long num_samples, size_t sample_pool_size, CSR<VID_T> * csr, int lrd_strategy, float starting_learning_rate, KernelCallerLambda kernel){
        float learning_rate_ec;
        for (int i =0; i<num_rounds; i++){
            if (lrd_strategy<2){
                learning_rate_ec = max(float(1-float(i)*1.0/(num_samples/sample_pool_size)), 1e-4f)*(starting_learning_rate);
            } else {
                learning_rate_ec = starting_learning_rate;
            }
            int selected_pool = i%NUM_POOLS;
            std::unique_lock<std::mutex> device_lock(device_pool_mutex[selected_pool]);
            while (device_pool_full[selected_pool] != 2){
                device_pool_condition_variable_full[selected_pool].wait(device_lock);
            }
            device_pool_full[selected_pool] = 3;
            kernel(csr, d_positive_sample_pools[selected_pool], d_negative_sample_pools[selected_pool], learning_rate_ec, sample_pool_size, kernel_stream);
            call_back_pool_id *payload = new call_back_pool_id; payload->selected_pool = selected_pool; 
#ifdef _DEBUG
            payload->i = i; 
#endif
            payload->us = this;
            CUDA_CHECK(cudaStreamAddCallback(*kernel_stream, sample_pool_used, (void*)payload, 0));
        }
    }
};

template class random_walk_training<int, int>;
