#include "../../hnswlib/hnswlib.h"
#include <thread>


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);
static int8_t vector_dot_product(const void* a, const void* b, const void *qty_ptr) {

    uint32_t length = * (uint32_t*)qty_ptr;
    // 确保长度是 64 的倍数，因为每个 _mm512_dpbusd_epi32 操作处理 64 个元素
/*     if (length % 64 != 0) {
        fprintf(stderr, "Length must be a multiple of 64\n");
        return 0;
    } */

    int32_t final_result = 0;
    size_t i = 0;
    int8_t *a_tmp=(int8_t *)a;
    int8_t *b_tmp=(int8_t *)b;
    if(length>=64){
        __m512i sum = _mm512_setzero_si512();  // 初始化累加和为 0
        for (; i+64 <= length; i += 64) {
            // 加载数据
            __m512i va = _mm512_loadu_si512((void*)&a_tmp[i]);
            __m512i vb = _mm512_loadu_si512((void*)&b_tmp[i]);
            // 执行点积运算
            //std::cout << "we are 32 lines" <<std::endl;
            sum = _mm512_dpbusd_epi32(sum, va, vb);
        }

        // 将 SIMD 寄存器中的结果累积到一个标量值
        int32_t result[16]={0};
        _mm512_storeu_si512((void*)result, sum);
        // 累加所有部分结果
        for (int j = 0; j < 16; ++j) {
            final_result += result[j];
        }
    }
    for (; i < length; i++) {
        final_result += a_tmp[i] * b_tmp[i];
    }
    //printf("%d ",final_result);
/*     if(final_result>127) final_result=127;
    else if(final_result<-128) final_result=-128;   */
    return final_result;
}



static int8_t
Int8InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int32_t res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((int8_t *) pVect1)[i] * ((int8_t *) pVect2)[i];
    }
        // 如果需要，可以在这里进行截断或饱和处理
    if(res>127) res=127;
    else if(res<-128) res=-128; 
    return static_cast<int8_t>(res);
}

class Int8InnerProductSpace : public hnswlib::SpaceInterface<int8_t> {
    DISTFUNC<int8_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
 public:
    Int8InnerProductSpace(size_t dim) {
        fstdistfunc_ = vector_dot_product;
        dim_ = dim;
        data_size_ = dim * sizeof(int8_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int8_t> get_dist_func() {
        return fstdistfunc_;
    }
    void *get_dist_func_param() {
        return &dim_;
    }
    ~Int8InnerProductSpace() {}
};

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;


        int dimSizeperThread = (end-start)/numThreads;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                int mylength = (threadId == numThreads-1 ) ? end - start - threadId*dimSizeperThread : dimSizeperThread;
                for(int id = 0;id < mylength; id++ ){
                     size_t myid = id + start + threadId*dimSizeperThread ;

                    if (myid >= end) {
                        break;
                    }

                    try {
                        fn(myid, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

int call_scalar(hnswlib::HierarchicalNSW<int8_t>* alg_hnsw,Int8InnerProductSpace & space,int8_t* data,int dim, int max_elements,int top_k,int num_threads){
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}

int call_AMX(hnswlib::HierarchicalNSW<int8_t>* alg_hnsw,Int8InnerProductSpace & space,int8_t* data,int dim, int max_elements,int top_k,int num_threads){
    //init_onednn();
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnnAMX(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}

int main() {
    int dim = 1024;               // Dimension of the elements
    int max_elements = 1000000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 43;       // Number of threads for operations with index
    int nq = 1000*num_threads;

    int top_k=1;

    int iteration=10;

  
    // Initing index
    Int8InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<int8_t>* alg_hnsw = new hnswlib::HierarchicalNSW<int8_t>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_int_distribution<> distrib_int8(0, 10);
    int8_t* data =(int8_t*) aligned_alloc(64,dim * max_elements) ;
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = static_cast<int8_t>(distrib_int8(rng));
        //printf("%d ",data[i]);
    }

    // Add data to index
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    });

    // Query the elements for themselves and measure recall
    float correct = 0;

    std::cout << "Start Search" <<"\n";
    fflush(stdout);


    auto start_scalar = std::chrono::high_resolution_clock::now();
    for(int i=0;i<iteration;i++){
      call_scalar(alg_hnsw,space,data,dim,nq,top_k,num_threads);
    }
    
    auto end_scalar = std::chrono::high_resolution_clock::now();
    
    auto start_AMX = std::chrono::high_resolution_clock::now();
    for(int i=0;i<iteration;i++){
      call_AMX(alg_hnsw,space,data,dim,nq,top_k,num_threads);
    }
    auto end_AMX = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration_scalar = end_scalar - start_scalar;
    std::chrono::duration<double> duration_AMX = end_AMX - start_AMX;


    std::cout << "Time taken for scalar:" << duration_scalar.count()/iteration<<std::endl;
    std::cout << "Time taken for AMX:" << duration_AMX.count()/iteration<<std::endl;
    fflush(stdout);

/*     delete[] data;
    delete alg_hnsw; */
    return 0;


}
