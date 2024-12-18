#include "../../hnswlib/hnswlib.h"
#include <thread>


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);
static int8_t vector_dot_product(const void* a, const void* b, const void *qty_ptr) {

    uint32_t length = * (uint32_t*)qty_ptr;

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
    return final_result;
}
static int8_t vector_dot_product_opt_avx512(const void* a, const void* b, const void *qty_ptr) {
  const uint8_t* pvec_u8 = (const uint8_t*)a;
  const int8_t* pvec_s8 = (const int8_t*)a;
  size_t qty32 = *((uint32_t*)qty_ptr) / 64;
  const uint8_t* pend_u8 = pvec_u8 + 64 * qty32;
  // calc dot
  __m512i sum512 = _mm512_setzero_si512();
  __m512i v1, v2,v3;

  __m128i one = _mm_set1_epi16(1);
  __m512i agg_base = _mm512_broadcastw_epi16(one);
  while (pvec_u8 < pend_u8) {
    v1 = _mm512_loadu_si512(pvec_u8);
    v2 = _mm512_loadu_si512(pvec_s8);
    v3 = _mm512_maddubs_epi16(v1, v2);
    sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v3, agg_base));
    pvec_u8 += 64;
    pvec_s8 += 64;
  }
  int8_t dotsum = _mm512_reduce_add_epi32(sum512);
  // fetch from u8 qcode
/*   float* flt_u8 = (float*)((char*)qcode_u8 + dim);
  float scale_u8 = *flt_u8;
  flt_u8++;
  float offset_u8 = *flt_u8;
  // fetch from s8 qcode
  float* flt_s8 = (float*)((char*)qcode_s8 + dim);
  float scale_s8 = *flt_s8;
  flt_s8++;
  float sum_s8 = *flt_s8;
  float score = scale_u8 * scale_s8 * dotsum + offset_u8 * sum_s8; */
  return dotsum;
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
static int8_t fvec_inner_product_int8_avx2int8(const void* a, const void* b, const void *qty_ptr) {
  //exit(-1);
  const uint8_t* pvec_u8 = (const uint8_t*)a;
    const int8_t* pvec_s8 = (const int8_t*)b;
    size_t qty32 = *((size_t*)qty_ptr) / 32;
    const uint8_t* pend_u8 = pvec_u8 + 32 * qty32;

    // 初始化累加和为 0
    __m256i sum256 = _mm256_setzero_si256();
    __m256i v1, v2, v3;

    // 创建一个包含 1 的 128 位向量
    __m128i one = _mm_set1_epi16(1);
    // 广播 1 到 256 位向量
    __m256i agg_base = _mm256_broadcastw_epi16(one);

    while (pvec_u8 < pend_u8) {
        v1 = _mm256_loadu_si256((__m256i*)pvec_u8);
        v2 = _mm256_loadu_si256((__m256i*)pvec_s8);
        v3 = _mm256_maddubs_epi16(v1, v2);
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(v3, agg_base));
        pvec_u8 += 32;
        pvec_s8 += 32;
    }

    // 处理剩余数据
    for (size_t i = 32 * qty32; i < *((size_t*)qty_ptr); i++) {
        sum256 = _mm256_add_epi32(sum256, _mm256_set1_epi32(pvec_u8[i] * pvec_s8[i]));
    }

    // 将 SIMD 寄存器中的结果累积到一个标量值
    int32_t result[8];
    _mm256_storeu_si256((__m256i*)result, sum256);

    int8_t dotsum = 0;
    for (int i = 0; i < 8; ++i) {
        dotsum += result[i];
    }
    return dotsum;
}
class Int8InnerProductSpace : public hnswlib::SpaceInterface<int8_t> {
    DISTFUNC<int8_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
 public:
    Int8InnerProductSpace(size_t dim) {
        fstdistfunc_ = fvec_inner_product_int8_avx2int8;
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
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
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
    int max_elements = 100000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    int nq = max_elements;
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 56;       // Number of threads for operations with index

    int top_k=5;

    int iteration=10;

  
    // Initing index
    Int8InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<int8_t>* alg_hnsw = new hnswlib::HierarchicalNSW<int8_t>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_int_distribution<> distrib_int8(0, 3);
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
