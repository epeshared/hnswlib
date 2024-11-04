#include "../../hnswlib/hnswlib.h"
#include <iostream>
#include <random>
#include <vector>

// 自定义 L2 距离计算函数
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);
static int8_t
Int8InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int32_t res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((int8_t *) pVect1)[i] * ((int8_t *) pVect2)[i];
    }
        // 如果需要，可以在这里进行截断或饱和处理
    if (res > 127) {
        res = 127;
    } else if (res < -128) {
        res = -128;
    }

    return static_cast<int8_t>(-res);
}
/* static int8_t
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
} */

static int8_t Int8L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr){
    int8_t *pVect1 = (int8_t *) pVect1v;
    int8_t *pVect2 = (int8_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    int8_t res = 0;
    for (size_t i = 0; i < qty; i++) {
        int8_t t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }

    //printf("res:%d",res);
    return (res);
}
class Int8L2Space : public hnswlib::SpaceInterface<int8_t> {

    DISTFUNC<int8_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
 public:
    Int8L2Space(size_t dim) {
        fstdistfunc_ = Int8L2Sqr;
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
    ~Int8L2Space() {}
};

class Int8InnerProductSpace : public hnswlib::SpaceInterface<int8_t> {
    DISTFUNC<int8_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
 public:
    Int8InnerProductSpace(size_t dim) {
        fstdistfunc_ = Int8InnerProduct;
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

int call_scalar(hnswlib::HierarchicalNSW<int8_t>* alg_hnsw,Int8InnerProductSpace & space,int8_t* data,int dim, int max_elements,int top_k){
    //init_onednn();
    int correct=0;
    for(int j=0;j<20;j++){
      for (int i = 0; i < max_elements; i++) {
          std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, top_k);
          hnswlib::labeltype label = result.top().second;

          //printf("label:%d i:%d \n",label,i);
          if (label == i) correct++;
      }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index
    std::string hnsw_path = "hnsw_int8.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<int8_t>(&space, hnsw_path);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, top_k);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    //delete[] data;
    //delete alg_hnsw;
    return 0;
}

int call_AMX(hnswlib::HierarchicalNSW<int8_t>* alg_hnsw,Int8InnerProductSpace & space,int8_t* data,int dim, int max_elements,int top_k){
    //init_onednn();

    
    int correct=0;
    for(int j=0;j<20;j++){
      for (int i = 0; i < max_elements; i++) {
          std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnnAMX(data + i * dim, top_k);
          hnswlib::labeltype label = result.top().second;

          //printf("label:%d i:%d \n",label,i);
          if (label == i) correct++;
      }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index
    std::string hnsw_path = "hnsw_int8.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<int8_t>(&space, hnsw_path);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnnAMX(data + i * dim, top_k);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    //delete[] data;
    //delete alg_hnsw;
    return 0;
}

int main() {
    int dim = 128;              // Dimension of the elements
    int max_elements = 100000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    int top_k=1;
    // Initing index
    Int8InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<int8_t>* alg_hnsw = new hnswlib::HierarchicalNSW<int8_t>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_int_distribution<> distrib_int8(0, 10);
    int8_t* data = new int8_t[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = static_cast<int8_t>(distrib_int8(rng));
        //printf("%d ",data[i]);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;

    std::cout << "Start Search" <<"\n";
    fflush(stdout);


    auto start_scalar = std::chrono::high_resolution_clock::now();
    call_scalar(alg_hnsw,space,data,dim,max_elements,top_k);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    
    auto start_AMX = std::chrono::high_resolution_clock::now();
    call_AMX(alg_hnsw,space,data,dim,max_elements,top_k);
    auto end_AMX = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration_scalar = end_scalar - start_scalar;
    std::chrono::duration<double> duration_AMX = end_AMX - start_AMX;


    std::cout << "Time taken for scalar:" << duration_scalar.count()<<std::endl;
    std::cout << "Time taken for AMX:" << duration_AMX.count()<<std::endl;

    delete[] data;
    delete alg_hnsw;
}

