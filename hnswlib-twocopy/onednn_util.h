
#pragma once
#include <stdlib.h>
#include <mutex>
#include <shared_mutex>
#include <cstring> 
#include "oneapi/dnnl/dnnl.hpp"
#include <immintrin.h> 
#include <omp.h>
#include <sys/time.h>
#include <sys/syscall.h> 
#include <unistd.h>


static dnnl::engine cpu_engine;
static dnnl::stream engine_stream;
static bool is_onednn_init = false;
static std::mutex init_mutex;

#define u64 unsigned long long
#define u8  unsigned char
#define u16 unsigned short int

#define XFEATURE_XTILECFG           17
#define XFEATURE_XTILEDATA          18
#define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM         0x1022
#define ARCH_REQ_XCOMP_PERM         0x1023        

uint64_t get_amx_xcr_reg(void) {
    // 调用 _xgetbv 内建函数，参数0表示读取XCR0
    return _xgetbv(0);
}
int enable_amx() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return 0;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
        return 1;
    }
    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(WRITE) error" << std::endl;
        return 0;
    }
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return 0;
    }
    return 1;
}
void amx_int8_mul(void *cfg, void *ma, void *mb, int64_t a_stride, int64_t b_stride, void *mc) {

/*    if (cfg != NULL) {

        // 加载 TILE 配置
        // 这里你需要使用 oneAPI 或其他库提供的 API 来完成这个操作
        // 例如，使用 _ldtilecfg 函数（如果存在）
        // _ldtilecfg((tile_config*)cfg);
         _tile_loadconfig((void *)cfg);
    } */
/*                printf("we are 69\n");
  fflush(stdout); */

    // 使用 AMX 指令进行矩阵乘法
    // 注意：这里的函数名是假设性的，实际使用时需要根据 oneAPI 文档来确定
    _tile_loadd(0,ma, a_stride);
    _tile_loadd(1,mb, b_stride);
    _tile_dpbuud(2,0,1);
    //_tile_stored(2, mc, b_stride);
    //_tile_release();

    // 使用内联汇编实现AMX操作
/*     __asm__ volatile (
        "test %1, %1 \n\t"  // 检查 cfg 是否为 0
        "jz 1f \n\t"        // 如果 cfg 为 0，则跳过配置加载
        "ldtilecfg (%1) \n\t"  // 加载 tile 配置
        "1: \n\t"           // 跳转标签
        "tileloadd tmm0, (%2,%5,1) \n\t"  // 从 ma 加载数据到 tmm0
        "tileloadd tmm1, (%3,%7,1) \n\t"  // 从 mb 加载数据到 tmm1
        "tdpbuud tmm2, tmm0, tmm1 \n\t"   // 执行 int8 矩阵乘法
        "tilestored (%6,%7,1), tmm2 \n\t"// 将结果存储到 mc
        "mov %0, %6 \n\t"                 // 返回 mc 的地址
        "tilerelease \n\t"                // 释放所有 tiles
        : "=r" (mc)                       // 输出：mc 的新值通过 rax 返回
        : "r" (cfg), "r" (ma), "r" (mb), "r" (a_stride), "r" (b_stride), "r" (mc)
        : "tmm0", "tmm1", "tmm2", "memory"  // 告诉编译器哪些寄存器被修改了
    ); */

    return ;  // 返回 mc 的地址
}

int parall_memcpy(int8_t* dst, std::vector<int8_t*> src, size_t blocksize,size_t offset, size_t len){
     for (size_t i = offset; i+ offset < len; i++)
     {
        std::memcpy(dst + i * blocksize,src[i],blocksize);
     }

     return 0;
     
}

int parall_memcpy_bf16(uint16_t* dst, std::vector<uint16_t*>& src, size_t blocksize, size_t offset, size_t len) {
     for (size_t i = offset; i+ offset < len; i++)
     {
        std::memcpy(dst + i * blocksize,src[i],blocksize);
     }

    return 0;
}

static void omp_memcpy_bf16(size_t xrow, size_t xcol, size_t blocksize, uint16_t* reserve, std::vector<uint16_t*> bf16_vec) {
#pragma omp parallel for
    for (size_t i = 0; i < xrow; i++) {
        std::copy(bf16_vec[i], bf16_vec[i] + blocksize, reserve + i * blocksize);    
    }                
}

static bool is_amxbf16_supported() {
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
    return edx & (1 << 22);
}
/* 
static void init_onednn() {
    std::unique_lock<std::mutex> lock(init_mutex);

    if (is_onednn_init) {
        return;
    }

    // init dnnl engine
    cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    engine_stream = dnnl::stream(cpu_engine);

    is_onednn_init = true;
}

__attribute__((constructor)) static void library_load() {
    // this functionn will be automatically called when the library is loaded
    //printf("Library loaded.\n");
    init_onednn();
} */

/**
 * @brief Compute float32 matrix inner product with in8_t intermediate results to
 * accelerate
 * @details The main idea is:
 * 1. Define float32 memory layout for input and output
 * 2. Create low precision bf16 memory descriptors as inner product input
 * 3. Generate inner product primitive descriptor
 * 4. Execute s8 => float32 chain operation, isolate different precision data, accelerate inner
 * product
 * 5. Pipeline execution via streams for asynchronous scheduling
 *
 * @param xrow Row number of input matrix X
 * @param xcol Column number of input matrix X
 * @param yrow Row number of weight matrix Y
 * @param ycol Column number of weight matrix Y
 * @param in_s8_1 Input matrix pointer in int8_t type
 * @param in_s8_2 Weight matrix pointer in int8_t type
 * @param out_f32 Output matrix pointer for result in float32 type
 * @return None
 */
static void  compute_s8s8f32_inner_product(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        int8_t* in_s8_1,
        int8_t* in_s8_2,
        int8_t* relay_out,
        float* out_f32) {
    dnnl::memory::desc s8_md1 = dnnl::memory::desc(
            {xrow, xcol},
            dnnl::memory::data_type::s8,
            dnnl::memory::format_tag::ab);
    dnnl::memory::desc s8_md2 = dnnl::memory::desc(
            {yrow, ycol},
            dnnl::memory::data_type::s8,
            dnnl::memory::format_tag::any);
    dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc(
            {xrow, yrow},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab);

    //dnnl::memory s8_mem1 = dnnl::memory(s8_md1, cpu_engine, in_s8_1);
    //dnnl::memory s8_mem2 = dnnl::memory(s8_md2, cpu_engine, in_s8_2);

    dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32);


    dnnl::inner_product_forward::primitive_desc inner_product_pd =
            dnnl::inner_product_forward::primitive_desc(
                    cpu_engine,
                    dnnl::prop_kind::forward_training,
                    s8_md1,
                    s8_md2,
                    f32_dst_md2);

    dnnl::inner_product_forward inner_product_prim =
            dnnl::inner_product_forward(inner_product_pd);

    
    
/*     dnnl::memory s8_mem1 =
            dnnl::memory(inner_product_pd.src_desc(), cpu_engine,in_s8_1);
    dnnl::memory s8_mem2 =
            dnnl::memory(inner_product_pd.weights_desc(), cpu_engine,in_s8_2);   */
          
   
    dnnl::memory s8_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine,NULL);
    //dnnl::memory s8_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine); 
    dnnl::memory s8_mem2(
                {{yrow, ycol}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab}, cpu_engine,NULL);
    dnnl::memory b_s8_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine,NULL);
     
    s8_mem1.set_data_handle(in_s8_1);
    s8_mem2.set_data_handle(in_s8_2);
    b_s8_mem2.set_data_handle(relay_out);

    
    dnnl::reorder(s8_mem2, b_s8_mem2).execute(engine_stream, s8_mem2, b_s8_mem2);

      
   // write_to_dnnl_memory(in_s8_1,s8_mem1);
    //write_to_dnnl_memory(in_s8_2,s8_mem2);        
    
/* 
    inner_product_prim.execute(
            engine_stream,
            {{DNNL_ARG_SRC, s8_mem1},
             {DNNL_ARG_WEIGHTS, b_s8_mem2},
             {DNNL_ARG_DST, f32_dst_mem}}); */
    inner_product_prim.execute(
            engine_stream,
            {{DNNL_ARG_SRC, s8_mem1},
             {DNNL_ARG_WEIGHTS, s8_mem2},
             {DNNL_ARG_DST, f32_dst_mem}});

    // Wait for the computation to finalize.
    engine_stream.wait(); 

    // for (size_t i = 0; i < xrow * yrow; i++)
    // {
    //     printf("out_f32[%ld]=%ld\n", i, (uint64_t)out_f32[i]);
    // }    
}

/* static void  compute_s8s8f32_inner_product_memcp(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        std::vector<int8_t*> in_s8_1_vec,
        int8_t* in_s8_2,
        float* out_f32) {

    std::unique_ptr<int8_t[]> in_s8_1(new int8_t[xrow * xcol]);
    size_t blocksize = xcol * sizeof(int8_t);

    int8_t* dest = in_s8_1.get();
    for (const auto& block : in_s8_1_vec) {
        std::memcpy(dest, block, blocksize);
        dest += blocksize;
    } 
    compute_s8s8f32_inner_product(xrow,xcol,yrow,ycol,in_s8_1.get(),in_s8_2,out_f32);       
} */

inline void prefetch_data(int8_t* data, size_t size) {
    
    for (size_t i = 0; i < size; i+=256) {
        _mm_prefetch(reinterpret_cast<const char*>(&data[i]), _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(&data[i+64]), _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(&data[i+128]), _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(&data[i+192]), _MM_HINT_T2);

    }
}

int32_t add_all(int32_t *results,int32_t *result, uint64_t batchSize) __attribute__((optimize("-O0")));
int32_t add_all(int32_t *results,int32_t *result, uint64_t batchSize){
  for(int i=0;i<batchSize;i++){
    results[i]+=result[i];
  }
  return 0;
}

float amx_inner_product_matrix_int8( int8_t *libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;
  int blockCount=(dims)/DIM;
  int tailCount=dims%DIM;
  thread_local unsigned char *maInt8=NULL;
  thread_local unsigned char *mbInt8=NULL,*mbTemp=NULL;
  thread_local bool init_mem=false;

  thread_local int preBatchSizeA =0;
  thread_local int preBatchSizeB =0;


  thread_local char cfg[64]={0};
/*   int32_t *result;
  result=re */

/*   cfg[0]=1;
  cfg[16]=DIM;
  cfg[48] = batchSizeA;  // row->M batchsizeA(16) *DIM(64)  X  DIM(64)*batchsizeB(1) 
  // matrix B need a layout rearragement
  cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
  cfg[48+1]   = DIM/4;   // row = K/4

  cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
  cfg[48+2] = batchSizeA;  */


  if(!init_mem){
    if(maInt8) free(maInt8);
    maInt8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    preBatchSizeA=batchSizeA;
    if(mbInt8) free(mbInt8);
    if(mbTemp) free(mbTemp);
    mbInt8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    mbTemp = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);

    preBatchSizeB=batchSizeB;
     
    cfg[0]=1;
    cfg[16]=DIM;
    cfg[48] = 16;  // row->M batchsizeA(16) *DIM(64)  X  DIM(64)*batchsizeB(1) 
    // matrix B need a layout rearragement
    cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
    cfg[48+1]   = DIM/4;   // row = K/4

    cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
    cfg[48+2] = 16;  

    _tile_loadconfig((void*)cfg); 
    init_mem=true;
    //printf("we are 318 lines\n");
  }
 // _tile_loadconfig((void*)cfg); 
  memset(maInt8,0,16*DIM);

  
  for(int i=0;i<blockCount;i++){
    int32_t stride=i*DIM;
    
    for(int j=0;j<batchSizeA;j++){  
      
      __m512i sa=_mm512_loadu_epi64(libraryMatrix+j*dims+stride);
      _mm512_store_epi64(maInt8+j*DIM,sa);
      //_mm_prefetch((char *) (maInt8 + (j+1)*DIM), _MM_HINT_T0);
/*       uint64_t* sa = (uint64_t*)(libraryMatrix[j]+stride);  
      uint64_t* da = (uint64_t*)(maInt8+j*DIM);
      da[0]=sa[0];da[1]=sa[1];da[2]=sa[2];da[3]=sa[3];da[4]=sa[4];da[5]=sa[5];da[6]=sa[6];da[7]=sa[7]; */
      //memcpy(maInt8+j*DIM, libraryMatrix[j]+i*DIM,sizeof(char)*DIM);
     }
 /*   __m512i sa=_mm512_loadu_epi64(libraryMatrix[batchSizeA-1]+stride);
    _mm512_store_epi64(maInt8+(batchSizeA-1)*DIM,sa); */
    for(int j=0;j<batchSizeB;j++){
      __m512i sb=_mm512_loadu_epi64(queryMatrix+j*dims+stride);
      _mm512_store_epi64(mbInt8+j*DIM,sb);
/*       uint64_t* sb = (uint64_t*)(queryMatrix+j*dims+stride);
      uint64_t* db = (uint64_t*)(mbInt8+j*DIM);  
      db[0]=sb[0];db[1]=sb[1];db[2]=sb[2];db[3]=sb[3];db[4]=sb[4];db[5]=sb[5];db[6]=sb[6];db[7]=sb[7]; */
      //memcpy(mbInt8+j*DIM, queryMatrix+j*dims+i*DIM,sizeof(char)*DIM);
    }
    
    int KPACK=4/sizeof(int8_t);

/*     for (int k = 0; k < DIM; k++) {
      for (int j = 0; j < batchSizeB; j++) {        
          *((int8_t*)(mbTemp+(k/KPACK*batchSizeB*KPACK+j*KPACK+k%KPACK))) = *((int8_t*)(mbInt8+(j*DIM+k)));
          //std::cout<<    *((u16*)(mbInt8+2*(k*batchSizeB+j))) << "     ";
      }
    }  */
/*     for (int k = 0; k < DIM; k++) {
      for (int j = 0; j < batchSizeA; j++) {        
          *((int8_t*)(mbTemp+(k/KPACK*batchSizeA*KPACK+j*KPACK+k%KPACK))) = *((int8_t*)(maInt8+(j*DIM+k)));
          //std::cout<<    *((u16*)(mbInt8+2*(k*batchSizeB+j))) << "     ";
      }
    }  */

    //amx_int8_mul((u64*) cfg, mbInt8,mbTemp,DIM,batchSizeB*2*2,(uint8_t*)result);
    amx_int8_mul((u64*) cfg, maInt8,mbInt8,DIM,batchSizeB*4,(void*)results);
  }
  _tile_stored(2,results,batchSizeB*4);
  _tile_zero(2);
/*   for(int j=0;j<batchSizeA*batchSizeB;j++){
        //printf("cfg : %lld ,batchsizea : %d id :%d %d\n",(uint64_t*)cfg[2], batchSizeA,j,result[j]);
      results[j]+=result[j];
  }  */
  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        for(int i=0;i<tailCount;i++){
          results[k*batchSizeB+l]+=libraryMatrix[k*dims+DIM*blockCount+i]*queryMatrix[l*dims+DIM*blockCount+i];
        }
      }
    }
  }
  return 0;
}


static int32_t vector_dot_product_int32_t(const void* a, const void* b, const void *qty_ptr) {

    uint32_t length = * (uint32_t*)qty_ptr;
    // 确保长度是 64 的倍数，因为每个 _mm512_dpbusd_epi32 操作处理 64 个元素
    if (length % 64 != 0) {
        fprintf(stderr, "Length must be a multiple of 64\n");
        return 0;
    }

    __m512i sum = _mm512_setzero_si512();  // 初始化累加和为 0

    int8_t *a_tmp=(int8_t *)a;
    int8_t *b_tmp=(int8_t *)b;

    for (size_t i = 0; i < length; i += 64) {
        // 加载数据
        __m512i va = _mm512_loadu_si512((__m512i*)&a_tmp[i]);
        __m512i vb = _mm512_loadu_si512((__m512i*)&b_tmp[i]);

        // 执行点积运算
        sum = _mm512_dpbusd_epi32(sum, va, vb);
    }

    // 将 SIMD 寄存器中的结果累积到一个标量值
    int32_t result[16];
    _mm512_storeu_si512((__m512i*)result, sum);

    // 累加所有部分结果
    int32_t final_result = 0;
    for (int i = 0; i < 16; ++i) {
        final_result += result[i];
    }

    return final_result;
}

float devided_batch_amx_inner_product(int8_t **libraryMatrix, int8_t **queryMatrix,uint64_t dims,uint64_t nSize,uint64_t mSize,int32_t* results_amx){
  int batchSizeA=16,batchSizeB=16;
  int batchCountA= (nSize-1)/batchSizeA+1;
  int batchCountB= (mSize-1)/batchSizeB+1;


  int32_t * results_ptr=results_amx;

  for(int i=0;i<batchCountA;i++){
  //maBf16+=i*batchSizeA;

    if (i == batchCountA-1){
      batchSizeA = (nSize%batchSizeA == 0) ? batchSizeA : nSize%batchSizeA;
    }
    for(int j=0;j<batchCountB;j++){
    
      if (j == batchCountB-1){
        batchSizeB = (mSize%batchSizeB == 0) ? batchSizeB : mSize%batchSizeB;
      }
      //printf("i: %d j:%d value:\n",i,j,queryMatrix+j*16*dims));
      amx_inner_product_matrix_int8( *libraryMatrix+i*16*dims,*queryMatrix+j*16*dims,dims, batchSizeA, batchSizeB, results_ptr);
      results_ptr+=batchSizeB*batchSizeA;
    }
  }
  return 0;
}


/* static void omp_pure_memcpy(size_t xrow, size_t blocksize, int8_t* reserve, const std::vector<int8_t*>& in_s8_1_vec) {
    size_t chunk_size = blocksize / sizeof(uint8_t);
 
    #pragma omp parallel for 
    for (size_t i = 0; i < xrow; ++i) {
        int8_t* dest = reserve + i * blocksize;
        int8_t* src = in_s8_1_vec[i];
 
        #pragma omp simd
        for (size_t j = 0; j < chunk_size; ++j) {
            dest[j] = src[j];
        }
    }
}

static void omp_memcpy(size_t xrow, size_t blocksize, int8_t* reserve, std::vector<int8_t*> in_s8_1_vec){
   //#pragma omp parallel num_threads(16)
   //{ 
    //#pragma omp for
    //omp_set_num_threads(1);
    // #pragma omp parallel for num_threads(16)
    for (size_t i = 0; i < xrow; ++i) {
        std::memcpy(reserve + i * blocksize, in_s8_1_vec[i], blocksize);
       // int id = omp_get_thread_num();
        // printf("Thread %d\n", id);
       // if (i + 4 < xrow) {
       //    prefetch_data(in_s8_1_vec[i + 4], blocksize);
       // }
    
    }
   //}
    
}



static void compute_bf16bf16f32_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    uint16_t* in_bf16_1, uint16_t* in_bf16_2, float* out_f32) {

    // Initialize CPU engine and stream
    // dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
    // dnnl::stream engine_stream(cpu_engine);

    dnnl::memory::desc bf16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
    dnnl::memory::desc bf16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
    dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32);

    dnnl::inner_product_forward::primitive_desc inner_product_pd = dnnl::inner_product_forward::primitive_desc(
        cpu_engine, 
        dnnl::prop_kind::forward_training,
        bf16_md1, bf16_md2, f32_dst_md2);

    dnnl::inner_product_forward inner_product_prim = dnnl::inner_product_forward(inner_product_pd);

    dnnl::memory bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine, in_bf16_1);
    dnnl::memory bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine, in_bf16_2);

    inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                               {DNNL_ARG_WEIGHTS, bf16_mem2},
                                               {DNNL_ARG_DST, f32_dst_mem}});
    engine_stream.wait();

    // for (size_t i = 0; i < xrow; i++) {
    //     float* data = out_f32 + i * yrow;
    //     for (size_t j = 0; j < yrow; ++j) {
    //         printf("%f ", data[j]);
    //     }
    //     printf("\n");
    // }
}


static void compute_bf16bf16f32_inner_product_reorder(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        uint16_t* in_bf16_1,
        uint16_t* in_bf16_2,
        float* out_f32) {
    dnnl::memory::desc bf16_md1 = dnnl::memory::desc(
            {xrow, xcol},
            dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::ab);
    dnnl::memory::desc bf16_md2 = dnnl::memory::desc(
            {yrow, ycol},
            dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::any);
    dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc(
            {xrow, yrow},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab);
 
    dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32);

    dnnl::inner_product_forward::primitive_desc inner_product_pd =
            dnnl::inner_product_forward::primitive_desc(
                    cpu_engine,
                    dnnl::prop_kind::forward_training,
                    bf16_md1,
                    bf16_md2,
                    f32_dst_md2);
 
    dnnl::inner_product_forward inner_product_prim =
            dnnl::inner_product_forward(inner_product_pd);
   
    dnnl::memory bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine,in_bf16_1);
    dnnl::memory bf16_mem2(
                {{yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab}, cpu_engine);
    dnnl::memory bf16_weight_mem = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine);
    bf16_mem2.set_data_handle(in_bf16_2);
    dnnl::reorder(bf16_mem2, bf16_weight_mem).execute(engine_stream, bf16_mem2, bf16_weight_mem);
 
 
    inner_product_prim.execute(
            engine_stream,
            {{DNNL_ARG_SRC, bf16_mem1},
             {DNNL_ARG_WEIGHTS, bf16_weight_mem},
             {DNNL_ARG_DST, f32_dst_mem}});
 
    // Wait for the computation to finalize.
    engine_stream.wait();
   
    // printf("comput_f32bf16f32_inner_product finished#######>\n");
}
 

static void compute_bf16bf16f32_inner_product_cm_memcp(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        std::vector<uint16_t*> in_bf16_1_vec,
        uint16_t* in_bf16_2,
        float* out_f32,
        uint16_t* reserve) {

    // size_t blocksize = xcol * sizeof(uint16_t);
    size_t blocksize = xcol;
    // size_t chunk_size = blocksize / sizeof(__m256i);
#ifdef OUTPUT
    // printf("start copying\n");
    struct timeval t11,t22;
    gettimeofday(&t11,NULL);
#endif    
    omp_memcpy_bf16(xrow, xcol, blocksize,  reserve, in_bf16_1_vec);   
#ifdef OUTPUT    
    gettimeofday(&t22,NULL);
    double cpy_timeuse = ((t22.tv_sec - t11.tv_sec) + (double)(t22.tv_usec - t11.tv_usec)/1000000.0);
    std::cout << "===> omp memcpy elpased: " << cpy_timeuse << std::endl; 
#endif 
    // printf("finish copy\n");
    // for (size_t i = 0; i < xcol * xrow; i++) {
    //     printf("reserve[%ld]=%ld, in_bf16_1_vec[%ld]=%ld\n", i, (uint16_t)reserve[i], i, (uint16_t)in_bf16_2[i]);
    // }
//    for (size_t i = 0; i <  xrow; ++i) {
//         uint16_t* data = reserve+i*blocksize;
        
//         // Generate random data and store it
//         for (size_t j = 0; j < xcol; ++j) {
//             printf("%ld ", data[j]);
//         }
//         printf("\n");
//     }           
    // printf("start amx onednn\n");
#ifdef OUTPUT 
    gettimeofday(&t11,NULL);
#endif      
    compute_bf16bf16f32_inner_product(xrow, xcol, yrow, ycol, reserve, in_bf16_2,  out_f32);
#ifdef OUTPUT     
    gettimeofday(&t22,NULL);
    double ip_timeuse = ((t22.tv_sec - t11.tv_sec) + (double)(t22.tv_usec - t11.tv_usec)/1000000.0);
    std::cout << "===>inner product elpased: " << ip_timeuse << std::endl;  
    std::cout << "===>total elpased: " << ip_timeuse + cpy_timeuse << std::endl;  
#endif          
    // printf("finish compute_bf16bf16f32_inner_product\n");
}

static void compute_bf16bf16f32_inner_product_cm_memcp_pool(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        std::vector<uint16_t*> in_bf16_1_vec,
        uint16_t* in_bf16_2,
        float* out_f32,
        uint16_t* reserve,
        ThreadPool& pool,
        size_t threads_num) {
    
    // 确定每个线程需要处理的数据块大小
    size_t chunk_size = xrow / threads_num;
    size_t remaining = xrow % threads_num;

    // 创建一个vector来存储future对象，以确保主线程可以等待所有任务完成
    std::vector<std::future<void>> futures;

    // 为每个线程分配任务
    // printf("start thread pool memcpy\n");
    // struct timeval t11,t22;
    // gettimeofday(&t11,NULL);    
    for (size_t t = 0; t < threads_num; ++t) {
        size_t start_row = t * chunk_size;
        size_t end_row = (t + 1) * chunk_size;
        if (t == threads_num - 1) {
            end_row += remaining; // 最后一个线程处理剩余的行
        }

        // 创建一个任务并提交给线程池
        futures.push_back(pool.enqueue([start_row, end_row, xcol, &in_bf16_1_vec, reserve]() {
            for (size_t i = start_row; i < end_row; ++i) {
                std::memcpy(&reserve[i * xcol], in_bf16_1_vec[i], xcol * sizeof(uint16_t));
            }
        }));
    }      

    // 等待所有任务完成
    for (auto &f : futures) {
        f.get();
    }
    // gettimeofday(&t22,NULL);
    // double timeuse = ((t22.tv_sec - t11.tv_sec) + (double)(t22.tv_usec - t11.tv_usec)/1000000.0);
    // std::cout << "===> thread pool memcpy elpased: " << timeuse << std::endl;  
    // printf("start ip\n");
    // gettimeofday(&t11,NULL);
    compute_bf16bf16f32_inner_product(yrow, ycol, xrow, xcol, in_bf16_2, reserve, out_f32);
    // gettimeofday(&t22,NULL);
    // double ip_timeuse = ((t22.tv_sec - t11.tv_sec) + (double)(t22.tv_usec - t11.tv_usec)/1000000.0);
    // std::cout << "===>inner product elpased: " << ip_timeuse << std::endl;       
}


static void compute_s8s8f32_inner_product_cm_memcp_omp(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        std::vector<int8_t*> in_s8_1_vec,
        int8_t* in_s8_2,
        float* out_f32,
        int8_t* reserve) {

    size_t blocksize = xcol * sizeof(int8_t);
    size_t offset = 0;

    // 预取前几个块的数据
    for (size_t i = 0; i < 4 && i < xrow; ++i) {
        prefetch_data(in_s8_1_vec[i], blocksize);
    }

    // 使用 OpenMP 并行化 memcpy
    //#pragma omp parallel for
    //omp_set_dynamic(0);
    omp_memcpy(xrow,  blocksize,  reserve, in_s8_1_vec);
    int proc_nums = omp_get_num_procs();
   // printf("dynamic is %d\n",dy);
    //omp_set_num_threads(32);
    compute_s8s8f32_inner_product( yrow, ycol, xrow, xcol, in_s8_2, reserve,out_f32);
}

static void compute_s8s8f32_inner_product_cm_memcp_pool(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        std::vector<int8_t*> in_s8_1_vec,
        int8_t* in_s8_2,
        float* out_f32,
        int8_t* reserve,
        ThreadPool& pool,
        size_t threads_num) {

    size_t blocksize = xcol * sizeof(int8_t);
    size_t offset = 0;

    // 预取前几个块的数据
    for (size_t i = 0; i < 4 && i < xrow; ++i) {
        prefetch_data(in_s8_1_vec[i], blocksize);
    }

    // 使用 thradpool 并行化 memcpy
     int slice = xrow / threads_num;
     int remainder = xrow % threads_num;
     std::vector< std::future<int> > results;
     for (int i = 0; i < threads_num-1; ++i) {
        results.emplace_back(pool.enqueue(parall_memcpy,reserve + i * slice * blocksize,in_s8_1_vec,blocksize,i*slice,slice));
    }
    results.emplace_back(pool.enqueue(parall_memcpy,reserve + (threads_num-1) * slice * blocksize,in_s8_1_vec,blocksize,(threads_num-1) * slice ,slice+remainder));

       
    for (auto& result : results) {
        result.get();
    }

    compute_s8s8f32_inner_product(xrow, xcol, yrow, ycol, reserve, in_s8_2, out_f32);
}

static void  compute_s8s8f32_inner_product_cm_memcp(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        std::vector<int8_t*> in_s8_1_vec,
        int8_t* in_s8_2,
        float* out_f32,
        int8_t* reserve) {

    size_t blocksize = xcol * sizeof(int8_t);
    //std::memset(reserve, 0, xrow * xcol * sizeof(int8_t));
    size_t offset = 0;
    
    for (const auto& block : in_s8_1_vec) {
         //_mm_prefetch(reinterpret_cast<const char*>(block), _MM_HINT_T2);

        std::memcpy(reserve + offset, block, blocksize);


        offset += blocksize;
    } 
    
  
    compute_s8s8f32_inner_product(xrow,xcol,yrow,ycol,reserve,in_s8_2,out_f32);       
}

extern int8_t in_s8_cm[];
extern std::unique_ptr<int8_t[]> in_s8_pt;
 */