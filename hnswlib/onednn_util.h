
#pragma once
#include <stdlib.h>
#include <mutex>
#include <shared_mutex>
#include <cstring> 
//#include "oneapi/dnnl/dnnl.hpp"
#include <immintrin.h> 
#include <omp.h>
#include <sys/time.h>
#include <sys/syscall.h> 
#include <unistd.h>


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



int32_t add_all(int32_t *results,int32_t *result, uint64_t batchSize) __attribute__((optimize("-O0")));
int32_t add_all(int32_t *results,int32_t *result, uint64_t batchSize){
  for(int i=0;i<batchSize;i++){
    results[i]+=result[i];
  }
  return 0;
}

float fvec_inner_product_int8_avx2int8_tail(const void* a, const void* b, const void *qty_ptr) {
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



    // 将 SIMD 寄存器中的结果累积到一个标量值
    int32_t result[8];
    _mm256_storeu_si256((__m256i*)result, sum256);

    float dotsum = 0;
    for (int i = 0; i < 8; ++i) {
        dotsum += result[i];
    }


    for (size_t i =0; i < *((size_t*)qty_ptr)-32 * qty32; i++) {
        dotsum +=pvec_u8[i] * pvec_s8[i];
    }
    return dotsum;
}
static int32_t vector_dot_product_int32_t(const void* a, const void* b, const void *qty_ptr) {

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
float amx_inner_product_matrix_int8_1( int8_t **libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;

  int blockDim = 384;
  int blockCount=((dims))/blockDim;
  int tailCount=dims%DIM;
  thread_local bool init_mem=false;
  thread_local char cfg[64]={0};

  thread_local int8_t *preQuery=NULL;


  unsigned char ma1Int8[1024] __attribute__((aligned(64)));
  unsigned char ma2Int8[1024] __attribute__((aligned(64)));
  unsigned char ma3Int8[1024] __attribute__((aligned(64)));
  unsigned char ma4Int8[1024] __attribute__((aligned(64)));
  unsigned char ma5Int8[1024] __attribute__((aligned(64)));
  unsigned char ma6Int8[1024] __attribute__((aligned(64)));
  
  if(!init_mem){
    cfg[0]=1;
    //0
    cfg[16]=64;
    cfg[48] = DIM/4;

    //1
    cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
    cfg[48+1]   = DIM/4;    // row = K/4
    //2
    cfg[16+2*2] = 16*2*2;; // N*sizeof(int32)
    cfg[48+2] = DIM/4;
    //3
    cfg[22]=  16*2*2;;
    cfg[51] =  DIM/4;  // row->M
    
    //4
    cfg[24] =  (16*4);   // col = N*4
    cfg[52]   = 16;   // row = K/4
    //5
    cfg[26]=(16*4);;
    cfg[53] = 16;  // row->M
    //6
    cfg[28] = (16*4);;   // col = N*4
    cfg[54]   = 16;   // row = K/4

    cfg[30]=batchSizeB*2*2;;
    cfg[55]=16;

    _tile_loadconfig((void*)cfg); 
    init_mem=true;
  }
  
  //memset(ma1Int8,0,16*DIM);  

  __m512i sa;
  for(int i=0;i<blockCount;i++){
    for(int j=0;j<batchSizeA;j++){
      sa=_mm512_load_si512(libraryMatrix[j]+i*blockDim);
      _mm512_store_si512(ma1Int8+j*DIM,sa);
      sa=_mm512_load_si512(libraryMatrix[j]+i*blockDim+64);
      _mm512_store_si512(ma2Int8+j*DIM,sa);
      sa=_mm512_load_si512(libraryMatrix[j]+i*blockDim+128);
      _mm512_store_si512(ma3Int8+j*DIM,sa);
      sa=_mm512_load_si512(libraryMatrix[j]+i*blockDim+192);
      _mm512_store_si512(ma4Int8+j*DIM,sa);
      sa=_mm512_load_si512(libraryMatrix[j]+i*blockDim+256);
      _mm512_store_si512(ma5Int8+j*DIM,sa);
      sa=_mm512_load_si512(libraryMatrix[j]+i*blockDim+320);
      _mm512_store_si512(ma6Int8+j*DIM,sa);
    }


    _tile_loadd(0,ma1Int8, 64);
    _tile_loadd(2,ma2Int8, 64);
    _tile_loadd(3,ma3Int8, 64);
    _tile_loadd(4,ma4Int8, 64);
    _tile_loadd(5,ma5Int8, 64);
    _tile_loadd(6,ma6Int8, 64);

    _tile_loadd(1,queryMatrix + i * blockDim , 4);
    _tile_dpbuud(7,0,1);
    _tile_loadd(1,queryMatrix + i * blockDim + 64 , 4);
    _tile_dpbuud(7,2,1);
    _tile_loadd(1,queryMatrix + i * blockDim + 128 , 4);
    _tile_dpbuud(7,3,1);
    _tile_loadd(1,queryMatrix + i * blockDim + 192, 4);
    _tile_dpbuud(7,4,1);
    _tile_loadd(1,queryMatrix + i * blockDim + 256, 4);
    _tile_dpbuud(7,5,1);
    _tile_loadd(1,queryMatrix + i * blockDim + 320, 4);
    _tile_dpbuud(7,6,1);   
  }
  _tile_stored(7,results,batchSizeB*4);
  _tile_zero(7);

  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        results[k*batchSizeB+l]+=vector_dot_product_int32_t(libraryMatrix[k]+DIM*blockCount,queryMatrix+l*dims+DIM*blockCount,&tailCount);
      }
    }
  }
  return 0;
} 
/* float amx_inner_product_matrix_int8( int8_t **libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;

  //int blockDim = 192;
  int blockCount=((dims))/DIM;
  int tailCount=dims%DIM;
  thread_local unsigned char *ma1Int8=NULL, *ma2Int8=NULL, *ma3Int8=NULL;
  thread_local bool init_mem=false;
  thread_local char cfg[64]={0};

  thread_local int8_t *preQuery=NULL;

  
  if(!init_mem){
    if(ma1Int8) free(ma1Int8);
    ma1Int8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    ma2Int8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    ma3Int8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    cfg[0]=1;
    cfg[16]=DIM;
    cfg[48] = 16;  // row->M batchsizeA(16) *DIM(64)  X  DIM(64)*batchsizeB(1) 
    // matrix B need a layout rearragement
    cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
    cfg[48+1]   = DIM/4;   // row = K/4

    cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
    cfg[48+2] = 16;  

    cfg[22]=(batchSizeB*4);
    cfg[51] = 16;  // row->M
    // matrix B need a layout rearragement
    cfg[24] = batchSizeB*2*2;   // col = N*4
    cfg[52]   =16;   // row = K/4

    cfg[26]=(batchSizeB*4);
    cfg[53] = 16;  // row->M
    // matrix B need a layout rearragement
    cfg[28] = batchSizeB*2*2;   // col = N*4
    cfg[54]   = 16; 

    cfg[30] = (batchSizeB*4); // N*sizeof(int32)
    cfg[55] = 16;  

    _tile_loadconfig((void*)cfg); 
    init_mem=true;
  }

  if(preQuery!=queryMatrix){
    switch(7-blockCount){
      case 1: _tile_loadd(6,queryMatrix+320, 4);
      case 2: _tile_loadd(5,queryMatrix+256, 4);
      case 3: _tile_loadd(4,queryMatrix+192, 4);
      case 4: _tile_loadd(3,queryMatrix+128, 4);
      case 5: _tile_loadd(2,queryMatrix+64 , 4);
      case 6: _tile_loadd(1,queryMatrix+0 , 4);
    }

    preQuery=queryMatrix;
  }
  
  memset(ma1Int8,0,16*DIM);  

  __m512i sa;
  for(int i=0;i<blockCount;i++){
    for(int j=0;j<batchSizeA;j++){
      sa=_mm512_loadu_si512(libraryMatrix[j]+i*DIM);
      _mm512_store_si512(ma1Int8+j*DIM,sa);
    }

    _tile_loadd(0,ma1Int8,64);

    switch(i+1){
      case 1: _tile_dpbuud(7,0,1); break;
      case 2: _tile_dpbuud(7,0,2); break;
      case 3: _tile_dpbuud(7,0,3); break;
      case 4: _tile_dpbuud(7,0,4); break;
      case 5: _tile_dpbuud(7,0,5); break;
      case 6: _tile_dpbuud(7,0,6); break;
    }
    
  }
  _tile_stored(7,results,batchSizeB*4);
  _tile_zero(7);

  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        results[k*batchSizeB+l]+=vector_dot_product_int32_t(libraryMatrix[k]+DIM*blockCount,queryMatrix+l*dims+DIM*blockCount,&tailCount);
      }
    }
  }
  return 0;
} */

float amx_inner_product_matrix_int8( int8_t **libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;

  int blockDim = 192;
  int blockCount=((dims))/blockDim;
  size_t tailCount=dims%DIM;
  int tailBlock=dims%blockDim;
  //thread_local unsigned char *ma1Int8=NULL, *ma2Int8=NULL, *ma3Int8=NULL;
  thread_local bool init_mem=false;
  thread_local char cfg[64]={0};

  
  unsigned char ma1Int8[1024] __attribute__((aligned(64)));
  unsigned char ma2Int8[1024] __attribute__((aligned(64)));
  unsigned char ma3Int8[1024] __attribute__((aligned(64)));
  
  if(!init_mem){
    //if(ma1Int8) free(ma1Int8);
/*     ma1Int8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16*32);
    ma2Int8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    ma3Int8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16); */
    cfg[0]=1;
    cfg[16]=DIM;
    cfg[48] = 16;  // row->M batchsizeA(16) *DIM(64)  X  DIM(64)*batchsizeB(1) 
    // matrix B need a layout rearragement
    cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
    cfg[48+1]   = DIM/4;   // row = K/4

    cfg[22]=DIM;
    cfg[51] = 16;  // row->M
    // matrix B need a layout rearragement
    cfg[24] = batchSizeB*2*2;   // col = N*4
    cfg[52]   = DIM/4;   // row = K/4

    cfg[26]=DIM;
    cfg[53] = 16;  // row->M
    // matrix B need a layout rearragement
    cfg[28] = batchSizeB*2*2;   // col = N*4
    cfg[54]   = DIM/4; 

    cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
    cfg[48+2] = 16;  

    _tile_loadconfig((void*)cfg); 
    init_mem=true;
  }
  
/*   memset(ma1Int8,0,16*DIM);  
  memset(ma2Int8,0,16*DIM); 
  memset(ma3Int8,0,16*DIM);   */

  // for(auto j=0;j<batchSizeA;j++){
  //   _mm_prefetch(libraryMatrix[j], _MM_HINT_T0);  
  // }
  for(int i=0;i<blockCount;i++){

    //int32_t stride=i*DIM;
    __m512i sa;
    size_t offset = i * blockDim;
    

    
    for(int j=0;j<batchSizeA;j++){  
      
      size_t destOffset1 = j * DIM;
      _mm512_store_si512(ma1Int8 + destOffset1, _mm512_loadu_si512(libraryMatrix[j] + offset));
      _mm512_store_si512(ma2Int8 + destOffset1, _mm512_loadu_si512(libraryMatrix[j] + offset + 64));
      _mm512_store_si512(ma3Int8 + destOffset1, _mm512_loadu_si512(libraryMatrix[j] + offset + 128));
    } 

    _tile_loadd(1,queryMatrix + i * blockDim , 4);
    _tile_loadd(4,queryMatrix + i * blockDim + 64 , 4);
    _tile_loadd(6,queryMatrix + i * blockDim + 128, 4);
    _tile_loadd(0,ma1Int8, 64);
    _tile_loadd(3,ma2Int8, 64);
    _tile_loadd(5,ma3Int8, 64);
    _tile_dpbuud(2,3,4);
    _tile_dpbuud(2,0,1);
    _tile_dpbuud(2,5,6);
    //amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);
  }
  if(tailBlock >= DIM){
    for(int i=0;i<tailBlock/DIM;i++){
      __m512i sa;
      for(int j=0;j<batchSizeA;j++){  
        sa=_mm512_loadu_si512(libraryMatrix[j]+blockCount*blockDim+i*DIM);
        _mm512_store_si512(ma1Int8+j*DIM,sa);
      }
      _tile_loadd(0,ma1Int8, 64);
      _tile_loadd(1,queryMatrix + blockCount*blockDim + i * DIM , 4);
      _tile_dpbuud(2,0,1);
    }
  }

  _tile_stored(2,results,batchSizeB*4);
  _tile_zero(2);

  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        results[k*batchSizeB+l]+=(int32_t)vector_dot_product_int32_t(libraryMatrix[k]+(dims/DIM)*DIM,queryMatrix+l*dims+(dims/DIM)*DIM,&tailCount);
      }
    }
  }
  return 0;
}

float amx_inner_product_matrix_int8_2( int8_t **libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;
  int blockCount=(dims)/DIM;
  int tailCount=dims%DIM;
  thread_local unsigned char *maInt8,*mbTemp=NULL,*mbInt8;
  thread_local bool init_mem=false;
  thread_local char cfg[64]={0};

  thread_local int8_t *preQuery=NULL;

  thread_local int32_t result[256]={0};

  if(!init_mem){
    if(maInt8) free(maInt8);
    maInt8 = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);
    mbTemp = (unsigned char*) aligned_alloc (64,sizeof(char)*DIM*16);

    cfg[0]=1;
    //0
    cfg[16]=64;
    cfg[48] = DIM/4;

    //1
    cfg[16+1*2] = 16*2*2;   // col = N*4
    cfg[48+1]   = DIM/4;   // row = K/4
    //2
    cfg[16+2*2] = 16*2*2;; // N*sizeof(int32)
    cfg[48+2] = DIM/4;
    //3
    cfg[22]=  16*2*2;;
    cfg[51] =  DIM/4;  // row->M
    
    //4
    cfg[24] =  (16*4);   // col = N*4
    cfg[52]   = 16;   // row = K/4
    //5
    cfg[26]=(16*4);;
    cfg[53] = 16;  // row->M
    //6
    cfg[28] = (16*4);;   // col = N*4
    cfg[54]   = 16;   // row = K/4

    _tile_loadconfig((void*)cfg); 
    init_mem=true;


  } 
  thread_local int count=0;
  if(queryMatrix!=preQuery){
    int  KPACK=4;
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < DIM; k++) {
        int8_t tmp =125;

        *((char*)(mbTemp+(k/KPACK*16*KPACK+j*KPACK+k%KPACK))) = *((int8_t*)(mbInt8+(j*DIM+k)));
        //std::cout<<(int32_t)queryMatrix[j*DIM+k]<<" ";
        //*((char*)(mbTemp+(k/KPACK*16*KPACK+j*KPACK+k%KPACK))) = *((char*)&tmp);
      }
    }
     _tile_loadd(0, mbTemp, 64);
     preQuery=queryMatrix;
  }

  for(int j = 0; j < batchSizeA/3; j++){

      //_mm_prefetch((char *) (libraryMatrix[j+4]), _MM_HINT_T0);
      _tile_loadd(1,libraryMatrix[3*j],64);
      _tile_loadd(2,libraryMatrix[3*j+1],64);
      _tile_loadd(3,libraryMatrix[3*j+2],64);

      _tile_dpbuud(4, 1, 0);
      _tile_dpbuud(5, 2, 0);
      _tile_dpbuud(6, 3, 0);

      _tile_stored(4, result, 16*2*2);
      int32_t res=0;
      for(int k = 0; k < 16; k++){
          res+=result[k*17];
      }
      results[3*j]=res;

      _tile_stored(5, result, 16*2*2);
      res=0;
      for(int k = 0; k < 16; k++){
          res+=result[k*17];
      }
      results[3*j+1]=res;

      _tile_stored(6, result, 16*2*2);
      res=0;
      for(int k = 0; k < 16; k++){
          res+=result[k*17];
      }
      results[3*j+2]=res;
      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6); 

  }

  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        results[k*batchSizeB+l]+=vector_dot_product_int32_t(libraryMatrix[k]+DIM*blockCount,queryMatrix+l*dims+DIM*blockCount,&tailCount);
      }
    }
  }
  return 0;
} 


#if 0
float amx_inner_product_matrix_int8( int8_t **libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;
  int blockCount=(dims)/DIM;
  int tailCount=dims%DIM;
  //thread_local unsigned char *maInt8=NULL;

  unsigned char maInt8[1024] __attribute__((aligned(64)));
  thread_local bool init_mem=false;
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
  }
  

  for(int i=0;i<blockCount;i++){

    int32_t stride=i*DIM;
    __m512i sa;
    for(int j=0;j<batchSizeA;j++){  
      sa=_mm512_loadu_si512(libraryMatrix[j]+stride);
      _mm512_store_si512(maInt8+j*DIM,sa);
    } 
    amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);
  }
  _tile_stored(2,results,batchSizeB*4);
  _tile_zero(2);

  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        results[k*batchSizeB+l]+=vector_dot_product_int32_t(libraryMatrix[k]+DIM*blockCount,queryMatrix+l*dims+DIM*blockCount,&tailCount);
      }
    }
  }
  return 0;
}
//#else 

float amx_inner_product_matrix_int8( int8_t **libraryMatrix, int8_t *queryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, int32_t *results){
  int DIM=64;
  int blockCount=(dims)/DIM;
  int tailCount=dims%DIM;
  thread_local unsigned char *maInt8=NULL;
 // thread_local unsigned char *mbInt8=NULL,*mbTemp=NULL;
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
     
    cfg[0]=1;
    cfg[16]=DIM;
    cfg[48] = 16;  // row->M batchsizeA(16) *DIM(64)  X  DIM(64)*batchsizeB(1) 
    // matrix B need a layout rearragement
    cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
    cfg[48+1]   = DIM/4;   // row = K/4

    cfg[22]=DIM;
    cfg[51] = 16;  // row->M
    // matrix B need a layout rearragement
    cfg[24] = batchSizeB*2*2;   // col = N*4
    cfg[52]   = DIM/4;   // row = K/4

    cfg[26]=DIM;
    cfg[53] = 16;  // row->M
    // matrix B need a layout rearragement
    cfg[28] = batchSizeB*2*2;   // col = N*4
    cfg[54]   = DIM/4;

    cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
    cfg[48+2] = 16;  

    _tile_loadconfig((void*)cfg); 
    init_mem=true;
  }

  memset(maInt8,0,16*DIM); 
  int i;

/*   for(int j=0;j<batchSizeA;j++){  
      _mm_prefetch((char *) (libraryMatrix[j]), _MM_HINT_T1);
  }  */
  for(i=0;i<blockCount/3;i++){
    __m512i sa;
    for(int j=0;j<batchSizeA;j++){  
      //_mm_prefetch((char *) (libraryMatrix[j]+(3*(i+1))*DIM), _MM_HINT_T0);
      sa=_mm512_loadu_si512(libraryMatrix[j] + 3 * i * DIM);
      _mm512_store_si512(maInt8+j*DIM,sa);
    } 
    _tile_loadd(0,maInt8, 64);
    _tile_loadd(1,queryMatrix + 3 * i * DIM , 4);
    for(int j=0;j<batchSizeA;j++){  
      sa=_mm512_loadu_si512(libraryMatrix[j]+(3 * i + 1) * DIM);
      _mm512_store_si512(maInt8+j*DIM,sa);
    } 
    _tile_loadd(3,maInt8, 64);
    _tile_loadd(4,queryMatrix + ( 3 * i + 1) * DIM , 4);  

    for(int j=0;j<batchSizeA;j++){  
      sa=_mm512_loadu_si512(libraryMatrix[j]+ (3 * i + 2) * DIM);
      _mm512_store_si512(maInt8+j*DIM,sa);
    } 
    _tile_loadd(5,maInt8, 64);
    _tile_loadd(6,queryMatrix + (3 * i + 2) * DIM , 4);


    _tile_dpbuud(2,0,1);
    _tile_dpbuud(2,3,4);
    _tile_dpbuud(2,5,6);
  }

  switch(blockCount%3){
    case 0: break;
    case 1: 
      for(int j = 0; j < batchSizeA; j++) {
          __m512i data = _mm512_loadu_si512(libraryMatrix[j] + 3 * i * DIM);
          _mm512_store_si512(maInt8 + j * DIM , data);
      }
      _tile_loadd(0,maInt8, 64);
      _tile_loadd(1,queryMatrix + 3 * i * DIM , 4); 
      _tile_dpbuud(2,0,1);  
      break;

    case 2:         
      for(int j = 0; j < batchSizeA; j++) {
          __m512i data = _mm512_loadu_si512(libraryMatrix[j] + 3 * i * DIM);
          _mm512_store_si512(maInt8 + j * DIM , data);
      }
      _tile_loadd(0,maInt8, 64);
      _tile_loadd(1,queryMatrix + 3* i * DIM , 4);
      for(int j = 0; j < batchSizeA; j++) {
          __m512i data = _mm512_loadu_si512(libraryMatrix[j] + ( 3 * i + 1 ) * DIM);
          _mm512_store_si512(maInt8 + j * DIM , data);
      }
      _tile_loadd(3,maInt8, 64);
      _tile_loadd(4,queryMatrix + (3 * i + 1 ) * DIM , 4);
      _tile_dpbuud(2,0,1);
      _tile_dpbuud(2,3,4);
      break;
  }
/*     for (int k = 0; k < DIM; k++) {
      for (int j = 0; j < batchSizeB; j++) {        
          *((int8_t*)(mbTemp+(k/KPACK*batchSizeB*KPACK+j*KPACK+k%KPACK))) = *((int8_t*)(mbInt8+(j*DIM+k)));
          //std::cout<<    *((u16*)(mbInt8+2*(k*batchSizeB+j))) << "     ";
      }
    }  */
    // /amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);


  _tile_stored(2,results,batchSizeB*4);
  _tile_zero(2);

  if(tailCount!=0){
    for(int k=0;k<batchSizeA;k++){
      for(int l=0;l<batchSizeB;l++){
        results[k*batchSizeB+l]+=vector_dot_product_int32_t(libraryMatrix[k]+DIM*blockCount,queryMatrix+l*dims+DIM*blockCount,&tailCount);
      }
    }
  }
  return 0;
}
#endif




float devided_batch_amx_inner_product_int8(int8_t **libraryMatrix, int8_t **queryMatrix,uint64_t dims,uint64_t nSize,uint64_t mSize,int32_t* results_amx){
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
      
      amx_inner_product_matrix_int8( libraryMatrix+i*16,*queryMatrix+j*16*dims,dims, batchSizeA, batchSizeB, results_ptr);
      results_ptr+=batchSizeB*batchSizeA;
    }
  }
  return 0;
}
float amx_inner_product_matrix_fp32_1( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results){
    int DIM=32;
    int blockDim = 96;
    int blockCount=((dims))/blockDim;
    size_t tailCount=dims%DIM;
    int tailBlock=dims%blockDim;

    thread_local char cfg[64]={0};
    thread_local bool init_mem=false;

    unsigned char ma1Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma2Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma3Bf16[1024] __attribute__((aligned(64)));
    unsigned char mb1Bf16[1024] __attribute__((aligned(64)));
    unsigned char mb2Bf16[1024] __attribute__((aligned(64)));
    unsigned char mb3Bf16[1024] __attribute__((aligned(64)));
 
    if(!init_mem){
        cfg[0]=1;
        cfg[16]=DIM*2;
        cfg[48] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
        cfg[48+1]   = DIM/2;   // row = K/4
 
        cfg[22]=DIM*2;
        cfg[51] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[24] = batchSizeB*2*2;   // col = N*4
        cfg[52]   = DIM/2;   // row = K/4
 
        cfg[26]=DIM*2;
        cfg[53] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[28] = batchSizeB*2*2;   // col = N*4
        cfg[54]   = DIM/2;   // row = K/4
 
        cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
        cfg[48+2] = 16;
        init_mem = true;
        _tile_loadconfig((void *)cfg);
    }
    __m512i high_bits;
    __m512i low_bits;
    __m512i all_bits;
    __m512i temp;
 
    for(int i=0;i<blockCount;i++){
      size_t offset = i * blockDim * 4;
      
      for(int j=0;j<batchSizeA;j++){  
        size_t destOffset1 = j * DIM *2;
        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  offset),16);
        low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  offset + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(ma1Bf16 + destOffset1 , all_bits);

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  offset + 128),16);
        low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  offset + 192);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(ma2Bf16 + destOffset1 , all_bits);

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  offset + 256),16);
        low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  offset + 320);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(ma3Bf16 + destOffset1 , all_bits);
      } 


      high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  offset),16);
      low_bits = _mm512_loadu_si512(floatQueryMatrix +  offset + 64);
      all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
      _mm512_store_si512(mb1Bf16 , all_bits);

      high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  offset + 128),16);
      low_bits = _mm512_loadu_si512(floatQueryMatrix +  offset + 192);
      all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
      _mm512_store_si512(mb2Bf16 , all_bits);
            
      high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  offset + 256),16);
      low_bits = _mm512_loadu_si512(floatQueryMatrix +  offset + 320);
      all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
      _mm512_store_si512(mb3Bf16 , all_bits);


      _tile_loadd(0,ma1Bf16, 64);
      _tile_loadd(3,ma2Bf16, 64);
      _tile_loadd(5,ma3Bf16, 64);
      _tile_loadd(1,mb1Bf16, 4);
      _tile_loadd(4,mb2Bf16, 4);
      _tile_loadd(6,mb3Bf16, 4);

      _tile_dpbf16ps(2,0,1);  
      _tile_dpbf16ps(2,3,4);
      _tile_dpbf16ps(2,5,6);
    //amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);
    }
    if(tailBlock >= DIM){
      for(int i=0;i<tailBlock/DIM;i++){
        __m512i sa;
        int32_t offset= blockCount * blockDim * 4 + i * DIM * 4;
        for(int j=0;j<batchSizeA;j++){  

          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  offset),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  offset + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(ma1Bf16 + j * DIM * 2, all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  offset),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  offset + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mb1Bf16 , all_bits);

        _tile_loadd(0,ma1Bf16, 64);
        _tile_loadd(1,mb1Bf16 , 4);
        _tile_dpbf16ps(2,0,1);
      }
    }
/*     for( i = 0; i < blockCount; i+=1) {
      for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 2);
          _mm512_store_si512(maBf16 + j * DIM * 2 , high_bits);
      }
      high_bits =_mm512_loadu_si512(floatQueryMatrix +  i * DIM * 2);
      _mm512_store_si512(mbBf16 , high_bits);
      _tile_loadd(0,maBf16, 64);
      _tile_loadd(1,mbBf16, 4);
      _tile_dpbf16ps(2,0,1);

    } */

    /* for( int i = 0; i < dims/32; i+=1) {
      for(int j = 0; j < batchSizeA; j++) {
//            high_bits=_mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4);
//           low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4 + 64);
//           all_bits=_mm512_cvtne2ps_pbh(*(__m512*)&high_bits,*(__m512*)&low_bits);
//           _mm512_store_si512(maBf16 + j * DIM * 2 ,*(__m512i*)&all_bits); 
          temp=_mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4);
          high_bits = _mm512_srli_epi32(temp,16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(ma1Bf16 + j * DIM * 2 , all_bits);
      }
      // high_bits=_mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4);
      // low_bits = _mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4 + 64);
      // all_bits=_mm512_cvtne2ps_pbh(*(__m512*)&high_bits,*(__m512*)&low_bits);
      // _mm512_store_si512(mbBf16 , *(__m512i*)&all_bits);
      high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4),16);
      low_bits = _mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4 + 64);
      all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
      _mm512_store_si512(mb1Bf16 , all_bits);
      _tile_loadd(0,ma1Bf16, 64);
      _tile_loadd(1,mb1Bf16, 4);
      _tile_dpbf16ps(2,0,1);

    } */
    _tile_stored(2, results, batchSizeB*2*2);
    _tile_zero(2);
   
    if (tailCount != 0) {
        for (int k = 0; k < batchSizeA; k++) {
            for (int l = 0; l < batchSizeB; l++) {
                __m512 result_vec = _mm512_setzero_ps();
                for (int i = 0; i < tailCount; i += 16) {
                    __m512 lib_vec = _mm512_loadu_ps((float *)(floatLibraryMatrix[k])  + DIM * blockCount + i);
                    __m512 query_vec = _mm512_loadu_ps((float *)(floatQueryMatrix + DIM * blockCount + i));
                    result_vec = _mm512_fmadd_ps(lib_vec, query_vec, result_vec);
                }
                results[k * batchSizeB + l] += _mm512_reduce_add_ps(result_vec);
            }
        }
    }
 
    return 0;
}
float amx_inner_product_matrix_fp32( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results){
    int DIM=32;
    int blockCount=(dims)/DIM;
    int tailCount=dims%DIM;
    unsigned char maBf16[1024] __attribute__((aligned(64)));
    unsigned char mbBf16[1024] __attribute__((aligned(64)));

    //thread_local unsigned char *mbBf16=NULL;
    thread_local char *preQuery=NULL;
    thread_local char cfg[64]={0};
    thread_local bool init_mem=false;
 
    if(!init_mem){

/*         if(!mbBf16){
          mbBf16 =(unsigned char *)aligned_alloc(64,sizeof(char)*dims*4);
        } */
        cfg[0]=1;
        cfg[16]=DIM*2;
        cfg[48] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
        cfg[48+1]   = DIM/2;   // row = K/4
 
        cfg[22]=DIM*2;
        cfg[51] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[24] = batchSizeB*2*2;   // col = N*4
        cfg[52]   = DIM/2;   // row = K/4
 
        cfg[26]=DIM*2;
        cfg[53] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[28] = batchSizeB*2*2;   // col = N*4
        cfg[54]   = DIM/2;   // row = K/4
 
        cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
        cfg[48+2] = 16;
 
        init_mem = true;
 
        _tile_loadconfig((void *)cfg);
    }
    __m512i high_bits;
    __m512i low_bits;
    __m512i all_bits;
/*     for( i = 0; i < blockCount; i+=1) {
      for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
      }
      high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4),16);
      low_bits = _mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4 + 64);
      all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
      _mm512_store_si512(mbBf16 , all_bits);
      _tile_loadd(0,maBf16, 64);
      _tile_loadd(1,mbBf16, 4);
      _tile_dpbf16ps(2,0,1);

    } */

/*     if(preQuery!=floatQueryMatrix){
      preQuery=floatQueryMatrix;
      for(int i=0;i<dims/DIM;i++){
        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16+i*64 , all_bits);
      }
    } */
    int i=0;
    for( i = 0; i < blockCount/3; i+=1) {
        int index=3*i;
        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  index * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  index * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  index * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  index * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(0,maBf16, 64);
        _tile_loadd(1,mbBf16 , 4);

        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  (index+1) * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  (index+1) * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  (index+1) * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  (index+1) * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(3,maBf16, 64);
        _tile_loadd(4,mbBf16 , 4);

        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  (index+2) * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  (index+2) * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  (index+2) * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  (index+2) * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(5,maBf16, 64);
        _tile_loadd(6,mbBf16 , 4);

        _tile_dpbf16ps(2,0,1);  
        _tile_dpbf16ps(2,3,4);
        _tile_dpbf16ps(2,5,6);
    }
    switch(blockCount%3){
      case 0: break;
      case 1:
        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  3 * i * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  3 * i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(0,maBf16, 64);
        _tile_loadd(1,mbBf16, 4);
        _tile_dpbf16ps(2,0,1);  
        break;
 
      case 2:        
        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  3*i * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  3*i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(0,maBf16, 64);
        _tile_loadd(1,mbBf16, 4);

        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  (3*i+1) * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  (3*i+1) * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  (3*i+1) * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  (3*i+1) * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(3,maBf16, 64);
        _tile_loadd(4,mbBf16, 4);
        _tile_dpbf16ps(2,0,1);  
        _tile_dpbf16ps(2,3,4);
        break;
    }
  
   
    _tile_stored(2, results, batchSizeB*2*2);
    _tile_zero(2);
   
    if (tailCount != 0) {
        for (int k = 0; k < batchSizeA; k++) {
            for (int l = 0; l < batchSizeB; l++) {
                __m512 result_vec = _mm512_setzero_ps();
                for (int i = 0; i < tailCount; i += 16) {
                    __m512 lib_vec = _mm512_loadu_ps((float *)(floatLibraryMatrix[k])  + DIM * blockCount + i);
                    __m512 query_vec = _mm512_loadu_ps((float *)(floatQueryMatrix + DIM * blockCount + i));
                    result_vec = _mm512_fmadd_ps(lib_vec, query_vec, result_vec);
                }
                results[k * batchSizeB + l] += _mm512_reduce_add_ps(result_vec);
            }
        }
    }
 
    return 0;
}
float amx_inner_product_matrix_bf16( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results){
    int DIM=32;
    int blockDim = 96;
    int blockCount=((dims))/blockDim;
    size_t tailCount=dims%DIM;
    int tailBlock=dims%blockDim;

    thread_local char cfg[64]={0};
    thread_local bool init_mem=false;

    unsigned char ma1Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma2Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma3Bf16[1024] __attribute__((aligned(64)));
 
    if(!init_mem){
        cfg[0]=1;
        cfg[16]=DIM*2;
        cfg[48] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
        cfg[48+1]   = DIM/2;   // row = K/4
 
        cfg[22]=DIM*2;
        cfg[51] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[24] = batchSizeB*2*2;   // col = N*4
        cfg[52]   = DIM/2;   // row = K/4
 
        cfg[26]= DIM*2;
        cfg[53] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[28] = batchSizeB*2*2;   // col = N*4
        cfg[54]   = DIM/2;   // row = K/4
 
        cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
        cfg[48+2] = 16;
        init_mem = true;
 
        _tile_loadconfig((void *)cfg);
    }
    //memset(maBf16,0,16*DIM*2);
 
    int i=0;
    for(int i=0;i<blockCount;i++){

      //int32_t stride=i*DIM;
      __m512i sa;
      size_t offset = i * blockDim *2;
      
      for(int j=0;j<batchSizeA;j++){  
        size_t destOffset1 = j * DIM * 2;
        _mm512_store_si512(ma1Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset));
        _mm512_store_si512(ma2Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 64));
        _mm512_store_si512(ma3Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 128));
      } 

      _tile_loadd(1,floatQueryMatrix + offset , 4);
      _tile_loadd(4,floatQueryMatrix + offset + 64 , 4);
      _tile_loadd(6,floatQueryMatrix + offset + 128, 4);
      _tile_loadd(0,ma1Bf16, 64);
      _tile_loadd(3,ma2Bf16, 64);
      _tile_loadd(5,ma3Bf16, 64);
      _tile_dpbf16ps(2,3,4);
      _tile_dpbf16ps(2,0,1);
      _tile_dpbf16ps(2,5,6);
    //amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);
    }
    if(tailBlock >= DIM){
      for(int i=0;i<tailBlock/DIM;i++){
        __m512i sa;
        for(int j=0;j<batchSizeA;j++){  
          sa=_mm512_loadu_si512(floatLibraryMatrix[j]+blockCount*blockDim * 2 + i * DIM * 2 );
          _mm512_store_si512(ma1Bf16+j*DIM*2,sa);
        }
        _tile_loadd(0,ma1Bf16, 64);
        _tile_loadd(1,floatQueryMatrix + blockCount*blockDim*2 + i * DIM*2 , 4);
        _tile_dpbf16ps(2,0,1);
      }
    }

/*     for( i = 0; i < blockCount; i+=1) {
      for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
      }
      high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4),16);
      low_bits = _mm512_loadu_si512(floatQueryMatrix +  i * DIM * 4 + 64);
      all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
      _mm512_store_si512(mbBf16 , all_bits);
      _tile_loadd(0,maBf16, 64);
      _tile_loadd(1,mbBf16, 4);
      _tile_dpbf16ps(2,0,1);

    } */
   
    _tile_stored(2, results, batchSizeB*2*2);
    _tile_zero(2);
   
    if (tailCount != 0) {
        for (int k = 0; k < batchSizeA; k++) {
            for (int l = 0; l < batchSizeB; l++) {
                __m512 result_vec = _mm512_setzero_ps();
                for (int i = 0; i < tailCount; i += 16) {
                    __m512 lib_vec = _mm512_loadu_ps((float *)(floatLibraryMatrix[k])  + DIM * blockCount + i);
                    __m512 query_vec = _mm512_loadu_ps((float *)(floatQueryMatrix + DIM * blockCount + i));
                    result_vec = _mm512_fmadd_ps(lib_vec, query_vec, result_vec);
                }
                results[k * batchSizeB + l] += _mm512_reduce_add_ps(result_vec);
            }
        }
    }
 
    return 0;
}
float devided_batch_amx_inner_product(char **floatLibraryMatrix, char *floatqueryMatrix, 
    uint64_t dims, uint64_t nSize, uint64_t mSize, float *results_amx) { 

    int batchSizeA = 16, batchSizeB = 16;
    int batchCountA = (nSize - 1) / batchSizeA + 1;
    int batchCountB = (mSize - 1) / batchSizeB + 1;

    int lastBatchSizeA = (nSize % batchSizeA == 0) ? batchSizeA : nSize % batchSizeA;
    int lastBatchSizeB = (mSize % batchSizeB == 0) ? batchSizeB : mSize % batchSizeB;

    int offsetA = batchSizeA * dims * 2;
    int offsetB = batchSizeB * dims * 2;

    float *results_ptr = results_amx;

    for (int i = 0; i < batchCountA; i++) {
        int currentBatchSizeA = (i == batchCountA - 1) ? lastBatchSizeA : batchSizeA;
        char **currentLibraryMatrixPtr = floatLibraryMatrix + i * 16;

        for (int j = 0; j < batchCountB; j++) {
            int currentBatchSizeB = (j == batchCountB - 1) ? lastBatchSizeB : batchSizeB;
            char *currentQueryMatrixPtr = floatqueryMatrix + j * offsetB;

            amx_inner_product_matrix_bf16(currentLibraryMatrixPtr, currentQueryMatrixPtr, dims, currentBatchSizeA, currentBatchSizeB, results_ptr);

            results_ptr += currentBatchSizeB * currentBatchSizeA;
        }
    }

    return 0;
}

float devided_batch_amx_inner_product_fp32(char **floatLibraryMatrix, char *floatqueryMatrix, 
    uint64_t dims, uint64_t nSize, uint64_t mSize, float *results_amx) { 

    int batchSizeA = 16, batchSizeB = 16;
    int batchCountA = (nSize - 1) / batchSizeA + 1;
    int batchCountB = (mSize - 1) / batchSizeB + 1;

    int lastBatchSizeA = (nSize % batchSizeA == 0) ? batchSizeA : nSize % batchSizeA;
    int lastBatchSizeB = (mSize % batchSizeB == 0) ? batchSizeB : mSize % batchSizeB;

    int offsetA = batchSizeA * dims * 4;
    int offsetB = batchSizeB * dims * 4;

    float *results_ptr = results_amx;

    for (int i = 0; i < batchCountA; i++) {
        int currentBatchSizeA = (i == batchCountA - 1) ? lastBatchSizeA : batchSizeA;
        char **currentLibraryMatrixPtr = floatLibraryMatrix + i * 16;

        for (int j = 0; j < batchCountB; j++) {
            int currentBatchSizeB = (j == batchCountB - 1) ? lastBatchSizeB : batchSizeB;
            char *currentQueryMatrixPtr = floatqueryMatrix + j * offsetB;

            amx_inner_product_matrix_fp32(currentLibraryMatrixPtr, currentQueryMatrixPtr, dims, currentBatchSizeA, currentBatchSizeB, results_ptr);

            results_ptr += currentBatchSizeB * currentBatchSizeA;
        }
    }

    return 0;
}