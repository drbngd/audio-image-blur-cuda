#ifndef UTIL_H
#define UTIL_H

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
        exit(code);
   }
}


inline void check_launch(const char *kernel_name) {
  cudaDeviceSynchronize();
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert in kernel launch %s: %s\n", kernel_name, cudaGetErrorString(code));
    exit(1);
  }
}

#endif