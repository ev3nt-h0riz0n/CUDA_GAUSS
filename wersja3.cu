%%writefile porownanie.cu

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <nvtx3/nvToolsExt.h>


struct Gauss_gen{
    thrust::pair<int,float> gauss_pair; //1-center,2-sigma 

    __host__ __device__ float operator()(int i){
        int x=i-gauss_pair.first;
        return exp(-(x*x)/(2.0f*gauss_pair.second*gauss_pair.second));
    }
};

struct Gauss_gen_norm{
    thrust::pair<int,float> gauss_pair; //1-center,2-sigma 
    float sum;

    __host__ __device__ float operator()(int i){
        int x=i-gauss_pair.first;
        float gauss=exp(-(x*x)/(2.0f*gauss_pair.second*gauss_pair.second));
        return gauss/sum;
    }
};


std::vector<float> GaussianKernel(int size, float sigma){
    std::vector<float> kernel(size);
    int center= size/2;
    float sum= 0;

    for(int i=0;i<size; i++) {
        int x = i-center;
        kernel[i] = exp(-(x*x)/(2.0f*sigma*sigma));
        sum= sum+ kernel[i];
    }

    for(int i=0;i<size;i++){ kernel[i] = kernel[i]/sum;}
    return kernel;
}

__global__ void Horizontal(uchar3* in, uchar3* out, float* kernel, int width, int height, int ksize) {
    int x= blockIdx.x* blockDim.x+ threadIdx.x;
    int y= blockIdx.y* blockDim.y+ threadIdx.y;

    if(x>=width || y>=height) return;

    int half = ksize/2;
    float sumR= 0, sumG=0, sumB=0;

    for(int i=0; i<ksize;i++){
        int px= x+(i-half);
        if(px<0) px=0;
        if(px>=width) px=width-1;

        uchar3 pixel=in[y*width+px];
        sumB= sumB+pixel.x*kernel[i];
        sumG= sumG+ pixel.y*kernel[i];
        sumR = sumR+pixel.z*kernel[i];
    }
    out[y*width+x] = make_uchar3(sumB,sumG,sumR);

}
__global__ void Vertical(uchar3* in, uchar3* out, float* kernel, int width, int height, int ksize) {
    int x= blockIdx.x* blockDim.x+ threadIdx.x;
    int y= blockIdx.y* blockDim.y+ threadIdx.y;

    if(x>=width || y>=height) return;

    int half = ksize/2;
    float sumR= 0, sumG=0, sumB=0;

    for(int i=0; i<ksize;i++){
        int py= y+(i-half);
        if(py<0) py=0;
        if(py>=height) py=height-1;

        uchar3 pixel=in[py*width+x];

        sumB= sumB+pixel.x*kernel[i];
        sumG= sumG+ pixel.y*kernel[i];
        sumR = sumR+pixel.z*kernel[i];
    }
    out[y*width+x] = make_uchar3(sumB,sumG,sumR);
}

int GaussianBlurFastCV(cv::Mat& source,cv::Mat& final, cv::Size size, float sigma){
    int height= source.rows;
    int width= source.cols;
    int center_x=size.width/2;
    int center_y=size.height/2;

    thrust::device_vector<float> d_kernel_x(size.width);

    thrust::counting_iterator<int> count(0);
    auto gauss_vec_x=thrust::make_transform_iterator(count,Gauss_gen{thrust::make_pair(center_x,sigma)});//generuje wektor z wartosciami gaussa
    float sum_x=thrust::reduce(thrust::device,gauss_vec_x,gauss_vec_x+size.width);//suma wektora

    thrust::transform(thrust::device,count,count+size.width,d_kernel_x.begin(),Gauss_gen_norm{thrust::make_pair(center_x,sigma),sum_x});
    float* d_kernel_xx=thrust::raw_pointer_cast(d_kernel_x.data());

    thrust::device_vector<float> d_kernel_y(size.height);

    auto gauss_vec_y=thrust::make_transform_iterator(count,Gauss_gen{thrust::make_pair(center_y,sigma)});//generuje wektor z wartosciami gaussa
    float sum_y=thrust::reduce(thrust::device,gauss_vec_y,gauss_vec_y+size.height);//suma wektora

    thrust::transform(thrust::device,count,count+size.height,d_kernel_y.begin(),Gauss_gen_norm{thrust::make_pair(center_y,sigma),sum_y});
    float* d_kernel_yy=thrust::raw_pointer_cast(d_kernel_y.data());


    uchar3 *d_input, *d_temp, *d_output;

    size_t imgSize= width* height* sizeof(uchar3);

//To do zostawienia już w spokoju
    cudaError_t err = cudaMalloc(&d_input, imgSize);
    if (err!= cudaSuccess){
        printf("cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_temp, imgSize);
    if (err!= cudaSuccess){
        printf("cudaMalloc d_temp failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err =cudaMalloc(&d_output, imgSize);
    if (err!= cudaSuccess){
        printf("cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(d_input, source.ptr(), imgSize, cudaMemcpyHostToDevice);
    if (err!= cudaSuccess){
        printf("cudaMempcpy H2D d_input failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaGetLastError(); // Clear any previous errors - czysci spis bledow

    dim3 block(16,16);
    dim3 grid((width+ block.x-1)/ block.x, (height+ block.y-1)/ block.y);

    Horizontal<<<grid, block>>>(d_input, d_temp, d_kernel_xx, width, height, size.width);
    err = cudaGetLastError();
    if (err != cudaSuccess) { // Error horizontal kernela
        printf("Kernel Horizontal launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    Vertical<<<grid, block>>>(d_temp, d_output, d_kernel_yy, width, height, size.height);
    err = cudaGetLastError(); //Error vertical kernela
    if (err != cudaSuccess) {
        printf("Kernel Vertical error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    //Wait for kernel to complete - Sprawdzenie czy kernele się wykonały
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    final.create(source.size(), source.type());
    err = cudaMemcpy(final.ptr(), d_output, imgSize, cudaMemcpyDeviceToHost);
        if (err!= cudaSuccess){
        printf("cudaMempcy H2D final failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    //Cleanup
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);

    return 0;
//Już git tu wszystko
}


int main()
{
  // Obliczanie czasu przetwarzania obrazu dla CPU (opencv)
  cv::Mat zdj = cv::imread("test (2).jpg");
  cv::Mat zdj_opencv;
  cv::Mat zdj_fastcv;

  auto start = std::chrono::high_resolution_clock::now();
  cv::GaussianBlur(zdj, zdj_opencv, cv::Size(23,23), 10);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diffOpen = end - start;
  std::cout<<"Czas wykonania OpenCV: " << diffOpen.count() << "s\n";
  GaussianBlurFastCV(zdj, zdj_fastcv,cv::Size(23,23),10);
  auto start2 = std::chrono::high_resolution_clock::now();
  GaussianBlurFastCV(zdj, zdj_fastcv,cv::Size(23,23),10);
  auto end2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diffFast = end2 - start2;
  std::cout<<"Czas wykonania FastCV: " << diffFast.count() << "s\n";

  cv::imwrite("zdj_opencv.jpg", zdj_opencv);
  cv::imwrite("zdj_fastcv.jpg", zdj_fastcv);

  return 0;
}
