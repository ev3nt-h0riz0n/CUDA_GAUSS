%%writefile wersja1.cu

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <nvToolsExt.h>



std::vector<std::vector<double>> GaussianKernel(int height,int width,double sigma) //Utworzenie kernela
{
    nvtxRangePush("Tworzenie_Kernela");
    std::vector<std::vector<double>> kernel(height, std::vector<double>(width)); //device_vector tu
    double sum=0.0;
    int half_width=width/2;
    int half_height=height/2;

    for(int row=0; row<height; row++){ //petla liczaca macierz do filtru
        for (int col=0; col<width; col++)
        {
            int matrix_x=col-half_width;
            int matrix_y=row-half_height;

            kernel[row][col]= (1/(2*M_PI*sigma*sigma))*exp(-(matrix_x*matrix_x+matrix_y*matrix_y)/(2*sigma*sigma)); //Wzor Gaussa
            sum= sum+kernel[row][col];


        }
    } for(int row=0; row<height; row++){ //petla normalizujaca
        for (int col=0; col<width; col++)
        {
            kernel[row][col]=kernel[row][col]/sum;
        }
    }
    nvtxRangePop();
    return kernel;
}

void GaussianBlurFastCV(cv::Mat& source_picture,cv::Mat& final_picture,cv::Size kernelSize,float sigma) //glowna funckja zamazania Gaussowskiego
{
    nvtxRangePush("FastCV");
    int height= source_picture.rows;
    int width= source_picture.cols;

    final_picture.create(source_picture.size(), source_picture.type());

    std::vector<std::vector<double>> kernel = GaussianKernel(kernelSize.height, kernelSize.width, sigma);

    nvtxRangePush("Konwolucja");
    int half_width=kernelSize.width/2;
    int half_height=kernelSize.height/2;

    for(int row=0; row<height; row++){ //petla przechodzenia przez piksele zdjecia
        for (int col=0; col<width; col++){

            double sumR=0.0;
            double sumG=0.0;
            double sumB=0.0;

            for(int kernel_row=0; kernel_row<kernelSize.height; kernel_row++){ //petla przechodzenia przez piksele kernela
                for (int kernel_col=0; kernel_col<kernelSize.width; kernel_col++){
                    int pct_row=row+(kernel_row-half_height);
                    int pct_col= col+(kernel_col-half_width);

                    if(pct_row<0) pct_row=0;
                    if(pct_row>=height)pct_row=height-1;
                    if(pct_col<0) pct_col=0;
                    if(pct_col>=width)pct_col=width-1;

                    cv::Vec3b pixel=source_picture.at<cv::Vec3b>(pct_row,pct_col);
                    double weight = kernel[kernel_row][kernel_col];

                    sumB+=pixel[0]*weight;
                    sumG+=pixel[1]*weight;
                    sumR+=pixel[2]*weight;

                }
            }
        cv::Vec3b new_pixel;
        new_pixel[0]=cv::saturate_cast<uchar>(sumB);
        new_pixel[1]=cv::saturate_cast<uchar>(sumG);
        new_pixel[2]=cv::saturate_cast<uchar>(sumR);
        final_picture.at<cv::Vec3b>(row,col)=new_pixel;
        }

    }
    nvtxRangePop();
    nvtxRangePop();
}

int main()
{
  // Obliczanie czasu przetwarzania obrazu dla CPU (opencv)
  cv::Mat zdj = cv::imread("test.jpg", cv::IMREAD_COLOR);
  cv::Mat zdj_opencv;
  cv::Mat zdj_fastcv;

  auto start = std::chrono::high_resolution_clock::now();
  cv::GaussianBlur(zdj, zdj_opencv, cv::Size(7,7), 10);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diffOpen = end - start;
  std::cout<<"Czas wykonania OpenCV: " << diffOpen.count() << "s\n";

  auto start2 = std::chrono::high_resolution_clock::now();
  GaussianBlurFastCV(zdj, zdj_fastcv, cv::Size(7,7),10);
  auto end2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diffFast = end2 - start2;
  std::cout<<"Czas wykonania FastCV: " << diffFast.count() << "s\n";

  cv::imwrite("zdj_opencv.jpg", zdj_opencv);
  cv::imwrite("zdj_fastcv.jpg", zdj_fastcv);
  return 0;
}
