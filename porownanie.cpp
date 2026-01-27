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

    for(int row=0; row<height; row++){ //petla liczaca macierz do filtru
        for (int col=0; col<width; col++)
        {
            int matrix_x=col-half_width;
            int matrix_y=row-half_height;

            kernel[row][col]= (1/(2*M_PI*sigma*sigma))*exp(-(matrix_x*matrix_x+matrix_y*matrix_y)/(2*sigma*sigma)); //Wzor Gaussa
            sum= sum+kernel[row][col];
    


        }
    } for(int row=0; row<height; row++){ //petla normalizujaca
@@ -33,21 +35,24 @@ std::vector<std::vector<double>> GaussianKernel(int height,int width,double sigm
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

@@ -80,8 +85,10 @@ void GaussianBlurFastCV(cv::Mat& source_picture,cv::Mat& final_picture,cv::Size
        new_pixel[2]=cv::saturate_cast<uchar>(sumR);
        final_picture.at<cv::Vec3b>(row,col)=new_pixel;
        }
        

    }
    nvtxRangePop();
    nvtxRangePop();
}

int main()
@@ -104,7 +111,7 @@ int main()

  std::chrono::duration<double> diffFast = end2 - start2;
  std::cout<<"Czas wykonania FastCV: " << diffFast.count() << "s\n";
  

  cv::imwrite("zdj_opencv.jpg", zdj_opencv);
  cv::imwrite("zdj_fastcv.jpg", zdj_fastcv);
  return 0;
