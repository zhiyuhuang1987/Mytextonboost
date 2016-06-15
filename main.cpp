#include <iostream>
#include<math.h>
#include<conio.h>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include <opencv2/core/mat.hpp>
#include "cv.h"

using namespace std;
using namespace cv;

const string fileFolder = "C:/Users/Wang/Documents/Qt/Mytextonboost/mri_data";
void imshow(Mat image);
void toLoadImage(const string fileFolder,vector<Mat> &image,vector<Mat> &truth);
void toGaussianBlur(vector<Mat> input,vector<Mat> &output);
void toWavelet(vector<Mat> input,vector<Mat> &output);
void toKmeans();
class wavelet;
class wavelet2;

int main()
{
    vector<Mat> mri_image,mri_truth;
    vector<Mat> mri_gauss,mri_wavelet;
    toLoadImage(fileFolder,mri_image,mri_truth);
    toGaussianBlur(mri_image,mri_gauss);
    toWavelet(mri_gauss,mri_wavelet);

    waitKey(0);

    return 0;
}

void imshow(Mat image)
{
    imshow("Test", image);
    waitKey(0);
}

void toLoadImage(const string fileFolder,vector<Mat> &image,vector<Mat> &truth)
{
    printf("Loading Images Starts\n");
    string trainImageFolder = fileFolder + "/trainData/image";
    string trainTruthFolder = fileFolder + "/trainData/truth";
    vector<String> imageFilename;
    vector<String> truthFilename;
    glob(trainImageFolder,imageFilename);
    glob(trainTruthFolder,truthFilename);
    for(size_t i = 0; i < imageFilename.size(); i++)
    {
        Mat im = imread(imageFilename[i]);
        Mat gt = imread(truthFilename[i]);
        if(im.empty()||gt.empty())
        {
            printf("loading image %d error! \n",i);
            waitKey(0);
            break;
        }
        image.push_back(im.clone());
        truth.push_back(gt.clone());
    }
    printf("Loading Images Starts\n");
}

void toGaussianBlur(vector<Mat> input,vector<Mat> &output)
{
    Mat output_mat;
    for(size_t i = 0; i < input.size();i++)
    {
        GaussianBlur(input[i],output_mat,Size(3,3),0,0);
        output.push_back(output_mat.clone());
    }
}

class wavelet
{
public:
    Mat src,dst;
    int NIter;
    //--------------------------------
    // Wavelet transform
    //--------------------------------
    void cvHaarWavelet()
    {
        float c,dh,dv,dd;
        src.copyTo(dst);
        //assert( src.type() == CV_32FC1 );
        //assert( dst.type() == CV_32FC1 );
        int width = src.cols;
        int height = src.rows;
        imshow(src);
        for (int k=0;k<NIter;k++)
        {
            for (int y=0;y<(height>>(k+1));y++)
            {
                for (int x=0; x<(width>>(k+1));x++)
                {
                    c=(src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)+src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5;
                    dst.at<float>(y,x)=c;

                    dh=(src.at<float>(2*y,2*x)+src.at<float>(2*y+1,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x+1))*0.5;
                    dst.at<float>(y,x+(width>>(k+1)))=dh;

                    dv=(src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)-src.at<float>(2*y+1,2*x+1))*0.5;
                    dst.at<float>(y+(height>>(k+1)),x)=dv;

                    dd=(src.at<float>(2*y,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5;
                    dst.at<float>(y+(height>>(k+1)),x+(width>>(k+1)))=dd;
                }
            }
            imshow(dst);
            //dst.copyTo(src);
        }
    }
    //--------------------------------
    //Inverse wavelet transform
    //--------------------------------
    void cvInvHaarWavelet()
    {
        float c,dh,dv,dd;
        //assert( src.type() == CV_32FC1 );
        //assert( dst.type() == CV_32FC1 );
        int width = src.cols;
        int height = src.rows;
        //--------------------------------
        // NIter - number of iterations
        //--------------------------------
        for (int k=NIter;k>0;k--)
        {
            for (int y=0;y<(height>>k);y++)
            {
                for (int x=0; x<(width>>k);x++)
                {
                    c=0;//src.at<float>(y,x);
                    dh=src.at<float>(y,x+(width>>k));
                    dv=src.at<float>(y+(height>>k),x);
                    dd=src.at<float>(y+(height>>k),x+(width>>k));

                    //-------------------
                    dst.at<float>(y*2,x*2)=0.5*(c+dh+dv+dd);
                    dst.at<float>(y*2,x*2+1)=0.5*(c-dh+dv-dd);
                    dst.at<float>(y*2+1,x*2)=0.5*(c+dh-dv-dd);
                    dst.at<float>(y*2+1,x*2+1)=0.5*(c-dh-dv+dd);
                }
            }
            //Mat C=src(Rect(0,0,width>>(k-1),height>>(k-1)));
            //Mat D=dst(Rect(0,0,width>>(k-1),height>>(k-1)));
            //D.copyTo(C);
        }
    }
};

class wavelet2
{
public:
    Mat im,im1,im2,im3,im4,im5,im6,temp,im11,im12,im13,im14,imi,imd,imr;
    float a,b,c,d;
    int getim()
    {
        //im=imread("lena.jpg",0); //Load image in Gray Scale
        imi=Mat::zeros(im.rows,im.cols,CV_8U);
        im.copyTo(imi);

        im.convertTo(im,CV_32F,1.0,0.0);
        im1=Mat::zeros(im.rows/2,im.cols,CV_32F);
        im2=Mat::zeros(im.rows/2,im.cols,CV_32F);
        im3=Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im4=Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im5=Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im6=Mat::zeros(im.rows/2,im.cols/2,CV_32F);

        //--------------Decomposition-------------------

        for(int rcnt=0;rcnt<im.rows;rcnt+=2)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt++)
            {

                a=im.at<float>(rcnt,ccnt);
                b=im.at<float>(rcnt+1,ccnt);
                c=(a+b)*0.707;
                d=(a-b)*0.707;
                int _rcnt=rcnt/2;
                im1.at<float>(_rcnt,ccnt)=c;
                im2.at<float>(_rcnt,ccnt)=d;
            }
        }

        for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt+=2)
            {

                a=im1.at<float>(rcnt,ccnt);
                b=im1.at<float>(rcnt,ccnt+1);
                c=(a+b)*0.707;
                d=(a-b)*0.707;
                int _ccnt=ccnt/2;
                im3.at<float>(rcnt,_ccnt)=c;
                im4.at<float>(rcnt,_ccnt)=d;
            }
        }

        for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt+=2)
            {

                a=im2.at<float>(rcnt,ccnt);
                b=im2.at<float>(rcnt,ccnt+1);
                c=(a+b)*0.707;
                d=(a-b)*0.707;
                int _ccnt=ccnt/2;
                im5.at<float>(rcnt,_ccnt)=c;
                im6.at<float>(rcnt,_ccnt)=d;
            }
        }

        imr=Mat::zeros(im.rows,im.cols,CV_32F);
        imd=Mat::zeros(im.rows,im.cols,CV_32F);
        imshow(im3);
        imshow(im4);
        imshow(im5);
        imshow(im6);
        im3.copyTo(imd(Rect(0,0,im.cols/2,im.rows/2)));
        im4.copyTo(imd(Rect(im.cols/2-1,0,im.cols/2,im.rows/2)));
        im5.copyTo(imd(Rect(0,im.rows/2-1,im.cols/2,im.rows/2)));
        im6.copyTo(imd(Rect(im.cols/2-1,im.rows/2-1,im.cols/2,im.rows/2)));


        //---------------------------------Reconstruction-------------------------------------

        im11=Mat::zeros(im.rows/2,im.cols,CV_32F);
        im12=Mat::zeros(im.rows/2,im.cols,CV_32F);
        im13=Mat::zeros(im.rows/2,im.cols,CV_32F);
        im14=Mat::zeros(im.rows/2,im.cols,CV_32F);

        for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        {
            for(int ccnt=0;ccnt<im.cols/2;ccnt++)
            {
                int _ccnt=ccnt*2;
                im11.at<float>(rcnt,_ccnt)=im3.at<float>(rcnt,ccnt);     //Upsampling of stage I
                im12.at<float>(rcnt,_ccnt)=im4.at<float>(rcnt,ccnt);
                im13.at<float>(rcnt,_ccnt)=im5.at<float>(rcnt,ccnt);
                im14.at<float>(rcnt,_ccnt)=im6.at<float>(rcnt,ccnt);
            }
        }


        for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt+=2)
            {

                a=im11.at<float>(rcnt,ccnt);
                b=im12.at<float>(rcnt,ccnt);
                c=(a+b)*0.707;
                im11.at<float>(rcnt,ccnt)=c;
                d=(a-b)*0.707;                           //Filtering at Stage I
                im11.at<float>(rcnt,ccnt+1)=d;
                a=im13.at<float>(rcnt,ccnt);
                b=im14.at<float>(rcnt,ccnt);
                c=(a+b)*0.707;
                im13.at<float>(rcnt,ccnt)=c;
                d=(a-b)*0.707;
                im13.at<float>(rcnt,ccnt+1)=d;
            }
        }

        temp=Mat::zeros(im.rows,im.cols,CV_32F);

        for(int rcnt=0;rcnt<im.rows/2;rcnt++)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt++)
            {

                int _rcnt=rcnt*2;
                imr.at<float>(_rcnt,ccnt)=im11.at<float>(rcnt,ccnt); //Upsampling at stage II
                temp.at<float>(_rcnt,ccnt)=im13.at<float>(rcnt,ccnt);
            }
        }

        for(int rcnt=0;rcnt<im.rows;rcnt+=2)
        {
            for(int ccnt=0;ccnt<im.cols;ccnt++)
            {

                a=imr.at<float>(rcnt,ccnt);
                b=temp.at<float>(rcnt,ccnt);
                c=(a+b)*0.707;
                imr.at<float>(rcnt,ccnt)=c;    //Filtering at Stage II
                d=(a-b)*0.707;
                imr.at<float>(rcnt+1,ccnt)=d;
            }
        }

        imd.convertTo(imd,CV_8U);
        namedWindow("Input Image",1);
        imshow("Input Image",imi);
        namedWindow("Wavelet Decomposition",1);
        imshow("Wavelet Decomposition",imd);
        imr.convertTo(imr,CV_8U);
        namedWindow("Wavelet Reconstruction",1);
        imshow("Wavelet Reconstruction",imr);
        waitKey(0);
        return 0;
    }
};

void toWavelet(vector<Mat> input,vector<Mat> &output)
{
    wavelet2 mri;
    for(size_t i = 0; i < input.size();i++)
    {
        mri.im = input[i];
        mri.getim();
        //imshow(mri.src);
        //mri.cvHaarWavelet();
        //imshow(mri.dst);
        //mri.cvInvHaarWavelet();
        //output.push_back(mri.dst.clone());
        //imshow(output[i]);
    }
}
void toKmeans()
{}
