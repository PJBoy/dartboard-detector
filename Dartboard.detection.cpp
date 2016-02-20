#include <iostream>
#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

void image_double_to_uchar(const Mat&, Mat&);
void convolve(const Mat&, Mat&, const Mat&);
int sobel(std::string, Mat &);
int hough_circle(Mat &, String);
int hough(Mat &, Mat &, Mat &, int, String);


int main(int argc, char* argv[])
{
    std::string filenames[] =
    {
        "dart0.jpg",
        "dart1.jpg",
        "dart2.jpg",
        "dart3.jpg",
        "dart4.jpg",
        "dart5.jpg",
        "dart6.jpg",
        "dart7.jpg",
        "dart8.jpg",
        "dart9.jpg",
        "dart10.jpg",
        "dart11.jpg",
        "dart12.jpg",
        "dart13.jpg",
        "dart14.jpg",
        "dart15.jpg"
    };
#pragma omp parallel for
    for (int i = 0; i < sizeof(filenames) / sizeof(*filenames); ++i)
    {
        Mat phase;
        sobel(filenames[i], phase);
        hough_circle(phase, filenames[i]);
    }
}

//read the magnitude, threshold it and detect circles.
int hough_circle(Mat &phase, String originalName)
{
    Mat magnitude = imread("magnitude " + originalName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat original = imread(originalName);
    int maxDimension = original.rows>original.cols ? original.rows : original.cols;
    while (maxDimension > 1000)
    {
        resize(original, original, Size(), 0.9, 0.9, INTER_CUBIC);
        maxDimension = original.rows>original.cols ? original.rows : original.cols;
    }
    threshold(magnitude, magnitude, 12345678997, 255, THRESH_BINARY | THRESH_OTSU);
    imwrite("threshold " + originalName + ".png", magnitude);

    if (hough(original, phase, magnitude, 8, originalName))
        return EXIT_FAILURE;
    imwrite("transform " + originalName + ".png", original);

    return 0;
}

int hough(Mat &original, Mat &phase, Mat &magnitude, int thresholdValue, String originalName)
{
    int coinCount = 0;
    const int radius = 150, min_radius = 0;
    int dims[] = {original.cols + 2 * radius, original.rows + 2 * radius, radius};
    Mat counter(3, dims, CV_8U, Scalar::all(0));

    //creates the hough space
    for (int y = 0; y < magnitude.rows; ++y)
        for (int x = 0; x < magnitude.cols; ++x)
            if (magnitude.at<uchar>(y, x) == 255)
                for (int r = min_radius; r < radius; ++r)
                {
                    double x0 = x - r*std::sin(phase.at<double>(y, x)) + radius;
                    double x1 = x + r*std::sin(phase.at<double>(y, x)) + radius;
                    double y0 = y + r*std::cos(phase.at<double>(y, x)) + radius;
                    double y1 = y - r*std::cos(phase.at<double>(y, x)) + radius;
                    //if (x0 >= 0 && x0 < magnitude.cols + radius && y0 >= 0 && y0 < magnitude.rows + radius)
                    counter.at<uchar>((int)x0, (int)y0, r) += 1;
                    //if (x1 >= 0 && x1 < magnitude.cols && y1 >= 0 && y1 < magnitude.rows)
                    counter.at<uchar>((int)x1, (int)y1, r) += 1;
                }

    copyMakeBorder(original, original, radius, radius, radius, radius, BORDER_CONSTANT, 0);

    {
        Mat1d hough_space_plot_double(dims[1], dims[0], 0.0);
        for (int r = min_radius; r < dims[2]; ++r)
            for (int y = 0; y < dims[1]; ++y)
                for (int x = 0; x < dims[0]; ++x)
                    hough_space_plot_double[y][x] += counter.at<uchar>(x, y, r);

        Mat1b hough_space_plot_uchar;
        image_double_to_uchar(hough_space_plot_double, hough_space_plot_uchar);
        imwrite("hough space " + originalName + ".png", hough_space_plot_uchar);
    }

    //finds coin centres
    //the first triple nested loop looks for possible coin centres.
    for (int x = radius / 2; x < original.cols - radius / 2; ++x)
        for (int y = radius / 2; y < original.rows - radius / 2; ++y)
            for (int r = min_radius; r < radius; ++r)
            {
                //if a possible centre is found, count how many possible centres there are in the vicinity of that pixel and average their values.
                if (counter.at<uchar>(x, y, r) > thresholdValue)
                {
                    int sumX = 0;
                    int sumY = 0;
                    int sumR = 0;
                    int sumCount = 0;
                    // a green point = a possible centre coordinate
                    circle(original, Point(x, y), 3, Scalar(0, 255, 0, 0.5), 1);
                    for (int r1 = min_radius; r1 < radius; ++r1)
                        for (int x1 = x - radius / 3; x1 < x + radius / 3; ++x1)
                            for (int y1 = y - radius / 3; y1 < y + radius / 3; ++y1)
                                if (counter.at<uchar>(x1, y1, r1) > thresholdValue)
                                {
                                    sumX += x1;
                                    sumY += y1;
                                    sumR += r1;
                                    sumCount++;
                                }
                    sumX /= sumCount;
                    sumY /= sumCount;
                    sumR /= sumCount;
                    sumR -= 1;

                    //if there are only 9 or less possible centres in the vicinity, skip the current X and Y coordinates.
                    if (sumCount <= 9)
                    {
                        for (int r1 = min_radius; r1 < radius; ++r1)
                            counter.at<uchar>(sumX, sumY, r1) = 0;
                        continue;
                    }


                    //take the average of the possible centres, and look for possible centres in the vicinity of the average.
                    int sumX1 = 0;
                    int sumY1 = 0;
                    int sumR1 = 0;
                    int sumCount1 = 0;
                    //a red point/circle = the average centre/circle out of the previous possible centres.
                    circle(original, Point(sumX, sumY), 2, Scalar(0, 0, 255, 0.5), 1);
                    circle(original, Point(sumX, sumY), sumR, Scalar(0, 0, 255, 0.5), 1);
                    //get all the possible centres within the circle and average them.
                    for (int x1 = sumX - sumR; x1 < sumX + sumR; ++x1)
                        for (int y1 = sumY - sumR; y1 < sumY + sumR; ++y1)
                            for (int r1 = min_radius; r1 < radius; ++r1)
                                if (counter.at<uchar>(x1, y1, r1) > thresholdValue)
                                {
                                    sumX1 += x1;
                                    sumY1 += y1;
                                    sumR1 += r1;
                                    sumCount1++;

                                }
                    if (!sumCount1)
                        continue;
                    sumX1 /= sumCount1;
                    sumY1 /= sumCount1;
                    sumR1 /= sumCount1;
                    //a blue point/circle = the final circle around a coin.
                    circle(original, Point(sumX1, sumY1), 1, Scalar(255, 0, 0, 0.5), 1);
                    circle(original, Point(sumX1, sumY1), sumR1 * 3, Scalar(255, 0, 0, 0.5), 1);
                    coinCount++;

                    //remove all the possible centres from the vicinity of the centre of the detected coin.
                    for (int x1 = sumX1 - sumR1 * 3 / 4; x1 < sumX1 + sumR1 * 3 / 4; ++x1)
                        for (int y1 = sumY1 - sumR1 * 3 / 4; y1 < sumY1 + sumR1 * 3 / 4; ++y1)
                            for (int r1 = min_radius; r1 < radius; ++r1)
                                counter.at<uchar>(x1, y1, r1) = 0;
                }
            }
    std::cout << "Dartboards in " << originalName << ": " << coinCount << std::endl;
    return 0;
}

int sobel(std::string image_path, Mat &phase)
{
    Mat input, output;

    input = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
    int maxDimension = input.rows>input.cols ? input.rows : input.cols;
    while (maxDimension > 1000)
    {
        resize(input, input, Size(), 0.9, 0.9, INTER_CUBIC);
        maxDimension = input.rows>input.cols ? input.rows : input.cols;
    }

    if (!input.data)
    {
        std::cerr << "No image data\n";
        return EXIT_FAILURE;
    }

    const double
        partial_x_array[] =
    {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    },
    partial_y_array[] =
        {
            -1, -2, -1,
            0, 0, 0,
            1, 2, 1
        };

        const Mat
            partial_x_matrix(Size(3, 3), CV_64F, (void*)partial_x_array),
            partial_y_matrix(Size(3, 3), CV_64F, (void*)partial_y_array);
        Mat
            partial_x_image_double, partial_x_image_uchar,
            partial_y_image_double, partial_y_image_uchar,
            magnitude, direction;

        convolve(input, partial_x_image_double, partial_x_matrix);
        image_double_to_uchar(partial_x_image_double, partial_x_image_uchar);
        imwrite("partial_x " + image_path + ".png", partial_x_image_uchar);

        convolve(input, partial_y_image_double, partial_y_matrix);
        image_double_to_uchar(partial_y_image_double, partial_y_image_uchar);
        imwrite("partial_y " + image_path + ".png", partial_y_image_uchar);

        magnitude = partial_x_image_double.mul(partial_x_image_double) + partial_y_image_double.mul(partial_y_image_double);
        for (int y = 0; y < magnitude.rows; ++y)
            for (int x = 0; x < magnitude.cols; ++x)
                magnitude.at<double>(y, x) = std::sqrt(magnitude.at<double>(y, x));
        image_double_to_uchar(magnitude, magnitude);
        imwrite("magnitude " + image_path + ".png", magnitude);

        direction.create(input.size(), CV_64F);
        for (int y = 0; y < direction.rows; ++y)
            for (int x = 0; x < direction.cols; ++x)
            {
                direction.at<double>(y, x) = std::atan2(partial_y_image_double.at<double>(y, x), partial_x_image_double.at<double>(y, x));
                //printf("%f  ", direction.at<double>(y, x));
            }
        phase = direction;
        image_double_to_uchar(direction, direction);
        imwrite("direction " + image_path + ".png", direction);
        return EXIT_SUCCESS;
}

//just a convolution
void convolve(const Mat& input, Mat& output, const Mat& kernel)
{
    const Mat kernel_abs(abs(kernel));
    kernel /= std::accumulate(kernel_abs.begin<double>(), kernel_abs.end<double>(), 0.0);

    // Create a padded version of the input to prevent border effects
    const int
        kernelRadiusX = (kernel.size[0] - 1) / 2,
        kernelRadiusY = (kernel.size[1] - 1) / 2;

    Mat input_padded;
    copyMakeBorder(input, input_padded, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, BORDER_REPLICATE);

    output.create(input.size(), CV_64F);

    // Do the convolution
    for (int y = 0; y < input.rows; ++y)
        for (int x = 0; x < input.cols; ++x)
        {
            output.at<double>(y, x) = 0;
            for (int y_offset = -kernelRadiusX; y_offset <= kernelRadiusX; ++y_offset)
                for (int x_offset = -kernelRadiusY; x_offset <= kernelRadiusY; ++x_offset)
                {
                    const int x_k = y_offset + kernelRadiusX;
                    const int y_k = x_offset + kernelRadiusY;
                    output.at<double>(y, x) += input_padded.at<uchar>(y + x_k, x + y_k) * kernel.at<double>(x_k, y_k);
                }
        }
}

//scaling the image values to fit into uchar.
void image_double_to_uchar(const Mat& input, Mat& output)
{
    const double
        min = *std::min_element(input.begin<double>(), input.end<double>()),
        max = *std::max_element(input.begin<double>(), input.end<double>()),
        scale = 255 / (max - min),
        delta = -min * scale;
    input.convertTo(output, CV_8U, scale, delta);
}
