/*
 * CS5330 Computer Vision - Project 1
 * Authors: [Your Name], Shyam S (shyams612)
 * Date: May 2026
 *
 * Purpose: Implements all image processing filter functions for Project 1.
 *          Covers Tasks 4-9: custom greyscale, sepia, 5x5 blur (two versions),
 *          Sobel X/Y, gradient magnitude, and blur+quantize.
 */

#include "filter.h"
#include <cmath>

/*
 * greyscale: Custom greyscale using blue-emphasis weights.
 *
 * OpenCV's cvtColor uses 0.299R + 0.587G + 0.114B (luminance standard).
 * This function uses 0.15R + 0.35G + 0.50B, which heavily weights the blue
 * channel. Warm/red scenes appear darker and blue/sky scenes appear brighter
 * compared to standard greyscale — a clearly distinct visual result.
 * All three output BGR channels are set to the same computed value.
 */
int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(r);
        cv::Vec3b       *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            uchar B = srcRow[c][0];
            uchar G = srcRow[c][1];
            uchar R = srcRow[c][2];

            // Custom weighting: emphasize blue, de-emphasize red
            int grey = (int)(0.15f * R + 0.35f * G + 0.50f * B);
            grey = std::min(255, std::max(0, grey));

            dstRow[c] = cv::Vec3b((uchar)grey, (uchar)grey, (uchar)grey);
        }
    }
    return 0;
}

/*
 * sepia: Applies the classic sepia tone matrix to each pixel.
 *
 * All three original channel values (B, G, R) are read before any writes
 * so that the new channel values are always computed from the original data.
 * This prevents using a modified channel value in subsequent calculations.
 * Values exceeding 255 are clamped. OpenCV stores pixels as BGR.
 */
int sepia(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(r);
        cv::Vec3b       *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            // Read original BGR values first
            float B = srcRow[c][0];
            float G = srcRow[c][1];
            float R = srcRow[c][2];

            // Apply sepia matrix (coefficients given for R,G,B inputs)
            int newB = (int)(0.272f * R + 0.534f * G + 0.131f * B);
            int newG = (int)(0.349f * R + 0.686f * G + 0.168f * B);
            int newR = (int)(0.393f * R + 0.769f * G + 0.189f * B);

            // Clamp to valid uchar range
            dstRow[c][0] = (uchar)std::min(255, newB);
            dstRow[c][1] = (uchar)std::min(255, newG);
            dstRow[c][2] = (uchar)std::min(255, newR);
        }
    }
    return 0;
}

/*
 * blur5x5_1: Naive 5x5 Gaussian blur using the .at<> accessor.
 *
 * Applies the integer Gaussian kernel:
 *   [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1] (sum = 100)
 * for each color channel independently. The outer two rows and columns
 * are left as copies of src (no blur at the border).
 * NOTE: .at<> has bounds-checking overhead — this is intentionally slow
 * for comparison against the optimized blur5x5_2.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    dst = src.clone();

    // 5x5 Gaussian integer kernel (sum = 100)
    int kernel[5][5] = {
        {1,  2,  4,  2, 1},
        {2,  4,  8,  4, 2},
        {4,  8, 16,  8, 4},
        {2,  4,  8,  4, 2},
        {1,  2,  4,  2, 1}
    };

    for (int r = 2; r < src.rows - 2; r++) {
        for (int c = 2; c < src.cols - 2; c++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Accumulate weighted sum over the 5x5 neighborhood
            for (int kr = -2; kr <= 2; kr++) {
                for (int kc = -2; kc <= 2; kc++) {
                    cv::Vec3b px = src.at<cv::Vec3b>(r + kr, c + kc);
                    int w = kernel[kr + 2][kc + 2];
                    sumB += w * px[0];
                    sumG += w * px[1];
                    sumR += w * px[2];
                }
            }

            // Divide by kernel sum (100) and write to dst
            dst.at<cv::Vec3b>(r, c) = cv::Vec3b(
                (uchar)(sumB / 100),
                (uchar)(sumG / 100),
                (uchar)(sumR / 100)
            );
        }
    }
    return 0;
}

/*
 * blur5x5_2: Optimized 5x5 Gaussian blur using separable passes + row pointers.
 *
 * Decomposes the 2D Gaussian into two 1D passes with filter [1 2 4 2 1] (sum=10):
 *   Pass 1: apply filter horizontally across each row → temp image
 *   Pass 2: apply filter vertically down each column → dst image
 * Uses cv::Mat::ptr<>() for direct row pointer access, avoiding .at<> overhead.
 * Outer two rows/cols receive non-zero values copied from src.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    // Intermediate buffer for horizontal pass
    cv::Mat temp = src.clone();
    dst = src.clone();

    // Pass 1: horizontal filter [1 2 4 2 1] / 10
    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b *srcRow  = src.ptr<cv::Vec3b>(r);
        cv::Vec3b       *tmpRow  = temp.ptr<cv::Vec3b>(r);

        for (int c = 2; c < src.cols - 2; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int val = (int)srcRow[c - 2][ch]
                        + 2 * (int)srcRow[c - 1][ch]
                        + 4 * (int)srcRow[c    ][ch]
                        + 2 * (int)srcRow[c + 1][ch]
                        +     (int)srcRow[c + 2][ch];
                tmpRow[c][ch] = (uchar)(val / 10);
            }
        }
    }

    // Pass 2: vertical filter [1 2 4 2 1] / 10
    for (int r = 2; r < src.rows - 2; r++) {
        const cv::Vec3b *r0  = temp.ptr<cv::Vec3b>(r - 2);
        const cv::Vec3b *r1  = temp.ptr<cv::Vec3b>(r - 1);
        const cv::Vec3b *r2  = temp.ptr<cv::Vec3b>(r    );
        const cv::Vec3b *r3  = temp.ptr<cv::Vec3b>(r + 1);
        const cv::Vec3b *r4  = temp.ptr<cv::Vec3b>(r + 2);
        cv::Vec3b       *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 2; c < src.cols - 2; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int val = (int)r0[c][ch]
                        + 2 * (int)r1[c][ch]
                        + 4 * (int)r2[c][ch]
                        + 2 * (int)r3[c][ch]
                        +     (int)r4[c][ch];
                dstRow[c][ch] = (uchar)(val / 10);
            }
        }
    }
    return 0;
}

/*
 * sobelX3x3: 3x3 Sobel X filter via two separable 1x3 passes.
 *
 * The Sobel X kernel [-1 0 1; -2 0 2; -1 0 1] decomposes into:
 *   Horizontal: [-1 0 1]  (applied first)
 *   Vertical:   [1 2 1]   (applied second)
 * Positive output = right-going gradient (bright on right, dark on left).
 * Output is CV_16SC3 (signed short) to hold values in [-1020, 1020].
 * Pixels directly access rows via ptr<> for speed.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    // Intermediate: result of horizontal [-1 0 1] pass (short to avoid overflow)
    cv::Mat temp(src.rows, src.cols, CV_16SC3, cv::Scalar(0));
    dst = cv::Mat(src.rows, src.cols, CV_16SC3, cv::Scalar(0));

    // Pass 1: horizontal [-1 0 1]
    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b  *srcRow = src.ptr<cv::Vec3b>(r);
        cv::Vec3s        *tmpRow = temp.ptr<cv::Vec3s>(r);

        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                tmpRow[c][ch] = (short)(-(int)srcRow[c - 1][ch]
                                        + (int)srcRow[c + 1][ch]);
            }
        }
    }

    // Pass 2: vertical [1 2 1]
    for (int r = 1; r < src.rows - 1; r++) {
        const cv::Vec3s *t0  = temp.ptr<cv::Vec3s>(r - 1);
        const cv::Vec3s *t1  = temp.ptr<cv::Vec3s>(r    );
        const cv::Vec3s *t2  = temp.ptr<cv::Vec3s>(r + 1);
        cv::Vec3s       *dstRow = dst.ptr<cv::Vec3s>(r);

        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                dstRow[c][ch] = (short)((int)t0[c][ch]
                                      + 2 * (int)t1[c][ch]
                                      + (int)t2[c][ch]);
            }
        }
    }
    return 0;
}

/*
 * sobelY3x3: 3x3 Sobel Y filter via two separable 1x3 passes.
 *
 * The Sobel Y kernel (positive up) [1 2 1; 0 0 0; -1 -2 -1] decomposes into:
 *   Horizontal: [1 2 1]   (applied first)
 *   Vertical:   [1 0 -1]  (applied second, positive = upward gradient)
 * "Positive up" means: top pixels brighter than bottom → positive output.
 * In image coords, top = lower row index, so vertical filter is [1; 0; -1].
 * Output is CV_16SC3 (signed short).
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    cv::Mat temp(src.rows, src.cols, CV_16SC3, cv::Scalar(0));
    dst = cv::Mat(src.rows, src.cols, CV_16SC3, cv::Scalar(0));

    // Pass 1: horizontal [1 2 1]
    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b  *srcRow = src.ptr<cv::Vec3b>(r);
        cv::Vec3s        *tmpRow = temp.ptr<cv::Vec3s>(r);

        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                tmpRow[c][ch] = (short)((int)srcRow[c - 1][ch]
                                      + 2 * (int)srcRow[c][ch]
                                      + (int)srcRow[c + 1][ch]);
            }
        }
    }

    // Pass 2: vertical [1 0 -1] (positive up)
    for (int r = 1; r < src.rows - 1; r++) {
        const cv::Vec3s *t0  = temp.ptr<cv::Vec3s>(r - 1);
        const cv::Vec3s *t2  = temp.ptr<cv::Vec3s>(r + 1);
        cv::Vec3s       *dstRow = dst.ptr<cv::Vec3s>(r);

        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                dstRow[c][ch] = (short)((int)t0[c][ch] - (int)t2[c][ch]);
            }
        }
    }
    return 0;
}

/*
 * magnitude: Euclidean gradient magnitude from Sobel X and Y images.
 *
 * For each pixel and channel: dst = sqrt(sx^2 + sy^2), clamped to [0, 255].
 * Input images must be CV_16SC3 (output of sobelX3x3 / sobelY3x3).
 * Output is CV_8UC3 suitable for direct display with imshow.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) return -1;
    if (sx.size() != sy.size())   return -1;

    dst.create(sx.rows, sx.cols, CV_8UC3);

    for (int r = 0; r < sx.rows; r++) {
        const cv::Vec3s *sxRow  = sx.ptr<cv::Vec3s>(r);
        const cv::Vec3s *syRow  = sy.ptr<cv::Vec3s>(r);
        cv::Vec3b       *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < sx.cols; c++) {
            for (int ch = 0; ch < 3; ch++) {
                float mag = std::sqrt((float)(sxRow[c][ch] * sxRow[c][ch])
                                    + (float)(syRow[c][ch] * syRow[c][ch]));
                dstRow[c][ch] = (uchar)std::min(255.0f, mag);
            }
        }
    }
    return 0;
}

/*
 * depthFog: Exponential depth-based fog.
 *
 * DA2 outputs higher values for closer objects, so distance = 1 - depth/255.
 * Fog amount per pixel: f = 1 - exp(-k * distance)
 * Final color: dst = (1-f)*src + f*fog_color
 * Fog color is a light grey (200, 200, 200).
 */
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, float k) {
    if (src.empty() || depth.empty()) return -1;
    if (src.size() != depth.size())   return -1;

    dst.create(src.rows, src.cols, CV_8UC3);

    const cv::Vec3b fogColor(200, 200, 200);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b    *srcRow = src.ptr<cv::Vec3b>(r);
        const unsigned char *depRow = depth.ptr<unsigned char>(r);
        cv::Vec3b           *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            float distance = 1.0f - depRow[c] / 255.0f;  // 0=close, 1=far
            float fog = 1.0f - std::exp(-k * distance);

            for (int ch = 0; ch < 3; ch++) {
                float blended = (1.0f - fog) * srcRow[c][ch] + fog * fogColor[ch];
                dstRow[c][ch] = (uchar)std::min(255.0f, std::max(0.0f, blended));
            }
        }
    }
    return 0;
}

/*
 * emboss: Emboss effect via Sobel dot product with a light direction.
 *
 * For each pixel, compute the dot product of the (sx, sy) gradient vector
 * with a fixed light direction (lx, ly) = (0.7071, 0.7071) — upper-left light.
 * The result is averaged across the three channels then shifted by +128 so
 * that flat regions map to mid-grey, raised edges facing the light are bright,
 * and edges facing away are dark. Output is clamped to [0, 255].
 */
int emboss(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    dst.create(src.rows, src.cols, CV_8UC3);

    const float lx = 0.7071f;
    const float ly = 0.7071f;

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(r);
        const cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(r);
        cv::Vec3b       *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            float dot = 0.0f;
            for (int ch = 0; ch < 3; ch++) {
                dot += sxRow[c][ch] * lx + syRow[c][ch] * ly;
            }
            dot /= 3.0f;  // average across channels

            int val = (int)(dot + 128.0f);
            uchar out = (uchar)std::min(255, std::max(0, val));
            dstRow[c] = cv::Vec3b(out, out, out);
        }
    }
    return 0;
}

/*
 * negative: Pixel-wise color negative.
 *
 * Each channel value is inverted: dst[c] = 255 - src[c].
 * No neighborhood access needed — purely per-pixel.
 */
int negative(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(r);
        cv::Vec3b       *dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            dstRow[c] = cv::Vec3b(255 - srcRow[c][0],
                                  255 - srcRow[c][1],
                                  255 - srcRow[c][2]);
        }
    }
    return 0;
}

/*
 * blurQuantize: Blurs a color image then quantizes each channel.
 *
 * Step 1: Apply blur5x5_2 to smooth the image.
 * Step 2: Quantize each channel into `levels` discrete values:
 *           bucket = 255 / levels
 *           xt = pixel_val / bucket        (integer division → bin index)
 *           xf = xt * bucket               (map back to quantized value)
 * A good default for `levels` is 10, producing a cartoon/posterize effect.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || levels <= 0) return -1;

    // Step 1: blur
    cv::Mat blurred;
    blur5x5_2(src, blurred);

    // Step 2: quantize
    dst = blurred.clone();
    int bucket = 255 / levels;

    for (int r = 0; r < blurred.rows; r++) {
        const cv::Vec3b *blurRow = blurred.ptr<cv::Vec3b>(r);
        cv::Vec3b       *dstRow  = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < blurred.cols; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int xt = blurRow[c][ch] / bucket;
                dstRow[c][ch] = (uchar)(xt * bucket);
            }
        }
    }
    return 0;
}
