// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "platform.h"
#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

using std::vector;

struct FaceObject
{
    cv::Rect_<float> rect;
	//cv::Point2f landmark[5];
    float prob;
};

static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// copy from src/layer/proposal.cpp
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);//round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}


static ncnn::Mat generate_anchors(const int32_t image_w, const int32_t image_h,
		const float OCTAVE, const float SCALE_PER_OCTAVE, const vector<float>& ASPRCT_RATIO,
		const vector<int>& ANCHOR_SIZE, const vector<int>& STRIDE){
	int num_anchors = 0;
	for (std::size_t j = 0; j < STRIDE.size(); ++j)
	{
		num_anchors += image_w * image_h / STRIDE[j] / STRIDE [j] * SCALE_PER_OCTAVE * ASPRCT_RATIO.size(); 
	}
	ncnn::Mat anchors;
	anchors.create(4, num_anchors);
	int anchor_count = 0;

	for (std::size_t i = 0; i < STRIDE.size(); ++i) 
	{
		vector<vector<float>> anchors_per_point;
		int anchor_size = ANCHOR_SIZE[i];
		vector<float> base_box = {0.0, 0.0, STRIDE[i] - 1.0, STRIDE[i] - 1.0};
		for (std::size_t j = 0; j < ASPRCT_RATIO.size(); ++j)
		{
			float aspect_ratio = ASPRCT_RATIO[j];
			float w = base_box[2] - base_box[0] + 1.0;
			float h = base_box[3] - base_box[1] + 1.0;
			float x_ctr = base_box[0] + 0.5 * (w - 1.0);
			float y_ctr = base_box[1] + 0.5 * (h - 1.0);
			float size = w * h;
			float size_ratio = size / aspect_ratio;
			float ws = sqrt(size_ratio);
			ws -= remainder(ws, 1.0);
			float hs = ws * aspect_ratio;
			hs -= remainder(hs, 1.0);

			for (int oct = 0; oct < SCALE_PER_OCTAVE; oct ++)
			{
				float area = anchor_size * pow(OCTAVE, oct / SCALE_PER_OCTAVE);
				float scales = area / float(STRIDE[i]);
				float ws_s = ws * scales;
				float hs_s = hs * scales;
				vector<float> temp = {
					ws_s, hs_s, x_ctr+0.5, y_ctr+0.5
				}; // w, h , x, y of each anchors

				anchors_per_point.push_back(temp);
			}// area
		} // aspect

		// walk through each points
		for (int y = 0; y < image_h; y += STRIDE[i])
		{
			for (int x = 0; x < image_w; x += STRIDE[i])
			{
				for (vector<vector<float>>::iterator it = anchors_per_point.begin();
						it != anchors_per_point.end(); it ++)
				{
					float* anchor = anchors.row(anchor_count);
					anchor[0] = (*it)[0];
					anchor[1] = (*it)[1];
					anchor[2] = x + (*it)[2];
					anchor[3] = y + (*it)[3];
					anchor_count ++ ;
				}
			}
		}
	}
	return anchors;

}

static void generate_proposals(const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& anchors, float prob_threshold, std::vector<FaceObject>& faceobjects){
	int num_anchors = score_blob.h;
	for (int i=0; i<num_anchors; i++){
		float prob = (1 / (1 + exp(-score_blob[i])));
		if (prob >= prob_threshold)
		{
			const float* ptr = bbox_blob.row(i);
			const float* anchor_ptr = anchors.row(i); // w, h, x, y

			float dx = ptr[0] * 0.1;
			float dy = ptr[1] * 0.1;
			float dw = ptr[2] * 0.2;
			float dh = ptr[3] * 0.2;
			
			float pb_cx = anchor_ptr[2] + anchor_ptr[0] * dx;
			float pb_cy = anchor_ptr[3] + anchor_ptr[1] * dy;

			float pb_w = anchor_ptr[0] * exp(dw);
			float pb_h = anchor_ptr[1] * exp(dh);

			float x0 = pb_cx - pb_w * 0.5f;
			float y0 = pb_cy - pb_h * 0.5f;
			float x1 = pb_cx + pb_w * 0.5f - 1;
			float y1 = pb_cy + pb_h * 0.5f - 1;
			FaceObject obj;
			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0 + 1;
			obj.rect.height = y1 - y0 + 1;
			obj.prob = prob;
			faceobjects.push_back(obj);
		}
		
	}	
}

static int detect_retinaperson(const cv::Mat& bgr, const ncnn::Mat& anchors, std::vector<FaceObject>& faceobjects, const int resize_w, const int resize_h)
{
    ncnn::Net retinaperson;

#if NCNN_VULKAN
    retinaperson.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // mobile1.0x: input 0, score 545, bbox 546
    retinaperson.load_param("/models/mobile1.0x.param");
    retinaperson.load_model("/models/mobile1.0x.bin");

    const float prob_threshold = 0.5f;
    const float nms_threshold = 0.5f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, resize_w, resize_h);
	const float mean_vals[3] = {102.9801f, 115.9465f, 122.7717f};
	in.substract_mean_normalize(mean_vals, 0);
    ncnn::Extractor ex = retinaperson.create_extractor();

    //ex.input("x.1", in);
    ex.input("0", in);

	ncnn::Mat score_blob, bbox_blob;
	//ex.extract("270", score_blob);
	//ex.extract("271", bbox_blob);
	ex.extract("545", score_blob);
	ex.extract("546", bbox_blob);
	//printf("anchors %d, %d\n", anchors.w, anchors.h);

	std::vector<FaceObject> faceproposals;
	generate_proposals(score_blob, bbox_blob, anchors, prob_threshold, faceproposals);
	//printf("bboxs: %d\n", faceproposals.size());
	qsort_descent_inplace(faceproposals);
	// apply nms with nms_threshold
	std::vector<int> picked;
	nms_sorted_bboxes(faceproposals, picked, nms_threshold);
	
	int _count = picked.size();

	faceobjects.resize(_count);

	float ratio_width = float(img_w) / float(resize_w);
	float ratio_height = float(img_h) / float(resize_h);
	for (int i = 0; i < _count; i++)
	{
		faceobjects[i] = faceproposals[picked[i]];
        // resize and clip to image size
        float x0 = faceobjects[i].rect.x;
        float y0 = faceobjects[i].rect.y;
        float x1 = x0 + faceobjects[i].rect.width;
        float y1 = y0 + faceobjects[i].rect.height;
		x0 *= ratio_width;
		x1 *= ratio_width;
		y0 *= ratio_height;
		y1 *= ratio_height;

        x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;
	}
	printf("num of boxes: %d", faceobjects.size());

    return 0;
}

static void draw_faceobjects(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

	cv::imwrite("retinaperson_output.jpg", image);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN
	
	// settings for anchors generator
	const int resize_h = 512, resize_w = 928; // (288x480) || (384x672) || (512x928)
	const float OCTAVE = 2.0;
	const float SCALE_PER_OCTAVE = 3;
	const vector<float> ASPRCT_RATIO {0.8, 1.5, 2.5, 3.5};
	const vector<int> ANCHOR_SIZE {32, 64, 128};
	const vector<int> STRIDE {8, 16, 32};
	const ncnn::Mat anchors = generate_anchors(resize_w, resize_h, OCTAVE, SCALE_PER_OCTAVE, ASPRCT_RATIO, ANCHOR_SIZE, STRIDE);
	printf("anchor shape : %d %d %d\n", anchors.w, anchors.h, anchors.c);


    std::vector<FaceObject> faceobjects;
    detect_retinaperson(m, anchors, faceobjects, resize_w, resize_h);

#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

	draw_faceobjects(m, faceobjects);

    return 0;
}
