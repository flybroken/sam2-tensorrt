#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrackInSam2{
public:
	STrackInSam2(vector<float> tlwh_, float score,int label_id);
	~STrackInSam2();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrackInSam2*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	
	void single_predict();
	
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> mean_to_xyxy();
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();
	
	void activate(int frame_id);

	void re_activate(STrackInSam2 &new_track, int frame_id, bool new_id = false);
	void update(vector<float> tlwh_tmp, int frame_id);
	std::vector<float> compute_iou(vector<vector<float>> high_bboxs);

public:
	bool is_activated;
	int track_id;
	int label_id;
	int state;

	int stable_frames = 0;

	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	dataSam2Type::KAL_MEAN mean;
	dataSam2Type::KAL_COVA covariance;
	float score;

private:
	byte_kalman::KalmanFilter kalman_filter;
};

float _compute_iou(vector<float>bbox1, vector<float>bbox2);