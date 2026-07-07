#include "STrack.h"

STrackInSam2::STrackInSam2(vector<float> tlwh_, float score,int label_id){
	this->kalman_filter = byte_kalman::KalmanFilter();
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;
	track_id = 0;
	state = TrackState::New;

	frame_id = 0;
	tracklet_len = 0;
	this->score = score;
	this->label_id = label_id;
	start_frame = 0;
}

STrackInSam2::~STrackInSam2()
{
}

void STrackInSam2::activate(int frame_id)
{
	this->track_id = this->next_id();

	vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	dataSam2Type::DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.initiate(xyah_box);

	this->mean = mc.first;
	this->covariance = mc.second;

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	//if (frame_id == 1)
	{
		this->is_activated = true;
	}
	//this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

void STrackInSam2::re_activate(STrackInSam2 &new_track, int frame_id, bool new_id)
{
	vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	dataSam2Type::DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	if (new_id)
		this->track_id = next_id();
}

void STrackInSam2::update(vector<float> tlwh_tmp, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;

	vector<float> xyah = tlwh_to_xyah(tlwh_tmp);
	dataSam2Type::DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	this->state = TrackState::Tracked;
	this->is_activated = true;
}

void STrackInSam2::static_tlwh()
{
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];

	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

void STrackInSam2::static_tlbr()
{
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}

vector<float> STrackInSam2::tlwh_to_xyah(vector<float> tlwh_tmp)
{
	vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;
	tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

vector<float> STrackInSam2::mean_to_xyxy()
{
	std::vector<float> kf_bbox(this->mean.head<4>().data(), this->mean.head<4>().data() + 4);
    vector<float> xyxy(4, 0); // 初始化输出向量，包含4个元素，全部初始化为0

    // 中心点坐标
    float x_center = kf_bbox[0];
    float y_center = kf_bbox[1];

    // 宽高比和高度
    float aspect_ratio = kf_bbox[2];
    float height = kf_bbox[3];

    // 计算宽度
    float width = aspect_ratio * height;

    // 计算左上角和右下角坐标
    xyxy[0] = x_center - width / 2; // 左上角 x 坐标
    xyxy[1] = y_center - height / 2; // 左上角 y 坐标
    xyxy[2] = x_center + width / 2; // 右下角 x 坐标
    xyxy[3] = y_center + height / 2; // 右下角 y 坐标

    return xyxy;
}

vector<float> STrackInSam2::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

vector<float> STrackInSam2::tlbr_to_tlwh(vector<float> &tlbr)
{
	tlbr[2] -= tlbr[0];
	tlbr[3] -= tlbr[1];
	return tlbr;
}

void STrackInSam2::mark_lost()
{
	state = TrackState::Lost;
}

void STrackInSam2::mark_removed()
{
	state = TrackState::Removed;
}

int STrackInSam2::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

int STrackInSam2::end_frame()
{
	return this->frame_id;
}

void STrackInSam2::multi_predict(vector<STrackInSam2*> &stracks, byte_kalman::KalmanFilter &kalman_filter)
{
	for (int i = 0; i < stracks.size(); i++)
	{
		if (stracks[i]->state != TrackState::Tracked)
		{
			stracks[i]->mean[7] = 0;
		}
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
	}
}

void STrackInSam2::single_predict()
{
	this->kalman_filter.predict(this->mean, this->covariance);
}

std::vector<float> STrackInSam2::compute_iou(vector<vector<float>> high_bboxs)
{
	std::vector<float> pred_bbox = this->mean_to_xyxy();
	std::vector<float> ious;

	for(size_t i = 0; i < high_bboxs.size(); i++)
	{
		float iou = _compute_iou(pred_bbox, high_bboxs[i]);
		ious.push_back(iou);
	}
	return ious;
}

float _compute_iou(vector<float>bbox1, vector<float>bbox2) {
    // 计算交集的左上角和右下角
    float x_min_intersection = std::max(bbox1[0], bbox2[0]);
    float y_min_intersection = std::max(bbox1[1], bbox2[1]);
    float x_max_intersection = std::min(bbox1[2], bbox2[2]);
    float y_max_intersection = std::min(bbox1[3], bbox2[3]);

    // 如果交集不存在（即交集区域的右下角小于左上角），返回0
    if (x_min_intersection >= x_max_intersection || y_min_intersection >= y_max_intersection) {
        return 0.0f;
    }

    // 计算交集面积
    float intersection_area = (x_max_intersection - x_min_intersection) * (y_max_intersection - y_min_intersection);

    // 计算两个框的面积
    float area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

    // 计算并集面积
    float union_area = area_bbox1 + area_bbox2 - intersection_area;

    // 计算并返回 IoU
    return intersection_area / union_area;
}
