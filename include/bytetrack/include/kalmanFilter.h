#pragma once

#include "dataType.h"

namespace byte_kalman
{
	class KalmanFilter
	{
	public:
		static const double chi2inv95[10];
		KalmanFilter();
		dataSam2Type::KAL_DATA initiate(const dataSam2Type::DETECTBOX& measurement);
		void predict(dataSam2Type::KAL_MEAN& mean, dataSam2Type::KAL_COVA& covariance);
		dataSam2Type::KAL_HDATA project(const dataSam2Type::KAL_MEAN& mean, const dataSam2Type::KAL_COVA& covariance);
		dataSam2Type::KAL_DATA update(const dataSam2Type::KAL_MEAN& mean,
			const dataSam2Type::KAL_COVA& covariance,
			const dataSam2Type::DETECTBOX& measurement);

		Eigen::Matrix<float, 1, -1> gating_distance(
			const dataSam2Type::KAL_MEAN& mean,
			const dataSam2Type::KAL_COVA& covariance,
			const std::vector<dataSam2Type::DETECTBOX>& measurements,
			bool only_position = false);

	private:
		Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
		Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
		float _std_weight_position;
		float _std_weight_velocity;
	};
}