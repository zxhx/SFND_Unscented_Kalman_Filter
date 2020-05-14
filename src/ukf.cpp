#include "ukf.h"
#include "Eigen/Dense"
#include "measurement_package.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  use_laser_ = true; // if this is false, laser measurements will be ignored (except during init)
  use_radar_ = true;  // if this is false, radar measurements will be ignored (except during init)
  is_initialized_ = false; // false until first measurement is taken
  x_ = VectorXd(5); // initial state vector
  P_ = MatrixXd(5, 5); // initial covariance matrix
  std_a_ = 3; // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 1;  // Process noise standard deviation yaw acceleration in rad/s^2
  std_laspx_ = 0.15; // Laser measurement noise standard deviation position1 in m
  std_laspy_ = 0.15; // Laser measurement noise standard deviation position2 in m
  std_radr_ = 0.3; // Radar measurement noise standard deviation radius in m
  std_radphi_ = 0.03; // Radar measurement noise standard deviation angle in rad
  std_radrd_ = 0.3; // Radar measurement noise standard deviation radius change in m/s
  n_x_ = 5; // State dimension for CTR model
  n_aug_ = 7; // Augmented state dimension for CTR model
  lambda_ = 3 - n_aug_; // Lambda governs sigma point spreading parameter

  // laser measurement matrix
  H_ = Eigen::MatrixXd(2,5);
  Ht_ = Eigen::MatrixXd(5,2);
  H_ << 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0;
  Ht_ = H_.transpose();

  // noise matrices
  R_laser_ = Eigen::MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_, 0.0,
              0.0, std_laspy_*std_laspy_;
  
  R_radar_ = Eigen::MatrixXd(3,3);
  R_radar_ <<  std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0,std_radrd_*std_radrd_;

  // identity matrix for regular Kalman
  I_ = MatrixXd::Identity(n_x_, n_x_);

  // weights for unscented predictions
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i=1;i<2*n_aug_+1;i++) {
    weights_(i) = 1/(2*(lambda_+n_aug_));
  }

  // Initialize prediciton matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    // set the state with the initial location and zero velocity
    x_ << meas_package.raw_measurements_[0], 
          meas_package.raw_measurements_[1], 
          0, 
          0,
          0;

    // guesstimate by checking, assume some correlation between turning radius and its acceleration
    P_ <<  0.25, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.25, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.25, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.25, 0.01,
           0.0, 0.0, 0.0, 0.01, 0.25;

    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;

  } else if (!is_initialized_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
    // set the state with the initial location and zero velocity
    x_ << meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]), 
          meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]), 
          0, 
          0,
          0;

    // guesstimate by checking, assume some correlation between turning radius and its acceleration
    P_ <<  0.25, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.25, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.25, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.25, 0.01,
           0.0, 0.0, 0.0, 0.01, 0.25;

    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;

  } else {
    float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000;
    previous_timestamp_ = meas_package.timestamp_;

    UKF::Prediction(dt);
    
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      UKF::UpdateLidar(meas_package);
    } else {
      UKF::UpdateRadar(meas_package);
    }
    return;
  }
}

void UKF::Prediction(double delta_t) {

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_*std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_*std_yawdd_;

  // create matrix： A
  MatrixXd A = P_aug.llt().matrixL();
  A = A*sqrt(lambda_ + n_x_ + 2);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);
  Xsig_aug.col(0) = x_aug;
  for (int i = 1; i < n_aug_ + 1; i++) {
      Xsig_aug.col(i) = x_aug + A.col(i-1);
      Xsig_aug.col(i+n_aug_) = x_aug - A.col(i-1);
  }

  // create matrix with predicted sigma points as columns
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      VectorXd x = Xsig_aug.col(i);
      if (abs(x(4)) < 0.00001) {
        Xsig_pred_(0,i) = x(0) + x(2)*cos(x(3))*delta_t + 0.5*pow(delta_t,2)*cos(x(3))*x(5);
        Xsig_pred_(1,i) = x(1) + x(2)*sin(x(3))*delta_t + 0.5*pow(delta_t,2)*sin(x(3))*x(5);
        Xsig_pred_(2,i) = x(2) + delta_t*x(5);
        Xsig_pred_(3,i) = x(3) + 0.5*pow(delta_t,2)*x(6);
        Xsig_pred_(4,i) = x(4) + delta_t*x(6);
      } else {
        Xsig_pred_(0,i) = x(0) + x(2)/x(4)*(sin(x(3)+x(4)*delta_t) - sin(x(3))) + 0.5*pow(delta_t,2)*cos(x(3))*x(5);
        Xsig_pred_(1,i) = x(1) + x(2)/x(4)*(-cos(x(3)+x(4)*delta_t) + cos(x(3))) + 0.5*pow(delta_t,2)*sin(x(3))*x(5);
        Xsig_pred_(2,i) = x(2) + delta_t*x(5);
        Xsig_pred_(3,i) = x(3) + x(4)*delta_t + 0.5*pow(delta_t,2)*x(6);
        Xsig_pred_(4,i) = x(4) + delta_t*x(6);
      }
  }

  // create vector for predicted state
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);
  for (int i=0;i<2*n_aug_+1;i++) {
    x_pred += weights_(i)*Xsig_pred_.col(i);
  }

  // create covariance matrix for prediction
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);
  for (int i=0;i<2*n_aug_+1;i++) {

    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x_pred;
  P_ = P_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {

  // map prediction to measure space using standard (linear) Kalman filter
  VectorXd z_laser = H_ * x_;
  VectorXd y = meas_package.raw_measurements_ - z_laser;
  MatrixXd S_laser = H_ * P_ * Ht_ + R_laser_;
  MatrixXd Si = S_laser.inverse();
  MatrixXd PHt = P_ * Ht_;
  MatrixXd K_laser = PHt * Si;

  //new estimate
  x_ = x_ + (K_laser * y);
  P_ = (I_ - K_laser * H_) * P_;

  // NIS calculation
  nis_ = y.transpose()*Si*y;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // n_z is three for radar
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

   // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_radar_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i=0;i<2*n_aug_+1;++i) {
      VectorXd x_diff = (Xsig_pred_.col(i) - x_);
      VectorXd z_diff = (Zsig.col(i) - z_pred);

      // angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      Tc += weights_(i)*x_diff*z_diff.transpose();
  }

  // Kalman matrix
  MatrixXd K = Tc*S.inverse();

  // new prediction and covariance
  x_ = x_ + K*(meas_package.raw_measurements_ - z_pred);
  P_ = P_ - K*S*K.transpose();

  // nis calculation
  nis_ = (meas_package.raw_measurements_ - z_pred).transpose()*S.inverse()*(meas_package.raw_measurements_ - z_pred);
}
