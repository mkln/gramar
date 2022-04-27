#define ARMA_DONT_PRINT_ERRORS

#ifndef GRAMAR 
#define GRAMAR

// uncomment to disable openmp on compilation
#undef _OPENMP

#include <RcppArmadillo.h>
#include "meshedgp.h"

class Gramar {
public:
  
  // meta
  unsigned int n; // number of locations, total
  unsigned int p; // number of covariates
  unsigned int q; // number of outcomes
  
  unsigned int dd; // dimension
  unsigned int n_blocks; // number of blocks
  
  // data
  arma::mat y, X;
  Eigen::MatrixXd ye, Xe;
  
  // coords
  arma::mat coords;

  // indexing info
  arma::field<arma::vec> axis_partition;
  
  //arma::umat na_mat;
  
  // variable data
  //arma::field<arma::uvec> ix_by_q_a; // storing indices using only available data
  
  //int n_loc_ne_blocks;
  
  // DAG
  //arma::field<arma::sp_mat> Ib;

  //arma::uvec u_predicts;
  
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  
  // utils
  arma::uvec oneuv;
  double hl2pi;
    
  // params
  arma::mat yhat;
  arma::mat offsets;
  
  //arma::mat rand_norm_mat;
  arma::vec rand_unif;
  arma::vec rand_unif2;
  
  arma::mat XB; // by outcome
  arma::mat linear_predictor;
  
  arma::mat XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  arma::mat Bcoeff; // sampled
  
  // setup
  bool predicting;
  //bool cached;
  
  bool verbose;
  bool debug;
  
  // predictions
  arma::field<arma::cube> Hpred;
  arma::field<arma::mat> Rcholpred;

  MGP mgp;
  
  //double logpost;
  
  // changing the values, no sampling;
  //void theta_update(MeshDataLMC&, const arma::mat&);
  void beta_update();
  
  // --------------------------------------------------------------- from Gaussian

  // tausq 
  arma::vec tausq_ab;
  arma::vec tausq_inv; // tausq for the l=q variables
  
  // MCMC
  // ModifiedPP-like updates for tausq -- used if not forced_grid
  int tausq_mcmc_counter;
  RAMAdapt tausq_adapt;
  arma::mat tausq_unif_bounds;
  
  
  bool calc_ywlogdens(MeshDataLMC& data);

  
  
  //void logpost_refresh_after_gibbs(MGP& mgp, MeshDataLMC& data, bool sample=true); 
  
  // Predictions for W and Y
  void predict(bool sample=true);
  
  // metropolis update vector and utils
  void metrop_thetastar();
  bool thetastar_adapt_active;
  int thetastar_mcmc_counter;
  RAMAdapt thetastar_adapt;
  arma::mat thetastar_unif_bounds;
  arma::mat thetastar_metrop_sd;
  
  // need to adjust these
  int npars;
  //int nugget_in;
  
  // --------------------------------------------------------------- timers
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  void mgp_initialize_param();
  void metrop_theta_collapsed(bool sample_w);
  void mgp_post_solver_initialize();
  
  // --------------------------------------------------------------- constructors
  
  //Gramar(){};
  Gramar(
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    
    const arma::mat& coords_in, 
    const arma::mat& xcoords_in,
    
    const arma::field<arma::vec>& axis_partition_in,
    
    const arma::mat& beta_in,
    const arma::mat& theta_in,
    const arma::vec& tausq_inv_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& tausq_ab_in,
    
    bool adapting_theta,
    const arma::mat& metrop_theta_sd,
    const arma::mat& metrop_theta_bounds,
    
    bool verbose_in,
    bool debugging,
    int num_threads);
};

#endif
