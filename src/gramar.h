#define ARMA_DONT_PRINT_ERRORS

#ifndef M1DSP 
#define M1DSP

// uncomment to disable openmp on compilation
#undef _OPENMP

#include <RcppArmadillo.h>
#include "meshedgp.h"
#include "mcmc_hmc_sample.h"



class Gramar {
public:
  
  arma::uvec familyid;
  
  // meta
  unsigned int n; // number of locations, total
  unsigned int p; // number of covariates
  unsigned int q; // number of outcomes
  unsigned int nfact; // number of factors
  unsigned int dd; // dimension
  unsigned int n_blocks; // number of blocks
  
  // data
  arma::mat y;
  
  arma::mat X;
  arma::mat Z;
  
  Eigen::MatrixXd yXBe, Xe;
  
  // coords
  arma::mat coords;

  // indexing info
  arma::field<arma::vec> axis_partition;
  
  arma::umat na_mat;
  
  // variable data
  arma::field<arma::uvec> ix_by_q_a; // storing indices using only available data
  
  int n_loc_ne_blocks;
  
  // DAG
  arma::field<arma::sp_mat> Ib;
  //arma::field<arma::uvec>   parents; // i = parent block names for i-labeled block (not ith block)
  //arma::field<arma::uvec>   children; // i = children block names for i-labeled block (not ith block)
  //arma::vec                 block_names; //  i = block name (+1) of block i. all dependence based on this name
  //arma::uvec                ref_block_names;
  //arma::vec                 block_groups; // same group = sample in parallel given all others
  
  //int                       n_gibbs_groups;
  //arma::field<arma::vec>    u_by_block_groups;
  //int                       predict_group_exists;
  
  arma::uvec u_predicts;
  
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  
  
  
  // utils
  arma::uvec oneuv;
  double hl2pi;
    
  // params
  arma::mat yhat;
  arma::mat offsets;
  arma::mat rand_norm_mat;
  arma::vec rand_unif;
  arma::vec rand_unif2;
  // regression
  arma::mat Lambda;
  arma::umat Lambda_mask; // 1 where we want lambda to be nonzero
  arma::mat LambdaHw; // part of the regression mean explained by the latent process
  //arma::mat wU; // nonreference locations
  
  arma::mat XB; // by outcome
  arma::mat linear_predictor;
  
  arma::field<arma::mat> XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  
  arma::mat Bcoeff; // sampled
  
  
  // setup
  bool predicting;
  bool cached;
  
  bool verbose;
  bool debug;
  
  // predictions
  arma::field<arma::cube> Hpred;
  arma::field<arma::mat> Rcholpred;
  
  // init / indexing
  //void init_indexing();
  //void na_study();
  //void make_gibbs_groups();
  //void init_gibbs_index();
  //void init_matern(int num_threads, int matern_twonu_in, bool use_ps);
  void init_for_mcmc();
  
  //void init_cache(MeshDataLMC&, const arma::mat& target_coords, bool cached);
  //void init_meshdata(MeshDataLMC&, const arma::mat&, int);
  
  MGP mgp;
  
  double logpost;
  
  bool use_ps;
  
  // changing the values, no sampling;
  //void theta_update(MeshDataLMC&, const arma::mat&);
  void beta_update(const arma::vec&);
  void tausq_update(double);
  
  
  
  // --------------------------------------------------------------- from Gaussian
  

  // tausq 
  arma::vec tausq_ab;
  arma::vec tausq_inv; // tausq for the l=q variables
  
  // MCMC
  // ModifiedPP-like updates for tausq -- used if not forced_grid
  int tausq_mcmc_counter;
  RAMAdapt tausq_adapt;
  arma::mat tausq_unif_bounds;
  
  // tausq for Beta regression
  arma::vec brtausq_mcmc_counter;
  std::vector<RAMAdapt> opt_tausq_adapt;
  
  int lambda_mcmc_counter;
  int n_lambda_pars;
  arma::uvec lambda_sampling;
  arma::mat lambda_unif_bounds; // 1x2: lower and upper for off-diagonal
  RAMAdapt lambda_adapt;
  
  void init_betareg();
  
  //void update_lly(int, MeshDataLMC&, const arma::mat& LamHw, bool map=false);
  //void calc_DplusSi(int, MeshDataLMC& data, const arma::mat& Lam, const arma::vec& tsqi);
  void update_block_w_cache(int, MeshDataLMC& data);
  void refresh_w_cache(MeshDataLMC& data);
  
  // W -- spatial process at data layer
  int which_hmc;
  bool w_do_hmc;
  bool w_hmc_nuts;
  bool w_hmc_rm;
  bool w_hmc_srm;
  
  void deal_with_w(MGP& mgp);
  void gaussian_w(MGP& mgp);
  void nongaussian_w(MGP& mgp);
  
  void w_prior_sample(MeshDataLMC& data);
  
  std::vector<NodeDataW> w_node;
  arma::vec hmc_eps;
  std::vector<AdaptE> hmc_eps_adapt;
  arma::uvec hmc_eps_started_adapting;
  
  bool calc_ywlogdens(MeshDataLMC& data);
  
  // Beta & Lambda for data layer
  void deal_with_BetaLambdaTau(MeshDataLMC& data, bool sample, 
                               bool sample_beta, bool sample_lambda, bool sample_tau);
  arma::vec sample_BetaLambda_row(bool sample, int j, const arma::mat& rnorm_precalc);
  void sample_hmc_BetaLambdaTau(bool sample, 
                                bool sample_beta, bool sample_lambda, bool sample_tau);

  
  std::vector<NodeDataB> lambda_node; // std::vector
  std::vector<AdaptE> lambda_hmc_adapt; // std::vector
  arma::uvec lambda_hmc_started;
  
  
  // Tausq
  void deal_with_tausq();
  void gibbs_sample_tausq_std();
  
  //void logpost_refresh_after_gibbs(MGP& mgp, MeshDataLMC& data, bool sample=true); 
  
  // Predictions for W and Y
  void predict(bool sample=true);
  void predicty(bool sample=true);
  
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
  
  void metrop_theta_collapsed();
  // --------------------------------------------------------------- constructors
  
  //Gramar(){};
  Gramar(
    const arma::mat& y_in, 
    const arma::uvec& familyid,
    
    const arma::mat& X_in, 
    
    const arma::mat& coords_in, 
    const arma::mat& xcoords_in,
    
    const arma::field<arma::vec>& axis_partition_in,
    int k_in,
    
    const arma::mat& w_in,
    const arma::mat& beta_in,
    const arma::mat& lambda_in,
    const arma::umat& lambda_mask_in,
    
    const arma::mat& theta_in,
    const arma::vec& tausq_inv_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& tausq_ab_in,
    
    int which_hmc_in,
    bool adapting_theta,
    const arma::mat& metrop_theta_sd,
    const arma::mat& metrop_theta_bounds,
    
    bool use_cache,
    bool use_ps,
    
    bool verbose_in,
    bool debugging,
    int num_threads);
};

#endif
