#ifndef MGPCLASS 
#define MGPCLASS

// uncomment to disable openmp on compilation
//#undef _OPENMP

#include "RcppArmadillo.h"

#include "distrib_vecrandom.h"
#include "mcmc_ramadapt.h"

#include "covariance_lmc.h"
#include "utils_caching.h"
#include "utils_others.h"
#include "utils_mgplmc.h"


#include <RcppEigen.h>
#include <Eigen/CholmodSupport>

//#include "utils_field_v_concatm.h"

using namespace std;

arma::uvec uturbocolthreshold(const arma::vec& col1, const arma::vec& thresholds);

arma::field<arma::uvec> split_ap(arma::uvec& membership, const arma::mat& coords, 
         const arma::field<arma::vec>& thresholds, int offset=0);

arma::field<arma::uvec> reorder_by_block(arma::field<arma::uvec> indexing,
                                         arma::uvec& order, 
                                         arma::uvec& restore_order,
                                         const arma::uvec& membership,
                                         int offset=0);

struct MeshDataLMC {
  arma::mat theta; 
  
  arma::vec nu;
  
  // x coordinates here
  // p parents coordinates
  
  arma::field<arma::cube> CC_cache; // C(x,x)
  arma::field<arma::cube> Kxxi_cache; // Ci(x,x)
  arma::field<arma::cube> H_cache; // C(x,p) Ci(p,p)
  arma::field<arma::cube> Ri_cache; // ( C(x,x) - C(x,p)Ci(p,p)C(p,x) )^{-1}
  arma::field<arma::cube> chR_cache;
  arma::field<arma::cube> chRi_cache;
  arma::field<arma::cube> Kppi_cache; // Ci(p,p)
  arma::vec Ri_chol_logdet;
  
  std::vector<arma::cube *> w_cond_chR_ptr;
  std::vector<arma::cube *> w_cond_chRi_ptr;
  std::vector<arma::cube *> w_cond_prec_ptr;
  std::vector<arma::cube *> w_cond_mean_K_ptr;
  std::vector<arma::cube *> w_cond_prec_parents_ptr;
  
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::mat wcore; 
  arma::mat loglik_w_comps;
  
  //arma::vec ll_y;
  
  double loglik_w; // will be pml
  
  // for collapsed sampler
  Eigen::SparseMatrix<double> Citsqi;
  //Eigen::SparseMatrix<double> Cidebug;
  //Eigen::SparseMatrix<double> Ciplustau2i;
  double Citsqi_ldet;
  arma::mat Citsqi_solveX;
  arma::mat Citsqi_solvey;
  arma::mat Citsqi_solvern;
  arma::mat bpost_Sichol;
  arma::mat bpost_meancore;
  double collapsed_ldens;
  
  // w cache
  arma::field<arma::mat> Sigi;
  arma::field<arma::mat> Smu;
  
  arma::field<arma::field<arma::cube> > AK_uP;
  
};

inline Eigen::MatrixXd armamat_to_matrixxd(arma::mat arma_A){
  
  Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                        arma_A.n_rows,
                                                        arma_A.n_cols);
  
  return eigen_B;
}
inline arma::mat matrixxd_to_armamat(Eigen::MatrixXd eigen_A){
  
  arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
                               true, false);
  
  return arma_B;
}


class MGP {
public:
  
  // utils
  int k; // not used
  int n; // number of in-sample observations
  int n_blocks;
  int dd;
  double hl2pi;
  bool use_ps;
  
  arma::uvec block_order;
  //arma::uvec block_order_restore;
  
  // kernel inputs
  arma::mat xcoords;
  
  // meshing inputs
  arma::mat coords;
  
  // value of process
  arma::mat w;
  
  // -----------------------------
  // -- partitioning, DAG 
  // -----------------------------
  // objects for blocking/partitioning/graphical model 
  arma::field<arma::vec> thresholds;
  arma::field<arma::uvec> indexing; 
  arma::field<arma::uvec> parents_indexing;
  
  arma::uvec membership; // block membership for each coordinate
  // objects for graphical representation of conditional independence
  arma::field<arma::vec> children;
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f;
  
  
  // 0 if no available obs in this block, >0=count how many available
  arma::umat block_ct_obs; 
  
  // initializing indexing and NA counts. we do this at any change in coordinates
  void init_partitiondag(const arma::mat& target_coords, 
                    const arma::field<arma::vec>& axis_partition);
  void refresh_partitiondag(const arma::umat& na_mat);
  // -----------------------------  
  // -- covariance stuff that does not depend on value of hyperparams
  // -----------------------------
  // objects for caching the Gaussian conditional density matrices
  arma::uvec coords_caching; 
  arma::uvec coords_caching_ix;
  arma::uvec kr_caching;
  arma::uvec kr_caching_ix;
  arma::uvec cx_and_kr_caching; // merge of coords and kr
  arma::uvec findkr;
  arma::uvec findcc;
  unsigned int starting_kr;

  void init_cache(bool use_cache);
  
  
  // covariance stuff that depends on value of hyperparams
  MeshDataLMC param_data;
  MeshDataLMC alter_data;
  
  // updating these parameters
  void init_meshdata(MeshDataLMC& data, const arma::mat& theta_in);
  bool refresh_cache(MeshDataLMC& data);
  bool get_mgplogdens_comps(MeshDataLMC& data);
  void update_block_covpars(int u, MeshDataLMC& data);
  void update_all_block_covpars(MeshDataLMC& data);
  void update_block_wlogdens(int u, MeshDataLMC& data);
  bool calc_mgplogdens(MeshDataLMC& data);
  void logpost_refresh_after_gibbs(MeshDataLMC& data);
  
  // utilities to get density of mgp
  bool calc_mgplogdens(MeshDataLMC& data, const arma::mat& w);
  
  // objects to update covariance hyperparameters
  // RAMA for theta
  void metrop_theta();
  bool theta_adapt_active;
  int theta_mcmc_counter;
  RAMAdapt theta_adapt;
  arma::mat theta_unif_bounds;
  arma::mat theta_metrop_sd;
  void accept_make_change();
  
  // -- utilities for getting full conditional mean and covariance
  
  
  // -- prior component
  // full conditional precision
  arma::mat block_fullconditional_prior_ci(int u, MeshDataLMC& data);
  // full conditional Smu and gradient
  arma::mat conj_block_fullconditional_m(int u, MeshDataLMC& data);
  double eval_block_fullconditional_m(int u, const arma::mat& x, MeshDataLMC& data);
  arma::mat block_grad_fullconditional_prior_m(double& logdens,
                                               const arma::mat& x, int u, MeshDataLMC& data);
  
  // likelihood component in the conjugate regression case
  void block_fullconditional_regdata(int u, const arma::mat& y,
                                          const arma::mat& XB,
                                          const arma::umat& na_mat,
                                          const arma::mat& Lambda,
                                          const arma::vec& tausq_inv,
                                          MeshDataLMC& data);
  
  // calculate quadratic forms using mgp precision
  double quadratic_form(const arma::mat& w1, const arma::mat& w2, MeshDataLMC& data);
  
  // MGP precision matrix components
  Eigen::SparseMatrix<double> He, Rie;
  arma::uvec sortsort_order;
  arma::umat linear_sort_map;
  void new_precision_matrix_product(MeshDataLMC& data);
  void new_precision_matrix_direct(MeshDataLMC& data);
  void update_precision_matrix(MeshDataLMC& data);
  
  // sampling from MGP prior
  void prior_sample(MeshDataLMC& data);
  
  // utility for sampling posterior without cholesky
  arma::mat posterior_sample_cholfree(MeshDataLMC& data);
  
  // collapsed sampler
  Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > solver;
  void solver_initialize();
  //void precision_cholesky_solve(const arma::vec& tausq);
  void collapsed_logdensity(MeshDataLMC& data, 
                            const Eigen::MatrixXd& ye, 
                                 const Eigen::MatrixXd& Xe,
                                 const arma::mat& y,
                                 const arma::mat& x,
                                 const arma::mat& xb,
                                 const arma::mat& Vbi,
                                 const arma::mat& XtX,
                                 const arma::vec& tausq_inv,
                                 bool also_sample_w=false);
  bool get_collapsed_logdens_comps(MeshDataLMC& data, 
                                   const Eigen::MatrixXd& ye, 
                                   const Eigen::MatrixXd& Xe,
                                   const arma::mat& y,
                                   const arma::mat& x,
                                   const arma::mat& xb,
                                   const arma::mat& Vbi,
                                   const arma::mat& XtX,
                                   const arma::vec& tausq_inv,
                                   bool also_sample_w=false);
  arma::vec metrop_theta_collapsed(const Eigen::MatrixXd& ye, 
                                   const Eigen::MatrixXd& Xe,
                                   const arma::mat& y,
                                   const arma::mat& x,
                                   const arma::mat& xb,
                                   const arma::mat& Vbi,
                                   const arma::mat& XtX,
                              const arma::vec& tausq_inv,
                              bool also_sample_w=false);
  
  arma::vec gibbs_beta_collapsed();
  
  
  MGP();
  MGP(const arma::mat& target_coords, 
      const arma::mat& x_coords,
      const arma::field<arma::vec>& axis_partition,
      const arma::mat& theta_in,
      bool adapting_theta,
      const arma::mat& metrop_theta_sd,
      const arma::mat& metrop_theta_bounds,
      int space_dim, bool v_in, bool d_in);
  
  // predictions
  arma::mat pred_coords;
  arma::mat pred_xcoords;
  arma::mat io_xcoords;
  int no;
  arma::field<arma::uvec> pred_indexing;
  arma::field<arma::uvec> pred_parents_indexing;
  arma::field<arma::uvec> pred_indexing_reorder;
  arma::field<arma::uvec> pred_parents_indexing_reorder;
  arma::field<arma::vec> pred_children;
  arma::umat pred_block_ct_obs; 
  arma::uvec pred_membership;
  arma::field<arma::field<arma::field<arma::uvec> > > pred_u_is_which_col_f;
  
  MGP(const arma::mat& target_coords, 
      const arma::mat& x_coords,
      const arma::mat& coordsout,
      const arma::mat& x_out,
      const arma::field<arma::vec>& axis_partition,
      const arma::mat& theta_in,
      int space_dim, bool v_in, bool d_in);
  void init_partitiondag_prediction(const arma::mat& target_coords, const arma::mat& coordsout,
                         const arma::field<arma::vec>& axis_partition);
  void prediction_sample(const arma::vec& theta,
                         const arma::mat& xobs, const arma::vec& wobs, 
                         const arma::field<arma::uvec>& obs_indexing);
  //arma::mat predict_via_precision_product(const MGP& out_mgp, const arma::vec& theta);
  //arma::mat predict_via_precision_orig(const MGP& out_mgp, const arma::vec& theta);
  //arma::mat predict_via_precision_direct(const arma::vec& theta);
  arma::mat predict_via_precision_part(const arma::vec& theta);
  //Eigen::SparseMatrix<double> PP_all, HH_o, PP_o, PP_ox, Ciprediction;
  arma::uvec pred_block_order;
  
  arma::uvec block_order_all;
  arma::uvec block_order_reverse;
  //arma::uvec pred_block_order_restore;
  
  
  bool verbose;
  bool debug;
};

inline MGP::MGP(){
  
  Eigen::SparseMatrix<double> In(1,1);
  In.setIdentity();
  
  //solver = Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> >(In);
  solver.compute(In);
  
  
  
}


#endif
