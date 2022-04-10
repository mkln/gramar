#ifndef MGPCLASS 
#define MGPCLASS

// uncomment to disable openmp on compilation
#undef _OPENMP

#include "RcppArmadillo.h"


#include "distrib_vecrandom.h"
#include "mcmc_ramadapt.h"

#include "covariance_lmc.h"
#include "utils_caching.h"
#include "utils_others.h"
#include "utils_mgplmc.h"


#include <RcppEigen.h>
//#include <Eigen/CholmodSupport>

//#include "utils_field_v_concatm.h"

using namespace std;

arma::uvec uturbocolthreshold(const arma::vec& col1, const arma::vec& thresholds);

arma::field<arma::uvec> split_ap(const arma::mat& coords, const arma::field<arma::vec>& thresholds);

struct MeshDataLMC {
  arma::mat theta; 
  arma::vec nu;
  
  // x coordinates here
  // p parents coordinates
  
  arma::field<arma::cube> CC_cache; // C(x,x)
  arma::field<arma::cube> Kxxi_cache; // Ci(x,x)
  arma::field<arma::cube> H_cache; // C(x,p) Ci(p,p)
  arma::field<arma::cube> Ri_cache; // ( C(x,x) - C(x,p)Ci(p,p)C(p,x) )^{-1}
  arma::field<arma::cube> Kppi_cache; // Ci(p,p)
  arma::vec Ri_chol_logdet;
  
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
  //Eigen::SparseMatrix<double> Ciplustau2i;
  double Citsqi_ldet;
  arma::mat Citsqi_solveX;
  arma::mat Citsqi_solveyXB;
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
                               false, false);
  
  return arma_B;
}


class MGP {
public:
  
  // utils
  int k;
  int n_blocks;
  int dd;
  double hl2pi;
  bool use_ps;
  
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
  
  // -- objects for information about data at each block
  // at least one of q available
  arma::field<arma::uvec> na_1_blocks; 
  // at least one of q missing
  arma::field<arma::uvec> na_0_blocks; 
  // indices of avails
  arma::field<arma::uvec> na_ix_blocks;
  // 0 if no available obs in this block, >0=count how many available
  arma::umat block_ct_obs; 
  
  // initializing indexing and NA counts. we do this at any change in coordinates
  void init_partitiondag(const arma::mat& target_coords, 
                    const arma::field<arma::vec>& axis_partition,
                    const arma::umat& na_mat);
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
  Eigen::SparseMatrix<double> H, Ri;
  arma::uvec sortsort_order;
  arma::umat linear_sort_map;
  void new_precision_matrix_product(MeshDataLMC& data);
  void new_precision_matrix_direct(MeshDataLMC& data);
  void update_precision_matrix(MeshDataLMC& data);
  
  // sampling from MGP prior
  void prior_sample(MeshDataLMC& data);
  
  // collapsed sampler
  Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > solver;
  void solver_initialize();
  //void precision_cholesky_solve(const arma::vec& tausq);
  void collapsed_logdensity(MeshDataLMC& data, const Eigen::MatrixXd& yXBe, 
                                 const Eigen::MatrixXd& Xe,
                                 const arma::vec& tausq_inv);
  bool get_collapsed_logdens_comps(MeshDataLMC& data, const Eigen::MatrixXd& yXBe, 
                                   const Eigen::MatrixXd& Xe,
                                   const arma::vec& tausq_inv);
  void metrop_theta_collapsed(const Eigen::MatrixXd& yXBe, const Eigen::MatrixXd& Xe,
                              const arma::vec& tausq_inv);
  
  
  MGP();
  MGP(const arma::mat& target_coords, 
      const arma::mat& x_coords,
      const arma::field<arma::vec>& axis_partition,
      const arma::umat& na_mat,
      bool cached, 
      const arma::mat& theta_in,
      bool adapting_theta,
      const arma::mat& metrop_theta_sd,
      const arma::mat& metrop_theta_bounds,
      bool use_ps_in,
      int space_dim, bool v_in, bool d_in);
  
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
