#include "gramar.h"
using namespace std;


Gramar::Gramar(
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
  
  bool verbose_in=false,
  bool debugging=false,
  int num_threads = 1){
  
  oneuv = arma::ones<arma::uvec>(1);//utils
  hl2pi = -.5 * log(2.0 * M_PI);
  
  verbose = verbose_in;
  debug = debugging;
  
  if(verbose & debug){
    Rcpp::Rcout << "Gramar::Gramar initialization.\n";
  }
  
  // data
  y = y_in;
  //Z = arma::ones(y.n_rows);
  X = X_in;
  XtX = X.t() * X;
  
  Xe = armamat_to_matrixxd(X);
  ye = armamat_to_matrixxd(y);
 
  offsets = arma::zeros(arma::size(y));
  
  //na_mat = arma::zeros<arma::umat>(arma::size(y));
  //na_mat.elem(arma::find_finite(y)).fill(1);
  
  p = X.n_cols;
  // spatial coordinates and dimension
  coords = coords_in;
  dd = coords.n_cols;
  q = y.n_cols;
  
  // Partitioning/DAG
  axis_partition = axis_partition_in;
  
  tausq_inv = tausq_inv_in;
  
  XB = arma::zeros(coords.n_rows, q);
  linear_predictor = arma::zeros(coords.n_rows, q);
  
  Bcoeff = beta_in; 
  for(unsigned int j=0; j<q; j++){
    XB.col(j) = X * Bcoeff.col(j);
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "Beta size: " << arma::size(Bcoeff) << "\n"; 
  }
  
  // prior params
   Vi    = beta_Vi_in;
   bprim = arma::zeros(p);
   Vim   = Vi * bprim;
  
  tausq_ab = tausq_ab_in;
  
  // init
  //predicting = true;
  
  Rcpp::Rcout << "--- " << endl;
  Rcpp::Rcout << arma::size(theta_in) << " " << arma::size(metrop_theta_sd) << " " << arma::size(metrop_theta_bounds) << endl;
  
  // standard level
  mgp.~MGP(); // cleanup
  new (&mgp) MGP(coords, xcoords_in, axis_partition, 
                theta_in, 
                adapting_theta, metrop_theta_sd, metrop_theta_bounds,
                dd, verbose, debug);

  n_blocks = mgp.n_blocks;
  
  // now we know where NAs are, we can erase them
  //y.elem(arma::find_nonfinite(y)).fill(0);
  n = y.n_rows;
  yhat = arma::zeros(n, q);

  //LambdaHw = mgp.w * Lambda.t(); // arma::zeros(warpcx.n_rows, q); 
  //wU = w;
  
  rand_unif = arma::zeros(n_blocks);
  
  // metropolis updates
  // *** add tausq to the mix
  thetastar_mcmc_counter = 0;
  thetastar_unif_bounds = metrop_theta_bounds;
  thetastar_metrop_sd = metrop_theta_sd;
  thetastar_adapt = RAMAdapt(theta_in.n_elem, thetastar_metrop_sd, 0.24);
  thetastar_adapt_active = adapting_theta;
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "Gramar::Gramar initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
}

void Gramar::beta_update(){ 
  Bcoeff = mgp.gibbs_beta_collapsed();
  XB = X * Bcoeff;
}

/*
void expand_grid_with_values_(arma::umat& locs,
                              arma::vec& vals,
                              int rowstart, int rowend,
                              const arma::uvec& x1,
                              const arma::uvec& x2,
                              const arma::mat& values){
  
  for(int i=rowstart; i<rowend; i++){
    arma::uvec ix;
    try {
      ix = arma::ind2sub(arma::size(values), i-rowstart);
    } catch (...) {
      Rcpp::Rcout << arma::size(values) << " " << i-rowstart << " " << i <<" " << rowstart << " " << rowend << endl;
      throw 1;
    }
    locs(0, i) = x1(ix(0));
    locs(1, i) = x2(ix(1));
    vals(i) = values(ix(0), ix(1));
  }
}
*/

void Gramar::metrop_theta_collapsed(bool sample_w){
  tausq_inv = mgp.metrop_theta_collapsed(ye, Xe, y, X, XB, Vi, XtX, tausq_inv, sample_w);
}

void Gramar::mgp_initialize_param(){
  // needs to use this function the first time
  // because get_collapsed_logdens_comps assumes precisionmat and pattern
  // have already been computed
  bool acceptable = mgp.get_mgplogdens_comps(mgp.param_data);
  //Rcpp::Rcout << "alter_data initialize " << endl;
  acceptable = mgp.get_mgplogdens_comps(mgp.alter_data);
}

void Gramar::mgp_post_solver_initialize(){
  mgp.collapsed_logdensity(mgp.param_data, ye, Xe, y, X, XB, Vi, XtX, tausq_inv);
  mgp.collapsed_logdensity(mgp.alter_data, ye, Xe, y, X, XB, Vi, XtX, tausq_inv);
}

