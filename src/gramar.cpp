#include "gramar.h"
using namespace std;


Gramar::Gramar(
  const arma::mat& y_in, 
  const arma::uvec& familyid_in,
  
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
  
  bool use_cache=true,
  
  bool use_ps_in=true,
  
  bool verbose_in=false,
  bool debugging=false,
  int num_threads = 1){
  
  oneuv = arma::ones<arma::uvec>(1);//utils
  hl2pi = -.5 * log(2.0 * M_PI);
  
  verbose = verbose_in;
  debug = debugging;
  
  cached = use_cache;
  
  if(verbose & debug){
    Rcpp::Rcout << "Gramar::Gramar initialization.\n";
  }
  
  // data
  y = y_in;
  Z = arma::ones(y.n_rows);
  X = X_in;
  
  Xe = armamat_to_matrixxd(X);
  yXBe = armamat_to_matrixxd(y);
  
  familyid = familyid_in;
  offsets = arma::zeros(arma::size(y));
  
  na_mat = arma::zeros<arma::umat>(arma::size(y));
  na_mat.elem(arma::find_finite(y)).fill(1);
  
  p = X.n_cols;
  // spatial coordinates and dimension
  coords = coords_in;
  dd = coords.n_cols;
  q = y.n_cols;
  nfact = k_in;
  
  Lambda = lambda_in; 
  Lambda_mask = lambda_mask_in;
  
  // NAs at blocks of outcome variables 
  ix_by_q_a = arma::field<arma::uvec>(q);
  for(unsigned int j=0; j<q; j++){
    ix_by_q_a(j) = arma::find_finite(y.col(j));
    if(verbose){
      Rcpp::Rcout << "Y(" << j+1 << ") : " << ix_by_q_a(j).n_elem << " observed locations.\n";
    }
  }
  
  // Partitioning/DAG
  axis_partition = axis_partition_in;
  
  
  
  if(verbose & debug){
    Rcpp::Rcout << "Lambda size: " << arma::size(Lambda) << "\n";
  }
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
  predicting = true;
  use_ps = use_ps_in;
  
  Rcpp::Rcout << "--- " << endl;
  Rcpp::Rcout << arma::size(theta_in) << " " << arma::size(metrop_theta_sd) << " " << arma::size(metrop_theta_bounds) << endl;
  
  // standard level
  mgp.~MGP(); // cleanup
  new (&mgp) MGP(coords, xcoords_in, axis_partition, na_mat, false, 
                theta_in, 
                adapting_theta, metrop_theta_sd, metrop_theta_bounds,
                use_ps,
                dd, verbose, debug);

  n_blocks = mgp.n_blocks;
  
  // now we know where NAs are, we can erase them
  y.elem(arma::find_nonfinite(y)).fill(0);
  n = y.n_rows;
  yhat = arma::zeros(n, q);

  LambdaHw = mgp.w * Lambda.t(); // arma::zeros(warpcx.n_rows, q); 
  //wU = w;
  
  rand_norm_mat = arma::zeros(coords.n_rows, nfact);
  rand_unif = arma::zeros(n_blocks);
  
  
  if(arma::any(familyid == 3) || arma::any(familyid == 4)){
    init_betareg();
  }
  which_hmc = which_hmc_in;
  init_for_mcmc();
  
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

void Gramar::init_betareg(){
  if(verbose & debug){
    Rcpp::Rcout << "init_betareg \n";
  }
  tausq_unif_bounds = arma::join_horiz(1e-4 * arma::ones(q), 1e4 * arma::ones(q));
  opt_tausq_adapt.reserve(q);
  brtausq_mcmc_counter = arma::zeros(q);
  
  for(unsigned int i=0; i<q; i++){
    //if(familyid(i) == 3){
      
      RAMAdapt brtsq(1, arma::eye(1,1)*.1, .4);
      opt_tausq_adapt.push_back(brtsq);
      
    //}
  }
}


void Gramar::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void Gramar::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

// --- 
void Gramar::init_for_mcmc(){
  if(verbose & debug){
    Rcpp::Rcout << "[init_for_mcmc]\n";
  }
  
  w_do_hmc = arma::any(familyid > 0);
  
  // defaults 
  w_hmc_nuts = false;
  w_hmc_rm = true;
  w_hmc_srm = true;
  
  if(w_do_hmc){
    // let user choose what to use
    if(which_hmc == 1){
      // mala
      if(verbose){
        Rcpp::Rcout << "Using MALA" << endl;
      }
      w_hmc_nuts = false;
      w_hmc_rm = false;
      w_hmc_srm = false;
    }
    if(which_hmc == 2){
      // nuts
      if(verbose){
        Rcpp::stop("Nuts removed");
        Rcpp::Rcout << "Using NUTS" << endl;
      }
      w_hmc_nuts = true;
      w_hmc_rm = false;
      w_hmc_srm = false;
    }
    if(which_hmc == 3){
      // rm-mala
      if(verbose){
        Rcpp::Rcout << "Using simplified manifold MALA" << endl;
      }
      w_hmc_nuts = false;
      w_hmc_rm = true;
      w_hmc_srm = false;
    }
    if(which_hmc == 4){
      // rm-mala then s-mmala
      if(verbose){
        Rcpp::Rcout << "Using simplified-preconditioned MALA" << endl;
      }
      w_hmc_nuts = false;
      w_hmc_rm = true;
      w_hmc_srm = true;
    }
  }
  
  //beta_node.reserve(q); // for beta
  lambda_node.reserve(q); // for lambda
  
  // start with small epsilon for a few iterations,
  // then find reasonable and then start adapting
  
  //beta_hmc_started = arma::zeros<arma::uvec>(q);
  lambda_hmc_started = arma::zeros<arma::uvec>(q);
  
  for(unsigned int j=0; j<q; j++){
    arma::vec yj_obs = y( ix_by_q_a(j), oneuv * j );
    arma::mat X_obs = X.rows(ix_by_q_a(j));
    arma::mat offsets_obs = offsets(ix_by_q_a(j), oneuv * j);
    arma::vec lw_obs = LambdaHw(ix_by_q_a(j), oneuv * j);
    
    arma::vec offsets_for_beta = offsets_obs + lw_obs;
    int family = familyid(j);
    
    // Lambda
    NodeDataB new_lambda_block(yj_obs, offsets_for_beta, X_obs, family);
    lambda_node.push_back(new_lambda_block);
    
    // *** sampling beta and lambda together so we use p+k here
    arma::uvec subcols = arma::find(Lambda_mask.row(j) == 1);
    int n_lambdas = subcols.n_elem;
    AdaptE new_lambda_adapt;
    new_lambda_adapt.init(.05, p+n_lambdas, w_hmc_srm, w_hmc_nuts);
    lambda_hmc_adapt.push_back(new_lambda_adapt);
  }
  
  
  if(w_do_hmc){
    if(verbose & debug){
      Rcpp::Rcout << "[init nongaussian outcome]\n";
    }
    
    w_node.reserve(n_blocks); // for w
    hmc_eps = .025 * arma::ones(n_blocks);
    hmc_eps_started_adapting = arma::zeros<arma::uvec>(n_blocks);
    
    //Rcpp::Rcout << " Initializing HMC for W -- 1" << endl;
    for(unsigned int i=0; i<n_blocks; i++){
      
      NodeDataW new_block;
      w_node.push_back(new_block);
      
      int blocksize = mgp.indexing(i).n_elem * nfact;
      AdaptE new_eps_adapt;
      
      new_eps_adapt.init(hmc_eps(i), blocksize, w_hmc_srm, w_hmc_nuts);
      hmc_eps_adapt.push_back(new_eps_adapt);
    }
    
    
    
    //Rcpp::Rcout << " Initializing HMC for W -- 2" << endl;
    arma::mat offset_for_w = offsets + XB;
    //#pragma omp parallel for
    for(unsigned int i=0; i<n_blocks; i++){
      int u = i;
      
      arma::uvec indexing_target = mgp.indexing(u);
      
      NodeDataW new_block(y, na_mat, //Z.rows(indexing(u)), 
                              offset_for_w,
                              indexing_target,
                              familyid, nfact, false);
      
      new_block.update_mv(offset_for_w, 1.0 / tausq_inv, Lambda);
      
      w_node.at(u) = new_block;
    }
  }
  
}

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


void Gramar::metrop_thetastar(){
  if(verbose & debug){
    Rcpp::Rcout << "[metrop_theta] start\n";
  }
  
  int k = mgp.param_data.theta.n_cols;
  
  thetastar_adapt.count_proposal();
  
  arma::vec param = arma::vectorise(mgp.param_data.theta);
  arma::vec new_param = arma::vectorise(mgp.param_data.theta);
  
  Rcpp::RNGScope scope;
  arma::vec U_update = mrstdnorm(new_param.n_elem, 1);
  
  
  // theta
  new_param = par_huvtransf_back(par_huvtransf_fwd(param, thetastar_unif_bounds) + 
    thetastar_adapt.paramsd * U_update, thetastar_unif_bounds);
  
  //new_param(1) = 1; //***
  
  //bool out_unif_bounds = unif_bounds(new_param, theta_unif_bounds);
  
  arma::mat thetastar_proposal = 
    arma::mat(new_param.memptr(), new_param.n_elem/k, k);
  
  mgp.alter_data.theta = thetastar_proposal;
  
  bool acceptable = mgp.get_mgplogdens_comps(mgp.alter_data );
  
  bool accepted = false;
  double logaccept = 0;
  double current_loglik = 0;
  double new_loglik = 0;
  double prior_logratio = 0;
  double jacobian = 0;
  
  if(acceptable){
    new_loglik = mgp.alter_data.loglik_w;
    
    current_loglik = mgp.param_data.loglik_w;
    
    prior_logratio = 0;
    if(mgp.param_data.theta.n_rows > 5){
      for(int i=0; i<mgp.param_data.theta.n_rows-2; i++){
        prior_logratio += arma::accu( -mgp.alter_data.theta.row(i) +mgp.param_data.theta.row(i) ); // exp
      }
    }
    
    jacobian  = calc_jacobian(new_param, param, thetastar_unif_bounds);
    logaccept = new_loglik - current_loglik + 
      prior_logratio +
      jacobian;
    
    
    accepted = do_I_accept(logaccept);
    
  } else {
    accepted = false;
    //num_chol_fails ++;
    if(verbose & debug){
      Rcpp::Rcout << "[warning] numerical failure at MH proposal -- auto rejected\n";
    }
  }
  
  if(accepted){
    thetastar_adapt.count_accepted();
    
    std::swap(mgp.param_data, mgp.alter_data);
    mgp.param_data.theta = thetastar_proposal;
    
    if(debug & verbose){
      Rcpp::Rcout << "[theta] accepted (log accept. " << logaccept << " : " << new_loglik << " " << current_loglik << 
        " " << prior_logratio << " " << jacobian << ")\n";
    }
  } else {
    if(debug & verbose){
      Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << " : " << new_loglik << " " << current_loglik << 
        " " << prior_logratio << " " << jacobian << ")\n";
    }
  }
  
  thetastar_adapt.update_ratios();
  
  if(thetastar_adapt_active){
    thetastar_adapt.adapt(U_update, acceptable*exp(logaccept), thetastar_mcmc_counter); 
  }
  thetastar_mcmc_counter++;
  if(verbose & debug){
    Rcpp::Rcout << "[metrop_theta] end\n";
  }
}

void Gramar::metrop_theta_collapsed(){
  
  arma::mat yXB = y - XB;
  yXBe = armamat_to_matrixxd(yXB);
  
  mgp.metrop_theta_collapsed(yXBe, Xe, tausq_inv);
}

