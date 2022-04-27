#define ARMA_DONT_PRINT_ERRORS

#include "gramar.h"
#include "utils_interrupt_handler.h"
#include "utils_parametrize.h"

using namespace std;

//[[Rcpp::export]]
Rcpp::List mgp_precision(const arma::mat& coords, 
                         const arma::mat& xcoords_in,
                         
                         const arma::field<arma::vec>& axis_partition,
                         
                         const arma::mat& theta,
                         const arma::vec& tausq,
                         
                         int num_threads = 1,
                         bool use_cache=true,
                         
                         bool verbose=false,
                         bool debug=false){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  arma::mat metrop_theta_sd_btm = arma::eye(3,3);
  arma::mat metrop_theta_bounds = arma::zeros(1);
  
  arma::umat na_mat = arma::ones<arma::umat>(coords.n_rows, 1);
  
  MGP mgp(coords, xcoords_in, 
          axis_partition, 
          theta, false, metrop_theta_sd_btm,
          metrop_theta_bounds, 2, verbose, debug);
  
  bool acceptable = mgp.get_mgplogdens_comps(mgp.param_data );
  
  
  Rcpp::Rcout << "--- product: " << endl;
  mgp.new_precision_matrix_product(mgp.param_data);
  Eigen::SparseMatrix<double> Ci_prd = mgp.param_data.Citsqi;

  return Rcpp::List::create(
    Rcpp::Named("H") = mgp.H,
    Rcpp::Named("Ri") = mgp.Ri,
    Rcpp::Named("Ci_prd") = Ci_prd,
    Rcpp::Named("linear_sort_map") = mgp.linear_sort_map,
    Rcpp::Named("membership") = mgp.membership
  );
}






//[[Rcpp::export]]
Rcpp::List gramar_mcmc_collapsed(
    const arma::mat& y, 
    const arma::mat& X, 
    
    const arma::mat& coords, 
    
    const arma::field<arma::vec>& axis_partition,
    const arma::mat& set_unif_bounds_in,
    const arma::mat& beta_Vi,
    
    const arma::vec& sigmasq_ab,
    const arma::vec& tausq_ab,
    
    const arma::mat& theta,
    const arma::mat& beta,
    const arma::vec& tausq,
    
    const arma::mat& mcmcsd,
    
    int mcmc_keep = 100,
    int mcmc_burn = 100,
    int mcmc_thin = 1,
    
    int mcmc_startfrom = 0,
    
    int num_threads = 1,
    
    bool adapting=false,
    
    bool verbose=false,
    bool debug=false,
    int print_every=false,
    
    bool sample_beta=true,
    bool sample_theta=true){
  
  if(verbose & debug){
    Rcpp::Rcout << "Initializing.\n";
  }
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  // timers
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  // ------
  
  bool printall = print_every == 1;
  bool verbose_mcmc = printall;
  
  //unsigned int n = coords.n_rows;
  unsigned int d = coords.n_cols;
  unsigned int q  = y.n_cols;
  
  if(verbose & debug){
    Rcpp::Rcout << "Limits to MCMC search for theta:\n";
    Rcpp::Rcout << set_unif_bounds_in << endl;
  }
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;

  arma::mat start_theta = theta;
  if(verbose & debug){
    Rcpp::Rcout << "start theta \n" << theta;
  }
  
  Gramar msp(y, 
             X, coords, X, axis_partition, 
             
             beta,
             start_theta, 
             1.0/tausq, 
             beta_Vi, tausq_ab,
             
             adapting,
             mcmcsd,
             set_unif_bounds_in,
             
             verbose, debug, num_threads);
  
  int par_size = msp.mgp.param_data.theta.n_elem;
  
  
  arma::mat b_mcmc = arma::zeros(X.n_cols, mcmc_thin*mcmc_keep);
  arma::vec tausq_mcmc = arma::zeros(mcmc_thin*mcmc_keep);
  arma::mat theta_mcmc = arma::zeros(par_size, mcmc_thin*mcmc_keep);
  arma::mat w_mcmc = arma::zeros(X.n_rows, mcmc_keep);
  arma::vec llsave = arma::zeros(mcmc_thin*mcmc_keep);
  
  bool acceptable = false;

  msp.mgp_initialize_param(); // gets mgp components
  msp.mgp.solver_initialize(); // analyzePattern and precision matrix
  msp.mgp_post_solver_initialize(); // first density calc for param and alter


  bool interrupted = false;
  
  if(verbose){
    Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations.\n\n";
  }
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; 
  int mcmc_saved = 0; 
  bool sample_w = true;

  try {
    
    for(m=0; (m<mcmc) & (!interrupted); m++){
      
      mx = m-mcmc_burn;
      msp.predicting = false;
      
      if(printall){
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      if(sample_theta){
        start = std::chrono::steady_clock::now();
        msp.metrop_theta_collapsed(sample_w);
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & verbose){
          Rcpp::Rcout << "[theta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us.\n";
        }
      }
      if(sample_beta){
        msp.beta_update();
      }

      if(mx >= 0){
        tausq_mcmc(mx) = 1.0 / msp.tausq_inv(0);
        b_mcmc.col(mx) = msp.Bcoeff;
        theta_mcmc.col(mx) = msp.mgp.param_data.theta;
        llsave(mx) = msp.mgp.param_data.collapsed_ldens;
        
        mx++;
        if(mx % mcmc_thin == 0){
          w_mcmc.col(mcmc_saved) = msp.mgp.w;
          mcmc_saved++;
        }
      }
       
      interrupted = checkInterrupt();
      if(interrupted){
        Rcpp::stop("Interrupted by the user.");
      }
      
      if((m>0) & (mcmc > 100)){
        
        bool print_condition = (print_every>0);
        if(print_condition){
          print_condition = print_condition & (!(m % print_every));
        };
        
        if(print_condition){
          end_mcmc = std::chrono::steady_clock::now();
          
          int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
          int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
          msp.mgp.theta_adapt.print_summary(time_tick, time_mcmc, m, mcmc);
          
          tick_mcmc = std::chrono::steady_clock::now();
          unsigned int printlimit = 10;
          
          msp.mgp.theta_adapt.print_acceptance();
          Rprintf("  theta = ");
          unsigned int n_theta = msp.mgp.param_data.theta.n_elem;
          unsigned int n_print_theta = min(printlimit, n_theta);
          for(unsigned int pp=0; pp<n_print_theta; pp++){
            Rprintf("%.3f ", msp.mgp.param_data.theta(pp));
          }
          
          Rprintf("\n  tausq = ");
          unsigned int n_print_tsq = min(printlimit, q);
          for(unsigned int pp=0; pp<n_print_tsq; pp++){
            Rprintf("(%d) %.6f ", pp+1, 1.0/msp.tausq_inv(pp));
          }
          
          Rprintf("\n\n");
        } 
      } else {
        tick_mcmc = std::chrono::steady_clock::now();
      }
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    if(print_every>0){
      Rcpp::Rcout << "MCMC done [" << mcmc_time/1000.0 <<  "s]\n";
    }
    
    return Rcpp::List::create(
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("paramsd") = msp.mgp.theta_adapt.paramsd,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("indexing") = msp.mgp.indexing,
      //Rcpp::Named("Ci") = msp.mgp.param_data.Cidebug,
      Rcpp::Named("Citsqi") = msp.mgp.param_data.Citsqi,
      Rcpp::Named("mcmc") = mcmc,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("marglik") = llsave,
      Rcpp::Named("success") = true
    );
    
  } catch(const std::exception& e) {
    Rcpp::Rcout << "Caught exception \"" << e.what() << "\"\n";
    
    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::warning("MCMC has been interrupted. Returning partial saved results if any.\n");
    
    return Rcpp::List::create(
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("paramsd") = msp.mgp.theta_adapt.paramsd,
      Rcpp::Named("mcmc") = mcmc,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("success") = false
    );
  }
}

//[[Rcpp::export]]
Rcpp::List gramar_wpredict(
  const arma::mat& Xin,
  const arma::mat& coordsin,
  const arma::field<arma::uvec>& indexingin,
  
  const arma::mat& Xout,
  const arma::mat& coordsout,
  
  const arma::field<arma::vec>& axis_partition,
  
  const arma::mat& w_mcmc,
  const arma::mat& theta_mcmc,
  
  bool verbose=false,
  bool debug=false){
  
  Rcpp::Rcout << "init for prediction " << endl;
  
  int n_par = theta_mcmc.n_rows + 1;
  int dd = coordsout.n_cols;
  arma::mat metrop_theta_sd = arma::eye(n_par+1, n_par+1);
  arma::mat metrop_theta_bounds = arma::zeros(n_par+1, 2);
  metrop_theta_bounds.col(1) += 1;
  
  int mcmc = w_mcmc.n_cols;
  arma::mat w_predicted = arma::zeros(Xout.n_rows, mcmc);
  
  Rcpp::Rcout << "Create MGP obj " << endl;
  MGP out_mgp(coordsout, Xout, axis_partition, 
              theta_mcmc.col(0), false, metrop_theta_sd, metrop_theta_bounds,
              dd, verbose, debug);
  
  out_mgp.refresh_cache(out_mgp.param_data);
  out_mgp.update_all_block_covpars(out_mgp.param_data);
  out_mgp.new_precision_matrix_direct(out_mgp.param_data);
  
  for(int i=0; i<0; i++){
    arma::vec w = w_mcmc.col(i);
    arma::vec theta = theta_mcmc.col(i);
    
    out_mgp.param_data.theta = theta;
    out_mgp.refresh_cache(out_mgp.param_data);
    out_mgp.update_all_block_covpars(out_mgp.param_data);
    out_mgp.update_precision_matrix(out_mgp.param_data);
    
    Rcpp::Rcout << "save and return" << endl;
    w_predicted.col(i) = out_mgp.w;
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("w_predicted") = w_predicted,
    Rcpp::Named("indexing") = out_mgp.indexing
  );
}


//[[Rcpp::export]]
Rcpp::List gramar_wpredict_via_prec(
    const arma::mat& Xin,
    const arma::mat& coordsin,
    const arma::field<arma::uvec>& indexingin,
    
    const arma::mat& Xout,
    const arma::mat& coordsout,
    
    const arma::field<arma::vec>& axis_partition,
    
    const arma::mat& w_mcmc,
    const arma::mat& theta_mcmc,
    
    bool verbose=false,
    bool debug=false){
  
  Rcpp::Rcout << "init for prediction " << endl;
  
  int n_par = theta_mcmc.n_rows + 1;
  int dd = coordsout.n_cols;
  arma::mat metrop_theta_sd = arma::eye(n_par+1, n_par+1);
  arma::mat metrop_theta_bounds = arma::zeros(n_par+1, 2);
  metrop_theta_bounds.col(1) += 1;
  
  int mcmc = w_mcmc.n_cols;
  arma::mat w_predicted = arma::zeros(Xout.n_rows, mcmc);
  
  Rcpp::Rcout << "Create MGP obj " << endl;
  MGP in_mgp(coordsin, Xin, axis_partition, 
              theta_mcmc.col(0), false, metrop_theta_sd, metrop_theta_bounds,
              dd, verbose, debug);
  
  MGP out_mgp(coordsout, Xout, axis_partition, 
             theta_mcmc.col(0), false, metrop_theta_sd, metrop_theta_bounds,
             dd, verbose, debug);
  
  in_mgp.refresh_cache(in_mgp.param_data);
  in_mgp.update_all_block_covpars(in_mgp.param_data);

  /*
  in_mgp.w = w_mcmc.col(0);
  arma::vec theta = theta_mcmc.col(0);
  in_mgp.param_data.theta = theta;
  w_predicted.col(0) = in_mgp.predict_via_precision(out_mgp, theta);
  */
  for(int i=0; i<mcmc; i++){
    in_mgp.w = w_mcmc.col(i);
    arma::vec theta = theta_mcmc.col(i);
    
    in_mgp.param_data.theta = theta;
    in_mgp.refresh_cache(in_mgp.param_data);
    in_mgp.update_all_block_covpars(in_mgp.param_data);
  
    w_predicted.col(i) = in_mgp.predict_via_precision(out_mgp, theta);
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("w_predicted") = w_predicted,
    Rcpp::Named("indexing") = in_mgp.indexing,
    Rcpp::Named("H") = in_mgp.H,
    Rcpp::Named("Ri") = in_mgp.Ri,
    Rcpp::Named("PP_all") = in_mgp.PP_all,
    Rcpp::Named("PP_o") = in_mgp.PP_o,
    Rcpp::Named("PP_ox") = in_mgp.PP_ox
  );
}




