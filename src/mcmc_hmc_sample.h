#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

//#include "../distributions/mvnormal.h"

//#include "distparams.h"
#include "mcmc_hmc_nodes.h"
#include "mcmc_hmc_adapt.h"

// mala and rm-mala
template <class T>
inline arma::mat mala_cpp(arma::mat current_q, 
                          int u, MGP& mgp, 
                                   T& postparams,
                                   AdaptE& adaptparams, 
                                   const arma::mat& rnorm_mat,
                                   const double& runifvar,
                                   bool adapt=true,
                                   bool debug=false){
  
  int k = current_q.n_cols;
  // currents
  arma::vec xgrad;
  double joint0, eps1, eps2;
  
  // data
  xgrad = postparams.compute_dens_and_grad(joint0, current_q);
  // prior
  double logprior=0;
  
  arma::mat prior_neghess;
  if(u != -1){
    prior_neghess = mgp.block_fullconditional_prior_ci(u, mgp.param_data);
  } else {
    // betalambda calculate prior covariance from inside other function
    prior_neghess = arma::zeros(current_q.n_elem, current_q.n_elem);
  }
  
  arma::mat prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, current_q, u, mgp.param_data);
  
  joint0 += logprior;
  xgrad += prior_grad;


  eps2 = pow(adaptparams.eps, 2.0);
  eps1 = adaptparams.eps;

  
  if(xgrad.has_nan() || xgrad.has_inf() || std::isnan(joint0) || std::isinf(joint0)){
    adaptparams.alpha = 0.0;
    adaptparams.n_alpha = 1.0;
    adaptparams.adapt_step();
    return current_q;
  }
  
  arma::vec veccurq = arma::vectorise(current_q);
  arma::vec proposal_mean = veccurq + eps2 * 0.5 * xgrad;// / max(eps2, arma::norm(xgrad));
  
  // proposal value
  arma::vec p = arma::vectorise(rnorm_mat); 
  arma::vec q = proposal_mean + eps1 * p;
  arma::mat qmat = arma::mat(q.memptr(), q.n_elem/k, k);
  
  // proposal
  double joint1; // = postparams.logfullcondit(qmat);
  arma::vec revgrad;
  
  revgrad = postparams.compute_dens_and_grad(joint1, qmat);
  
  logprior=0;
  prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, qmat, u, mgp.param_data);
  
  joint1 += logprior;
  revgrad += prior_grad;
  
  
  
  if(revgrad.has_inf() || std::isnan(joint1) || std::isinf(joint1)){
    adaptparams.alpha = 0.0;
    adaptparams.n_alpha = 1.0;
    adaptparams.adapt_step();
    return current_q;
  }
  
  arma::vec reverse_mean = q + eps2 * 0.5 * revgrad; 
  double prop0to1 = -.5/(eps1*eps1) * arma::conv_to<double>::from(
    (q - proposal_mean).t() * (q - proposal_mean) );
  double prop1to0 = -.5/(eps1*eps1) * arma::conv_to<double>::from(
    (veccurq - reverse_mean).t() * (veccurq - reverse_mean) );
  
  adaptparams.alpha = std::min(1.0, exp(joint1 + prop1to0 - joint0 - prop0to1));
  adaptparams.n_alpha = 1.0;
  
  if(runifvar < adaptparams.alpha){ 
    current_q = qmat;
  } 
  
  adaptparams.adapt_step();
  return current_q;
}

template <class T>
inline arma::mat manifmala_cpp(arma::mat current_q, 
                               int u, MGP& mgp,
                                      T& postparams,
                                      AdaptE& adaptparams, 
                                      const arma::mat& rnorm_mat,
                                      const double& runifvar, const double& runifadapt, 
                                      bool adapt=true,
                                      bool debug=false){
  
  // with infinite adaptation
  
  int k = current_q.n_cols;
  // currents
  arma::vec xgrad;
  double joint0, eps1, eps2;
  arma::mat H_forward;
  arma::mat MM, Minvchol;//, Minv;
  
  bool chol_error = false;
  bool rev_chol_error = false;
  
  arma::mat prior_neghess;
  if(u != -1){
    prior_neghess = mgp.block_fullconditional_prior_ci(u, mgp.param_data);
  } else {
    // betalambda calculate prior covariance from inside other function
    prior_neghess = arma::zeros(current_q.n_elem, current_q.n_elem);
  }
  
  bool fixed_C_this_step = adaptparams.use_C_const(runifadapt);
  
  if(fixed_C_this_step) {
    // not adapting at this time
    
    xgrad = postparams.compute_dens_and_grad(joint0, current_q);
    
    double logprior=0;
    arma::mat prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, current_q, u, mgp.param_data);

    joint0 += logprior;
    xgrad += prior_grad;
    
    MM = adaptparams.C_const;
    Minvchol = adaptparams.Ccholinv_const;
  } else {
    // adapting at this time; 
    
    MM = postparams.compute_dens_grad_neghess(joint0, xgrad, current_q);
    
    double logprior=0;
    
    arma::mat prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, current_q, u, mgp.param_data);
    
    joint0 += logprior;
    xgrad += prior_grad;
    MM += prior_neghess;
    
    adaptparams.weight_average_C_temp(MM);
    try {
      Minvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(MM), "lower")));
    } catch (...) {
      Minvchol = arma::eye(current_q.n_elem, current_q.n_elem);
      chol_error = true;
    }
  }
  
  
  //if(adapt){
    eps2 = pow(adaptparams.eps, 2.0);
    eps1 = adaptparams.eps;
  //} 
  
  if(MM.has_nan() || xgrad.has_nan() || xgrad.has_inf() || std::isnan(joint0) || std::isinf(joint0)){
    adaptparams.alpha = 0.0;
    adaptparams.n_alpha = 1.0;
    adaptparams.adapt_step();
    return current_q;
  }
  
  arma::vec veccurq = arma::vectorise(current_q);
  arma::vec proposal_mean = veccurq + eps2 * 0.5 * Minvchol.t() * Minvchol * xgrad;// / max(eps2, arma::norm(xgrad));
  
  // proposal value
  arma::vec p = arma::vectorise(rnorm_mat); 
  arma::vec q = proposal_mean + eps1 * Minvchol.t() * p;
  arma::mat qmat = arma::mat(q.memptr(), q.n_elem/k, k);
  
  // proposal
  double joint1; // = postparams.logfullcondit(qmat);
  arma::vec revgrad;
  arma::mat H_reverse;
  arma::mat RR, Rinvchol, Rinv;
  
  if(fixed_C_this_step) {
    // after burn period keep constant variance
    revgrad = postparams.compute_dens_and_grad(joint1, qmat);
    
    double logprior=0;
    
    arma::mat prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, qmat, u, mgp.param_data);
    
    joint1 += logprior;
    revgrad += prior_grad;
    
    RR = adaptparams.C_const;
    Rinvchol = adaptparams.Ccholinv_const;
    //Rinv = Rinvchol.t() * Rinvchol;
  } else {
    // initial burn period use full riemann manifold
    RR = postparams.compute_dens_grad_neghess(joint1, revgrad, qmat);
    
    
    double logprior=0;
    
    arma::mat prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, qmat, u, mgp.param_data);
    
    joint1 += logprior;
    revgrad += prior_grad;
    RR += prior_neghess;
    
    adaptparams.weight_average_C_temp(RR);
    if(!chol_error){
      try {
        Rinvchol = arma::inv(arma::trimatl(arma::chol(arma::symmatu(RR), "lower")));
      } catch (...) {
        rev_chol_error = true;
      }
      
    } else {
      Rinvchol = arma::eye(current_q.n_elem, current_q.n_elem);
    }
  }
  
  if(rev_chol_error || revgrad.has_inf() || std::isnan(joint1) || std::isinf(joint1)){
    adaptparams.alpha = 0.0;
    adaptparams.n_alpha = 1.0;
    adaptparams.adapt_step();
    return current_q;
  }
  
  double Richoldet = arma::accu(log(Rinvchol.diag()));
  double Micholdet = arma::accu(log(Minvchol.diag()));
  arma::vec reverse_mean = q + eps2 * 0.5 * Rinvchol.t() * Rinvchol * revgrad; 
  double prop0to1 = Micholdet -.5/(eps1*eps1) * arma::conv_to<double>::from(
    (q - proposal_mean).t() * MM * (q - proposal_mean) );
  double prop1to0 = Richoldet -.5/(eps1*eps1) * arma::conv_to<double>::from(
    (veccurq - reverse_mean).t() * RR * (veccurq - reverse_mean) );
  
  adaptparams.alpha = std::min(1.0, exp(joint1 + prop1to0 - joint0 - prop0to1));
  adaptparams.n_alpha = 1.0;
  
  if(runifvar < adaptparams.alpha){ 
    current_q = qmat;
    if(!fixed_C_this_step){
      // accepted: send accepted covariance to adaptation
      adaptparams.update_C_const(RR, Rinvchol); 
    }
  } else {
    if(!fixed_C_this_step){
      // rejected: send original covariance to adaptation
      adaptparams.update_C_const(MM, Minvchol);
    }
  }
  
  adaptparams.adapt_step();
  
  
  return current_q;
}


inline arma::mat unvec(arma::vec q, int k){
  arma::mat result = arma::mat(q.memptr(), q.n_elem/k, k);
  return result;
}

// Position q and momentum p
struct pq_point {
  arma::vec q;
  arma::vec p;
  
  explicit pq_point(int n): q(n), p(n) {}
  pq_point(const pq_point& z): q(z.q.size()), p(z.p.size()) {
    q = z.q;
    p = z.p;
  }
  
  pq_point& operator= (const pq_point& z) {
    if (this == &z)
      return *this;
    
    q = z.q;
    p = z.p;
    
    return *this;
  }
};

template <class T>
inline void leapfrog(int u, MGP& mgp, pq_point &z, float epsilon, T& postparams,  int k=1){
  double logprior=0;
  arma::mat qmat = unvec(z.q, k);
  //arma::mat ehalfMi = epsilon * 0.5 * Minv;
  z.p += epsilon * 0.5  * (postparams.gradient_logfullcondit(qmat) + 
    mgp.block_grad_fullconditional_prior_m(logprior, qmat, u, mgp.param_data));
  //arma::vec qvecplus = arma::vectorise(z.q) + epsilon * z.p;
  z.q += epsilon * z.p;
  qmat = unvec(z.q, k);
  z.p += epsilon * 0.5  * (postparams.gradient_logfullcondit(qmat) +
    mgp.block_grad_fullconditional_prior_m(logprior, qmat, u, mgp.param_data));
}

template <class T>
inline double find_reasonable_stepsize(int u, MGP& mgp, const arma::mat& current_q, T& postparams, const arma::mat& rnorm_mat){
  int K = current_q.n_elem;
  
  arma::mat prior_neghess;
  
  if(u != -1){
    prior_neghess = mgp.block_fullconditional_prior_ci(u, mgp.param_data);
  } else {
    prior_neghess = arma::zeros(current_q.n_elem, current_q.n_elem);
  }
  
  pq_point z(K);
  arma::vec p0 = //Minvchol.t() *
    arma::vectorise(rnorm_mat); //arma::randn(K);
  
  double epsilon = 1;
  double logprior=0;
  arma::mat prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, current_q, u, mgp.param_data);
  
  arma::vec veccurq = arma::vectorise(current_q);
  
  z.q = veccurq;
  z.p = p0;
  
  double p_orig = (logprior + postparams.logfullcondit(current_q)) - 
    0.5* arma::conv_to<double>::from(z.p.t() * z.p);
    
  leapfrog(u, mgp, z, epsilon, postparams, current_q.n_cols);

  //Rcpp::Rcout << "done leapfrog " << endl;
  arma::mat newq = unvec(z.q, current_q.n_cols);
  
  prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, newq, u, mgp.param_data);
  double p_prop = (logprior + postparams.logfullcondit(newq)) - 0.5* arma::conv_to<double>::from(z.p.t() * z.p);//sum(z.p % z.p); 
  //Rcpp::Rcout << "after:  " << p_prop << "\n";
  
  double p_ratio = exp(p_prop - p_orig);
  double a = 2 * (p_ratio > .5) - 1;
  int it=0;
  bool condition = (pow(p_ratio, a) > pow(2.0, -a)) || std::isnan(p_ratio);
  
  while( condition & (it < 50) ){
    it ++;
    double twopowera = pow(2.0, a);
    epsilon = twopowera * epsilon;
    
    leapfrog(u, mgp, z, epsilon, postparams, current_q.n_cols);
    newq = unvec(z.q, current_q.n_cols);
    
    prior_grad = mgp.block_grad_fullconditional_prior_m(logprior, newq, u, mgp.param_data);
    p_prop = (logprior + postparams.logfullcondit(newq)) - 0.5* arma::conv_to<double>::from(z.p.t() * //MM * 
      z.p);//sum(z.p % z.p); 
    p_ratio = exp(p_prop - p_orig);
    
    condition = (pow(p_ratio, a)*twopowera > 1.0) || std::isnan(p_ratio);
    
    // reset
    z.q = veccurq;
    z.p = p0;
  }
  if(it == 50){
    epsilon = .01;
    
  }
  return epsilon/2.0;
} 

