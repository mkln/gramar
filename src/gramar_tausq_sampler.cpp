#include "gramar.h"

using namespace std;

void Gramar::deal_with_tausq(){
    gibbs_sample_tausq_std();
}


void Gramar::gibbs_sample_tausq_std(){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_tausq_std] start\n";
  }
  
  start = std::chrono::steady_clock::now();
  // note that at the available locations w already includes Lambda 
  
  double aprior = tausq_ab(0);
  double bprior = tausq_ab(1);
  
  //arma::mat LHW = w * Lambda.t();
  
  logpost = 0;
  for(unsigned int j=0; j<q; j++){
    if(familyid(j) == 0){
      // gibbs update
      arma::mat yrr = 
        y.submat(ix_by_q_a(j), oneuv*j) - 
        XB.submat(ix_by_q_a(j), oneuv*j) - 
        LambdaHw.submat(ix_by_q_a(j), oneuv*j); //***
      
      double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
      
      double aparam = aprior + ix_by_q_a(j).n_elem/2.0;
      double bparam = 1.0/( bprior + .5 * bcore );
      
      Rcpp::RNGScope scope;
      tausq_inv(j) = R::rgamma(aparam, bparam);
      logpost += 0.5 * (ix_by_q_a(j).n_elem + .0) * log(tausq_inv(j)) - 0.5*tausq_inv(j)*bcore;
      
      if(verbose & debug){
        Rcpp::Rcout << "[gibbs_sample_tausq] " << j << " | "
                    << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv(j)
                    << "\n";
      }
    }
  }
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_tausq] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us.\n";
  }
  
}
