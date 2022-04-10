#include "gramar.h"

using namespace std;


void Gramar::deal_with_w(MGP& mgp){
  
  // ensure this is independent on the number of threads being used
  Rcpp::RNGScope scope;
  rand_norm_mat = mrstdnorm(mgp.coords.n_rows, mgp.k);
  rand_unif = vrunif(mgp.n_blocks);
  rand_unif2 = vrunif(mgp.n_blocks);
  
  if(w_do_hmc){
    nongaussian_w(mgp);
  } else {
    gaussian_w(mgp);
  }
}



void Gramar::gaussian_w(MGP& mgp){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w] " << "\n";
  }
  
  start_overall = std::chrono::steady_clock::now();
  for(unsigned int i=0; i<mgp.n_blocks; i++){
    int u = i;
    
    if((mgp.block_ct_obs(u) > 0)){
      
      mgp.param_data.Sigi(u) = mgp.block_fullconditional_prior_ci(u, mgp.param_data);
      mgp.param_data.Smu(u) = mgp.conj_block_fullconditional_m(u, mgp.param_data);
      mgp.block_fullconditional_regdata(u, y, XB, na_mat, Lambda, tausq_inv, mgp.param_data);
      arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( mgp.param_data.Sigi(u) ), "lower")));
      // sample
      
      arma::vec wmean = Sigi_chol.t() * Sigi_chol * mgp.param_data.Smu(u);
      arma::vec wtemp = wmean;
      
      arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(mgp.indexing(u)));
      wtemp += Sigi_chol.t() * rnvec;
      
      mgp.w.rows(mgp.indexing(u)) = 
        arma::mat(wtemp.memptr(), wtemp.n_elem/mgp.k, mgp.k); 
    } 
  }
  
  LambdaHw = mgp.w * Lambda.t();
  
  if(false || verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}


void Gramar::nongaussian_w(MGP& mgp){
  if(verbose & debug){
    Rcpp::Rcout << "[hmc_sample_w] " << endl;
  }
  
  start_overall = std::chrono::steady_clock::now();

  
  int mala_timer = 0;
  
  arma::mat offset_for_w = offsets + XB;
  
  for(unsigned int i=0; i<n_blocks; i++){
    int u = i;
    
    if((mgp.block_ct_obs(u) > 0)){
      
      start = std::chrono::steady_clock::now();
      
      //Rcpp::Rcout << "u :  " << u << endl;
      w_node.at(u).update_mv(offset_for_w, 1.0/tausq_inv, Lambda);
      arma::mat w_current = mgp.w.rows(mgp.indexing(u));
      
      // adapting eps
      hmc_eps_adapt.at(u).step();
      
      //hmc_eps_started_adapting(u) == 1;/*
      if((hmc_eps_started_adapting(u) == 0) & (hmc_eps_adapt.at(u).i==10)){
        // wait a few iterations before starting adaptation
        //message("find reasonable");
        
        hmc_eps(u) = find_reasonable_stepsize(u, mgp, 
                w_current, w_node.at(u), rand_norm_mat.rows(mgp.indexing(u)));
        //message("found reasonable");
        int blocksize = mgp.indexing(u).n_elem * mgp.k;
        AdaptE new_adapting_scheme;
        new_adapting_scheme.init(hmc_eps(u), blocksize, w_hmc_srm, w_hmc_nuts, 1e4);
        
        hmc_eps_adapt.at(u) = new_adapting_scheme;
        hmc_eps_started_adapting(u) = 1;
      }
      
      
      bool do_gibbs = arma::all(familyid == 0);
      arma::mat w_temp = w_current;
      
      if(which_hmc == 1){
        // mala
        w_temp = mala_cpp(w_current, 
                          u, mgp,
                          w_node.at(u), hmc_eps_adapt.at(u),
                          rand_norm_mat.rows(mgp.indexing(u)),
                          rand_unif(u), true, debug);
      }
      if((which_hmc == 3) || (which_hmc == 4)){
        // some form of manifold mala
        
        w_temp = manifmala_cpp(w_current, 
                               u, mgp,
                               w_node.at(u), hmc_eps_adapt.at(u),
                               rand_norm_mat.rows(mgp.indexing(u)),
                               rand_unif(u), rand_unif2(u),
                               true, debug);
        
      }
      
      
      end = std::chrono::steady_clock::now();
      
      hmc_eps(u) = hmc_eps_adapt.at(u).eps;
      
      mgp.w.rows(mgp.indexing(u)) = w_temp;//arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
      
      
      mala_timer += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      
    }
    
  }
  
  LambdaHw = mgp.w * Lambda.t();
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[hmc_sample_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
    //Rcpp::Rcout << "mala: " << mala_timer << endl;
  }
  
}


/*
void Gramar::predict(bool sample){
  start_overall = std::chrono::steady_clock::now();
  if(predict_group_exists == 1){
    if(verbose & debug){
      Rcpp::Rcout << "[predict] start \n";
    }
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int i=0; i<u_predicts.n_elem; i++){ //*** subset to blocks with NA
      int u = u_predicts(i);// u_predicts(i);
      // only predictions at this block. 
      arma::uvec predict_parent_indexing, cx;
      arma::cube Kxxi_parents;
      

        // no observed locations, use line of sight
        predict_parent_indexing = parents_indexing(u);
        CviaKron_HRj_chol_bdiag(Hpred(i), Rcholpred(i), Kxxi_parents,
                                na_1_blocks(u),
                                warpcx, indexing(u), predict_parent_indexing, param_data.k, param_data.theta, matern);
      
      
      //Rcpp::Rcout << "step 1 "<< endl;
      arma::mat wpars = w.rows(predict_parent_indexing);
      
      for(unsigned int ix=0; ix<indexing(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 0){
          arma::rowvec wtemp = arma::sum(arma::trans(Hpred(i).slice(ix)) % wpars, 0);
          
          wtemp += arma::trans(Rcholpred(i).col(ix)) % rand_norm_mat.row(indexing(u)(ix));
          
          
          w.row(indexing(u)(ix)) = wtemp;
          
          
          LambdaHw.row(indexing(u)(ix)) = w.row(indexing(u)(ix)) * Lambda.t();
        }
      }
      
      //Rcpp::Rcout << "done" << endl;
      
      
    }
    
    if(verbose & debug){
      end_overall = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[predict] "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                  << "us. ";
    }
  }
}*/

void Gramar::predicty(bool sample){  
  int n = XB.n_rows;
  yhat.fill(0);
  Rcpp::RNGScope scope;
  arma::mat Lw = mgp.w*Lambda.t();
  for(unsigned int j=0; j<q; j++){
    linear_predictor.col(j) = XB.col(j) + Lw.col(j);
    if(familyid(j) == 0){
      // gaussian
      yhat.col(j) = linear_predictor.col(j) + pow(1.0/tausq_inv(j), .5) * mrstdnorm(n, 1);
    } else if(familyid(j) == 1){
      // poisson
      yhat.col(j) = vrpois(exp(linear_predictor.col(j)));
    } else if(familyid(j) == 2){
      // binomial
      yhat.col(j) = vrbern(1.0/(1.0 + exp(-linear_predictor.col(j))));
    } else if(familyid(j) == 3){
      arma::vec mu =  1.0/ (1.0 + exp(-linear_predictor.col(j)));
      arma::vec aa = tausq_inv(j) * mu;
      arma::vec bb = tausq_inv(j) * (1.0-mu);
      yhat.col(j) = vrbeta(aa, bb);
    } else if(familyid(j) == 4){
      arma::vec muvec = exp(linear_predictor.col(j));
      double alpha = 1.0/tausq_inv(j); 
      yhat.col(j) = vrnb(muvec, alpha);
    }
  }
}


