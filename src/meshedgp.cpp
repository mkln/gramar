#ifndef MGP_UTILS 
#define MGP_UTILS


#include "meshedgp.h"

using namespace std;

//[[Rcpp::export]]
arma::uvec uturbocolthreshold(const arma::vec& col1, const arma::vec& thresholds){
  arma::uvec result = arma::zeros<arma::uvec>(col1.n_elem);
  for(unsigned int i=0; i<col1.n_elem; i++){
    unsigned int overthreshold = 1;
    for(unsigned int j=0; j<thresholds.n_elem; j++){
      if(col1(i) >= thresholds(j)){
        overthreshold += 1;
      } else {
        break;
      }
    }
    result(i) = overthreshold;
  }
  return result;
}

//[[Rcpp::export]]
arma::field<arma::uvec> split_ap(arma::uvec& membership, const arma::mat& coords, 
                                 const arma::field<arma::vec>& thresholds, 
                                 int offset){
  arma::umat resultmat = arma::zeros<arma::umat>(arma::size(coords));
  membership = arma::zeros<arma::uvec>(coords.n_rows);
  
  for(unsigned int j=0; j<thresholds.n_elem; j++){
    arma::vec thresholds_col = thresholds(j);
    resultmat.col(j) = uturbocolthreshold(coords.col(j), thresholds_col);
  }
  
  unsigned int ni = thresholds(0).n_elem;
  unsigned int nj = thresholds(1).n_elem;
  arma::field<arma::uvec> splitmat(ni+1, nj+1);
  
  arma::uvec ones = arma::ones<arma::uvec>(1);
  for(unsigned int i=0; i<resultmat.n_rows; i++){
    unsigned int ixi = resultmat(i, 0);
    unsigned int ixj = resultmat(i, 1);
    splitmat(ixi-1, ixj-1) = arma::join_vert( splitmat(ixi-1, ixj-1), offset + 
      ones*i);
    membership(i) = arma::sub2ind(arma::size(ni+1, nj+1), ixi-1, ixj-1);
  }
  
  return splitmat;
}

arma::field<arma::uvec> reorder_by_block(arma::field<arma::uvec> indexing,
                                         arma::uvec& order, 
                                         arma::uvec& restore_order,
                                         const arma::uvec& membership,
                                         int offset){
  
  //arma::uvec sortsort_order = arma::sort_index(arma::sort_index(membership));
  int ctr=0;
  for(unsigned int i=0; i<indexing.n_elem; i++){
    //indexing(i) = offset+sortsort_order(indexing(i)-offset);
    indexing(i) = arma::regspace<arma::uvec>(offset + ctr, 
             offset + ctr + indexing(i).n_elem);
  }
  //order = sortsort_order;
  //restore_order = arma::sort_index(order);
  return indexing;
}

MGP::MGP(const arma::mat& target_coords, 
         const arma::mat& x_coords,
         const arma::field<arma::vec>& axis_partition,
         const arma::mat& theta_in,
         bool adapting_theta,
         const arma::mat& metrop_theta_sd,
         const arma::mat& metrop_theta_bounds,
         int space_dim, bool v_in, bool d_in){
  
  verbose = v_in;
  debug = d_in;
  hl2pi = -.5 * log(2.0 * M_PI);
  
  k = theta_in.n_cols;
  dd = space_dim;
  
  w = arma::zeros(target_coords.n_rows, k);
  
  if(verbose & debug){
    Rcpp::Rcout << "initializing MGP " << endl;  
  }
  
  n = x_coords.n_rows;
  xcoords = x_coords;
  
  thresholds = axis_partition;
  init_partitiondag(target_coords, axis_partition);
  init_cache(false);
  
  block_order = field_v_concatm(indexing);
  
  n_blocks = indexing.n_elem;
  
  init_meshdata(param_data, theta_in);
  alter_data = param_data;
  
  // RAMA for theta
  theta_mcmc_counter = 0;
  theta_unif_bounds = metrop_theta_bounds;
  
  int nt = metrop_theta_sd.n_rows;
  arma::mat param_theta_sd = arma::zeros(nt + 1, nt + 1);
  param_theta_sd.submat(0, 0, nt-1, nt-1) = metrop_theta_sd;
  param_theta_sd(nt, nt) = metrop_theta_sd(0,0);
  theta_metrop_sd = param_theta_sd;
  theta_adapt = RAMAdapt(theta_in.n_elem + 1, theta_metrop_sd, 0.24); // +1 for tausq
  theta_adapt_active = adapting_theta;
}


void MGP::init_partitiondag(const arma::mat& target_coords, 
                            const arma::field<arma::vec>& axis_partition){
  
  //int q = na_mat.n_cols;
  
  indexing = split_ap(membership, target_coords, axis_partition);

  coords = target_coords;//.rows(block_order);
  xcoords = xcoords;//.rows(block_order);
  
  n_blocks = (indexing.n_rows +.0)*(indexing.n_cols +.0);
  
  arma::field<arma::uvec> dim_by_parent(arma::size(indexing));
  parents_indexing = arma::field<arma::uvec> (arma::size(indexing));
  children = arma::field<arma::vec> (arma::size(indexing));
  
  // prepare stuff for NA management
  if(verbose & debug){
    Rcpp::Rcout << "[block_ct_obs] start \n"; 
  }
  block_ct_obs = arma::zeros<arma::umat>(indexing.n_rows, indexing.n_cols);
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      block_ct_obs(i,j) = indexing(i,j).n_elem;
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_indexing] parent_indexing\n";
  }
  Rcpp::Rcout << dd << endl;
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      arma::field<arma::uvec> pixs(dd);
      if(indexing(i, j).n_elem > 0){
        dim_by_parent(i, j) = arma::zeros<arma::uvec>(dd + 1);
        if(i>0){
          if(indexing(i-1, j).n_elem > 0){
            pixs(0) = indexing(i-1, j);
            dim_by_parent(i, j)(1) = indexing(i-1, j).n_elem;
          }
        }
        if(j>0){
          if(indexing(i, j-1).n_elem > 0){
            pixs(1) = indexing(i, j-1);
            dim_by_parent(i, j)(2) = indexing(i, j-1).n_elem;
          }
        }
        parents_indexing(i,j) = field_v_concat_uv(pixs);
        dim_by_parent(i,j) = arma::cumsum(dim_by_parent(i,j));
      }
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_indexing] u_is_which_col_f\n";
  }
  
  u_is_which_col_f = arma::field<arma::field<arma::field<arma::uvec> > > (arma::size(indexing));
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      if(indexing(i, j).n_elem > 0){
        
        u_is_which_col_f(i, j) = arma::field<arma::field<arma::uvec> > (dd);
        children(i, j) = arma::zeros(dd) - 1;
        
        if(i+1<indexing.n_rows){
          
          if(block_ct_obs(i+1, j) > 0){
            
            children(i, j)(0) = arma::sub2ind(arma::size(indexing.n_rows, indexing.n_cols), i+1, j); 
            
            // (i,j) is BTM of (i+1, j)
            u_is_which_col_f(i, j)(0) = arma::field<arma::uvec> (2);
            // (i+1,j) has (i,j) as first parent
            if(dim_by_parent(i+1, j)(1) - dim_by_parent(i+1, j)(0) > 0){
              u_is_which_col_f(i, j)(0)(0) = 
                arma::regspace<arma::uvec>(dim_by_parent(i+1, j)(0), dim_by_parent(i+1, j)(1)-1);  
            } else {
              //u_is_which_col_f(i, j)(0)(0) = arma::uvec(0);
            }
            if(dim_by_parent(i+1, j)(2) - dim_by_parent(i+1, j)(1) > 0){
              u_is_which_col_f(i, j)(0)(1) = 
                arma::regspace<arma::uvec>(dim_by_parent(i+1, j)(1), dim_by_parent(i+1, j)(2)-1);  
            } else {
              //u_is_which_col_f(i, j)(0)(1) = arma::uvec(0);
            }
          }
        } 
        
        if(j+1<indexing.n_cols){
          if(block_ct_obs(i, j+1) > 0){
            children(i, j)(1) = arma::sub2ind(arma::size(indexing.n_rows, indexing.n_cols), i, j+1);
            
            // (i,j) is LEFT of (i, j+1)
            u_is_which_col_f(i, j)(1) = arma::field<arma::uvec> (2);
            // (i+1,j) has (i,j) as second parent
            if(dim_by_parent(i, j+1)(2) - dim_by_parent(i, j+1)(1) > 0){
              u_is_which_col_f(i, j)(1)(0) = 
                arma::regspace<arma::uvec>(dim_by_parent(i, j+1)(1), dim_by_parent(i, j+1)(2)-1);
            } else {
              //u_is_which_col_f(i, j)(1)(0) = arma::uvec(0);
            }
            if(dim_by_parent(i, j+1)(1) - dim_by_parent(i, j+1)(0) > 0){
              u_is_which_col_f(i, j)(1)(1) = 
                arma::regspace<arma::uvec>(dim_by_parent(i, j+1)(0), dim_by_parent(i, j+1)(1)-1); 
            } else {
              //u_is_which_col_f(i, j)(1)(1) = arma::uvec(0);
            }
            
            
          }
        }
      }
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_indexing] done.\n";
  }
  
}



void MGP::refresh_partitiondag(const arma::umat& na_mat){
  if(verbose & debug){
    Rcpp::Rcout << "[refresh_partitiondag] start" << endl;
  }
  int q = na_mat.n_cols;
  
  arma::field<arma::uvec> dim_by_parent(arma::size(indexing));
  parents_indexing = arma::field<arma::uvec> (arma::size(indexing));
  children = arma::field<arma::vec> (arma::size(indexing));
  
  block_ct_obs = arma::zeros<arma::umat>(indexing.n_rows, indexing.n_cols);
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      block_ct_obs(i,j) = indexing(i,j).n_elem;
    }
  }
  
  
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      arma::field<arma::uvec> pixs(dd);
      if(indexing(i, j).n_elem > 0){
        dim_by_parent(i, j) = arma::zeros<arma::uvec>(dd + 1);
        if(i>0){
          if(indexing(i-1, j).n_elem > 0){
            pixs(0) = indexing(i-1, j);
            dim_by_parent(i, j)(1) = indexing(i-1, j).n_elem;
          }
        }
        if(j>0){
          if(indexing(i, j-1).n_elem > 0){
            pixs(1) = indexing(i, j-1);
            dim_by_parent(i, j)(2) = indexing(i, j-1).n_elem;
          }
        }
        parents_indexing(i,j) = field_v_concat_uv(pixs);
        dim_by_parent(i,j) = arma::cumsum(dim_by_parent(i,j));
      }
    }
  }
  
  
  u_is_which_col_f = arma::field<arma::field<arma::field<arma::uvec> > > (arma::size(indexing));
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      if(indexing(i, j).n_elem > 0){
        
        u_is_which_col_f(i, j) = arma::field<arma::field<arma::uvec> > (dd);
        children(i, j) = arma::zeros(dd) - 1;
        
        if(i+1<indexing.n_rows){
          
          if(block_ct_obs(i+1, j) > 0){
            
            children(i, j)(0) = arma::sub2ind(arma::size(indexing.n_rows, indexing.n_cols), i+1, j); 
            
            // (i,j) is BTM of (i+1, j)
            u_is_which_col_f(i, j)(0) = arma::field<arma::uvec> (2);
            // (i+1,j) has (i,j) as first parent
            if(dim_by_parent(i+1, j)(1) - dim_by_parent(i+1, j)(0) > 0){
              u_is_which_col_f(i, j)(0)(0) = 
                arma::regspace<arma::uvec>(dim_by_parent(i+1, j)(0), dim_by_parent(i+1, j)(1)-1);  
            } else {
              //u_is_which_col_f(i, j)(0)(0) = arma::uvec(0);
            }
            if(dim_by_parent(i+1, j)(2) - dim_by_parent(i+1, j)(1) > 0){
              u_is_which_col_f(i, j)(0)(1) = 
                arma::regspace<arma::uvec>(dim_by_parent(i+1, j)(1), dim_by_parent(i+1, j)(2)-1);  
            } else {
              //u_is_which_col_f(i, j)(0)(1) = arma::uvec(0);
            }
          }
        } 
        
        if(j+1<indexing.n_cols){
          if(block_ct_obs(i, j+1) > 0){
            children(i, j)(1) = arma::sub2ind(arma::size(indexing.n_rows, indexing.n_cols), i, j+1);
            
            // (i,j) is LEFT of (i, j+1)
            u_is_which_col_f(i, j)(1) = arma::field<arma::uvec> (2);
            // (i+1,j) has (i,j) as second parent
            if(dim_by_parent(i, j+1)(2) - dim_by_parent(i, j+1)(1) > 0){
              u_is_which_col_f(i, j)(1)(0) = 
                arma::regspace<arma::uvec>(dim_by_parent(i, j+1)(1), dim_by_parent(i, j+1)(2)-1);
            } else {
              //u_is_which_col_f(i, j)(1)(0) = arma::uvec(0);
            }
            if(dim_by_parent(i, j+1)(1) - dim_by_parent(i, j+1)(0) > 0){
              u_is_which_col_f(i, j)(1)(1) = 
                arma::regspace<arma::uvec>(dim_by_parent(i, j+1)(0), dim_by_parent(i, j+1)(1)-1); 
            } else {
              //u_is_which_col_f(i, j)(1)(1) = arma::uvec(0);
            }
            
            
          }
        }
      }
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[refresh_partitiondag] done.\n";
  }
  
}

void MGP::init_cache(bool use_cache){
  // coords_caching stores the layer names of those layers that are representative
  // coords_caching_ix stores info on which layers are the same in terms of rel. distance
  
  if(verbose & debug){
    Rcpp::Rcout << "init_cache start \n";
  }
  //coords_caching_ix = caching_pairwise_compare_uc(coords_blocks, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching_ix = caching_pairwise_compare_mi(coords, indexing, block_ct_obs, use_cache); // uses block_names(i)-1 !
  coords_caching = arma::unique(coords_caching_ix);
  
  //parents_caching_ix = caching_pairwise_compare_uc(parents_coords, block_names, block_ct_obs);
  //parents_caching_ix = caching_pairwise_compare_uci(coords, parents_indexing, block_names, block_ct_obs);
  //parents_caching = arma::unique(parents_caching_ix);
  
  arma::field<arma::mat> kr_pairing(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i = 0; i<n_blocks; i++){
    arma::mat cmat = coords.rows(indexing(i));
    if(parents_indexing(i).n_elem > 0){
      arma::mat pmat = coords.rows(parents_indexing(i));
      arma::mat kr_mat_c = arma::join_vert(cmat, pmat);
      kr_pairing(i) = kr_mat_c;
    } else {
      kr_pairing(i) = cmat;
    }
  }
  
  kr_caching_ix = caching_pairwise_compare_m(kr_pairing, block_ct_obs, use_cache);
  kr_caching = arma::unique(kr_caching_ix);
  
  starting_kr = 0;
  cx_and_kr_caching = kr_caching;
  
  // 
  findkr = arma::zeros<arma::uvec>(n_blocks);
  findcc = arma::zeros<arma::uvec>(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks; i++){
    int u = i;
    int kr_cached_ix = kr_caching_ix(u);
    arma::uvec cpx = arma::find( kr_caching == kr_cached_ix, 1, "first" );
    findkr(u) = cpx(0);
    
    int u_cached_ix = coords_caching_ix(u);
    arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
    findcc(u) = cx(0);
    
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "Caching c: " << coords_caching.n_elem 
                << " k: " << kr_caching.n_elem << "\n";
  }
}


void MGP::metrop_theta(){
  if(verbose & debug){
    Rcpp::Rcout << "[metrop_theta] start\n";
  }
  
  theta_adapt.count_proposal();
  
  arma::vec param = arma::vectorise(param_data.theta);
  arma::vec new_param = arma::vectorise(param_data.theta);
  
  Rcpp::RNGScope scope;
  arma::vec U_update = mrstdnorm(new_param.n_elem, 1);
  
  
  // theta
  new_param = par_huvtransf_back(par_huvtransf_fwd(param, theta_unif_bounds) + 
    theta_adapt.paramsd * U_update, theta_unif_bounds);
  
  //new_param(1) = 1; //***
  
  //bool out_unif_bounds = unif_bounds(new_param, theta_unif_bounds);
  
  arma::mat theta_proposal = 
    arma::mat(new_param.memptr(), new_param.n_elem/k, k);
  
  if(use_ps == false){
    theta_proposal.tail_rows(1).fill(1);
  }
  
  alter_data.theta = theta_proposal;
  
  
  bool acceptable = get_mgplogdens_comps(alter_data );
  
  bool accepted = false;
  double logaccept = 0;
  double current_loglik = 0;
  double new_loglik = 0;
  double prior_logratio = 0;
  double jacobian = 0;
  
  if(acceptable){
    new_loglik = alter_data.loglik_w;
    
    current_loglik = param_data.loglik_w;
    
    prior_logratio = calc_prior_logratio(
      alter_data.theta.tail_rows(1).t(), param_data.theta.tail_rows(1).t(), 2, 1); // sigmasq
    
    if(param_data.theta.n_rows > 5){
      for(int i=0; i<param_data.theta.n_rows-2; i++){
        prior_logratio += arma::accu( -alter_data.theta.row(i) +param_data.theta.row(i) ); // exp
      }
    }
    
    jacobian  = calc_jacobian(new_param, param, theta_unif_bounds);
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
    theta_adapt.count_accepted();
    
    accept_make_change();
    param_data.theta = theta_proposal;
    
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
  
  
  theta_adapt.update_ratios();
  
  if(theta_adapt_active){
    theta_adapt.adapt(U_update, acceptable*exp(logaccept), theta_mcmc_counter); 
  }
  theta_mcmc_counter++;
  if(verbose & debug){
    Rcpp::Rcout << "[metrop_theta] end\n";
  }
}


void MGP::prior_sample(MeshDataLMC& data){
  // sample from MGP prior
  w = arma::zeros(coords.n_rows, k);
  
  if(verbose & debug){
    Rcpp::Rcout << "[w_prior_sample] " << "\n";
  }
  //Rcpp::Rcout << "Lambda from:  " << Lambda_orig(0, 0) << " to  " << Lambda(0, 0) << endl;
  
  //int ns = coords.n_rows;
  
  bool acceptable = refresh_cache(data);
  if(!acceptable){
    Rcpp::stop("Something went wrong went getting the conditional Gaussians. Try different theta? ");
  }
  
  // assuming that the ordering in block_names is the ordering of the product of conditional densities
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      unsigned int u = arma::sub2ind(arma::size(indexing.n_rows, indexing.n_cols), i, j);
      update_block_covpars(u, data);
      arma::mat wtemp = arma::zeros(indexing(u).n_elem, k);
      arma::mat wparent;
      if(parents_indexing(u).n_elem > 0){
        wparent = w.rows(parents_indexing(u)); 
      }
      for(int h=0; h<k; h++){
        arma::mat w_mean = arma::zeros(indexing(u).n_elem);
        if(parents_indexing(u).n_elem > 0){
          w_mean = (*data.w_cond_mean_K_ptr.at(u)).slice(h) * wparent.col(h);
        } 
        arma::mat Sigi_chol = (*data.w_cond_chR_ptr.at(u)).slice(h); //
        // sample
        arma::vec rnvec = arma::randn(indexing(u).n_elem);
        wtemp.col(h) = w_mean + Sigi_chol.t() * rnvec;  
      }
      
      w.rows(indexing(u)) = wtemp;
    }
  }
  

  if(verbose & debug){
    Rcpp::Rcout << "[w_prior_sample] loops \n";
  }
  
}



arma::mat MGP::block_fullconditional_prior_ci(int u, MeshDataLMC& data){
  //
  // parents
  arma::mat result = build_block_diagonal_ptr(data.w_cond_prec_ptr.at(u));
  
  // children
  for(unsigned int c=0; c<children(u).n_elem; c++){
    //Rcpp::Rcout << children(xi, xj) << endl;
    int child = children(u)(c);
    if(child != -1){
      arma::cube AK_u = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(0));
      AKuT_x_R_ptr(data.AK_uP(u)(c), AK_u, data.w_cond_prec_ptr.at(child)); 
      add_AK_AKu_multiply_(result, data.AK_uP(u)(c), AK_u);  
    }
  }
  
  return result;
}


arma::mat MGP::conj_block_fullconditional_m(int u, MeshDataLMC& data){
  // parents
  arma::mat result = arma::zeros(k*indexing(u).n_elem, 1); // replace with fill(0)
  if(parents_indexing(u).n_elem>0){
    add_smu_parents_ptr_(result, data.w_cond_prec_ptr.at(u), data.w_cond_mean_K_ptr.at(u),
                         w.rows( parents_indexing(u) ));
  } 

  for(unsigned int c=0; c<dd; c++){
    int child = children(u)(c);
    if(child != -1){
      //Rcpp::Rcout << "child found " << c << endl;
      //---------------------
      arma::cube AK_u = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(0));
      
      arma::mat w_child = w.rows(indexing(child));
      arma::mat w_parchild = w.rows(parents_indexing(child));
      
      //---------------------
      if(parents_indexing(child).n_elem > indexing(u).n_elem){ // some other parent exists
        arma::cube AK_others = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(1));
        arma::mat w_parchild_others = w_parchild.rows(u_is_which_col_f(u)(c)(1));
        result += 
          arma::vectorise(AK_vec_multiply(data.AK_uP(u)(c), 
                                          w_child - AK_vec_multiply(AK_others, w_parchild_others)));
      } else {
        result += 
          arma::vectorise(AK_vec_multiply(data.AK_uP(u)(c), w_child));
      }
    }
  }
  
  return result;
}

double MGP::eval_block_fullconditional_m(int u, const arma::mat& x, MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "eval_block_fullconditional_m " << endl;  
  }
  
  double logdens = 0;
  if(u == -1){
    return logdens;
  }
  // parents
  //Rcpp::Rcout << u << " parents:" << endl;
  
  arma::mat wpars = w.rows( parents_indexing(u) );
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(parents_indexing(u).n_elem>0){ // meaning w_parents.n_rows > 0
      xcentered -= (*data.w_cond_mean_K_ptr.at(u)).slice(j) * wpars.col(j);
    } 
    arma::vec Rix = (*data.w_cond_prec_ptr.at(u)).slice(j) * xcentered;
    numer += arma::conv_to<double>::from( xcentered.t() * Rix );
    
  }
  
  //Rcpp::Rcout << " logdens parent MGP : " << -.5 * numer << endl;
  logdens = -.5 * numer;
  
  // children
  numer = 0;
  for(unsigned int c=0; c<dd; c++){
    int child = children(u)(c);
    if(child != -1){
      //Rcpp::Rcout << "child found " << c << endl;
      //---------------------
      arma::cube AK_u = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(0));
      
      arma::mat w_centered = w.rows(indexing(child)) - AK_vec_multiply(AK_u, x);
      arma::mat w_parchild = w.rows(parents_indexing(child));
      
      //---------------------
      if(parents_indexing(child).n_elem > indexing(u).n_elem){ // some other parent exists
        arma::cube AK_others = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(1));
        arma::mat w_parchild_others = w_parchild.rows(u_is_which_col_f(u)(c)(1));
        w_centered = w_centered - AK_vec_multiply(AK_others, w_parchild_others);
      } 
      
      numer += AK_vec_outer_ptr(data.w_cond_prec_ptr.at(child), w_centered);
    }
  }
  //Rcpp::Rcout << " logdens children MGP : " << -.5 * numer << endl;
  logdens += -.5 * numer;
  
  return logdens;
}


arma::mat MGP::block_grad_fullconditional_prior_m(double& logdens, 
                                                  const arma::mat& x,
                                                  int u, MeshDataLMC& data){
  //
  if(u == -1){
    logdens = 0;
    return arma::zeros(x.n_elem, 1);
  }
  
  // parents
  arma::mat norm_grad = arma::zeros(arma::size(x));
  arma::mat wpars = w.rows( parents_indexing(u) );
  double numer = 0;
  for(unsigned int j=0; j<x.n_cols; j++){
    arma::vec xcentered = x.col(j);
    if(parents_indexing(u).n_elem>0){ // meaning w_parents.n_rows > 0
      xcentered -= (*data.w_cond_mean_K_ptr.at(u)).slice(j) * wpars.col(j);
    } 
    arma::vec Rix = (*data.w_cond_prec_ptr.at(u)).slice(j) * xcentered;
    numer += arma::conv_to<double>::from( xcentered.t() * Rix );
    norm_grad.col(j) = - Rix;
  }
  
  //Rcpp::Rcout << " logdens parent MGP : " << -.5 * numer << endl;
  logdens = -.5 * numer;
  
  // children
  numer = 0;
  for(unsigned int c=0; c<dd; c++){
    int child = children(u)(c);
    if(child != -1){
      //Rcpp::Rcout << "child found " << c << endl;
      //---------------------
      arma::cube AK_u = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(0));
      
      arma::mat w_centered = w.rows(indexing(child)) - AK_vec_multiply(AK_u, x);
      arma::mat w_parchild = w.rows(parents_indexing(child));
      
      //---------------------
      if(parents_indexing(child).n_elem > indexing(u).n_elem){ // some other parent exists
        arma::cube AK_others = cube_cols_ptr(data.w_cond_mean_K_ptr.at(child), u_is_which_col_f(u)(c)(1));
        arma::mat w_parchild_others = w_parchild.rows(u_is_which_col_f(u)(c)(1));
        w_centered = w_centered - AK_vec_multiply(AK_others, w_parchild_others);
      } 
      norm_grad += AK_vec_multiply(data.AK_uP(u)(c), w_centered);
      numer += AK_vec_outer_ptr(data.w_cond_prec_ptr.at(child), w_centered);
    }
  }
  //Rcpp::Rcout << " logdens children MGP : " << -.5 * numer << endl;
  logdens += -.5 * numer;
  
  if(verbose & debug){
    Rcpp::Rcout << "update_block_w_cache done " << endl;  
  }
  return arma::vectorise(norm_grad);
}


void MGP::block_fullconditional_regdata(int u, 
                                        const arma::mat& y,
                                        const arma::mat& XB,
                                        const arma::umat& na_mat,
                                        const arma::mat& Lambda,
                                        const arma::vec& tausq_inv,
                                        MeshDataLMC& data){
  //data
  int q = y.n_cols;
  arma::mat u_tau_inv = arma::zeros(indexing(u).n_elem, q);
  arma::mat ytilde = arma::zeros(indexing(u).n_elem, q);
  
  for(unsigned int j=0; j<q; j++){
    for(unsigned int ix=0; ix<indexing(u).n_elem; ix++){
      if(na_mat(indexing(u)(ix), j) == 1){
        u_tau_inv(ix, j) = pow(tausq_inv(j), .5);
        ytilde(ix, j) = (y(indexing(u)(ix), j) - XB(indexing(u)(ix), j))*u_tau_inv(ix, j);
      }
    }
    // dont erase:
    //Sigi_tot += arma::kron( arma::trans(Lambda.row(j)) * Lambda.row(j), arma::diagmat(u_tau_inv%u_tau_inv));
    arma::mat LjtLj = arma::trans(Lambda.row(j)) * Lambda.row(j);
    arma::vec u_tausq_inv = u_tau_inv.col(j) % u_tau_inv.col(j);
    add_LtLxD(data.Sigi(u), LjtLj, u_tausq_inv);
    
    data.Smu(u) += arma::vectorise(arma::diagmat(u_tau_inv.col(j)) * ytilde.col(j) * Lambda.row(j));
  }
}



void MGP::init_meshdata(MeshDataLMC& data, const arma::mat& theta_in){
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_meshdata]\n";
  }
  
  
  
  data.Smu = arma::field<arma::mat>(n_blocks);
  data.Sigi = arma::field<arma::mat>(n_blocks);
  data.AK_uP = arma::field<arma::field<arma::cube> >(n_blocks);
  data.CC_cache = arma::field<arma::cube>(coords_caching.n_elem);
  
  data.Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<n_blocks; i++){
    data.Smu(i) = arma::zeros(k*indexing(i).n_elem, 1);
    data.Sigi(i) = arma::zeros(k*indexing(i).n_elem, k*indexing(i).n_elem);
    
    arma::uvec xyloc = arma::ind2sub(arma::size(indexing), i);
    unsigned int xi = xyloc(0);
    unsigned int xj = xyloc(1);
    data.AK_uP(i) = arma::field<arma::cube>(dd);
    if(xi+1 < indexing.n_rows){
      data.AK_uP(i)(0) = arma::zeros(indexing(xi, xj).n_elem, indexing(xi+1, xj).n_elem, k);
    }
    if(xj+1 < indexing.n_cols){
      data.AK_uP(i)(1) = arma::zeros(indexing(xi, xj).n_elem, indexing(xi, xj+1).n_elem, k);
    }
  }
  data.w_cond_chR_ptr.reserve(n_blocks);
  data.w_cond_chRi_ptr.reserve(n_blocks);
  data.w_cond_prec_ptr.reserve(n_blocks);
  data.w_cond_mean_K_ptr.reserve(n_blocks);
  data.w_cond_prec_parents_ptr.reserve(n_blocks);
  
  for(unsigned int i=0; i<n_blocks; i++){
    arma::cube jibberish = arma::zeros(1,1,1);
    data.w_cond_prec_ptr.push_back(&jibberish);
    data.w_cond_chR_ptr.push_back(&jibberish);
    data.w_cond_chRi_ptr.push_back(&jibberish);
    data.w_cond_mean_K_ptr.push_back(&jibberish);
    data.w_cond_prec_parents_ptr.push_back(&jibberish);
  }
  
  data.Kxxi_cache = arma::field<arma::cube>(coords_caching.n_elem);
  for(unsigned int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    data.Kxxi_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    if(block_ct_obs(u) > 0){
      data.CC_cache(i) = arma::cube(indexing(u).n_elem, indexing(u).n_elem, k);
    }
  }
  
  // loglik w for updating theta
  data.logdetCi_comps = arma::zeros(n_blocks);
  data.logdetCi = 0;
  
  // ***
  data.wcore = arma::zeros(n_blocks, 1);
  data.loglik_w_comps = arma::zeros(n_blocks, 1);
  data.loglik_w = 0; 
  data.theta = theta_in;//##
  
  data.H_cache = arma::field<arma::cube> (kr_caching.n_elem);
  data.Ri_cache = arma::field<arma::cube> (kr_caching.n_elem);
  data.chRi_cache = arma::field<arma::cube> (kr_caching.n_elem);
  data.chR_cache = arma::field<arma::cube> (kr_caching.n_elem);
  data.Kppi_cache = arma::field<arma::cube> (kr_caching.n_elem);
  for(unsigned int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    data.Ri_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    data.chRi_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    data.chR_cache(i) = 
      arma::zeros(indexing(u).n_elem, indexing(u).n_elem, k);
    if(parents_indexing(u).n_elem > 0){
      data.H_cache(i) = 
        arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem, k);
      data.Kppi_cache(i) = 
        arma::zeros(parents_indexing(u).n_elem, parents_indexing(u).n_elem, k);
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_meshdata] done.\n";
  }
  
}



bool MGP::refresh_cache(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "[refresh_cache] start.\n";
  }
  
  data.Ri_chol_logdet = arma::zeros(kr_caching.n_elem);
  
  int errtype = -1;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); 
    if(block_ct_obs(u) > 0){
      for(unsigned int j=0; j<k; j++){
        data.CC_cache(i).slice(j) = Correlationf(xcoords, indexing(u), indexing(u), 
                      data.theta.col(j), use_ps, true);
      }
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(unsigned int it=0; it<cx_and_kr_caching.n_elem; it++){
    int i = 0;
    if(it < starting_kr){
      // this means we are caching coords
      i = it;
      int u = coords_caching(i); // block name of ith representative
      try {
        CviaKron_invsympd_(data.Kxxi_cache(i),
                           xcoords, indexing(u), k, data.theta, use_ps);
      } catch (...) {
        errtype = 1;
      }
    } else {
      // this means we are caching kr
      i = it - starting_kr;
      int u = kr_caching(i);
      try {
        if(block_ct_obs(u) > 0){
          //int u_cached_ix = coords_caching_ix(u);
          //arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first");
          
          int ccfound = findcc(u);
          //arma::cube Cxx = CC_cache(ccfound);
          data.Ri_chol_logdet(i) = CviaKron_HRi_(data.H_cache(i), 
                              data.Ri_cache(i), data.chR_cache(i), data.chRi_cache(i),
                              data.Kppi_cache(i), data.CC_cache(ccfound),
                              xcoords, indexing(u), parents_indexing(u), k, data.theta, use_ps);
        }
      } catch (...) {
        errtype = 2;
      }
    }
  }
  
  if((verbose & debug)){
    Rcpp::Rcout << "[refresh_cache] \n";
  }
  
  //Rcpp::Rcout << "refresh_cache " << errtype << endl;
  
  if(errtype > 0){
    if(verbose & debug){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "theta: " << data.theta.t() << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  return true;
}

bool MGP::get_mgplogdens_comps(MeshDataLMC& data){
  bool acceptable = refresh_cache(data);
  if(acceptable){
    acceptable = calc_mgplogdens(data);
    return acceptable;
  } else {
    return acceptable;
  }
}



void MGP::update_block_covpars(int u, MeshDataLMC& data){
  //message("[update_block_covpars] start.");
  // given block u as input, this function updates H and R
  // which will be used later to compute logp(w | theta)
  int krfound = findkr(u);
  //w_cond_prec(u) = Ri_cache(krfound);
  
  data.w_cond_prec_ptr.at(u) = &data.Ri_cache(krfound);
  data.w_cond_chRi_ptr.at(u) = &data.chRi_cache(krfound);
  data.w_cond_chR_ptr.at(u) = &data.chR_cache(krfound);
  data.logdetCi_comps(u) = data.Ri_chol_logdet(krfound);
  
  if( parents_indexing(u).n_elem > 0 ){
    //w_cond_mean_K(u) = H_cache(krfound);
    data.w_cond_mean_K_ptr.at(u) = &data.H_cache(krfound);
    data.w_cond_prec_parents_ptr.at(u) = &data.Kppi_cache(krfound);
  } 
  
  //message("[update_block_covpars] done.");
}

void MGP::update_block_wlogdens(int u, MeshDataLMC& data){
  
  arma::mat wx = w.rows(indexing(u));
  arma::mat wcoresum = arma::zeros(1, k);
  
  if( parents_indexing(u).n_elem > 0 ){
    arma::mat wpar = w.rows(parents_indexing(u));
    for(unsigned int j=0; j<k; j++){
      wx.col(j) = wx.col(j) - 
        (*data.w_cond_mean_K_ptr.at(u)).slice(j) *
        //w_cond_mean_K(u).slice(j) * 
        wpar.col(j);
    }
  }
  
  for(unsigned int j=0; j<k; j++){
    wcoresum(j) = 
      arma::conv_to<double>::from(arma::trans(wx.col(j)) * 
      //w_cond_prec(u).slice(j) * 
      (*data.w_cond_prec_ptr.at(u)).slice(j) *
      wx.col(j));
  }
  
  data.wcore.row(u) = arma::accu(wcoresum);
  data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * arma::accu(wcoresum); //
  //arma::accu(wcore.slice(u).diag());
  
  //message("[update_block_wlogdens] done.");
}

double MGP::quadratic_form(const arma::mat& w1, const arma::mat& w2, MeshDataLMC& data){
  
  arma::mat wcoresum = arma::zeros(n_blocks, k);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    if(block_ct_obs(i) > 0){
      arma::mat wx1 = w1.rows(indexing(i));
      arma::mat wx2 = w2.rows(indexing(i));
      if( parents_indexing(i).n_elem > 0 ){
        arma::mat wpar1 = w1.rows(parents_indexing(i));
        arma::mat wpar2 = w2.rows(parents_indexing(i));
        for(unsigned int j=0; j<k; j++){
          wx1.col(j) = wx1.col(j) - 
            (*data.w_cond_mean_K_ptr.at(i)).slice(j) *
            wpar1.col(j);
          wx2.col(j) = wx2.col(j) - 
            (*data.w_cond_mean_K_ptr.at(i)).slice(j) *
            wpar2.col(j);
        }
      }
      
      for(unsigned int j=0; j<k; j++){
        wcoresum(i, j) = 
          arma::conv_to<double>::from(arma::trans(wx1.col(j)) * 
          (*data.w_cond_prec_ptr.at(i)).slice(j) *
          wx2.col(j));
      }
    }
  }
  return arma::accu(wcoresum);
}


void MGP::update_all_block_covpars(MeshDataLMC& data){
  // called for a proposal of theta
  // updates involve the covariances
  // and Sigma for adjusting the error terms
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    if(block_ct_obs(i) > 0){
      update_block_covpars(i, data);
    } 
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[update_all_block_covpars] done \n";
  }
}

bool MGP::calc_mgplogdens(MeshDataLMC& data){
  // called for a proposal of theta
  // updates involve the covariances
  // and Sigma for adjusting the error terms
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    if(block_ct_obs(i) > 0){
      update_block_covpars(i, data);
      update_block_wlogdens(i, data);
    } 
  }
  
  data.loglik_w = 
    arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps);
  
  if(std::isnan(data.loglik_w)){
    Rcpp::Rcout << "Logdens components: \n" <<
      arma::accu(data.logdetCi_comps) << " " << 
        arma::accu(data.loglik_w_comps) << endl;
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[calc_mgplogdens] done \n";
  }
  
  return true;
}


void MGP::logpost_refresh_after_gibbs(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "logpost_refresh_after_gibbs\n";
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    if(block_ct_obs(i) > 0){
      //update_block_covpars(i, data);
      update_block_wlogdens(i, data);
    }
  }
  
  data.loglik_w = arma::accu(data.logdetCi_comps) + 
    arma::accu(data.loglik_w_comps);
  
  if(verbose & debug){
    Rcpp::Rcout << "[logpost_refresh_after_gibbs] " << endl;
  }
}


void MGP::accept_make_change(){
  std::swap(param_data, alter_data);
}






#endif
