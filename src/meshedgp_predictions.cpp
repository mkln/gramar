#ifndef MGP_PREDICT
#define MGP_PREDICT

#include "meshedgp.h"

using namespace std;


MGP::MGP(const arma::mat& target_coords, 
         const arma::mat& x_coords,
         const arma::mat& coordsout,
         const arma::mat& x_out,
         const arma::field<arma::vec>& axis_partition,
         const arma::mat& theta_in,
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
  
  
  
  thresholds = axis_partition;
  init_partitiondag_prediction(target_coords, coordsout, axis_partition);
  
  xcoords = x_coords;//.rows(block_order);
  pred_xcoords = x_out;//.rows(pred_block_order);
  io_xcoords = arma::join_vert(xcoords, pred_xcoords);
  
  n = xcoords.n_rows;
  no = pred_xcoords.n_rows;
  
  n_blocks = indexing.n_elem;
  
  block_order = field_v_concatm(indexing);
  pred_block_order = field_v_concatm(pred_indexing);
  block_order_all = arma::sort_index(arma::join_vert(block_order, pred_block_order));
  block_order_reverse = arma::sort_index(block_order_all);
  
  // dont initialize meshdata because we need everything local to run 
  // prediction in parallel
  
  //init_meshdata(param_data, theta_in);
  //alter_data = param_data;
}


void MGP::prediction_sample(const arma::vec& theta,
                            const arma::mat& xobs, const arma::vec& wobs, 
                            const arma::field<arma::uvec>& obs_indexing){
  // sample from MGP prior
  w = arma::zeros(coords.n_rows, k);
  
  if(verbose & debug){
    Rcpp::Rcout << "[prediction_sample] " << "\n";
  }
  
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      unsigned int u = arma::sub2ind(arma::size(indexing.n_rows, indexing.n_cols), i, j);
      
      arma::uvec parent_set;
      arma::mat xpar, wpar;
      
      parent_set = obs_indexing(u);
      xpar = xobs.rows(obs_indexing(u)); 
      wpar = wobs.rows(obs_indexing(u));
      
      arma::mat xo = xcoords.rows(indexing(u));
      // 
        arma::mat Coo = Correlationf(xcoords, indexing(u), indexing(u), theta, false, true);
        arma::mat Cxxi = arma::inv_sympd(Correlationc(xpar, xpar, theta, false, true));
        arma::mat Cox = Correlationc(xo, xpar, theta, false, false);
        
        arma::mat H = Cox * Cxxi;
        arma::mat Rchol = arma::chol(arma::symmatu(Coo - H * Cox.t()), "lower");
        
        arma::mat pred_mean = H * wpar;
        arma::vec rnvec = arma::randn(indexing(u).n_elem);
        
        w.rows(indexing(u)) = pred_mean + Rchol * rnvec;
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[prediction_sample] loops \n";
  }
  
}



void MGP::init_partitiondag_prediction(const arma::mat& target_coords, 
                                       const arma::mat& out_coords,
                                       const arma::field<arma::vec>& axis_partition){
  
  //int q = na_mat.n_cols;
  coords = target_coords;
  n = coords.n_rows;
  indexing = split_ap(membership, target_coords, axis_partition, 0);
  n_blocks = (indexing.n_rows +.0)*(indexing.n_cols +.0);
  
  coords = target_coords;//.rows(block_order);
  
  arma::field<arma::uvec> dim_by_parent(arma::size(indexing));
  parents_indexing = arma::field<arma::uvec> (arma::size(indexing));
  children = arma::field<arma::vec> (arma::size(indexing));
  
  // prepare stuff for NA management
  if(verbose & debug){
    Rcpp::Rcout << "[init_partitiondag_prediction] start \n"; 
  }
  block_ct_obs = arma::zeros<arma::umat>(indexing.n_rows, indexing.n_cols);
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      block_ct_obs(i,j) = indexing(i,j).n_elem;
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_partitiondag_prediction] parent_indexing\n";
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
  
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_partitiondag_prediction] u_is_which_col_f prediction, 1\n";
  }
  
  // prediction coords
  pred_coords = out_coords;
  no = pred_coords.n_rows;
  
  pred_indexing = split_ap(pred_membership, pred_coords, axis_partition, n);
  //Rcpp::Rcout << indexing << endl;
  
  // reorder
  pred_indexing_reorder = pred_indexing;
  //reorder_by_block(pred_indexing_reorder, pred_block_order, pred_block_order_restore, pred_membership, n);
  //pred_coords = pred_coords.rows(pred_block_order);
  
  arma::field<arma::uvec> pred_dim_by_parent(arma::size(pred_indexing));
  pred_parents_indexing = arma::field<arma::uvec> (arma::size(pred_indexing));
  pred_children = arma::field<arma::vec> (arma::size(pred_indexing));
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_partitiondag_prediction] u_is_which_col_f prediction, 2\n";
  }
  pred_block_ct_obs = arma::zeros<arma::umat>(pred_indexing.n_rows, pred_indexing.n_cols);
  for(unsigned int i=0; i<pred_indexing.n_rows; i++){
    for(unsigned int j=0; j<pred_indexing.n_cols; j++){
      pred_block_ct_obs(i,j) = pred_indexing(i,j).n_elem;
      //Rcpp::Rcout << i << ", " << j << ": " << pred_indexing(i,j).n_elem << endl;
    }
  }
  //Rcpp::stop("stop here");
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_partitiondag_prediction] u_is_which_col_f prediction, 3\n";
  }
  for(unsigned int i=0; i<pred_indexing.n_rows; i++){
    for(unsigned int j=0; j<pred_indexing.n_cols; j++){
      arma::field<arma::uvec> pixs(dd+1);
      if(pred_indexing(i, j).n_elem > 0){
        pred_dim_by_parent(i, j) = arma::zeros<arma::uvec>(dd + 2);
        
        // predictions: first parent is same block in reference set,
        // plus the neighbor blocks in non-reference sets
        pred_dim_by_parent(i, j)(1) = indexing(i, j).n_elem;
        if(indexing(i, j).n_elem > 0){
          pixs(0) = indexing(i, j);
        }
        if(i>0){
          if(pred_indexing(i-1, j).n_elem > 0){
            pixs(1) = pred_indexing(i-1, j);
            pred_dim_by_parent(i, j)(2) = pred_indexing(i-1, j).n_elem;
          }
        }
        if(j>0){
          if(pred_indexing(i, j-1).n_elem > 0){
            pixs(2) = pred_indexing(i, j-1);
            pred_dim_by_parent(i, j)(3) = pred_indexing(i, j-1).n_elem;
          }
        }
        pred_parents_indexing(i,j) = field_v_concat_uv(pixs);
        pred_dim_by_parent(i,j) = arma::cumsum(pred_dim_by_parent(i,j));
      }
    }
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_partitiondag_prediction] u_is_which_col_f prediction, 4\n";
  }
  
  pred_u_is_which_col_f = arma::field<arma::field<arma::field<arma::uvec> > > (arma::size(pred_indexing));
  for(unsigned int i=0; i<pred_indexing.n_rows; i++){
    for(unsigned int j=0; j<pred_indexing.n_cols; j++){
      if(pred_indexing(i, j).n_elem > 0){
        
        pred_u_is_which_col_f(i, j) = arma::field<arma::field<arma::uvec> > (dd); //number of children
        pred_children(i, j) = arma::zeros(dd) - 1;
        
        if(i+1<pred_indexing.n_rows){
          if(pred_block_ct_obs(i+1, j) > 0){
            
            pred_children(i, j)(0) = n_blocks + arma::sub2ind(arma::size(pred_indexing.n_rows, pred_indexing.n_cols), i+1, j); 
            
            // (i,j) is BTM of (i+1, j)
            pred_u_is_which_col_f(i, j)(0) = arma::field<arma::uvec> (2);
            // (i+1,j) has (i,j) as first parent (after reference parent)
            if(pred_dim_by_parent(i+1, j)(2) - pred_dim_by_parent(i+1, j)(1) > 0){
              pred_u_is_which_col_f(i, j)(0)(0) = 
                arma::regspace<arma::uvec>(pred_dim_by_parent(i+1, j)(1), pred_dim_by_parent(i+1, j)(2)-1);  
            } 
            arma::uvec first_other = arma::zeros<arma::uvec>(0);
            arma::uvec second_other = arma::zeros<arma::uvec>(0);
            if(pred_dim_by_parent(i+1, j)(1) - pred_dim_by_parent(i+1, j)(0) > 0){
              first_other = arma::regspace<arma::uvec>(pred_dim_by_parent(i+1, j)(0), pred_dim_by_parent(i+1, j)(1)-1);  
            }
            if(pred_dim_by_parent(i+1, j)(3) - pred_dim_by_parent(i+1, j)(2) > 0){
              second_other = arma::regspace<arma::uvec>(pred_dim_by_parent(i+1, j)(2), pred_dim_by_parent(i+1, j)(3)-1);  
            } 
            pred_u_is_which_col_f(i, j)(0)(1) = arma::join_vert(
              first_other, second_other);
          }
        } 
        
        if(j+1<pred_indexing.n_cols){
          if(pred_block_ct_obs(i, j+1) > 0){
            pred_children(i, j)(1) = n_blocks + arma::sub2ind(arma::size(pred_indexing.n_rows, pred_indexing.n_cols), i, j+1);
            
            // (i,j) is LEFT of (i, j+1)
            pred_u_is_which_col_f(i, j)(1) = arma::field<arma::uvec> (2);
            // (i+1,j) has (i,j) as second parent
            if(pred_dim_by_parent(i, j+1)(3) - pred_dim_by_parent(i, j+1)(2) > 0){
              pred_u_is_which_col_f(i, j)(1)(0) = 
                arma::regspace<arma::uvec>(pred_dim_by_parent(i, j+1)(2), pred_dim_by_parent(i, j+1)(3)-1);
            } else {
              //u_is_which_col_f(i, j)(1)(0) = arma::uvec(0);
            }
            if(pred_dim_by_parent(i, j+1)(2) - pred_dim_by_parent(i, j+1)(1) > 0){
              pred_u_is_which_col_f(i, j)(1)(1) = 
                arma::regspace<arma::uvec>(pred_dim_by_parent(i, j+1)(1), pred_dim_by_parent(i, j+1)(2)-1); 
            } else {
              //u_is_which_col_f(i, j)(1)(1) = arma::uvec(0);
            }
            
            if(pred_dim_by_parent(i, j+1)(2) - pred_dim_by_parent(i, j+1)(0) > 0){
              pred_u_is_which_col_f(i, j)(1)(1) = arma::regspace<arma::uvec>(pred_dim_by_parent(i, j+1)(0), pred_dim_by_parent(i, j+1)(2)-1);  
            }
          }
        }
      }
    }
  }
  
  
  
  // back to reference set
  
  u_is_which_col_f = arma::field<arma::field<arma::field<arma::uvec> > > (arma::size(indexing));
  for(unsigned int i=0; i<indexing.n_rows; i++){
    for(unsigned int j=0; j<indexing.n_cols; j++){
      if(indexing(i, j).n_elem > 0){
        
        u_is_which_col_f(i, j) = arma::field<arma::field<arma::uvec> > (dd+1);
        children(i, j) = arma::zeros(dd+1) - 1;
        
        // first child to i's left
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
        
        // second child at j's bottom
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
        
        // third child prediction
        if(pred_block_ct_obs(i, j) > 0){
          children(i, j)(2) = n_blocks + arma::sub2ind(arma::size(pred_indexing.n_rows, pred_indexing.n_cols), i, j);
          
          u_is_which_col_f(i, j)(2) = arma::field<arma::uvec> (2);
          if(pred_dim_by_parent(i, j)(1) - pred_dim_by_parent(i, j)(0) > 0){
            // this block is child's first parent
            u_is_which_col_f(i, j)(2)(0) = 
              arma::regspace<arma::uvec>(pred_dim_by_parent(i, j)(0), pred_dim_by_parent(i, j)(1)-1);
          }
          if(pred_dim_by_parent(i, j)(3) - pred_dim_by_parent(i, j)(1) > 0){
            // other parents
            u_is_which_col_f(i, j)(2)(1) = 
              arma::regspace<arma::uvec>(pred_dim_by_parent(i, j)(1), pred_dim_by_parent(i, j)(3)-1);
          }
        }
      }
    }
  }
  
  
  if(verbose & debug){
    Rcpp::Rcout << "[init_indexing] done.\n";
  }
  
}


/*
arma::mat MGP::predict_via_precision_orig(const MGP& out_mgp, const arma::vec& theta){
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  int n = coords.n_rows;
  int nk = n * k;
  
  arma::uvec linear_order = arma::regspace<arma::uvec>(0, n-1);
  arma::uvec sortsort_order = linear_order; //
    //arma::sort_index(arma::sort_index(membership));
  
  linear_sort_map = arma::join_horiz(linear_order, sortsort_order);
  
  start = std::chrono::steady_clock::now();
  // K_cholcp_cache = invchol(Ri)
  arma::field<arma::umat> H_blockrow_tripls(k * 2 * n_blocks);
  arma::field<arma::mat> H_blockrow_vals(k * 2 * n_blocks);
  arma::field<arma::umat> Ri_blockrow_tripls(k * 2 * n_blocks);
  arma::field<arma::mat> Ri_blockrow_vals(k * 2 * n_blocks);
  
  int h_ctr = 0;
  int ri_ctr = 0;
  
  arma::field<arma::uvec> o_indexing = out_mgp.indexing;
  arma::field<arma::uvec> o_par_indexing = out_mgp.parents_indexing;
  arma::mat o_xcoords = out_mgp.xcoords;
  arma::mat io_xcoords = arma::join_vert(xcoords, o_xcoords);
  
  int no = o_xcoords.n_rows;
  
  for(int i=0; i<n_blocks; i++){
    int ui = i;//block_names(i) - 1;
    int ni = indexing(ui).n_elem;
    int np = parents_indexing(ui).n_elem;
    
    arma::uvec sorted_index = indexing(ui);//arma::sort(sortsort_order(indexing(ui)));
    
    // in part
    if(np > 0){
      
      arma::uvec sorted_parents = parents_indexing(ui);//arma::sort(sortsort_order(parents_indexing(ui)));
      
      // locations to fill: indexing(ui) x parents_indexing(uj) 
      arma::umat H_tripl_locs(ni * np, 2);
      arma::mat H_tripl_val = arma::zeros(ni * np);
      
      for(int ix=0; ix < ni; ix++){
        for(int jx=0; jx < np; jx++){
          int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
          H_tripl_locs(vecix, 0) = sorted_index(ix);//sortsort_order(indexing(ui)(ix));
          H_tripl_locs(vecix, 1) = sorted_parents(jx);//sortsort_order(parents_indexing(ui)(jx));
          H_tripl_val(vecix, 0) = (*param_data.w_cond_mean_K_ptr.at(ui))(ix, jx, 0);
        }
      }
      H_blockrow_tripls(h_ctr) = H_tripl_locs;
      H_blockrow_vals(h_ctr) = H_tripl_val;
      h_ctr++;  
    }
    
    // locations to fill: indexing(ui) x parents_indexing(uj)
    arma::umat Ri_tripl_locs(ni * ni, 2);
    arma::mat Ri_tripl_val = arma::zeros(ni * ni);
    for(int ix=0; ix < ni; ix++){
      for(int jx=0; jx < ni; jx++){
        int vecix = arma::sub2ind(arma::size(ni, ni), ix, jx);
        Ri_tripl_locs(vecix, 0) = sorted_index(ix); //sortsort_order(indexing(ui)(ix));
        Ri_tripl_locs(vecix, 1) = sorted_index(jx); //sortsort_order(indexing(ui)(jx));
        Ri_tripl_val(vecix, 0) = (*param_data.w_cond_chRi_ptr.at(ui))(ix, jx, 0);
      }
    }
    Ri_blockrow_tripls(ri_ctr) = Ri_tripl_locs;
    Ri_blockrow_vals(ri_ctr) = Ri_tripl_val;
    ri_ctr++;
    
    // out part
    
    arma::uvec o_par_ix = n + o_par_indexing(ui);
    arma::uvec o_ix = n + o_indexing(ui);
    sorted_index = o_ix;
    arma::uvec parent_set = arma::join_vert(indexing(ui), o_par_ix);
    
    np = parent_set.n_elem;
    ni = o_ix.n_elem;
    
    arma::mat o_xcoords_ix = io_xcoords.rows(o_ix);
    arma::mat o_xcoords_par = io_xcoords.rows(parent_set);
    
    // 
      arma::mat Coo = Correlationf(io_xcoords, o_ix, o_ix, theta, false, true);
    arma::mat Cxx = Correlationc(o_xcoords_par, o_xcoords_par, theta, false, true);
    arma::mat Cxxi = arma::inv_sympd(Cxx);
    arma::mat Cox = Correlationc(o_xcoords_ix, o_xcoords_par, theta, false, false);
    
    arma::mat H = Cox * Cxxi;
    arma::mat Ri = arma::inv_sympd(arma::symmatu(Coo - H * Cox.t()));
    if(np > 0){
      
      arma::uvec sorted_parents = parent_set;//arma::sort(sortsort_order(parents_indexing(ui)));
      
      // locations to fill: indexing(ui) x parents_indexing(uj) 
      arma::umat H_tripl_locs = arma::umat(ni * np, 2);
      arma::mat H_tripl_val = arma::zeros(ni * np);
      
      for(int ix=0; ix < ni; ix++){
        for(int jx=0; jx < np; jx++){
          int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
          H_tripl_locs(vecix, 0) = sorted_index(ix);//sortsort_order(indexing(ui)(ix));
          H_tripl_locs(vecix, 1) = sorted_parents(jx);//sortsort_order(parents_indexing(ui)(jx));
          H_tripl_val(vecix, 0) = H(ix, jx);
        }
      }
      H_blockrow_tripls(h_ctr) = H_tripl_locs;
      H_blockrow_vals(h_ctr) = H_tripl_val;
      h_ctr++;  
    }
    
    // locations to fill: indexing(ui) x parents_indexing(uj)
    Ri_tripl_locs = arma::umat(ni * ni, 2);
    Ri_tripl_val = arma::zeros(ni * ni);
    for(int ix=0; ix < ni; ix++){
      for(int jx=0; jx < ni; jx++){
        int vecix = arma::sub2ind(arma::size(ni, ni), ix, jx);
        Ri_tripl_locs(vecix, 0) = sorted_index(ix); //sortsort_order(indexing(ui)(ix));
        Ri_tripl_locs(vecix, 1) = sorted_index(jx); //sortsort_order(indexing(ui)(jx));
        Ri_tripl_val(vecix, 0) = Ri(ix, jx);
      }
    }
    Ri_blockrow_tripls(ri_ctr) = Ri_tripl_locs;
    Ri_blockrow_vals(ri_ctr) = Ri_tripl_val;
    ri_ctr++;
    
  }
  
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "predict loop storing: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  
  int nall = io_xcoords.n_rows;
  He = Eigen::SparseMatrix<double>(nall, nall);
  Rie = Eigen::SparseMatrix<double>(nall, nall);
  
  arma::umat Hlocs = field_v_concatm(H_blockrow_tripls);
  arma::vec Hvals = field_v_concatm(H_blockrow_vals);
  arma::umat Rilocs = field_v_concatm(Ri_blockrow_tripls);
  arma::vec Rivals = field_v_concatm(Ri_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  tripletList_H.reserve(Hlocs.n_rows);
  for(int i=0; i<Hlocs.n_rows; i++){
    tripletList_H.push_back(T(Hlocs(i, 0), Hlocs(i, 1), Hvals(i)));
  }
  
  std::vector<T> tripletList_Ri;
  tripletList_Ri.reserve(Rilocs.n_rows);
  for(int i=0; i<Rilocs.n_rows; i++){
    tripletList_Ri.push_back(T(Rilocs(i, 0), Rilocs(i, 1), Rivals(i)));
  }
  
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  Rie.setFromTriplets(tripletList_Ri.begin(), tripletList_Ri.end());
  
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "setfromtriplets: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> I_eig(nall, nall);
  I_eig.setIdentity();
  //Ci = (I_eig - H).triangularView<Eigen::Lower>().transpose() *
    //  Ri * (I_eig - H).triangularView<Eigen::Lower>();
  PP_all = (I_eig - He).transpose() * Rie * (I_eig - He);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "product: " << timer << endl;
  
  // submatrices
  
  PP_o = PP_all.block(n, n, no, no);
  PP_ox = PP_all.block(n, 0, no, n);
  
  start = std::chrono::steady_clock::now();
  solver.analyzePattern(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "pattern: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  solver.factorize(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "factorize: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::MatrixXd we = armamat_to_matrixxd(w);
  Eigen::MatrixXd oxw = PP_ox * we;
  Eigen::MatrixXd pred_mean_e = -solver.solve(oxw);
  arma::mat pred_mean = matrixxd_to_armamat(pred_mean_e);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "solve & copy: " << timer << endl;
  
  return pred_mean;
}


arma::mat MGP::predict_via_precision_product(const MGP& out_mgp, const arma::vec& theta){
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_product] start " << endl;
  }
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  int n = coords.n_rows;
  int nk = n * k;
  
  arma::uvec linear_order = arma::regspace<arma::uvec>(0, n-1);
  arma::uvec sortsort_order = linear_order; //
    //arma::sort_index(arma::sort_index(membership));
  
  linear_sort_map = arma::join_horiz(linear_order, sortsort_order);
  
  start = std::chrono::steady_clock::now();
  // K_cholcp_cache = invchol(Ri)
  arma::field<arma::umat> H_blockrow_tripls(4 * n_blocks); //H blocks, R blocks, same for predictions 
  arma::field<arma::mat> H_blockrow_vals(4 * n_blocks);
  
  int h_ctr = 0;
  int ri_ctr = 0;
  
  arma::field<arma::uvec> o_indexing = out_mgp.indexing;
  arma::field<arma::uvec> o_par_indexing = out_mgp.parents_indexing;
  arma::mat o_xcoords = out_mgp.xcoords;
  arma::mat io_xcoords = arma::join_vert(xcoords, o_xcoords);
  
  int no = o_xcoords.n_rows;
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_product] begin loop " << endl;
  }
  for(int i=0; i<n_blocks; i++){
    int ui = i;//block_names(i) - 1;
    int ni = indexing(ui).n_elem;
    int np = parents_indexing(ui).n_elem;
    
    arma::uvec sorted_index = indexing(ui);//arma::sort(sortsort_order(indexing(ui)));
    
    // in part
    arma::mat chRit = ((*param_data.w_cond_chRi_ptr.at(ui)).slice(0));
    if(np > 0){
      
      arma::uvec sorted_parents = parents_indexing(ui);//arma::sort(sortsort_order(parents_indexing(ui)));
      
      // locations to fill: indexing(ui) x parents_indexing(uj) 
      arma::umat H_tripl_locs(ni * np, 2);
      arma::mat H_tripl_val = arma::zeros(ni * np);
      
      arma::mat Hmod = chRit * ((*param_data.w_cond_mean_K_ptr.at(ui)).slice(0));
      for(int ix=0; ix < ni; ix++){
        for(int jx=0; jx < np; jx++){
          int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
          H_tripl_locs(vecix, 0) = sorted_index(ix);//sortsort_order(indexing(ui)(ix));
          H_tripl_locs(vecix, 1) = sorted_parents(jx);//sortsort_order(parents_indexing(ui)(jx));
          H_tripl_val(vecix, 0) = -Hmod(ix, jx);
        }
      }
      H_blockrow_tripls(h_ctr) = H_tripl_locs;
      H_blockrow_vals(h_ctr) = H_tripl_val;
      h_ctr++;  
    }
    arma::umat H_tripl_locs(ni * ni, 2);
    arma::mat H_tripl_val = arma::zeros(ni * ni);
    for(int ix=0; ix < ni; ix++){
      for(int jx=0; jx < ni; jx++){
        int vecix = arma::sub2ind(arma::size(ni, ni), ix, jx);
        H_tripl_locs(vecix, 0) = sorted_index(ix); //sortsort_order(indexing(ui)(ix));
        H_tripl_locs(vecix, 1) = sorted_index(jx); //sortsort_order(indexing(ui)(jx));
        H_tripl_val(vecix, 0) = chRit(ix, jx);
      }
    }
    H_blockrow_tripls(h_ctr) = H_tripl_locs;
    H_blockrow_vals(h_ctr) = H_tripl_val;
    h_ctr++;  
    
    //Rcpp::Rcout << "2 " << endl;
    
    // out part
    
    arma::uvec o_par_ix = n + o_par_indexing(ui);
    arma::uvec o_ix = n + o_indexing(ui);
    sorted_index = o_ix;
    arma::uvec parent_set = arma::join_vert(indexing(ui), o_par_ix);
    
    np = parent_set.n_elem;
    ni = o_ix.n_elem;
    
    arma::mat o_xcoords_ix = io_xcoords.rows(o_ix);
    arma::mat o_xcoords_par = io_xcoords.rows(parent_set);
    
    // 
      arma::mat Coo = Correlationf(io_xcoords, o_ix, o_ix, theta, false, true);
    arma::mat Cxx = Correlationc(o_xcoords_par, o_xcoords_par, theta, false, true);
    arma::mat Cxxi = arma::inv_sympd(Cxx);
    arma::mat Cox = Correlationc(o_xcoords_ix, o_xcoords_par, theta, false, false);
    
    arma::mat H = Cox * Cxxi;
    //arma::mat Ri = arma::inv_sympd(arma::symmatu(Coo - H * Cox.t()));
    chRit = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo - H * Cox.t()), "lower"))));
    arma::mat Hmod = chRit * H;
    
    //Rcpp::Rcout << "3 " << endl;
    if(np > 0){
      
      arma::uvec sorted_parents = parent_set;//arma::sort(sortsort_order(parents_indexing(ui)));
      
      // locations to fill: indexing(ui) x parents_indexing(uj) 
      arma::umat H_tripl_locs = arma::umat(ni * np, 2);
      arma::mat H_tripl_val = arma::zeros(ni * np);
      
      for(int ix=0; ix < ni; ix++){
        for(int jx=0; jx < np; jx++){
          int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
          H_tripl_locs(vecix, 0) = sorted_index(ix);//sortsort_order(indexing(ui)(ix));
          H_tripl_locs(vecix, 1) = sorted_parents(jx);//sortsort_order(parents_indexing(ui)(jx));
          H_tripl_val(vecix, 0) = -Hmod(ix, jx);
        }
      }
      H_blockrow_tripls(h_ctr) = H_tripl_locs;
      H_blockrow_vals(h_ctr) = H_tripl_val;
      h_ctr++;  
    }
    H_tripl_locs = arma::umat(ni * ni, 2);
    H_tripl_val = arma::zeros(ni * ni);
    for(int ix=0; ix < ni; ix++){
      for(int jx=0; jx < ni; jx++){
        int vecix = arma::sub2ind(arma::size(ni, ni), ix, jx);
        H_tripl_locs(vecix, 0) = sorted_index(ix); //sortsort_order(indexing(ui)(ix));
        H_tripl_locs(vecix, 1) = sorted_index(jx); //sortsort_order(indexing(ui)(jx));
        H_tripl_val(vecix, 0) = chRit(ix, jx);
      }
    }
    H_blockrow_tripls(h_ctr) = H_tripl_locs;
    H_blockrow_vals(h_ctr) = H_tripl_val;
    h_ctr++;  
    
    
  }
  
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "predict loop storing: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  
  int nall = io_xcoords.n_rows;
  He = Eigen::SparseMatrix<double>(nall, nall);
  //Ri = Eigen::SparseMatrix<double>(nall, nall);
  
  arma::umat Hlocs = field_v_concatm(H_blockrow_tripls);
  arma::vec Hvals = field_v_concatm(H_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  tripletList_H.reserve(Hlocs.n_rows);
  for(int i=0; i<Hlocs.n_rows; i++){
    tripletList_H.push_back(T(Hlocs(i, 0), Hlocs(i, 1), Hvals(i)));
  }
  
  
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "setfromtriplets: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  PP_all = He.transpose() * He;
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "product 1: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> CC = He.block(n, 0, no, n);
  Eigen::SparseMatrix<double> DD = He.block(n, n, no, no);
  
  PP_o = DD.transpose() * DD;
  PP_ox = DD.transpose() * CC; 
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "product 2: " << timer << endl;
  
  // submatrices
  
  
  
  start = std::chrono::steady_clock::now();
  solver.analyzePattern(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "pattern: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  solver.factorize(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "factorize: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::MatrixXd we = armamat_to_matrixxd(w);
  Eigen::MatrixXd oxw = PP_ox * we;
  Eigen::MatrixXd pred_mean_e = -solver.solve(oxw);
  arma::mat pred_mean = matrixxd_to_armamat(pred_mean_e);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "solve & copy: " << timer << endl;
  
  return pred_mean;
}


arma::mat MGP::predict_via_precision_direct(const arma::vec& theta){
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] start 1" << endl;
  }
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  arma::uvec linear_order = arma::regspace<arma::uvec>(0, n-1);
  arma::uvec sortsort_order = linear_order; //
    //arma::sort_index(arma::sort_index(membership));
  
  linear_sort_map = arma::join_horiz(linear_order, sortsort_order);
  
  start = std::chrono::steady_clock::now();
  // K_cholcp_cache = invchol(Ri)
  arma::field<arma::umat> H_blockrow_tripls(4 * n_blocks); //H blocks, R blocks, same for predictions 
  arma::field<arma::mat> H_blockrow_vals(4 * n_blocks);
  
  arma::field<arma::umat> Ci_blockrow_tripls(2 * n_blocks);
  arma::field<arma::mat> Ci_blockrow_vals(2 * n_blocks);
  
  int h_ctr = 0;
  int ri_ctr = 0;
  arma::field<arma::mat> H(2 * n_blocks);
  arma::field<arma::mat> chRi(2 * n_blocks);
  arma::field<arma::mat> Ri(2 * n_blocks);
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] begin loop 1 " << endl;
  }
  
  for(int i=0; i<n_blocks; i++){
    if(parents_indexing(i).n_elem > 0){
      arma::mat Coo = Correlationf(io_xcoords, indexing(i), indexing(i), theta, false, true);
      arma::mat Cxx = Correlationf(io_xcoords, parents_indexing(i), parents_indexing(i), theta, false, true);
      arma::mat Cxxi = arma::inv_sympd(Cxx);
      arma::mat Cox = Correlationf(io_xcoords, indexing(i), parents_indexing(i), theta, false, false);
      
      H(i) = Cox * Cxxi;
      chRi(i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo - H(i) * Cox.t()), "lower"))));
    } else {
      arma::mat Coo = Correlationf(io_xcoords, indexing(i), indexing(i), theta, false, true);
      chRi(i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo), "lower"))));
    }
    Ri(i) = chRi(i).t() * chRi(i);
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] begin loop 2" << endl;
  }
  
  for(int i=0; i<n_blocks; i++){
    int pred_i = i+n_blocks;
    if((pred_indexing(i).n_elem > 0) & (pred_parents_indexing(i).n_elem > 0)){
      arma::mat Coo = Correlationf(io_xcoords, pred_indexing(i), pred_indexing(i), theta, false, true);
      arma::mat Cxx = Correlationf(io_xcoords, pred_parents_indexing(i), pred_parents_indexing(i), theta, false, true);
      arma::mat Cxxi = arma::inv_sympd(Cxx);
      arma::mat Cox = Correlationf(io_xcoords, pred_indexing(i), pred_parents_indexing(i), theta, false, false);
      H(pred_i) = Cox * Cxxi;
      chRi(pred_i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo - H(pred_i) * Cox.t()), "lower"))));
    } else {
      arma::mat Coo = Correlationf(io_xcoords, pred_indexing(i), pred_indexing(i), theta, false, true);
      chRi(pred_i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo), "lower"))));
    }
    Ri(pred_i) = chRi(pred_i).t() * chRi(pred_i);
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] begin loop 3" << endl;
  }
  
  for(int i=0; i<2*n_blocks; i++){
    int ui = i;//block_names(i) - 1;
    int ni, np;
    arma::uvec sorted_index, sorted_indexj, sorted_parents;
    arma::vec childreni, childrenj;
    arma::field<arma::field<arma::uvec> > u_is_which_i, u_is_which_j;
    
    if(i < n_blocks){
      ni = indexing(ui).n_elem;
      np = parents_indexing(ui).n_elem;
      sorted_index = indexing(ui);//arma::sort(sortsort_order(indexing(ui)));
      sorted_parents = parents_indexing(ui);
      childreni = children(ui);
      u_is_which_i = u_is_which_col_f(ui);
    } else {
      ni = pred_indexing(i - n_blocks).n_elem;
      np = pred_parents_indexing(i - n_blocks).n_elem;
      sorted_index = pred_indexing(i - n_blocks);
      sorted_parents = pred_parents_indexing(i - n_blocks);
      childreni = pred_children(i - n_blocks);
      u_is_which_i = pred_u_is_which_col_f(i - n_blocks);
    }
    
    arma::mat Hmod;
    // 
    if(ni > 0){
      if(np > 0){
        // locations to fill: indexing(ui) x parents_indexing(uj) 
        arma::umat H_tripl_locs(ni * np, 2);
        arma::mat H_tripl_val = arma::zeros(ni * np);
        
        //arma::mat Hmod = chRit * ((*param_data.w_cond_mean_K_ptr.at(ui)).slice(0));
        Hmod = chRi(i) * H(i);
        for(int ix=0; ix < ni; ix++){
          for(int jx=0; jx < np; jx++){
            int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
            H_tripl_locs(vecix, 0) = sorted_index(ix);//sortsort_order(indexing(ui)(ix));
            H_tripl_locs(vecix, 1) = sorted_parents(jx);//sortsort_order(parents_indexing(ui)(jx));
            H_tripl_val(vecix, 0) = -Hmod(ix, jx);
          }
        }
        H_blockrow_tripls(h_ctr) = H_tripl_locs;
        H_blockrow_vals(h_ctr) = H_tripl_val;
        h_ctr++;  
      }
      arma::umat H_tripl_locs(ni * ni, 2);
      arma::mat H_tripl_val = arma::zeros(ni * ni);
      for(int ix=0; ix < ni; ix++){
        for(int jx=0; jx < ni; jx++){
          if(chRi(i)(ix, jx) != 0){
            int vecix = arma::sub2ind(arma::size(ni, ni), ix, jx);
            H_tripl_locs(vecix, 0) = sorted_index(ix); //sortsort_order(indexing(ui)(ix));
            H_tripl_locs(vecix, 1) = sorted_index(jx); //sortsort_order(indexing(ui)(jx));
            H_tripl_val(vecix, 0) = chRi(i)(ix, jx);
          }
        }
      }
      H_blockrow_tripls(h_ctr) = H_tripl_locs;
      H_blockrow_vals(h_ctr) = H_tripl_val;
      h_ctr++;  
      
      /// get precision directly
      for(int j=i; (j<2*n_blocks)&(ni > 0); j++){
        // compute the upper triangular part of the precision matrix
        // col block name
        int uj = j;//block_names(j) - 1;
        int h = 0;
        
        int nj;
        if(j < n_blocks){
          nj = indexing(uj).n_elem;
          sorted_indexj = indexing(uj);
          childrenj = children(uj);
          u_is_which_j = u_is_which_col_f(uj);
        } else {
          nj = pred_indexing(j - n_blocks).n_elem;
          sorted_indexj = pred_indexing(j - n_blocks);
          childrenj = pred_children(j - n_blocks);
          u_is_which_j = pred_u_is_which_col_f(j - n_blocks);
        }
        if(nj > 0){
          arma::mat Ci_block;
          
          if(ui == uj){
            // block diagonal part
            Ci_block = Ri(ui);
            for(int c=0; c<childreni.n_elem; c++){
              int child = childreni(c);
              if(child != -1){
                arma::mat AK_u = H(child).cols(u_is_which_i(c)(0));
                Ci_block += AK_u.t() * Ri(child) * AK_u; 
              }
            }
            
            // locations to fill: indexing(ui) x indexing(uj) 
            arma::umat tripl_locs(ni * nj, 2);
            arma::mat tripl_val = arma::zeros(Ci_block.n_elem);
            
            for(int ix=0; ix < ni; ix++){
              for(int jx=0; jx < nj; jx++){
                int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
                tripl_locs(vecix, 0) = sorted_index(ix);//sortsort_order(indexing(ui)(ix));
                tripl_locs(vecix, 1) = sorted_indexj(jx);//ortsort_order(indexing(uj)(jx));
                tripl_val(vecix, 0) = Ci_block(ix, jx);
              }
            }
            
            //Rcpp::Rcout << ui << endl;
            //Rcpp::Rcout << arma::size(blockrow_tripls(i)) << " " << arma::size(tripl_locs) << endl;
            Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs);
            //Rcpp::Rcout << arma::size(blockrow_vals(i)) << " " << arma::size(tripl_val) << endl;
            Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val);
            //Rcpp::Rcout << "- " << endl;
            
          } else {
            bool nonempty=false;
            
            // is there anything between these two? 
            // they are either:
            // 1: ui is parent of uj 
            // 2: ui is child of uj <-- we're going by row so this should not be important?
            // 3: ui and uj have common child
            arma::uvec oneuv = arma::ones<arma::uvec>(1);
            arma::uvec ui_is_ujs_parent = arma::find(childreni == uj);
            //Rcpp::Rcout << parents(uj) << endl;
            if(ui_is_ujs_parent.n_elem > 0){
              nonempty = true;
              // ui is a parent of uj
              int c = ui_is_ujs_parent(0); // ui is uj's c-th parent
              //Rcpp::Rcout << " loc 5 b " << c << " " << arma::size(param_data.w_cond_mean_K(uj)) << endl;
              //Rcpp::Rcout << u_is_which_col_f(ui) << endl;
              arma::mat AK_u = H(uj).cols(u_is_which_i(c)(0));
              //Rcpp::Rcout << " loc 5 c " << endl;
              Ci_block = - AK_u.t() * Ri(uj);
            } else {
              // reference set commons
              // common children? in this case we can only have one common child
              arma::vec commons = arma::intersect(childrenj, childreni);
              commons = commons(arma::find(commons != -1));
              nonempty = false;
              for(int comc = 0; comc<commons.n_elem; comc++){
                nonempty = true;
                int child = commons(comc);
                arma::uvec find_ci = arma::find(childreni == child, 1, "first");
                int ci = find_ci(0);
                arma::uvec find_cj = arma::find(childrenj == child, 1, "first");
                int cj = find_cj(0);
                arma::mat AK_ui = H(child).cols(u_is_which_i(ci)(0));
                arma::mat AK_uj = H(child).cols(u_is_which_j(cj)(0));
                Ci_block = AK_ui.t() * Ri(child) * AK_uj; 
              } 
              // non reference set commons
              // only one possible common child
              
              // reference-nonreference cross commons
              // reference set ij has non-reference ij in common with non-reference (i-1,j) and (i,j-1)
            }
            
            if(nonempty){
              // locations to fill: indexing(ui) x indexing(uj) and the transposed lower block-triangular part
              arma::umat tripl_locs1(ni * nj, 2);
              arma::mat tripl_val1 = arma::vectorise(Ci_block);
              
              for(int jx=0; jx<nj; jx++){
                for(int ix=0; ix<ni; ix++){
                  int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
                  tripl_locs1(vecix, 0) = sorted_index(ix);
                  tripl_locs1(vecix, 1) = sorted_indexj(jx);
                }
              }
              
              Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs1);
              Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val1);
              
              arma::umat tripl_locs2(ni * nj, 2);
              arma::mat tripl_val2 = arma::vectorise(arma::trans(Ci_block));
              
              for(int jx=0; jx<nj; jx++){
                for(int ix=0; ix<ni; ix++){
                  int vecix = arma::sub2ind(arma::size(nj, ni), jx, ix);
                  tripl_locs2(vecix, 0) = sorted_indexj(jx);
                  tripl_locs2(vecix, 1) = sorted_index(ix);
                }
              }
              
              Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs2);
              Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val2);
            }
          }
        } // if observations
      } // inner block
    }
    
    
  
  }
  
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "predict loop storing: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  
  int nall = io_xcoords.n_rows;
  He = Eigen::SparseMatrix<double>(nall, nall);
  //Ri = Eigen::SparseMatrix<double>(nall, nall);
  
  arma::umat Hlocs = field_v_concatm(H_blockrow_tripls);
  arma::vec Hvals = field_v_concatm(H_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  tripletList_H.reserve(Hlocs.n_rows);
  for(int i=0; i<Hlocs.n_rows; i++){
    tripletList_H.push_back(T(Hlocs(i, 0), Hlocs(i, 1), Hvals(i)));
  }
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());

  Ciprediction = Eigen::SparseMatrix<double>(nall, nall);
  arma::umat Cilocs = field_v_concatm(Ci_blockrow_tripls);
  arma::vec Civals = field_v_concatm(Ci_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_Ci;
  tripletList_Ci.reserve(Cilocs.n_rows);
  for(int i=0; i<Cilocs.n_rows; i++){
    tripletList_Ci.push_back(T(Cilocs(i, 0), Cilocs(i, 1), Civals(i)));
  }
  Ciprediction.setFromTriplets(tripletList_Ci.begin(), tripletList_Ci.end());
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "setfromtriplets: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  PP_all = He.transpose() * He;
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "product 1: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> CC = He.block(n, 0, no, n);
  Eigen::SparseMatrix<double> DD = He.block(n, n, no, no);
  
  PP_o = DD.transpose() * DD;
  PP_ox = DD.transpose() * CC; 
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "product 2: " << timer << endl;
  
  // submatrices
  
  
  
  start = std::chrono::steady_clock::now();
  solver.analyzePattern(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "pattern: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  solver.factorize(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "factorize: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::MatrixXd we = armamat_to_matrixxd(w);
  Eigen::MatrixXd oxw = PP_ox * we;
  Eigen::MatrixXd pred_mean_e = -solver.solve(oxw);
  arma::mat pred_mean = matrixxd_to_armamat(pred_mean_e);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "solve & copy: " << timer << endl;
  
  return pred_mean;
}

*/
  
arma::mat MGP::predict_via_precision_part(const arma::vec& theta){
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] start 1" << endl;
  }
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;

  start = std::chrono::steady_clock::now();
  
  arma::field<arma::umat> H_blockrow_tripls(4 * n_blocks);
  arma::field<arma::mat> H_blockrow_vals(4 * n_blocks);
  
  arma::field<arma::umat> Ci_blockrow_tripls(2 * n_blocks);
  arma::field<arma::mat> Ci_blockrow_vals(2 * n_blocks);
  
  int h_ctr = 0;
  int ri_ctr = 0;
  arma::field<arma::mat> H(2 * n_blocks);
  arma::field<arma::mat> chRi(2 * n_blocks);
  arma::field<arma::mat> Ri(2 * n_blocks);
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] begin loop 1 " << endl;
  }
  
  for(int i=0; i<n_blocks; i++){
    if(parents_indexing(i).n_elem > 0){
      arma::mat Coo = Correlationf(io_xcoords, indexing(i), indexing(i), theta, false, true);
      arma::mat Cxx = Correlationf(io_xcoords, parents_indexing(i), parents_indexing(i), theta, false, true);
      arma::mat Cxxi = arma::inv_sympd(Cxx);
      arma::mat Cox = Correlationf(io_xcoords, indexing(i), parents_indexing(i), theta, false, false);
      
      H(i) = Cox * Cxxi;
      chRi(i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo - H(i) * Cox.t()), "lower"))));
    } else {
      arma::mat Coo = Correlationf(io_xcoords, indexing(i), indexing(i), theta, false, true);
      chRi(i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo), "lower"))));
    }
    Ri(i) = chRi(i).t() * chRi(i);
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] begin loop 2" << endl;
  }
  
  for(int i=0; i<n_blocks; i++){
    int pred_i = i+n_blocks;
    if((pred_indexing(i).n_elem > 0) & (pred_parents_indexing(i).n_elem > 0)){
      arma::mat Coo = Correlationf(io_xcoords, pred_indexing(i), pred_indexing(i), theta, false, true);
      arma::mat Cxx = Correlationf(io_xcoords, pred_parents_indexing(i), pred_parents_indexing(i), theta, false, true);
      arma::mat Cxxi = arma::inv_sympd(Cxx);
      arma::mat Cox = Correlationf(io_xcoords, pred_indexing(i), pred_parents_indexing(i), theta, false, false);
      H(pred_i) = Cox * Cxxi;
      chRi(pred_i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo - H(pred_i) * Cox.t()), "lower"))));
    } else {
      arma::mat Coo = Correlationf(io_xcoords, pred_indexing(i), pred_indexing(i), theta, false, true);
      chRi(pred_i) = (arma::inv(arma::trimatl(arma::chol(arma::symmatu(Coo), "lower"))));
    }
    Ri(pred_i) = chRi(pred_i).t() * chRi(pred_i);
  }
  
  if(verbose & debug){
    Rcpp::Rcout << "[predict_via_precision_direct] begin loop 3" << endl;
  }
  

  for(int i=0; i<2*n_blocks; i++){
    int ui = i;//block_names(i) - 1;
    int ni, np;
    arma::uvec *sorted_index, *sorted_indexj, *sorted_parents;
    arma::vec *childreni, *childrenj;
    arma::field<arma::field<arma::uvec> > *u_is_which_i, *u_is_which_j;
    
    if(i < n_blocks){
      ni = indexing(ui).n_elem;
      np = parents_indexing(ui).n_elem;
      sorted_index = &(indexing(ui));//arma::sort(sortsort_order(indexing(ui)));
      sorted_parents = &(parents_indexing(ui));
      childreni = &(children(ui));
      u_is_which_i = &(u_is_which_col_f(ui));
    } else {
      ni = pred_indexing(i - n_blocks).n_elem;
      np = pred_parents_indexing(i - n_blocks).n_elem;
      sorted_index = &pred_indexing(i - n_blocks);
      sorted_parents = &pred_parents_indexing(i - n_blocks);
      childreni = &pred_children(i - n_blocks);
      u_is_which_i = &pred_u_is_which_col_f(i - n_blocks);
    }
    
    //
    arma::umat H_tripl_locs;
    arma::mat Hmod, H_tripl_val;
    if(ni > 0){
      // build H
      //if(i >= n_blocks){
      if(np > 0){
        // locations to fill: indexing(ui) x parents_indexing(uj) 
        Hmod = chRi(i) * H(i);
        H_tripl_locs = arma::umat(ni * np, 2);
        H_tripl_val = arma::zeros(ni * np);
        for(int ix=0; ix < ni; ix++){
          for(int jx=0; jx < np; jx++){
            int loci = (*sorted_index)(ix); 
            int locj = (*sorted_parents)(jx); 
            int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
            H_tripl_locs(vecix, 0) = block_order_all(loci);
            H_tripl_locs(vecix, 1) = block_order_all(locj);
            H_tripl_val(vecix, 0) = -Hmod(ix, jx);
          }
        }
        H_blockrow_tripls(h_ctr) = H_tripl_locs;
        H_blockrow_vals(h_ctr) = H_tripl_val;
        h_ctr++;
      }
      
    
      H_tripl_locs = arma::umat(ni * ni, 2);
      H_tripl_val = arma::zeros(ni * ni);
      for(int ix=0; ix < ni; ix++){
        for(int jx=0; jx < ni; jx++){ // lower triangular
          int loci = (*sorted_index)(ix); 
          int locj = (*sorted_index)(jx); 
          if(chRi(i)(ix, jx) != 0){
            int vecix = arma::sub2ind(arma::size(ni, ni), ix, jx);
            H_tripl_locs(vecix, 0) = block_order_all(loci);
            H_tripl_locs(vecix, 1) = block_order_all(locj);
            H_tripl_val(vecix, 0) = chRi(i)(ix, jx);
          }
        }
      }
      H_blockrow_tripls(h_ctr) = H_tripl_locs;
      H_blockrow_vals(h_ctr) = H_tripl_val;
      h_ctr++;  

      /// get precision directly
      for(int j=i; (j<2*n_blocks)&(ni > 0); j++){
        // compute the upper triangular part of the precision matrix
        // col block name
        int uj = j;//block_names(j) - 1;
        int h = 0;
        int nj;
        if(j < n_blocks){
          nj = indexing(uj).n_elem;
          sorted_indexj = &indexing(uj);
          childrenj = &children(uj);
          u_is_which_j = &u_is_which_col_f(uj);
        } else {
          nj = pred_indexing(j - n_blocks).n_elem;
          sorted_indexj = &pred_indexing(j - n_blocks);
          childrenj = &pred_children(j - n_blocks);
          u_is_which_j = &pred_u_is_which_col_f(j - n_blocks);
        }
        if(nj > 0){
          arma::mat Ci_block;
          
          if((i >= n_blocks)&(ui == uj)){
            // block diagonal part
            Ci_block = Ri(ui);
            for(int c=0; c<(*childreni).n_elem; c++){
              int child = (*childreni)(c);
              if(child != -1){
                arma::mat AK_u = H(child).cols((*u_is_which_i)(c)(0));
                Ci_block += AK_u.t() * Ri(child) * AK_u; 
              }
            }
            // locations to fill: indexing(ui) x indexing(uj) 
            arma::umat tripl_locs(ni * nj, 2);
            arma::mat tripl_val = arma::zeros(Ci_block.n_elem);
            for(int ix=0; ix < ni; ix++){
              for(int jx=0; jx < nj; jx++){
                int loci = (*sorted_index)(ix);
                int locj = (*sorted_indexj)(jx);
                int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
                tripl_locs(vecix, 0) = loci;
                tripl_locs(vecix, 1) = locj;
                tripl_val(vecix, 0) = Ci_block(ix, jx);
              }
            }
            Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs);
            Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val);
          }
          
          if((ui != uj)){
            //Rcpp::Rcout << "ok " << endl;
            bool nonempty=false;
            // is there anything between these two? 
            // they are either:
            // 1: ui is parent of uj 
            // 2: ui is child of uj <-- we're going by row so this should not be important?
            // 3: ui and uj have common child
            arma::uvec oneuv = arma::ones<arma::uvec>(1);
            arma::uvec ui_is_ujs_parent = arma::find((*childreni) == uj, 1, "first");
            //Rcpp::Rcout << parents(uj) << endl;
            if(ui_is_ujs_parent.n_elem > 0){
              nonempty = true;
              // ui is a parent of uj
              int c = ui_is_ujs_parent(0); // ui is uj's c-th parent
              arma::mat AK_u = H(uj).cols((*u_is_which_i)(c)(0));
              Ci_block = - AK_u.t() * Ri(uj);
            } else {
              // reference set commons
              // common children? in this case we can only have one common child
              arma::vec commons = arma::intersect((*childrenj), (*childreni));
              commons = commons(arma::find(commons != -1));
              nonempty = false;
              
              if(commons.n_elem > 0){
                nonempty = true;
                int child = commons(0);
                arma::uvec find_ci = arma::find((*childreni) == child, 1, "first");
                int ci = find_ci(0);
                arma::uvec find_cj = arma::find((*childrenj) == child, 1, "first");
                int cj = find_cj(0);
                arma::mat AK_ui = H(child).cols((*u_is_which_i)(ci)(0));
                arma::mat AK_uj = H(child).cols((*u_is_which_j)(cj)(0));
                Ci_block = AK_ui.t() * Ri(child) * AK_uj; 
              } 
            }
            
            if(nonempty){
              // locations to fill: indexing(ui) x indexing(uj) and the transposed lower block-triangular part
              if((i >= n_blocks) & (j >= n_blocks)){
                arma::umat tripl_locs1(ni * nj, 2);
                arma::mat tripl_val1 = arma::vectorise(Ci_block);
                for(int jx=0; jx<nj; jx++){
                  for(int ix=0; ix<ni; ix++){
                    int loci =(*sorted_index)(ix);
                    int locj = (*sorted_indexj)(jx);
                    int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
                    tripl_locs1(vecix, 0) = loci;
                    tripl_locs1(vecix, 1) = locj;
                  }
                }
                Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs1);
                Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val1);
              }
              
              arma::umat tripl_locs2(ni * nj, 2);
              arma::mat tripl_val2 = arma::vectorise(arma::trans(Ci_block));
              for(int jx=0; jx<nj; jx++){
                for(int ix=0; ix<ni; ix++){
                  int vecix = arma::sub2ind(arma::size(nj, ni), jx, ix);
                  tripl_locs2(vecix, 0) = (*sorted_indexj)(jx);
                  tripl_locs2(vecix, 1) = (*sorted_index)(ix);
                }
              }
              Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs2);
              Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val2);
            }
            
          }
        } // if observations
      } // inner block
    }
  }
  
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(verbose & debug){
    Rcpp::Rcout << "predict loop storing: " << timer << endl;
  }
  
  start = std::chrono::steady_clock::now();
  
  int nall = io_xcoords.n_rows;
  
  Eigen::SparseMatrix<double> He, Ciprediction, PP_o, PP_ox, HH_o;
  
  He = Eigen::SparseMatrix<double>(nall, nall);
  Ciprediction = Eigen::SparseMatrix<double>(nall, nall);
  
  arma::umat Hlocs = field_v_concatm(H_blockrow_tripls);
  arma::vec Hvals = field_v_concatm(H_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  tripletList_H.reserve(Hlocs.n_rows);
  for(int i=0; i<Hlocs.n_rows; i++){
    tripletList_H.push_back(T(Hlocs(i, 0), Hlocs(i, 1), Hvals(i)));
  }
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  
  arma::umat Cilocs = field_v_concatm(Ci_blockrow_tripls);
  arma::vec Civals = field_v_concatm(Ci_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_Ci;
  tripletList_Ci.reserve(Cilocs.n_rows);
  for(int i=0; i<Cilocs.n_rows; i++){
    tripletList_Ci.push_back(T(Cilocs(i, 0), Cilocs(i, 1), Civals(i)));
  }
  Ciprediction.setFromTriplets(tripletList_Ci.begin(), tripletList_Ci.end());
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(verbose & debug){
    Rcpp::Rcout << "setfromtriplets: " << timer << endl;
  }
  
  // submatrices
  start = std::chrono::steady_clock::now();
  PP_o = Ciprediction.block(n, n, no, no);
  PP_ox = Ciprediction.block(n, 0, no, n);
  HH_o = He.block(n, n, no, no);

  Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > predsolver;
  predsolver.analyzePattern(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(verbose & debug){
    Rcpp::Rcout << "submat and pattern: " << timer << endl;
  }
  
  start = std::chrono::steady_clock::now();
  predsolver.factorize(PP_o);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(verbose & debug){
    Rcpp::Rcout << "factorize: " << timer << endl;
  }
  
  start = std::chrono::steady_clock::now();
  Eigen::MatrixXd we = armamat_to_matrixxd(w);
  Eigen::MatrixXd oxw = PP_ox * we;
  Eigen::MatrixXd pred_mean_e = -predsolver.solve(oxw);
  arma::mat pred_mean = matrixxd_to_armamat(pred_mean_e);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(verbose & debug){
    Rcpp::Rcout << "pred_mean solve & copy: " << timer << endl;
  }
  
  start = std::chrono::steady_clock::now();
  arma::mat wrnd = arma::randn(no);
  Eigen::MatrixXd wrnd_e = armamat_to_matrixxd(wrnd);
  Eigen::MatrixXd wrnd_Ht_e = HH_o.triangularView<Eigen::Lower>().solve(wrnd_e);
  arma::mat wrnd_Ht = matrixxd_to_armamat(wrnd_Ht_e);
  // reorder
  arma::uvec sub_order = block_order_all.subvec(n, n+no-1)-n;
  wrnd_Ht = wrnd_Ht.rows(sub_order);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(verbose & debug){
    Rcpp::Rcout << "H solve & copy & sample: " << timer << endl;
  }
  
  return pred_mean + wrnd_Ht;
}



#endif