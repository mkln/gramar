#ifndef MGP_PREC
#define MGP_PREC

#include "meshedgp.h"

using namespace std;

void MGP::new_precision_matrix_product(MeshDataLMC& data){

  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  int n = coords.n_rows;
  int nk = n * k;
  
  arma::uvec linear_order = arma::regspace<arma::uvec>(0, n-1);
  arma::uvec sortsort_order = //linear_order; //
      arma::sort_index(arma::sort_index(membership));
  
  linear_sort_map = arma::join_horiz(linear_order, sortsort_order);
  
  start = std::chrono::steady_clock::now();
  // K_cholcp_cache = invchol(Ri)
  arma::field<arma::umat> H_blockrow_tripls(k * n_blocks);
  arma::field<arma::mat> H_blockrow_vals(k * n_blocks);
  arma::field<arma::umat> Ri_blockrow_tripls(k * n_blocks);
  arma::field<arma::mat> Ri_blockrow_vals(k * n_blocks);
  
  int h_ctr = 0;
  int ri_ctr = 0;
  
  for(int i=0; i<n_blocks; i++){
   int ui = i;//block_names(i) - 1;
   int ni = indexing(ui).n_elem;
   int np = parents_indexing(ui).n_elem;
     
     arma::uvec sorted_index = arma::sort(sortsort_order(indexing(ui)));
     
    for(int j=0; j<k; j++){
     if(np > 0){
       
       arma::uvec sorted_parents = arma::sort(sortsort_order(parents_indexing(ui)));
       
       // locations to fill: indexing(ui) x parents_indexing(uj) 
       arma::umat H_tripl_locs(ni * np, 2);
       arma::mat H_tripl_val = arma::zeros(ni * np);
       
       for(int ix=0; ix < ni; ix++){
         for(int jx=0; jx < np; jx++){
           int vecix = arma::sub2ind(arma::size(ni, np), ix, jx);
           H_tripl_locs(vecix, 0) = n * j + sorted_index(ix);//sortsort_order(indexing(ui)(ix));
           H_tripl_locs(vecix, 1) = n * j + sorted_parents(jx);//sortsort_order(parents_indexing(ui)(jx));
           H_tripl_val(vecix, 0) = (*data.w_cond_mean_K_ptr.at(ui))(ix, jx, j);
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
         Ri_tripl_locs(vecix, 0) = n * j + sorted_index(ix); //sortsort_order(indexing(ui)(ix));
         Ri_tripl_locs(vecix, 1) = n * j + sorted_index(jx); //sortsort_order(indexing(ui)(jx));
         Ri_tripl_val(vecix, 0) = (*data.w_cond_prec_ptr.at(ui))(ix, jx, j);
       }
     }
     Ri_blockrow_tripls(ri_ctr) = Ri_tripl_locs;
     Ri_blockrow_vals(ri_ctr) = Ri_tripl_val;
     ri_ctr++;
   }
  }

  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "loop storing: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  
  H = Eigen::SparseMatrix<double>(nk, nk);
  Ri = Eigen::SparseMatrix<double>(nk, nk);
  
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
  
  H.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  Ri.setFromTriplets(tripletList_Ri.begin(), tripletList_Ri.end());

  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "setfromtriplets: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> I_eig(nk, nk);
  I_eig.setIdentity();
  //Ci = (I_eig - H).triangularView<Eigen::Lower>().transpose() *
  //  Ri * (I_eig - H).triangularView<Eigen::Lower>();
  data.Citsqi = (I_eig - H).transpose() * Ri * (I_eig - H);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "product: " << timer << endl;

  
}

void MGP::new_precision_matrix_direct(MeshDataLMC& data){
  if(verbose & debug){
    Rcpp::Rcout << "[new_precision_matrix_direct] start" << endl;
  }
  // building the precision matrix directly:
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  int n = coords.n_rows;
  int nk = n * k;
  
  arma::uvec linear_order = arma::regspace<arma::uvec>(0, n-1);
  
  // use sort index to get "nice" block matrix for visualization
  sortsort_order = linear_order; //
    //arma::sort_index(arma::sort_index(membership));
    
  linear_sort_map = arma::join_horiz(linear_order, sortsort_order);
  
  start = std::chrono::steady_clock::now();
  // K_cholcp_cache = invchol(Ri)
  arma::field<arma::umat> Ci_blockrow_tripls(k * n_blocks);
  arma::field<arma::mat> Ci_blockrow_vals(k * n_blocks);
  
  for(int i=0; i<n_blocks; i++){
    int ui = i;//block_names(i) - 1;
    //Ci_blockrow_tripls(i) = arma::zeros<arma::umat>(0, 2);
    //Ci_blockrow_vals(i) = arma::zeros(0, 1);
    int ni = indexing(ui).n_elem;
    
    for(int j=i; j<n_blocks; j++){
      // compute the upper triangular part of the precision matrix
      // col block name
      int uj = j;//block_names(j) - 1;
      int nj = indexing(uj).n_elem;
      
      for(int h=0; h<k; h++){
        arma::mat Ci_block;
        
        if(ui == uj){
          // block diagonal part
          Ci_block = (*data.w_cond_prec_ptr.at(ui)).slice(h);
          
          for(int c=0; c<children(ui).n_elem; c++){
            int child = children(ui)(c);
            if(child != -1){
              arma::mat AK_u = (*data.w_cond_mean_K_ptr.at(child)).slice(h).cols(u_is_which_col_f(ui)(c)(0));
              Ci_block += AK_u.t() * (*data.w_cond_prec_ptr.at(child)).slice(h) * AK_u; 
            }
          }
          
          // locations to fill: indexing(ui) x indexing(uj) 
          arma::umat tripl_locs(indexing(ui).n_elem * indexing(uj).n_elem, 2);
          arma::mat tripl_val = arma::zeros(Ci_block.n_elem);
          
          for(int ix=0; ix<indexing(ui).n_elem; ix++){
            for(int jx=0; jx<indexing(uj).n_elem; jx++){
              int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
              tripl_locs(vecix, 0) = n * h + sortsort_order(indexing(ui)(ix));
              tripl_locs(vecix, 1) = n * h + sortsort_order(indexing(uj)(jx));
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
          arma::uvec ui_is_ujs_parent = arma::find(children(ui) == uj);
          //Rcpp::Rcout << parents(uj) << endl;
          if(ui_is_ujs_parent.n_elem > 0){
            nonempty = true;
            
            // ui is a parent of uj
            int c = ui_is_ujs_parent(0); // ui is uj's c-th parent
            //Rcpp::Rcout << " loc 5 b " << c << " " << arma::size(param_data.w_cond_mean_K(uj)) << endl;
            //Rcpp::Rcout << u_is_which_col_f(ui) << endl;
            arma::mat AK_u = (*data.w_cond_mean_K_ptr.at(uj)).slice(h).cols(u_is_which_col_f(ui)(c)(0));
            //Rcpp::Rcout << " loc 5 c " << endl;
            Ci_block = - AK_u.t() * (*data.w_cond_prec_ptr.at(uj)).slice(h);
          } else {
            // common children? in this case we can only have one common child
            arma::vec commons = arma::intersect(children(uj), children(ui));
            commons = commons(arma::find(commons != -1));
            if(commons.n_elem > 0){
              nonempty = true;
              
              int child = commons(0);
              arma::uvec find_ci = arma::find(children(ui) == child);
              int ci = find_ci(0);
              arma::uvec find_cj = arma::find(children(uj) == child);
              int cj = find_cj(0);
              arma::mat AK_ui = (*data.w_cond_mean_K_ptr.at(child)).slice(h).cols(u_is_which_col_f(ui)(ci)(0));
              arma::mat AK_uj = (*data.w_cond_mean_K_ptr.at(child)).slice(h).cols(u_is_which_col_f(uj)(cj)(0));
              Ci_block = AK_ui.t() * (*data.w_cond_prec_ptr.at(child)).slice(h) * AK_uj; 
              
            }
          }
          
          if(nonempty){
            // locations to fill: indexing(ui) x indexing(uj) and the transposed lower block-triangular part
            arma::umat tripl_locs1(ni * nj, 2);
            arma::mat tripl_val1 = arma::vectorise(Ci_block);
            
            for(int jx=0; jx<nj; jx++){
              for(int ix=0; ix<ni; ix++){
                int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
                tripl_locs1(vecix, 0) = n * h + sortsort_order(indexing(ui)(ix));
                tripl_locs1(vecix, 1) = n * h + sortsort_order(indexing(uj)(jx));
              }
            }
            
            Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs1);
            Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val1);
            
            arma::umat tripl_locs2(ni * nj, 2);
            arma::mat tripl_val2 = arma::vectorise(arma::trans(Ci_block));
            
            for(int jx=0; jx<nj; jx++){
              for(int ix=0; ix<ni; ix++){
                int vecix = arma::sub2ind(arma::size(nj, ni), jx, ix);
                tripl_locs2(vecix, 0) = n * h + sortsort_order(indexing(uj)(jx));
                tripl_locs2(vecix, 1) = n * h + sortsort_order(indexing(ui)(ix));
              }
            }
            
            Ci_blockrow_tripls(i) = arma::join_vert(Ci_blockrow_tripls(i), tripl_locs2);
            Ci_blockrow_vals(i) = arma::join_vert(Ci_blockrow_vals(i), tripl_val2);
          }
        }
      } // factor loop
    } // inner block
  } // outer block
  
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "loop storing: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  
  data.Citsqi = Eigen::SparseMatrix<double>(nk, nk);
  
  arma::umat Cilocs = field_v_concatm(Ci_blockrow_tripls);
  arma::vec Civals = field_v_concatm(Ci_blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_Ci;
  tripletList_Ci.reserve(Cilocs.n_rows);
  for(int i=0; i<Cilocs.n_rows; i++){
    tripletList_Ci.push_back(T(Cilocs(i, 0), Cilocs(i, 1), Civals(i)));
  }
  
  data.Citsqi.setFromTriplets(tripletList_Ci.begin(), tripletList_Ci.end());
  //Ci = Ci.selfadjointView<Eigen::Lower>();
  
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "setfromtriplets: " << timer << endl;
  
  if(verbose & debug){
    Rcpp::Rcout << "[new_precision_matrix_direct] end" << endl;
  }
}

void MGP::update_precision_matrix(MeshDataLMC& data){
  // building the precision matrix directly:
  if(verbose & debug){
    Rcpp::Rcout << "[update_precision_matrix] start " << endl;
  }
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  int n = coords.n_rows;
  int nk = n * k;
  
  start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int ui = i;//block_names(i) - 1;
    //Ci_blockrow_tripls(i) = arma::zeros<arma::umat>(0, 2);
    //Ci_blockrow_vals(i) = arma::zeros(0, 1);
    
    int ni = indexing(ui).n_elem;
    
    for(int j=i; j<n_blocks; j++){
      // compute the upper triangular part of the precision matrix
      // col block name
      int uj = j;//block_names(j) - 1;
      int nj = indexing(uj).n_elem;
      
      for(int h=0; h<k; h++){
        arma::mat Ci_block;
        
        if(ui == uj){
          // block diagonal part
          Ci_block = (*data.w_cond_prec_ptr.at(ui)).slice(0);
          for(int c=0; c<children(ui).n_elem; c++){
            int child = children(ui)(c);
            if(child != -1){
              arma::mat AK_u = (*data.w_cond_mean_K_ptr.at(child)).slice(h).cols(u_is_which_col_f(ui)(c)(0));
              Ci_block += AK_u.t() * (*data.w_cond_prec_ptr.at(child)).slice(h) * AK_u; 
            }
          }
          
          // locations to fill: indexing(ui) x indexing(uj) 
          for(int ix=0; ix < ni; ix++){
            for(int jx=0; jx < nj; jx++){
              int row_ix = n * h + sortsort_order(indexing(ui)(ix));
              int col_ix = n * h + sortsort_order(indexing(uj)(jx));
              data.Citsqi.coeffRef(row_ix, col_ix) = Ci_block(ix, jx);
            }
          }
        } else {
          bool nonempty=false;
          
          // is there anything between these two? 
          // they are either:
          // 1: ui is parent of uj 
          // 2: ui is child of uj <-- we're going by row so this should not be important?
          // 3: ui and uj have common child
          arma::uvec oneuv = arma::ones<arma::uvec>(1);
          arma::uvec ui_is_ujs_parent = arma::find(children(ui) == uj, 1, "first");
          //Rcpp::Rcout << parents(uj) << endl;
          if(ui_is_ujs_parent.n_elem > 0){
            nonempty = true;
            
            // ui is a parent of uj
            int c = ui_is_ujs_parent(0); // ui is uj's c-th parent
            //Rcpp::Rcout << " loc 5 b " << c << " " << arma::size(param_data.w_cond_mean_K(uj)) << endl;
            //Rcpp::Rcout << u_is_which_col_f(ui) << endl;
            arma::mat AK_u = (*data.w_cond_mean_K_ptr.at(uj)).slice(h).cols(u_is_which_col_f(ui)(c)(0));
            //Rcpp::Rcout << " loc 5 c " << endl;
            Ci_block = - AK_u.t() * (*data.w_cond_prec_ptr.at(uj)).slice(h);
          } else {
            // common children? in this case we can only have one common child
            arma::vec commons = arma::intersect(children(uj), children(ui));
            commons = commons(arma::find(commons != -1));
            if(commons.n_elem > 0){
              nonempty = true;
              
              int child = commons(0);
              arma::uvec find_ci = arma::find(children(ui) == child, 1, "first");
              int ci = find_ci(0);
              arma::uvec find_cj = arma::find(children(uj) == child, 1, "first");
              int cj = find_cj(0);
              arma::mat AK_ui = (*data.w_cond_mean_K_ptr.at(child)).slice(h).cols(u_is_which_col_f(ui)(ci)(0));
              arma::mat AK_uj = (*data.w_cond_mean_K_ptr.at(child)).slice(h).cols(u_is_which_col_f(uj)(cj)(0));
              Ci_block = AK_ui.t() * (*data.w_cond_prec_ptr.at(child)).slice(h) * AK_uj; 
            }
          }
          
          if(nonempty){
            // locations to fill: indexing(ui) x indexing(uj) and the transposed lower block-triangular part
            arma::umat tripl_locs1(ni * nj, 2);
            arma::mat tripl_val1 = arma::vectorise(Ci_block);
            //arma::mat tripl_val2 = arma::vectorise(arma::trans(Ci_block));
            
            for(int jx=0; jx<nj; jx++){
              for(int ix=0; ix<ni; ix++){
                int row_ix = n * h + sortsort_order(indexing(ui)(ix));
                int col_ix = n * h + sortsort_order(indexing(uj)(jx));
                int vecix = arma::sub2ind(arma::size(ni, nj), ix, jx);
                data.Citsqi.coeffRef(row_ix, col_ix) = tripl_val1(vecix);
                data.Citsqi.coeffRef(col_ix, row_ix) = tripl_val1(vecix);
              }
            }
          }
        }
        
      } // factor loop
    } // inner loop
  } // outer loop

  
  
  if(verbose & debug){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[update_precision_matrix] end " << 
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << endl;
  }
}


void MGP::solver_initialize(){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  
  new_precision_matrix_direct(param_data);
  new_precision_matrix_direct(alter_data);
  
  // single pattern forever
  start = std::chrono::steady_clock::now();
  solver.analyzePattern(param_data.Citsqi);
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "analyzePattern: " << timer << endl;
}

void MGP::collapsed_logdensity(MeshDataLMC& data, const Eigen::MatrixXd& yXBe, 
                               const Eigen::MatrixXd& Xe,
                               const arma::vec& tausq_inv){
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  
  data.Citsqi.diagonal().array() += tausq_inv(0);
  
  start = std::chrono::steady_clock::now();
  solver.factorize(data.Citsqi);
  data.Citsqi_ldet = solver.logDeterminant();
  end = std::chrono::steady_clock::now();
  double timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "factorize: " << timer << endl;
  
  start = std::chrono::steady_clock::now();
  Eigen::MatrixXd Citsqi_solveX_e = solver.solve(Xe);
  Eigen::MatrixXd Citsqi_solveyXB_e = solver.solve(yXBe);
  data.Citsqi_solveX = matrixxd_to_armamat(Citsqi_solveX_e);
  data.Citsqi_solveyXB = matrixxd_to_armamat(Citsqi_solveyXB_e);
  end = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  Rcpp::Rcout << "solve: " << timer << endl;
  
  // get logdensity
  double ldetCi = arma::accu(data.logdetCi_comps) * 2.0;
  double ldet_Ctsq_i = coords.n_rows * 1.0/tausq_inv(0) - ldetCi  + data.Citsqi_ldet;

  arma::mat yXB = matrixxd_to_armamat(yXBe);
  
  double yXB_yXB = arma::conv_to<double>::from(yXB.t() * yXB);
  double yXB_qform = arma::conv_to<double>::from(yXB.t() * data.Citsqi_solveyXB);
  data.collapsed_ldens = -0.5 * ldet_Ctsq_i - 0.5 * tausq_inv(0) * 
    yXB_yXB - pow(tausq_inv(0), 2.0) * yXB_qform;
}

bool MGP::get_collapsed_logdens_comps(MeshDataLMC& data, 
                                      const Eigen::MatrixXd& yXBe,
                                      const Eigen::MatrixXd& Xe,
                                      const arma::vec& tausq_inv){
  
  bool acceptable = refresh_cache(data);
  if(acceptable){
    acceptable = calc_mgplogdens(data);
    update_precision_matrix(data);
    collapsed_logdensity(data, yXBe, Xe, tausq_inv);
    return acceptable;
  } else {
    return acceptable;
  }
}

void MGP::metrop_theta_collapsed(const Eigen::MatrixXd& yXBe, const Eigen::MatrixXd& Xe, 
                                 const arma::vec& tausq_inv){
  if(verbose & debug){
    Rcpp::Rcout << "[metrop_theta_collapsed] start\n";
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
  
  bool acceptable = get_collapsed_logdens_comps(param_data, yXBe, Xe, tausq_inv);
  acceptable = get_collapsed_logdens_comps(alter_data, yXBe, Xe, tausq_inv);
  
  bool accepted = false;
  double logaccept = 0;
  double current_loglik = 0;
  double new_loglik = 0;
  double prior_logratio = 0;
  double jacobian = 0;
  
  if(acceptable){
    new_loglik = alter_data.collapsed_ldens;
    
    current_loglik = param_data.collapsed_ldens;
    
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


#endif