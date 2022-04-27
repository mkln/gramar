#include "covariance_lmc.h"

using namespace std;


// matern covariance with nu = p + 1/2, and p=0,1,2
void matern_halfint_inplace(arma::mat& res, 
                            const arma::mat& coords,
                            //const arma::mat& x, const arma::mat& y, 
                            const arma::uvec& ix, const arma::uvec& iy,
                            const double& phi, 
                            const double& sigmasq, const double& reparam,
                            bool same, int twonu){
  // 0 based indexing
  double sigmasq_reparam = sigmasq/reparam;
  
  if(same){
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i)); //x.row(i);
      for(unsigned int j=i; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j)); //y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          if(twonu == 1){
            res(i, j) = sigmasq_reparam * exp(-hphi);
          } else {
            if(twonu == 3){
              res(i, j) = sigmasq_reparam * exp(-hphi) * (1 + hphi);
            } else {
              if(twonu == 5){
                res(i, j) = sigmasq_reparam * (1 + hphi + hphi*hphi / 3.0) * exp(-hphi);
              }
            }
          }
        } else {
          res(i, j) = sigmasq_reparam;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i)); //x.row(i);
      for(unsigned int j=0; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j)); //y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          if(twonu == 1){
            res(i, j) = sigmasq_reparam * exp(-hphi);
          } else {
            if(twonu == 3){
              res(i, j) = sigmasq_reparam * exp(-hphi) * (1 + hphi);
            } else {
              if(twonu == 5){
                res(i, j) = sigmasq_reparam * (1 + hphi + hphi*hphi / 3.0) * exp(-hphi);
              }
            }
          }
        } else {
          res(i, j) = sigmasq_reparam;
        }
      }
    }
  }
  //return res;
}


// powered exponential nu<2
void powerexp_inplace(arma::mat& res, 
                      const arma::mat& coords,
                      const arma::uvec& ix, const arma::uvec& iy, 
                      const double& phi, const double& nu, const double& sigmasq, const double& reparam,
                      bool same){

  double sigmasq_reparam = sigmasq / reparam;
  
  if(same){
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=i; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hnuphi = pow(arma::norm(delta), nu) * phi;
        if(hnuphi > 0.0){
          res(i, j) = exp(-hnuphi) * sigmasq_reparam;
        } else {
          res(i, j) = sigmasq_reparam;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=0; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hnuphi = pow(arma::norm(delta), nu) * phi;
        if(hnuphi > 0.0){
          res(i, j) = exp(-hnuphi) * sigmasq_reparam;
        } else {
          res(i, j) = sigmasq_reparam;
        }
      }
    }
  }
  if(nu > 1.7){
    double nugg = 1e-6;
    res.diag() += nugg;  
  }
  
}


void kernelp_inplace(arma::mat& res,
                     const arma::mat& Xcoords, const arma::uvec& ind1, const arma::uvec& ind2, 
                     const arma::vec& theta, bool same){
  
  double sigmasq = theta(theta.n_elem-1);
  arma::rowvec kweights = arma::trans(theta.subvec(0, theta.n_elem-2));
  
  if(same){
    for(unsigned int i=0; i<ind1.n_elem; i++){
      arma::rowvec cri = Xcoords.row(ind1(i));
      for(unsigned int j=i; j<ind2.n_elem; j++){
        //arma::rowvec deltasq = kweights.t() % (cri - Xcoords.row(ind2(j)));
        //double weighted = sqrt(arma::accu(deltasq % deltasq));
        arma::rowvec delta = cri - Xcoords.row(ind2(j));
        double weighted = pow(sqrt(arma::accu(kweights % delta % delta)), 1.99);
        res(i, j) = sigmasq * exp(-weighted);// + (weighted == 0? 1e-6 : 0);
      }
    }
    res = arma::symmatu(res);
  } else {
    //int cc = 0;
    for(unsigned int i=0; i<ind1.n_elem; i++){
      arma::rowvec cri = Xcoords.row(ind1(i));
      for(unsigned int j=0; j<ind2.n_elem; j++){
        //arma::rowvec deltasq = kweights.t() % (cri - Xcoords.row(ind2(j)));
        //double weighted = sqrt(arma::accu(deltasq % deltasq));
        arma::rowvec delta = cri - Xcoords.row(ind2(j));
        double weighted = pow(sqrt(arma::accu(kweights % delta % delta)), 1.99);
        res(i, j) = sigmasq * exp(-weighted);// + (weighted == 0? 1e-6 : 0);
      }
    }
  }
}


arma::mat Correlationf(
    const arma::mat& coords,
    const arma::uvec& ix, const arma::uvec& iy,
    const arma::vec& theta,
    bool ps, bool same){
  // these are not actually correlation functions because they are reparametrized to have 
  // C(0) = 1/reparam
  arma::mat res = arma::zeros(ix.n_rows, iy.n_rows);
  
  if(false & (coords.n_cols == 2)){
    
      // exponential
      double phi = theta(0);
      double nu = theta(1);
      double sigmasq = theta(2);
      
      
      double reparam = 1.0; 
      
      if(ps){
        reparam = pow(phi, .0 + 1);
      }
      
      powerexp_inplace(res, coords, ix, iy, phi, nu, sigmasq, reparam, same);
      return res;
      
  } else {
    // p exposures, p+1 param
    
    kernelp_inplace(res, coords, ix, iy, theta, same);
    return res;
  }
}

//[[Rcpp::export]]
arma::mat Correlationc(
    const arma::mat& coordsx,
    const arma::mat& coordsy,
    const arma::vec& theta,
    bool ps, bool same){
  // inefficient
  
  if(same){
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    
    return Correlationf(coordsx, ix, ix, theta, ps, same);
  } else {
    arma::mat coords = arma::join_vert(coordsx, coordsy);
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    arma::uvec iy = arma::regspace<arma::uvec>(coordsx.n_rows, coords.n_rows-1);
    
    return Correlationf(coords, ix, iy, theta, ps, same);
  }
  
}


//[[Rcpp::export]]
arma::mat gpkernel(
    const arma::mat& coordsx,
    const arma::vec& theta){
  arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
  return Correlationf(coordsx, ix, ix, theta, false, true);
}


void CviaKron_invsympd_(arma::cube& CCi, 
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, bool ps){
  for(int j=0; j<k; j++){
    CCi.slice(j) = arma::inv_sympd( Correlationf(coords, indx, indx, 
                                    theta.col(j), ps, true) );
  }
}

void inv_det_via_chol(arma::mat& xinv, 
                      arma::mat& xchol,
                      arma::mat& xcholi, double& ldet, const arma::mat& x){
  xchol = arma::chol( arma::symmatu(x) , "lower");
  xcholi = arma::inv(arma::trimatl(xchol));
  ldet = arma::accu(log(xcholi.diag()));
  xinv = xcholi.t() * xcholi;
}

void inv_det_via_qr(arma::mat& xinv, double& ldet, const arma::mat& x){
  arma::mat Q;
  arma::mat R;
  arma::qr(Q, R, x);
  
  xinv = arma::symmatu(arma::inv(arma::trimatu(R)) * Q.t());
  ldet = - 0.5 * arma::accu(log(abs(R.diag())));
}

double CviaKron_HRi_(arma::cube& H, 
                     arma::cube& Ri, 
                     arma::cube& chR,
                     arma::cube& chRi,
                     arma::cube& Kppi, 
                     const arma::cube& Cxx,
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, bool ps){
  
  double logdet=0;
  for(int j=0; j<k; j++){
    arma::mat Rinverted, Rchol, Rcholi;
    if(indy.n_elem > 0){
      arma::mat Cxy = Correlationf(coords, indx, indy, 
                                   theta.col(j), ps, false);
      arma::mat Cyy_i = arma::inv_sympd( Correlationf(coords, indy, indy, 
                                                      theta.col(j), ps, true) );
      Kppi.slice(j) = Cyy_i;
      arma::mat Hloc = Cxy * Cyy_i;
      arma::mat Targmat = Cxx.slice(j) - Hloc * Cxy.t();
      
      // ? sometimes ill conditioned? -- condition number
      //if((matern.twonu > 1) || (theta.n_rows > 2)){
        //Targmat.diag() += 1e-10;
      //}
      // Rloc_ichol = arma::inv(arma::trimatl(arma::chol( arma::symmatu(Targmat) , "lower")));
      // logdet += arma::accu(log(Rloc_ichol.diag()));

      double temp_ldet = 0;
      //arma::mat Temp;
      //inv_det_via_qr(Rinverted, temp_ldet, Targmat);
      inv_det_via_chol(Rinverted, Rchol, Rcholi, temp_ldet, Targmat);
      
      logdet += temp_ldet;
    
      H.slice(j) = Hloc;
    } else {
      arma::mat Targmat = Cxx.slice(j);
      double temp_ldet = 0;
      //inv_det_via_qr(Rinverted, temp_ldet, Targmat);
      inv_det_via_chol(Rinverted, Rchol, Rcholi, temp_ldet, Targmat);
      logdet += temp_ldet;
    }
    Ri.slice(j) = Rinverted;//
    chR.slice(j) = Rchol;
    chRi.slice(j) = Rcholi;
  }
  return logdet;
}

void CviaKron_HRj_chol_bdiag(
    arma::cube& Hj, arma::mat& Rjchol, arma::cube& Kxxi,
    const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta, bool ps){

  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  Kxxi = arma::zeros(indy.n_elem, indy.n_elem, k);
  for(int j=0; j<k; j++){
    Kxxi.slice(j) = arma::inv_sympd( Correlationf(coords, indy, indy, 
                                     theta.col(j), ps, true) );
    for(unsigned int ix=0; ix<indx.n_rows; ix++){
      if(naix(ix) == 0){ // otherwise it's not missing
        arma::uvec indxi = oneuv * indx(ix);
        arma::mat Cxx = Correlationf(coords, indxi, indxi, 
                                     theta.col(j), ps, true);
        arma::mat Cxy = Correlationf(coords, indxi, indy,  
                                     theta.col(j), ps, false);
        arma::mat Hloc = Cxy * Kxxi.slice(j);
        
        Hj.slice(ix).row(j) = Hloc;
        double Rcholtemp = arma::conv_to<double>::from(
          Cxx - Hloc * Cxy.t() );
        Rcholtemp = Rcholtemp < 0 ? 0.0 : Rcholtemp;
        Rjchol(j, ix) = pow(Rcholtemp, .5); 
      }
    }
  }
}

