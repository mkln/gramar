#define ARMA_DONT_PRINT_ERRORS

#ifndef XCOV_LMC 
#define XCOV_LMC

#ifdef _OPENMP
#include <omp.h>
#endif


#include <RcppArmadillo.h>

using namespace std;

/*
struct MaternParams {
  bool using_ps;
  bool estimating_nu;
  double * bessel_ws;
  int twonu;
};*/

// matern
//arma::mat matern(const arma::mat& x, const arma::mat& y, const double& phi, const double& nu, double * bessel_ws, bool same);


arma::mat Correlationf(const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const arma::vec& theta, bool ps, bool same);

arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, 
                       const arma::vec& theta, bool ps, bool same);

// inplace functions
void CviaKron_invsympd_(arma::cube& CCi, 
                        const arma::mat& coords, const arma::uvec& indx, 
                        int k, const arma::mat& theta, bool ps);
  
double CviaKron_HRi_(arma::cube& H, arma::cube& Ri, arma::cube& Kppi, 
                     const arma::cube& Cxx,
                     const arma::mat& coords, 
                     const arma::uvec& indx, const arma::uvec& indy, 
                     int k, const arma::mat& theta, bool ps);

//double CviaKron_invsympd_wdet_(arma::cube& res,
//                         const arma::mat& coords, const arma::uvec& indx, 
//                         int k, const arma::mat& theta, bool ps);


void CviaKron_HRj_chol_bdiag(
    arma::cube& Hj, arma::mat& Rjchol, arma::cube& Kxxi_parents,
    const arma::uvec& naix,
    const arma::mat& coords, const arma::uvec& indx, const arma::uvec& indy, 
    int k, const arma::mat& theta, bool ps);

void inv_det_via_chol(arma::mat& xinv, double& ldet, const arma::mat& x);

#endif
