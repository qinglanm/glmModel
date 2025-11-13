#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

//' Efficiently computes X' * W * X where W is a diagonal matrix.
 //'
 //' This function is used to optimize the matrix calculation in the
 //' Iteratively Reweighted Least Squares (IRLS) algorithm.
 //'
 //' @param X The design matrix (n x p).
 //' @param W_diag The diagonal elements of the weight matrix W (n x 1 vector).
 //' @return The symmetric matrix X'WX (p x p).
 //' @export
 //' @name arma_XTWX
 // [[Rcpp::export]]
 arma::mat arma_XTWX(const arma::mat& X, const arma::vec& W_diag) {

   // 1. Compute X_weighted = X * W_diag (element-wise multiplication across columns)
   // This effectively computes W * X because W is diagonal.
   // X.each_col() allows element-wise operation column by column.
   arma::mat X_weighted = X.each_col() % W_diag;

   // 2. Compute X' * X_weighted = X' * (W * X) = X'WX
   arma::mat XTWX = X.t() * X_weighted;

   return XTWX;
 }
