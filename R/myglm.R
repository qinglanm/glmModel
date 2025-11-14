#' @title Fit a Simplified Generalized Linear Model (Logistic Regression)
#'
#' @description
#' Implements the Iteratively Reweighted Least Squares (IRLS) algorithm to fit
#' a Binary Logistic Regression model. **The core matrix algebra (X'WX) is optimized
#' using RcppArmadillo for significant performance gains.**
#'
#' @details
#' The function calculates coefficients by iteratively solving the weighted least
#' squares problem: \eqn{\beta^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} Z^{(t)}}.
#' **The calculation of \eqn{X^T W^{(t)} X} is delegated to an internal C++ function
#' powered by the Armadillo linear algebra library.**
#' Includes input validation and stabilization to prevent probabilities from reaching 0 or 1.
#'
#' @param formula An object of class \code{"formula"} (e.g., \code{y ~ x1 + x2}).
#' @param data A \code{data.frame} containing the variables in the model.
#' @param family A character string specifying the distribution family. Currently only \code{"binomial"} is supported.
#' @param link A character string specifying the link function. Currently only \code{"logit"} is supported.
#' @param tolerance A numeric value for the convergence tolerance (\code{max(|beta_new - beta|)}).
#' @param max_iter An integer specifying the maximum number of IRLS iterations.
#' @return A list of class \code{"my_glm"} containing the following components:
#' \item{coefficients}{The estimated model coefficients.}
#' \item{v_cov}{The estimated variance-covariance matrix of the coefficients.}
#' \item{iterations}{The number of IRLS iterations performed.}
#' \item{converged}{Logical, TRUE if the IRLS algorithm converged within \code{max_iter}.}
#' \item{fitted.values}{The fitted mean values on the response scale (\eqn{\mu}).}
#' \item{linear.predictors}{The fitted linear predictors (\eqn{X\beta}).}
#' \item{residuals}{The raw residuals (\eqn{Y - \mu}).}
#' \item{logLik}{The final log-likelihood value for the fitted model.}
#' \item{AIC}{The calculated Akaike Information Criterion.}
#' \item{formula}{The model formula.}
#' \item{family}{The distribution family used.}
#' \item{link}{The link function used.}
#'
#' @seealso \code{\link[stats]{glm}}, \code{\link[stats]{binomial}}
#'
#' @importFrom stats model.frame model.response model.matrix pnorm
#' @importFrom Rcpp sourceCpp
#' @useDynLib glmModel, .registration = TRUE
#' @export
#'
#' @examples
#' set.seed(625)
#' n <- 200
#' X1 <- rnorm(n)
#' X2 <- rnorm(n)
#' X3 <- sample(0:1, n, replace = TRUE)
#' LP <- 0.5 + 1.2 * X1 - 0.8 * X2 + 0.6 * X3
#' Prob <- 1 / (1 + exp(-LP))
#' Y <- rbinom(n, 1, Prob)
#' df <- data.frame(Y, X1, X2, X3)
#'
#' # Fit logistic regression using custom IRLS
#' my_model <- my_glm(Y ~ X1 + X2 + X3, data = df)
#'
#' # View detailed summary
#' summary(my_model)
#'
#' # Compare with base R glm
#' base_model <- glm(Y ~ X1 + X2 + X3, data = df, family = binomial(link = "logit"))
#'
#' # Check for correctness (used in the vignette)
#' all.equal(my_model$coefficients, coef(base_model), tolerance = 1e-4)
my_glm <- function(formula, data, family = "binomial", link = "logit",
                   tolerance = 1e-6, max_iter = 25) {
  #-----------------------------------------------------------
  # 1. Input checks
  #-----------------------------------------------------------
  if (!inherits(formula, "formula")) stop("formula must be a valid R formula.")
  if (!is.data.frame(data)) stop("data must be a data frame.")
  if (family != "binomial") stop("Only binomial family is currently supported.")
  if (link != "logit") stop("Only logit link is currently supported.")

  #-----------------------------------------------------------
  # 2. Data preparation
  #-----------------------------------------------------------
  mf <- model.frame(formula = formula, data = data)
  Y <- model.response(mf)
  X <- model.matrix(formula, data = mf)
  n <- nrow(X)
  p <- ncol(X)

  # check response
  if (any(!Y %in% c(0, 1))) stop("Response variable must be binary (0/1).")

  #-----------------------------------------------------------
  # 3. Initialize parameters
  #-----------------------------------------------------------
  beta <- matrix(0, p, 1)    # start from zeros
  converged <- FALSE         # track convergence

  #-----------------------------------------------------------
  # 4. IRLS algorithm (Rcpp Optimized)
  #-----------------------------------------------------------
  for (i in 1:max_iter) {
    # Linear predictor
    eta <- X %*% beta

    # Inverse logit transformation (mu = 1 / (1 + exp(-eta)))
    mu <- 1 / (1 + exp(-eta))

    # Stabilization
    mu <- pmin(pmax(mu, 1e-8), 1 - 1e-8)

    # Weights (W_diag) and Working Response (Z)
    W_diag <- as.vector(mu * (1 - mu))
    Z <- eta + (Y - mu) / W_diag

    # 1. OPTIMIZATION: Use Rcpp for X'WX (most computationally intensive part)
    XT_W_X <- arma_XTWX(X, W_diag)

    # 2. Optimized R for X'WZ (avoids slow diag(W_diag) matrix creation)
    XT_W_Z <- t(X) %*% (W_diag * Z)

    # Weighted least squares update
    beta_new <- solve(XT_W_X, XT_W_Z)

    # Check for convergence
    if (max(abs(beta_new - beta)) < tolerance) {
      converged <- TRUE
      beta <- beta_new
      break
    }
    beta <- beta_new
  }

  #-----------------------------------------------------------
  # 5. Post-estimation calculations
  #-----------------------------------------------------------
  # Ensure final calculations use the converged beta
  if (!converged) {
    # Recalculate based on final beta (i.e., beta_new from last iteration)
    eta <- X %*% beta
    mu <- 1 / (1 + exp(-eta))
    mu <- pmin(pmax(mu, 1e-8), 1 - 1e-8)
    W_diag <- as.vector(mu * (1 - mu))
    XT_W_X <- arma_XTWX(X, W_diag)
  }

  v_cov <- solve(XT_W_X)  # variance-covariance matrix
  # name V-Cov's row and column
  colnames(v_cov) <- colnames(X)
  rownames(v_cov) <- colnames(X)
  fitted <- mu
  residuals <- Y - mu

  # Log-likelihood for binomial model: sum(y*log(mu) + (1-y)*log(1-mu))
  loglik <- sum(Y * log(as.vector(fitted)) + (1 - Y) * log(1 - as.vector(fitted)))
  p_eff <- p # Effective degrees of freedom = number of coefficients
  aic <- -2 * loglik + 2 * p_eff

  coefficients_vec <- as.vector(beta)
  names(coefficients_vec) <- colnames(X)
  #-----------------------------------------------------------
  # 6. Output result
  #-----------------------------------------------------------
  result <- list(
    coefficients = coefficients_vec,
    v_cov = v_cov,
    iterations = i,
    converged = converged,
    fitted.values = fitted,
    linear.predictors = eta,
    residuals = residuals,
    logLik = loglik,
    AIC = aic,
    formula = formula,
    family = family,
    link = link
  )
  class(result) <- "my_glm"
  return(result)
}

#-----------------------------------------------------------
# 7. Summary method (S3 Generic)
#-----------------------------------------------------------
#' @export
summary.my_glm <- function(object, ...) {

  # Check if v_cov is valid before computing SE
  if (any(is.na(diag(object$v_cov))) || any(diag(object$v_cov) <= 0)) {
    warning("Singular VCV matrix detected. Standard errors and p-values may be unreliable.")
    se   <- rep(NA, length(object$coefficients))
    z    <- rep(NA, length(object$coefficients))
    pval <- rep(NA, length(object$coefficients))
  } else {
    se   <- sqrt(diag(object$v_cov))
    z    <- object$coefficients / se
    pval <- 2 * (1 - pnorm(abs(z)))
  }

  coef_tab <- cbind(
    Estimate    = object$coefficients,
    `Std. Error` = se,
    `z value`    = z,
    `Pr(>|z|)`   = pval
  )

  cat("Call:\n")
  print(object$formula)

  cat("\nCoefficients:\n")
  print(round(coef_tab, 4))

  cat("\n--- Model Statistics ---\n")
  cat("Log-likelihood:", round(object$logLik, 4), "\n")
  cat("AIC:",           round(object$AIC, 4),     "\n")
  cat("IRLS Iterations:", object$iterations,      "\n")
  cat("Converged:",      object$converged,        "\n")

  invisible(list(
    coefficients = coef_tab,
    logLik = object$logLik,
    AIC = object$AIC,
    converged = object$converged
  ))
}

#-----------------------------------------------------------
# 8. Print method (S3 Generic)
#-----------------------------------------------------------
#' @export
print.my_glm <- function(x, ...) {

  cat("Call:\n")
  print(x$formula)

  cat("\nCoefficients:\n")
  print(x$coefficients)

  cat(paste0(
    "\nIRLS converged in ", x$iterations,
    " iterations. Converged: ", x$converged, "\n"
  ))

  invisible(x)
}
