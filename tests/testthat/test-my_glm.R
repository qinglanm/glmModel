library(testthat)
library(glmModel) # Ensure this is your package name

# -------------------- Test Data Setup --------------------
# Generate data once for all tests
set.seed(625)
n <- 200
X1 <- rnorm(n)
X2 <- rnorm(n)
X3 <- sample(0:1, n, replace = TRUE)
LP <- 0.5 + 1.2 * X1 - 0.8 * X2 + 0.6 * X3
Prob <- 1 / (1 + exp(-LP))
Y <- rbinom(n, 1, Prob)
df <- data.frame(Y, X1, X2, X3)

# -------------------- Test 1: Validate Numerical Correctness --------------------
test_that("my_glm coefficients and V-Cov matrix match base R glm", {
  # Run your model and the benchmark model
  my_model <- my_glm(Y ~ X1 + X2 + X3, data = df)
  base_model <- glm(Y ~ X1 + X2 + X3, data = df, family = binomial(link = "logit"))

  # Validate Coefficients: Ignore name attribute differences, only compare values
  expect_equal(
    as.vector(my_model$coefficients),
    as.vector(coef(base_model)),
    tolerance = 1e-4
  )

  # Validate Variance-Covariance Matrix (V-Cov)
  expect_equal(
    my_model$v_cov,
    vcov(base_model),
    tolerance = 1e-4
  )
})

# -------------------- Test 2: Validate Input Validation --------------------
test_that("my_glm stops on non-binary response", {
  # Create a non-binary (non 0/1) response variable dataframe
  df_bad_y <- df
  df_bad_y$Y[1] <- 5

  # Expect the function to stop (error) and throw the correct message
  expect_error(
    my_glm(Y ~ X1, data = df_bad_y),
    "Response variable must be binary \\(0/1\\)." # This string must match the error message from your function
  )

  # Check for binomial family support only
  expect_error(
    my_glm(Y ~ X1, data = df, family = "gaussian"),
    "Only binomial family is currently supported."
  )
})

# -------------------- Test 3: Validate Convergence/Boundary Conditions --------------------
test_that("my_glm handles non-convergence gracefully", {
  # Set a very small max_iter to ensure non-convergence
  my_model_no_conv <- my_glm(Y ~ X1 + X2 + X3, data = df, max_iter = 1)

  # Expect the 'converged' flag to be FALSE
  expect_false(my_model_no_conv$converged)
  # Expect the number of iterations to equal the max limit
  expect_equal(my_model_no_conv$iterations, 1)
})

# -------------------- Test 4: Testing Family/Link --------------------
test_that("Input validation catches unsupported family/link", {
  # Load test data (assuming df is in your environment or defined here)
  set.seed(625)
  n <- 50 # Small data for quick test
  df_small <- data.frame(Y = rbinom(n, 1, 0.5), X1 = rnorm(n))

  # if (family != "binomial")
  expect_error(my_glm(Y ~ X1, data = df_small, family = "poisson"),
               "Only binomial family is currently supported.",
               fixed = TRUE)

  # if (link != "logit")
  expect_error(my_glm(Y ~ X1, data = df_small, link = "probit"),
               "Only logit link is currently supported.",
               fixed = TRUE)
})

# -------------------- Test 5: Testing Covergence --------------------
test_that("summary.my_glm handles singular VCV matrix", {
  # Create a dummy my_glm object with invalid v_cov to trigger the warning path
  dummy_model <- list(
    coefficients = c(Int = 1, X = 0.5),
    v_cov = matrix(c(1, 0, 0, -1), 2, 2, dimnames = list(c("(Intercept)", "X"), c("(Intercept)", "X"))),
    logLik = -10,
    AIC = 20,
    iterations = 5,
    converged = TRUE,
    formula = Y ~ X # Need formula for print()
  )
  class(dummy_model) <- "my_glm"

  # The summary should issue a warning and still produce a table with NA/NaN
  # (if check)
  expect_warning({
    s_output <- summary(dummy_model)
  }, "Singular VCV matrix detected. Standard errors and p-values may be unreliable.")

  # Check output structure
  expect_true(all(is.na(s_output$coefficients[, "Std. Error"])))

  # Check print.my_glm logic
  expect_output(print(dummy_model), "Coefficients:")
  expect_output(print(dummy_model), "IRLS converged in 5 iterations")
})
