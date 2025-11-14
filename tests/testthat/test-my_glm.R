library(testthat)
library(glmModel)

# -------------------- Test 1: Validate Numerical Correctness --------------------
test_that("my_glm coefficients and V-Cov matrix match base R glm", {
  # VITAL FIX: Data Setup moved inside the block
  set.seed(625)
  n <- 200
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  X3 <- sample(0:1, n, replace = TRUE)
  LP <- 0.5 + 1.2 * X1 - 0.8 * X2 + 0.6 * X3
  Prob <- 1 / (1 + exp(-LP))
  Y <- rbinom(n, 1, Prob)
  df <- data.frame(Y, X1, X2, X3)

  # Run your model and the benchmark model
  my_model <- my_glm(Y ~ X1 + X2 + X3, data = df)
  base_model <- glm(Y ~ X1 + X2 + X3, data = df, family = binomial(link = "logit"))

  # Validate Coefficients
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

# -------------------- Test 2: Validate Input Validation (Error Handling) --------------------
test_that("my_glm stops on non-binary response and unsupported family", {
  # VITAL FIX: Data Setup moved inside the block
  set.seed(625)
  n <- 200
  df <- data.frame(Y = rbinom(n, 1, 0.5), X1 = rnorm(n))

  # 1. Non-binary response test
  df_bad_y <- df
  df_bad_y$Y[1] <- 5
  expect_error(
    my_glm(Y ~ X1, data = df_bad_y),
    "Response variable must be binary \\(0/1\\)."
  )

  # 2. Check for binomial family support only (already correctly uses df)
  expect_error(
    my_glm(Y ~ X1, data = df, family = "gaussian"),
    "Only binomial family is currently supported."
  )
})

# -------------------- Test 3: Validate Convergence/Boundary Conditions --------------------
test_that("my_glm handles non-convergence gracefully", {
  # VITAL FIX: Data Setup moved inside the block
  set.seed(625)
  n <- 200
  df <- data.frame(Y = rbinom(n, 1, 0.5), X1 = rnorm(n), X2 = rnorm(n), X3 = sample(0:1, n, replace = TRUE))

  # Set a very small max_iter to ensure non-convergence
  my_model_no_conv <- my_glm(Y ~ X1 + X2 + X3, data = df, max_iter = 1)

  # Expect the 'converged' flag to be FALSE
  expect_false(my_model_no_conv$converged)
  # Expect the number of iterations to equal the max limit
  expect_equal(my_model_no_conv$iterations, 1)
})

# -------------------- Test 4: Testing Family/Link --------------------
test_that("Input validation catches unsupported family/link", {
  # VITAL FIX: Data Setup moved inside the block
  set.seed(625)
  n <- 50
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

# -------------------- Test 5: Summary/Print VCV Singular Warning --------------------
test_that("summary.my_glm handles singular VCV matrix", {
  # Data is NOT needed here as it uses a manually constructed dummy_model
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
  expect_warning({
    s_output <- summary(dummy_model)
  }, "Singular VCV matrix detected. Standard errors and p-values may be unreliable.")

  # Check output structure
  expect_true(all(is.na(s_output$coefficients[, "Std. Error"])))

  # Check print.my_glm logic
  expect_output(print(dummy_model), "Coefficients:")
  expect_output(print(dummy_model), "IRLS converged in 5 iterations")
})

# -------------------- Test 6: Formula and Data Coverage --------------------
test_that("my_glm catches incorrect input types (formula/data frame)", {
  # Data is NOT needed here as the error is thrown before data is processed fully

  # 1. Coverage formula check
  expect_error(my_glm("Y ~ X1", data = data.frame(Y=0,X1=1)),
               "formula must be a valid R formula.",
               fixed = TRUE)

  # 2. Coverage data check
  df_matrix <- matrix(1:4, ncol = 2)
  expect_error(my_glm(V1 ~ V2, data = df_matrix),
               "data must be a data frame.",
               fixed = TRUE)
})

# -------------------- Test 7: Summary Coverage (Normal Calculation) --------------------
test_that("Summary method executes normal SE/z/p-value calculation (non-singular VCV)", {
  # VITAL FIX: Define df_small here to avoid CI failure (This was the problem line 131)
  set.seed(625)
  n <- 50
  df_small <- data.frame(Y = rbinom(n, 1, 0.5), X1 = rnorm(n), X2 = rnorm(n))

  # Run model
  model <- my_glm(Y ~ X1 + X2, data = df_small)

  # Run summary and check if SE/z/pval exist (covers lines 201-203)
  s_output <- summary(model)

  # Check if Std. Error exists and it is not NA/NaN
  expect_true(all(!is.na(s_output$coefficients[, "Std. Error"])))

  # Check if z value and Pr(>|z|) exist
  expect_true("z value" %in% colnames(s_output$coefficients))
})
