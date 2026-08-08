# Secretary Bird Optimization Algorithm (SBOA) for MLP - Original R Code
# Reference: https://github.com/burakdilber/MLP-SBOA-Article-Source-Code

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

forward_pass <- function(weights, biases, X) {
  W1 <- weights$W1
  W2 <- weights$W2
  b1 <- biases$b1
  b2 <- biases$b2
  
  Z1 <- X %*% W1 + matrix(b1, nrow = nrow(X), ncol = length(b1), byrow = TRUE)
  A1 <- sigmoid(Z1)
  Z2 <- A1 %*% W2 + matrix(b2, nrow = nrow(A1), ncol = length(b2), byrow = TRUE)
  Y_hat <- sigmoid(Z2)
  
  list(A1 = A1, Y_hat = Y_hat)
}

mse_loss <- function(weights, biases, X, Y) {
  forward <- forward_pass(weights, biases, X)
  Y_hat <- forward$Y_hat
  mean((Y - Y_hat)^2)
}

fitness_function <- function(params, X_train, Y_train, input_dim, hidden_dim, output_dim) {
  W1 <- matrix(params[1:(input_dim * hidden_dim)], nrow = input_dim, ncol = hidden_dim)
  b1 <- params[(input_dim * hidden_dim + 1):(input_dim * hidden_dim + hidden_dim)]
  W2 <- matrix(params[(input_dim * hidden_dim + hidden_dim + 1):(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim)],
               nrow = hidden_dim, ncol = output_dim)
  b2 <- params[(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + 1):length(params)]
  
  weights <- list(W1 = W1, W2 = W2)
  biases <- list(b1 = b1, b2 = b2)
  
  mse_loss(weights, biases, X_train, Y_train)
}

SBOA <- function(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness) {
  lowerbound <- rep(lowerbound, dimension)
  upperbound <- rep(upperbound, dimension)
  
  X <- matrix(0, SearchAgents, dimension)
  for (i in 1:dimension) {
    X[, i] <- lowerbound[i] + runif(SearchAgents) * (upperbound[i] - lowerbound[i])
  }
  best_so_far <- numeric(Max_iterations)
  fit <- numeric(SearchAgents)
  for (i in 1:SearchAgents) {
    M <- X[i, ]
    fit[i] <- fitness(M, X_train, Y_train, input_dim, hidden_dim, output_dim)
  }
  
  for (t in 1:Max_iterations) {
    CF <- (1 - t / Max_iterations)^(2 * t / Max_iterations)
    
    best <- min(fit)
    location <- which.min(fit)
    
    if (t == 1) {
      Bast_P <- X[location, ]
      fbest <- best
    } else if (best < fbest) {
      fbest <- best
      Bast_P <- X[location, ]
    }
    
    for (i in 1:SearchAgents) {
      if (t < Max_iterations / 3) {
        Rn <- nrow(X)
        X_random_1 <- sample(1:Rn, 1)
        X_random_2 <- sample(1:Rn, 1)
        R1 <- runif(1, 0, 1)
        X1 <- X[i, ] + (X[X_random_1, ] - X[X_random_2, ]) * R1
        X1 <- pmax(X1, lowerbound)
        X1 <- pmin(X1, upperbound)
      } else if (t > Max_iterations / 3 && t < 2 * Max_iterations / 3) {
        RB <- rnorm(dimension)
        X1 <- Bast_P + exp((t / Max_iterations)^4) * (RB - 0.5) * (Bast_P - X[i, ])
        X1 <- pmax(X1, lowerbound)
        X1 <- pmin(X1, upperbound)
      } else {
        RL <- 0.5 * Levy(dimension)
        X1 <- Bast_P + CF * X[i, ] * RL
        X1 <- pmax(X1, lowerbound)
        X1 <- pmin(X1, upperbound)
      }
      
      f_newP1 <- fitness(X1, X_train, Y_train, input_dim, hidden_dim, output_dim)
      if (f_newP1 <= fit[i]) {
        X[i, ] <- X1
        fit[i] <- f_newP1
      }
    }
    
    r <- runif(1)
    k <- sample(1:SearchAgents, 1)
    Xrandom <- X[k, ]
    
    for (i in 1:SearchAgents) {
      if (r < 0.5) {
        RB <- runif(dimension, -1, 1)
        X2 <- Bast_P + (1 - t / Max_iterations)^2 * (2 * RB - 1) * X[i, ]
        X2 <- pmax(X2, lowerbound)
        X2 <- pmin(X2, upperbound)
      } else {
        K <- round(1 + runif(1))
        R2 <- runif(dimension)
        X2 <- X[i, ] + R2 * (Xrandom - K * X[i, ])
        X2 <- pmax(X2, lowerbound)
        X2 <- pmin(X2, upperbound)
      }
      
      f_newP2 <- fitness(X2, X_train, Y_train, input_dim, hidden_dim, output_dim)
      if (f_newP2 <= fit[i]) {
        X[i, ] <- X2
        fit[i] <- f_newP2
      }
    }
    
    best_so_far[t] <- fbest
    cat("Iteration", t, ": Best Cost =", best_so_far[t], "\n")
  }
  
  list(Best_score = fbest, Best_pos = Bast_P, SBOA_curve = best_so_far)
}

Levy <- function(d) {
  beta <- 1.5
  sigma <- (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
  u <- rnorm(d) * sigma
  v <- rnorm(d)
  step <- u / abs(v)^(1 / beta)
  return(step)
}

# Example validation run on Auto-MPG dataset
run_mpg_test <- function() {
  # Load data
  cat("Loading auto-mpg dataset...\n")
  data_path <- "data/cleaned/auto-mpg.csv"
  if (!file.exists(data_path)) {
     stop("Please run python experiments first to generate data/cleaned/auto-mpg.csv")
  }
  df <- read.csv(data_path)
  
  # Select features and targets
  features <- c("cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin")
  X <- as.matrix(df[, features])
  Y <- as.matrix(df$mpg)
  
  # Normalize
  min_X <- apply(X, 2, min)
  max_X <- apply(X, 2, max)
  X_norm <- scale(X, center = min_X, scale = max_X - min_X)
  
  min_Y <- min(Y)
  max_Y <- max(Y)
  Y_norm <- scale(Y, center = min_Y, scale = max_Y - min_Y)
  
  # SBOA settings
  hidden_dim <- 8
  SearchAgents <- 15
  Max_iterations <- 10
  dimension <- ncol(X_norm) * hidden_dim + hidden_dim + hidden_dim * ncol(Y_norm) + ncol(Y_norm)
  
  cat("Starting SBOA optimization in R...\n")
  result <- SBOA(X_norm, Y_norm, ncol(X_norm), hidden_dim, ncol(Y_norm), SearchAgents, Max_iterations, -1, 1, dimension, fitness_function)
  
  # Save curves
  dir.create("outputs/r_curves", showWarnings = FALSE, recursive = TRUE)
  write.csv(result$SBOA_curve, "outputs/r_curves/sboa_convergence_r.csv", row.names = FALSE)
  cat("Saved R output curve to: outputs/r_curves/sboa_convergence_r.csv\n")
}

run_mpg_test()
