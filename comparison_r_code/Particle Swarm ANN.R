library(rsample)
library(tidyverse)
# library(tidymodels)
# library(baguette)
# library(bonsai)
library(yardstick)

library(readxl)


# Aktivasyon fonksiyonları
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

# İleri beslemeli sinir ağı (tek gizli katman)
forward_pass <- function(weights, biases, X) {
  W1 <- weights$W1
  W2 <- weights$W2
  b1 <- biases$b1
  b2 <- biases$b2
  
  # Girişten gizli katmana
  Z1 <- X %*% W1 + matrix(b1, nrow = nrow(X), ncol = length(b1), byrow = TRUE)
  A1 <- sigmoid(Z1)
  
  # Gizli katmandan çıkış katmanına
  Z2 <- A1 %*% W2 + matrix(b2, nrow = nrow(A1), ncol = length(b2), byrow = TRUE)
  Y_hat <- sigmoid(Z2)
  
  list(A1 = A1, Y_hat = Y_hat)
}

# Amaç fonksiyonu (MSE)
mse_loss <- function(weights, biases, X, Y) {
  forward <- forward_pass(weights, biases, X)
  Y_hat <- forward$Y_hat
  mean((Y - Y_hat)^2)
}

# HO algoritmasına uyarlanmış fitness fonksiyonu
fitness_function <- function(params, X_train, Y_train, input_dim, hidden_dim, output_dim) {
  # Parametreleri ağırlık ve biaslara böl
  W1 <- matrix(params[1:(input_dim * hidden_dim)], nrow = input_dim, ncol = hidden_dim)
  b1 <- params[(input_dim * hidden_dim + 1):(input_dim * hidden_dim + hidden_dim)]
  W2 <- matrix(params[(input_dim * hidden_dim + hidden_dim + 1):(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim)],
               nrow = hidden_dim, ncol = output_dim)
  b2 <- params[(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + 1):length(params)]
  
  weights <- list(W1 = W1, W2 = W2)
  biases <- list(b1 = b1, b2 = b2)
  
  # MSE değerini hesapla
  mse_loss(weights, biases, X_train, Y_train)
}

PSO <- function(X_train, Y_train, input_dim, hidden_dim, output_dim,
                SearchAgents, Max_iterations, lowerbound, upperbound,
                dimension, fitness, w = 1, wdamp = 0.99, c1 = 1.5, c2 = 2.0) {
  
  # Problem Definition
  var_size <- c(1, dimension)
  
  # Velocity Limits
  vel_max <- 0.1 * (upperbound - lowerbound)
  vel_min <- -upperbound
  
  # Particle Initialization
  particle <- vector("list", SearchAgents)
  for (i in 1:SearchAgents) {
    particle[[i]] <- list(
      position = runif(dimension, lowerbound, upperbound),
      velocity = rep(0, dimension),
      cost = Inf,
      best = list(position = NULL, cost = Inf)
    )
  }
  
  # Global Best Initialization
  global_best <- list(position = NULL, cost = Inf)
  
  # Evaluate Initial Particles
  for (i in 1:SearchAgents) {
    particle[[i]]$cost <- fitness(particle[[i]]$position, X_train, Y_train, input_dim, hidden_dim, output_dim)
    particle[[i]]$best$position <- particle[[i]]$position
    particle[[i]]$best$cost <- particle[[i]]$cost
    
    if (particle[[i]]$best$cost < global_best$cost) {
      global_best <- particle[[i]]$best
    }
  }
  
  # Best Cost History
  best_cost <- numeric(Max_iterations)
  
  # PSO Main Loop
  for (it in 1:Max_iterations) {
    for (i in 1:SearchAgents) {
      
      # Update Velocity
      particle[[i]]$velocity <- w * particle[[i]]$velocity +
        c1 * runif(dimension) * (particle[[i]]$best$position - particle[[i]]$position) +
        c2 * runif(dimension) * (global_best$position - particle[[i]]$position)
      
      # Apply Velocity Limits
      particle[[i]]$velocity <- pmax(particle[[i]]$velocity, vel_min)
      particle[[i]]$velocity <- pmin(particle[[i]]$velocity, vel_max)
      
      # Update Position
      particle[[i]]$position <- particle[[i]]$position + particle[[i]]$velocity
      
      # Velocity Mirror Effect
      is_outside <- (particle[[i]]$position < lowerbound | particle[[i]]$position > upperbound)
      particle[[i]]$velocity[is_outside] <- -particle[[i]]$velocity[is_outside]
      
      # Apply Position Limits
      particle[[i]]$position <- pmax(particle[[i]]$position, lowerbound)
      particle[[i]]$position <- pmin(particle[[i]]$position, upperbound)
      
      # Evaluate
      particle[[i]]$cost <- fitness(particle[[i]]$position, X_train, Y_train, input_dim, hidden_dim, output_dim)
      
      # Update Personal Best
      if (particle[[i]]$cost < particle[[i]]$best$cost) {
        particle[[i]]$best$position <- particle[[i]]$position
        particle[[i]]$best$cost <- particle[[i]]$cost
        
        # Update Global Best
        if (particle[[i]]$best$cost < global_best$cost) {
          global_best <- particle[[i]]$best
        }
      }
    }
    
    # Store the Best Cost of the Current Iteration
    best_cost[it] <- global_best$cost
    
    # Display Iteration Information
    cat("Iteration", it, ": Best Cost =", best_cost[it], "\n")
    
    # Dampen Inertia Weight
    w <- w * wdamp
  }
  
  return(list(best_solution = global_best, best_cost_history = best_cost))
}


# HO algoritmasını çalıştır
PSO_nn <- function(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations, lowerbound = -1, upperbound = 1) {
  # Parametre sayısı: W1, b1, W2, b2
  input_dim = ncol(X_train)
  output_dim = ncol(Y_train)
  
  data_X = X_train
  data_y = Y_train
  
  ## Normalization for X
  max_data_X <- apply(X_train, 2, max)
  min_data_X <- apply(X_train, 2, min)
  X_train <- as.matrix(scale(X_train, center = min_data_X, scale = max_data_X - min_data_X))
  
  ## Normalization for y
  max_data_y <- apply(Y_train, 2, max)
  min_data_y <- apply(Y_train, 2, min)
  Y_train <- as.matrix(scale(data_y, center = min_data_y, scale = max_data_y - min_data_y))
  
  num_params <- input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
  
  fitness_function = fitness_function
  # HO algoritmasını çağır
  pso_result <- PSO(X_train, Y_train, input_dim, hidden_dim, output_dim,
                   SearchAgents, Max_iterations, lowerbound, upperbound,
                   dimension = num_params, fitness = fitness_function)
  
  # En iyi ağırlık ve bias değerleri
  best_params <- pso_result$best_solution$position
  W1 <- matrix(best_params[1:(input_dim * hidden_dim)], nrow = input_dim, ncol = hidden_dim)
  b1 <- best_params[(input_dim * hidden_dim + 1):(input_dim * hidden_dim + hidden_dim)]
  W2 <- matrix(best_params[(input_dim * hidden_dim + hidden_dim + 1):(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim)],
               nrow = hidden_dim, ncol = output_dim)
  b2 <- best_params[(input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + 1):length(best_params)]
  
  weights <- list(W1 = W1, W2 = W2)
  biases <- list(b1 = b1, b2 = b2)
  
  NET <- forward_pass(weights, biases, X_train)
  y_hat_org = (NET$Y_hat) * (max(data_y) - min(data_y)) + min(data_y)
  predictions = as.data.frame(cbind(yhat = y_hat_org, y = data_y))
  RMSE.data <- rmse(predictions, V1, V2)
  MAE.data <- mae(predictions, V1, V2)
  MAPE.data <- mape(predictions, V1, V2)
  MASE.data <- mase(predictions, V1, V2)
  RSQ.data <- rsq(predictions, V1, V2)
  
  result = list(weights = list(W1 = W1, W2 = W2), biases = list(b1 = b1, b2 = b2),
                pso_curve = pso_result$best_cost_history,
                RMSE = RMSE.data, MAE = MAE.data, MAPE = MAPE.data,
                MASE = MASE.data, RSQ = RSQ.data,
                normalization_data_X = X_train, data_X = data_X, data_y = data_y)
  class(result) <- 'PSO_nn'
  return(result)
}

predict.PSO_nn <- function(object, newdata) {
  
  ## Normalization new data
  max_data_new <- apply(object$data_X, 2, max)
  min_data_new <- apply(object$data_X, 2, min)
  newdata <- as.matrix(scale(newdata, center = min_data_new, scale = max_data_new - min_data_new))
  
  nn <- nrow(newdata)
  data_all <- rbind(object$normalization_data_X, newdata)
  yhatall <- forward_pass(weights = object$weights, biases = object$biases, data_all)
  
  yhattestg = yhatall$Y_hat * (max(object$data_y) - min(object$data_y)) + min(object$data_y)
  
  list(yhattest = yhattestg[(length(yhattestg) - nn + 1):(nrow(object$normalization_data_X) + nn)])
}

#### ------------------ Approximation Data set ------------------------
## Sigmoid Data ----------------------------
# Sigmoid fonksiyonu tanımlama
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}


# Eğitim ve test verisi oluşturma
train_x_sigmoid <- seq(-3, 3, by = 0.1)
test_x_sigmoid <- seq(-3, 3, by = 0.05)

train_y_sigmoid <- sigmoid(train_x_sigmoid)
test_y_sigmoid <- sigmoid(test_x_sigmoid)

# Veri çerçeveleri oluşturma
df_train <- data.frame(x = train_x_sigmoid, y = train_y_sigmoid)
df_test <- data.frame(x = test_x_sigmoid, y = test_y_sigmoid)

X = as.matrix(df_train[,1])
y = as.matrix(df_train[,2])

# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim = 15
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)
runtime <- numeric(N)


# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))

  
  # MAE hesaplama
  MAE_pso <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso

  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sigmoid_mean_pso <- mean(runtime)
runtime_sigmoid_sd_pso<- sd(runtime)

# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_sig <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_sig[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_sig
min_values_pso_sig <- sapply(sb_values, min)

iter_min_pso_sig <- format(min(min_values_pso_sig), scientific = TRUE, digits = 3)
iter_mean_pso_sig <- format(mean(min_values_pso_sig), scientific = TRUE, digits = 3)
iter_max_pso_sig <- format(max(min_values_pso_sig), scientific = TRUE, digits = 3)
iter_sd_pso_sig <- format(sd(min_values_pso_sig), scientific = TRUE, digits = 3)


min_mae_pso_sig <- min(mae_values)
mean_mae_pso_sig <- mean(mae_values)
max_mae_pso_sig <- max(mae_values)
sd_mae_pso_sig <- sd(mae_values)



## Sinüs Data ----------------------------
# Sinus fonksiyonu tanımlama
sine_function <- function(x) {
  return(sin(2 * x))
}


# Eğitim ve test verisi oluşturma
train_x_sine <- seq(-2 * pi, 2 * pi, by = 0.1)
test_x_sine <- seq(-2 * pi, 2 * pi, by = 0.05)

train_y_sine <- sine_function(train_x_sine)
test_y_sine <- sine_function(test_x_sine)

# Veri çerçeveleri oluşturma
df_train <- data.frame(x = train_x_sine, y = train_y_sine)
df_test <- data.frame(x = test_x_sine, y = test_y_sine)

X = as.matrix(df_train[,1])
y = as.matrix(df_train[,2])

# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim = 15
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_pso <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sinus_mean_pso <- mean(runtime)
runtime_sinus_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_sin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_sin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_sin
min_values_pso_sin <- sapply(sb_values, min)

iter_min_pso_sin <- format(min(min_values_pso_sin), scientific = TRUE, digits = 3)
iter_mean_pso_sin <- format(mean(min_values_pso_sin), scientific = TRUE, digits = 3)
iter_max_pso_sin <- format(max(min_values_pso_sin), scientific = TRUE, digits = 3)
iter_sd_pso_sin <- format(sd(min_values_pso_sin), scientific = TRUE, digits = 3)


min_mae_pso_sin <- min(mae_values)
mean_mae_pso_sin <- mean(mae_values)
max_mae_pso_sin <- max(mae_values)
sd_mae_pso_sin <- sd(mae_values)



## Cosinus Data ----------------------------
# Cosinus fonksiyonu tanımlama
cosine_function <- function(x) {
  return((cos(x * pi / 2))^7)
}


# Eğitim ve test verisi oluşturma
train_x_cosine <- seq(1.25, 2.75, by = 0.05)
test_x_cosine <- seq(1.25, 2.75, by = 0.04)

train_y_cosine <- cosine_function(train_x_cosine)
test_y_cosine <- cosine_function(test_x_cosine)

# Veri çerçeveleri oluşturma
df_train <- data.frame(x = train_x_cosine, y = train_y_cosine)
df_test <- data.frame(x = test_x_cosine, y = test_y_cosine)

X = as.matrix(df_train[,1])
y = as.matrix(df_train[,2])

# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim = 15
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_pso <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cos_mean_pso <- mean(runtime)
runtime_cos_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_cosin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_cosin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_cosin
min_values_pso_cosin <- sapply(sb_values, min)

iter_min_pso_cosin <- format(min(min_values_pso_cosin), scientific = TRUE, digits = 3)
iter_mean_pso_cosin <- format(mean(min_values_pso_cosin), scientific = TRUE, digits = 3)
iter_max_pso_cosin <- format(max(min_values_pso_cosin), scientific = TRUE, digits = 3)
iter_sd_pso_cosin <- format(sd(min_values_pso_cosin), scientific = TRUE, digits = 3)


min_mae_pso_cosin <- min(mae_values)
mean_mae_pso_cosin <- mean(mae_values)
max_mae_pso_cosin <- max(mae_values)
sd_mae_pso_cosin <- sd(mae_values)

### ---------------------------- Concrete Comprehensive -------------------------

df <- read_excel("data/Concrete_Data.xls")
df <- as.data.frame(df)
df

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)


X = as.matrix(df_train[,1:8])
y = as.matrix(df_train[,9])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim = 17
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_pso <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_pso
  
  # MAE hesaplama
  MAE_pso <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_pso <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_concrete_mean_pso <- mean(runtime)
runtime_concrete_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_conc <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_conc[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_conc
min_values_pso_conc <- sapply(sb_values, min)

iter_min_pso_conc <- format(min(min_values_pso_conc), scientific = TRUE, digits = 3)
iter_mean_pso_conc <- format(mean(min_values_pso_conc), scientific = TRUE, digits = 3)
iter_max_pso_conc <- format(max(min_values_pso_conc), scientific = TRUE, digits = 3)
iter_sd_pso_conc <- format(sd(min_values_pso_conc), scientific = TRUE, digits = 3)


min_rmse_pso_conc <- min(rmse_values)
mean_rmse_pso_conc <- mean(rmse_values)
max_rmse_pso_conc <- max(rmse_values)
sd_rmse_pso_conc <- sd(rmse_values)


min_mae_pso_conc <- min(mae_values)
mean_mae_pso_conc <- mean(mae_values)
max_mae_pso_conc <- max(mae_values)
sd_mae_pso_conc <- sd(mae_values)


min_r2_pso_conc <- min(r2_values)
mean_r2_pso_conc <- mean(r2_values)
max_r2_pso_conc <- max(r2_values)
sd_r2_pso_conc <- sd(r2_values)



### ------------------------- Energy - Heating load ------------------

df <- read_excel("data/ENB2012_data.xlsx")
df <- as.data.frame(df)
summary(df)

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:8])
y = as.matrix(df_train[,9])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim <- 17
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_pso <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_pso
  
  # MAE hesaplama
  MAE_pso <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_pso <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_heat_mean_pso <- mean(runtime)
runtime_heat_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_heat <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_heat[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_heat
min_values_pso_heat <- sapply(sb_values, min)

iter_min_pso_heat <- format(min(min_values_pso_heat), scientific = TRUE, digits = 3)
iter_mean_pso_heat <- format(mean(min_values_pso_heat), scientific = TRUE, digits = 3)
iter_max_pso_heat <- format(max(min_values_pso_heat), scientific = TRUE, digits = 3)
iter_sd_pso_heat <- format(sd(min_values_pso_heat), scientific = TRUE, digits = 3)


min_rmse_pso_heat <- min(rmse_values)
mean_rmse_pso_heat <- mean(rmse_values)
max_rmse_pso_heat <- max(rmse_values)
sd_rmse_pso_heat <- sd(rmse_values)


min_mae_pso_heat <- min(mae_values)
mean_mae_pso_heat <- mean(mae_values)
max_mae_pso_heat <- max(mae_values)
sd_mae_pso_heat <- sd(mae_values)


min_r2_pso_heat <- min(r2_values)
mean_r2_pso_heat <- mean(r2_values)
max_r2_pso_heat <- max(r2_values)
sd_r2_pso_heat <- sd(r2_values)


### ------------------------- Energy - Cooling load ------------------

df <- read_excel("data/ENB2012_data.xlsx")
df <- as.data.frame(df)
summary(df)

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:8])
y = as.matrix(df_train[,10])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim <- 17
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  
  # RMSE hesaplama
  RMSE_pso <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_pso
  
  # MAE hesaplama
  MAE_pso <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_pso <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cool_mean_pso <- mean(runtime)
runtime_cool_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_cool <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_cool[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_cool
min_values_pso_cool <- sapply(sb_values, min)

iter_min_pso_cool <- format(min(min_values_pso_cool), scientific = TRUE, digits = 3)
iter_mean_pso_cool <- format(mean(min_values_pso_cool), scientific = TRUE, digits = 3)
iter_max_pso_cool <- format(max(min_values_pso_cool), scientific = TRUE, digits = 3)
iter_sd_pso_cool <- format(sd(min_values_pso_cool), scientific = TRUE, digits = 3)



min_rmse_pso_cool <- min(rmse_values)
mean_rmse_pso_cool <- mean(rmse_values)
max_rmse_pso_cool <- max(rmse_values)
sd_rmse_pso_cool <- sd(rmse_values)


min_mae_pso_cool <- min(mae_values)
mean_mae_pso_cool <- mean(mae_values)
max_mae_pso_cool <- max(mae_values)
sd_mae_pso_cool <- sd(mae_values)


min_r2_pso_cool <- min(r2_values)
mean_r2_pso_cool <- mean(r2_values)
max_r2_pso_cool <- max(r2_values)
sd_r2_pso_cool <- sd(r2_values)


### Auto - MPG ------------------------

df <- read_csv("data/auto-mpg.csv") %>% select(-c(`car name`))
df <- as.data.frame(df)
df$horsepower <- as.numeric(df$horsepower)
df$horsepower[is.na(df$horsepower)] <- mean(df$horsepower, na.rm = TRUE)

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,2:8])
y = as.matrix(df_train[,1])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim <- 15
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,2:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,1]))
  
  # RMSE hesaplama
  RMSE_pso <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_pso
  
  # MAE hesaplama
  MAE_pso <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_pso <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_auto_mean_pso <- mean(runtime)
runtime_auto_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_mpg <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_mpg[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_mpg
min_values_pso_mpg <- sapply(sb_values, min)

iter_min_pso_mpg <- format(min(min_values_pso_mpg), scientific = TRUE, digits = 3)
iter_mean_pso_mpg <- format(mean(min_values_pso_mpg), scientific = TRUE, digits = 3)
iter_max_pso_mpg <- format(max(min_values_pso_mpg), scientific = TRUE, digits = 3)
iter_sd_pso_mpg <- format(sd(min_values_pso_mpg), scientific = TRUE, digits = 3)



min_rmse_pso_mpg <- min(rmse_values)
mean_rmse_pso_mpg <- mean(rmse_values)
max_rmse_pso_mpg <- max(rmse_values)
sd_rmse_pso_mpg <- sd(rmse_values)


min_mae_pso_mpg <- min(mae_values)
mean_mae_pso_mpg <- mean(mae_values)
max_mae_pso_mpg <- max(mae_values)
sd_mae_pso_mpg <- sd(mae_values)


min_r2_pso_mpg <- min(r2_values)
mean_r2_pso_mpg <- mean(r2_values)
max_r2_pso_mpg <- max(r2_values)
sd_r2_pso_mpg <- sd(r2_values)



### House price ------------------------

df <- read_excel("data/real_estate.xlsx") %>% select(-c(`No`))

df <- as.data.frame(df)


set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:6])
y = as.matrix(df_train[,7])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function

hidden_dim <- 13
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
rmse_values <- numeric(N)
mae_values <- numeric(N)
r2_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:6]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,7]))
  
  # RMSE hesaplama
  RMSE_pso <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_pso
  
  # MAE hesaplama
  MAE_pso <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_pso
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_pso <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_house_mean_pso <- mean(runtime)
runtime_house_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_house <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_house[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_house
min_values_pso_house <- sapply(sb_values, min)

iter_min_pso_house <- format(min(min_values_pso_house), scientific = TRUE, digits = 3)
iter_mean_pso_house <- format(mean(min_values_pso_house), scientific = TRUE, digits = 3)
iter_max_pso_house <- format(max(min_values_pso_house), scientific = TRUE, digits = 3)
iter_sd_pso_house <- format(sd(min_values_pso_house), scientific = TRUE, digits = 3)



min_rmse_pso_house <- min(rmse_values)
mean_rmse_pso_house <- mean(rmse_values)
max_rmse_pso_house <- max(rmse_values)
sd_rmse_pso_house <- sd(rmse_values)


min_mae_pso_house <- min(mae_values)
mean_mae_pso_house <- mean(mae_values)
max_mae_pso_house <- max(mae_values)
sd_mae_pso_house <- sd(mae_values)


min_r2_pso_house <- min(r2_values)
mean_r2_pso_house <- mean(r2_values)
max_r2_pso_house <- max(r2_values)
sd_r2_pso_house <- sd(r2_values)

### Breast Cancer -------------------------------------

breast_cancer <- read.csv("data/breast-cancer-wisconsin.data", header=FALSE)
breast_cancer <- breast_cancer[ ,2:11]
breast_cancer[, ncol(breast_cancer)] <- ifelse(
  breast_cancer[, ncol(breast_cancer)] == 2, 0,
  ifelse(breast_cancer[, ncol(breast_cancer)] == 4, 1, breast_cancer[, ncol(breast_cancer)])
)

breast_cancer[breast_cancer == "?"] <- NA
colSums(is.na(breast_cancer))

# V7 sütununu sayısal hale getirme
breast_cancer$V7 <- as.numeric(as.character(breast_cancer$V7))

# Kayıp verileri ortalama ile doldurma
breast_cancer$V7[is.na(breast_cancer$V7)] <- mean(breast_cancer$V7, na.rm = TRUE)
colSums(is.na(breast_cancer))


df <- as.data.frame(breast_cancer)

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:9])
y = as.matrix(df_train[,10])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function
hidden_dim <- 19
N = 10


# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
accuracy_values <- numeric(N)
precision_values <- numeric(N)
recall_values <- numeric(N)
f1_score_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_pso <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_pso
  
  # precision hesaplama
  precision_pso <- TP / (TP + FP)
  precision_values[i] <- precision_pso
  
  # recall hesaplama
  recall_pso <- TP / (TP + FN)
  recall_values[i] <- recall_pso
  
  # f1 score
  f1_score_pso <- 2 * (precision_pso * recall_pso) / (precision_pso + recall_pso)
  f1_score_values[i] <- f1_score_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_breast_mean_pso <- mean(runtime)
runtime_breast_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_cancer <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_cancer[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_cancer
min_values_pso_cancer <- sapply(sb_values, min)

iter_min_pso_cancer <- format(min(min_values_pso_cancer), scientific = TRUE, digits = 3)
iter_mean_pso_cancer <- format(mean(min_values_pso_cancer), scientific = TRUE, digits = 3)
iter_max_pso_cancer <- format(max(min_values_pso_cancer), scientific = TRUE, digits = 3)
iter_sd_pso_cancer <- format(sd(min_values_pso_cancer), scientific = TRUE, digits = 3)


min_accuracy_pso_cancer <- min(accuracy_values)
mean_accuracy_pso_cancer <- mean(accuracy_values)
max_accuracy_pso_cancer <- max(accuracy_values)
sd_accuracy_pso_cancer <- sd(accuracy_values)


min_precision_pso_cancer <- min(precision_values)
mean_precision_pso_cancer <- mean(precision_values)
max_precision_pso_cancer <- max(precision_values)
sd_precision_pso_cancer <- sd(precision_values)


min_recall_pso_cancer <- min(recall_values)
mean_recall_pso_cancer <- mean(recall_values)
max_recall_pso_cancer <- max(recall_values)
sd_recall_pso_cancer <- sd(recall_values)

min_f1_score_pso_cancer <- min(f1_score_values)
mean_f1_score_pso_cancer <- mean(f1_score_values)
max_f1_score_pso_cancer <- max(f1_score_values)
sd_f1_score_pso_cancer <- sd(f1_score_values)


### Tic-tac-toe Dataset--------------------------------------

df <- read.csv("data/tic-tac-toe.data")
library(dplyr)

df <- df %>%
  mutate(across(everything(), ~ recode(., "x" = 1, "o" = 2, "b" = 3, "positive" = 1, "negative" = 0)))
df

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:9])
y = as.matrix(df_train[,10])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function
hidden_dim <- 19
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
accuracy_values <- numeric(N)
precision_values <- numeric(N)
recall_values <- numeric(N)
f1_score_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_pso <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_pso
  
  # precision hesaplama
  precision_pso <- TP / (TP + FP)
  precision_values[i] <- precision_pso
  
  # recall hesaplama
  recall_pso <- TP / (TP + FN)
  recall_values[i] <- recall_pso
  
  # f1 score
  f1_score_pso <- 2 * (precision_pso * recall_pso) / (precision_pso + recall_pso)
  f1_score_values[i] <- f1_score_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_tic_mean_pso <- mean(runtime)
runtime_tic_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_tictac <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_tictac[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_tictac
min_values_pso_tictac <- sapply(sb_values, min)

iter_min_pso_tictac <- format(min(min_values_pso_tictac), scientific = TRUE, digits = 3)
iter_mean_pso_tictac <- format(mean(min_values_pso_tictac), scientific = TRUE, digits = 3)
iter_max_pso_tictac <- format(max(min_values_pso_tictac), scientific = TRUE, digits = 3)
iter_sd_pso_tictac <- format(sd(min_values_pso_tictac), scientific = TRUE, digits = 3)

min_accuracy_pso_tictac <- min(accuracy_values)
mean_accuracy_pso_tictac <- mean(accuracy_values)
max_accuracy_pso_tictac <- max(accuracy_values)
sd_accuracy_pso_tictac <- sd(accuracy_values)

min_precision_pso_tictac <- min(precision_values)
mean_precision_pso_tictac <- mean(precision_values)
max_precision_pso_tictac <- max(precision_values)
sd_precision_pso_tictac <- sd(precision_values)

min_recall_pso_tictac <- min(recall_values)
mean_recall_pso_tictac <- mean(recall_values)
max_recall_pso_tictac <- max(recall_values)
sd_recall_pso_tictac <- sd(recall_values)

min_f1_score_pso_tictac <- min(f1_score_values)
mean_f1_score_pso_tictac <- mean(f1_score_values)
max_f1_score_pso_tictac <- max(f1_score_values)
sd_f1_score_pso_tictac <- sd(f1_score_values)


### Australian Dataset--------------------------------------

df <- read.table("data/australian.dat", quote="\"", comment.char="")
df

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:14])
y = as.matrix(df_train[,15])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function
hidden_dim <- 29
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
accuracy_values <- numeric(N)
precision_values <- numeric(N)
recall_values <- numeric(N)
f1_score_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:14]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,15]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_pso <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_pso
  
  # precision hesaplama
  precision_pso <- TP / (TP + FP)
  precision_values[i] <- precision_pso
  
  # recall hesaplama
  recall_pso <- TP / (TP + FN)
  recall_values[i] <- recall_pso
  
  # f1 score
  f1_score_pso <- 2 * (precision_pso * recall_pso) / (precision_pso + recall_pso)
  f1_score_values[i] <- f1_score_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_aust_mean_pso <- mean(runtime)
runtime_aust_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_australian <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_australian[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_australian
min_values_pso_australian <- sapply(sb_values, min)

iter_min_pso_australian <- format(min(min_values_pso_australian), scientific = TRUE, digits = 3)
iter_mean_pso_australian <- format(mean(min_values_pso_australian), scientific = TRUE, digits = 3)
iter_max_pso_australian <- format(max(min_values_pso_australian), scientific = TRUE, digits = 3)
iter_sd_pso_australian <- format(sd(min_values_pso_australian), scientific = TRUE, digits = 3)

min_accuracy_pso_australian <- min(accuracy_values)
mean_accuracy_pso_australian <- mean(accuracy_values)
max_accuracy_pso_australian <- max(accuracy_values)
sd_accuracy_pso_australian <- sd(accuracy_values)

min_precision_pso_australian <- min(precision_values)
mean_precision_pso_australian <- mean(precision_values)
max_precision_pso_australian <- max(precision_values)
sd_precision_pso_australian <- sd(precision_values)

min_recall_pso_australian <- min(recall_values)
mean_recall_pso_australian <- mean(recall_values)
max_recall_pso_australian <- max(recall_values)
sd_recall_pso_australian <- sd(recall_values)

min_f1_score_pso_australian <- min(f1_score_values)
mean_f1_score_pso_australian <- mean(f1_score_values)
max_f1_score_pso_australian <- max(f1_score_values)
sd_f1_score_pso_australian <- sd(f1_score_values)


### Banknote Dataset--------------------------------------

df <- read.csv("data/data_banknote_authentication.txt", header=FALSE)
df

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:4])
y = as.matrix(df_train[,5])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function
hidden_dim <- 9
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
accuracy_values <- numeric(N)
precision_values <- numeric(N)
recall_values <- numeric(N)
f1_score_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_pso <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_pso
  
  # precision hesaplama
  precision_pso <- TP / (TP + FP)
  precision_values[i] <- precision_pso
  
  # recall hesaplama
  recall_pso <- TP / (TP + FN)
  recall_values[i] <- recall_pso
  
  # f1 score
  f1_score_pso <- 2 * (precision_pso * recall_pso) / (precision_pso + recall_pso)
  f1_score_values[i] <- f1_score_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_bank_mean_pso <- mean(runtime)
runtime_bank_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_bank <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_bank[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_bank
min_values_pso_bank <- sapply(sb_values, min)

iter_min_pso_bank <- format(min(min_values_pso_bank), scientific = TRUE, digits = 3)
iter_mean_pso_bank <- format(mean(min_values_pso_bank), scientific = TRUE, digits = 3)
iter_max_pso_bank <- format(max(min_values_pso_bank), scientific = TRUE, digits = 3)
iter_sd_pso_bank <- format(sd(min_values_pso_bank), scientific = TRUE, digits = 3)

min_accuracy_pso_bank <- min(accuracy_values)
mean_accuracy_pso_bank <- mean(accuracy_values)
max_accuracy_pso_bank <- max(accuracy_values)
sd_accuracy_pso_bank <- sd(accuracy_values)

min_precision_pso_bank <- min(precision_values)
mean_precision_pso_bank <- mean(precision_values)
max_precision_pso_bank <- max(precision_values)
sd_precision_pso_bank <- sd(precision_values)

min_recall_pso_bank <- min(recall_values)
mean_recall_pso_bank <- mean(recall_values)
max_recall_pso_bank <- max(recall_values)
sd_recall_pso_bank <- sd(recall_values)

min_f1_score_pso_bank <- min(f1_score_values)
mean_f1_score_pso_bank <- mean(f1_score_values)
max_f1_score_pso_bank <- max(f1_score_values)
sd_f1_score_pso_bank <- sd(f1_score_values)



### Blood Dataset--------------------------------------

df <- read.csv("data/transfusion.data")
df

set.seed(123)
df_split <- initial_split(df, prop = 0.67)
df_split

df_train <- training(df_split)
df_test  <- testing(df_split)

X = as.matrix(df_train[,1:4])
y = as.matrix(df_train[,5])


# HO algoritmasını çalıştır
SearchAgents <- 200
Max_iterations <- 250

lowerbound = -10
upperbound = 10

fitness = fitness_function
hidden_dim <- 9
N = 10

# Değerleri saklamak için boş vektörler oluşturuyoruz
sb_values <- list()
accuracy_values <- numeric(N)
precision_values <- numeric(N)
recall_values <- numeric(N)
f1_score_values <- numeric(N)

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # pso ile YSA modelini eğitiyoruz
  result_pso <- PSO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_pso, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_pso <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_pso
  
  # precision hesaplama
  precision_pso <- TP / (TP + FP)
  precision_values[i] <- precision_pso
  
  # recall hesaplama
  recall_pso <- TP / (TP + FN)
  recall_values[i] <- recall_pso
  
  # f1 score
  f1_score_pso <- 2 * (precision_pso * recall_pso) / (precision_pso + recall_pso)
  f1_score_values[i] <- f1_score_pso
  
  # pso iterasyonlarındaki değerleri saklıyoruz
  pso_curve_iter <- result_pso$pso_curve
  sb_values[[i]] <- pso_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_blood_mean_pso <- mean(runtime)
runtime_blood_sd_pso<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_pso_blood <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_pso_blood[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_pso_blood
min_values_pso_blood <- sapply(sb_values, min)

iter_min_pso_blood <- format(min(min_values_pso_blood), scientific = TRUE, digits = 3)
iter_mean_pso_blood <- format(mean(min_values_pso_blood), scientific = TRUE, digits = 3)
iter_max_pso_blood <- format(max(min_values_pso_blood), scientific = TRUE, digits = 3)
iter_sd_pso_blood <- format(sd(min_values_pso_blood), scientific = TRUE, digits = 3)

min_accuracy_pso_blood <- min(accuracy_values)
mean_accuracy_pso_blood <- mean(accuracy_values)
max_accuracy_pso_blood <- max(accuracy_values)
sd_accuracy_pso_blood <- sd(accuracy_values)

min_precision_pso_blood <- min(precision_values)
mean_precision_pso_blood <- mean(precision_values)
max_precision_pso_blood <- max(precision_values)
sd_precision_pso_blood <- sd(precision_values)

min_recall_pso_blood <- min(recall_values)
mean_recall_pso_blood <- mean(recall_values)
max_recall_pso_blood <- max(recall_values)
sd_recall_pso_blood <- sd(recall_values)

min_f1_score_pso_blood <- min(f1_score_values)
mean_f1_score_pso_blood <- mean(f1_score_values)
max_f1_score_pso_blood <- max(f1_score_values)
sd_f1_score_pso_blood <- sd(f1_score_values)




runtime_concrete_mean_pso
runtime_concrete_sd_pso

runtime_heat_mean_pso
runtime_heat_sd_pso

runtime_cool_mean_pso
runtime_cool_sd_pso

runtime_auto_mean_pso
runtime_auto_sd_pso

runtime_house_mean_pso
runtime_house_sd_pso

runtime_sigmoid_mean_pso
runtime_sigmoid_sd_pso

runtime_cos_mean_pso
runtime_cos_sd_pso

runtime_sinus_mean_pso
runtime_sinus_sd_pso

runtime_breast_mean_pso
runtime_breast_sd_pso

runtime_tic_mean_pso
runtime_tic_sd_pso

runtime_aust_mean_pso
runtime_aust_sd_pso

runtime_bank_mean_pso
runtime_bank_sd_pso

runtime_blood_mean_pso
runtime_blood_sd_pso

