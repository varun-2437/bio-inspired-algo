library(rsample)
library(yardstick)
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
  Y_hat <- sigmoid(Z2)  # Çıkış lineer
  
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

ZOA <- function(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness) {
  
  lowerbound <- rep(lowerbound, dimension)  # Lower limit for variables
  upperbound <- rep(upperbound, dimension)  # Upper limit for variables
  
  # INITIALIZATION
  X <- matrix(0, nrow = SearchAgents, ncol = dimension)
  for (i in 1:dimension) {
    X[, i] <- lowerbound[i] + runif(SearchAgents) * (upperbound[i] - lowerbound[i])  # Initial population
  }
  
  fit <- numeric(SearchAgents)
  for (i in 1:SearchAgents) {
    L <- X[i,]
    fit[i] <- fitness(L, X_train, Y_train, input_dim, hidden_dim, output_dim)
  }
  
  best_so_far <- numeric(Max_iterations)
  average <- numeric(Max_iterations)
  
  # Start iterations
  for (t in 1:Max_iterations) {
    
    # Update the global best (fbest)
    best <- min(fit)
    location <- which.min(fit)
    if (t == 1) {
      PZ <- X[location,]  # Optimal location
      fbest <- best       # The optimization objective function
    } else if (best < fbest) {
      fbest <- best
      PZ <- X[location,]
    }
    
    # PHASE 1: Foraging Behaviour
    for (i in 1:SearchAgents) {
      I <- round(1 + runif(1))
      X_newP1 <- X[i,] + runif(dimension) * (PZ - I * X[i,])  # Eq(3)
      X_newP1 <- pmax(X_newP1, lowerbound)
      X_newP1 <- pmin(X_newP1, upperbound)
      
      # Updating X_i using (5)
      f_newP1 <- fitness(X_newP1, X_train, Y_train, input_dim, hidden_dim, output_dim)
      if (f_newP1 <= fit[i]) {
        X[i,] <- X_newP1
        fit[i] <- f_newP1
      }
    }
    
    # PHASE 2: Defense strategies against predators
    Ps <- runif(1)
    k <- sample(1:SearchAgents, 1)
    AZ <- X[k,]  # Attacked zebra
    
    for (i in 1:SearchAgents) {
      
      if (Ps < 0.5) {
        # S1: The lion attacks the zebra and thus the zebra chooses an escape strategy
        R <- 0.1
        X_newP2 <- X[i,] + R * (2 * runif(dimension) - 1) * (1 - t / Max_iterations) * X[i,]  # Eq.(5) S1
        X_newP2 <- pmax(X_newP2, lowerbound)
        X_newP2 <- pmin(X_newP2, upperbound)
        
      } else {
        # S2: Other predators attack the zebra and the zebra will choose the offensive strategy
        I <- round(1 + runif(1))
        X_newP2 <- X[i,] + runif(dimension) * (AZ - I * X[i,])  # Eq(5) S2
        X_newP2 <- pmax(X_newP2, lowerbound)
        X_newP2 <- pmin(X_newP2, upperbound)
      }
      
      f_newP2 <- fitness(X_newP2, X_train, Y_train, input_dim, hidden_dim, output_dim)  # Eq(6)
      if (f_newP2 <= fit[i]) {
        X[i,] <- X_newP2
        fit[i] <- f_newP2
      }
    }
    
    best_so_far[t] <- fbest
    average[t] <- mean(fit)
    
    cat("Iteration", t, ": Best Cost =", best_so_far[t], "\n")
  }
  
  Best_score <- fbest
  Best_pos <- PZ
  ZOA_curve <- best_so_far
  
  return(list(Best_score = Best_score, Best_pos = Best_pos, ZOA_curve = ZOA_curve))
}


# HO algoritmasını çalıştır
ZOA_nn <- function(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations, lowerbound = -1, upperbound = 1) {
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
  
  # HO algoritmasını çağır
  zoa_result <- ZOA(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, num_params, fitness_function)
  
  # En iyi ağırlık ve bias değerleri
  best_params <- zoa_result$Best_pos
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
  
  result = list(weights = list(W1 = W1, W2 = W2), biases = list(b1 = b1, b2 = b2), zoa_curve = zoa_result$ZOA_curve,
                RMSE = RMSE.data, MAE = MAE.data, MAPE = MAPE.data,
                MASE = MASE.data, RSQ = RSQ.data,
                normalization_data_X = X_train, data_X = data_X, data_y = data_y)
  class(result) <- 'ZOA_nn'
  return(result)
}

predict.ZOA_nn <- function(object, newdata) {
  
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

# Denemeleri gerçekleştirelim
for (i in 1:N) {
  start_time <- Sys.time()
  random_seed <- sample(1:10000, 1)  # Rastgele bir seed seç
  set.seed(random_seed)
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  # MAE hesaplama
  MAE_zoa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  

  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sigmoid_mean_zoa <- mean(runtime)
runtime_sigmoid_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_sig <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_sig[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_sig
min_values_zoa_sig <- sapply(sb_values, min)

iter_min_zoa_sig <- format(min(min_values_zoa_sig), scientific = TRUE, digits = 3)
iter_mean_zoa_sig <- format(mean(min_values_zoa_sig), scientific = TRUE, digits = 3)
iter_max_zoa_sig <- format(max(min_values_zoa_sig), scientific = TRUE, digits = 3)
iter_sd_zoa_sig <- format(sd(min_values_zoa_sig), scientific = TRUE, digits = 3)


min_mae_zoa_sig <- min(mae_values)
mean_mae_zoa_sig <- mean(mae_values)
max_mae_zoa_sig <- max(mae_values)
sd_mae_zoa_sig <- sd(mae_values)



## Sinüs Data ----------------------------
# Sinüs fonksiyonu tanımlama
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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_zoa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sinus_mean_zoa <- mean(runtime)
runtime_sinus_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_sin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_sin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_sin
min_values_zoa_sin <- sapply(sb_values, min)

iter_min_zoa_sin <- format(min(min_values_zoa_sin), scientific = TRUE, digits = 3)
iter_mean_zoa_sin <- format(mean(min_values_zoa_sin), scientific = TRUE, digits = 3)
iter_max_zoa_sin <- format(max(min_values_zoa_sin), scientific = TRUE, digits = 3)
iter_sd_zoa_sin <- format(sd(min_values_zoa_sin), scientific = TRUE, digits = 3)


min_mae_zoa_sin <- min(mae_values)
mean_mae_zoa_sin <- mean(mae_values)
max_mae_zoa_sin <- max(mae_values)
sd_mae_zoa_sin <- sd(mae_values)


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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_zoa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cos_mean_zoa <- mean(runtime)
runtime_cos_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_cosin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_cosin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_cosin
min_values_zoa_cosin <- sapply(sb_values, min)

iter_min_zoa_cosin <- format(min(min_values_zoa_cosin), scientific = TRUE, digits = 3)
iter_mean_zoa_cosin <- format(mean(min_values_zoa_cosin), scientific = TRUE, digits = 3)
iter_max_zoa_cosin <- format(max(min_values_zoa_cosin), scientific = TRUE, digits = 3)
iter_sd_zoa_cosin <- format(sd(min_values_zoa_cosin), scientific = TRUE, digits = 3)


min_mae_zoa_cosin <- min(mae_values)
mean_mae_zoa_cosin <- mean(mae_values)
max_mae_zoa_cosin <- max(mae_values)
sd_mae_zoa_cosin <- sd(mae_values)



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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_zoa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_zoa
  
  # MAE hesaplama
  MAE_zoa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_zoa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_concrete_mean_zoa <- mean(runtime)
runtime_concrete_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_conc <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_conc[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_conc
min_values_zoa_conc <- sapply(sb_values, min)

iter_min_zoa_conc <- format(min(min_values_zoa_conc), scientific = TRUE, digits = 3)
iter_mean_zoa_conc <- format(mean(min_values_zoa_conc), scientific = TRUE, digits = 3)
iter_max_zoa_conc <- format(max(min_values_zoa_conc), scientific = TRUE, digits = 3)
iter_sd_zoa_conc <- format(sd(min_values_zoa_conc), scientific = TRUE, digits = 3)


min_rmse_zoa_conc <- min(rmse_values)
mean_rmse_zoa_conc <- mean(rmse_values)
max_rmse_zoa_conc <- max(rmse_values)
sd_rmse_zoa_conc <- sd(rmse_values)


min_mae_zoa_conc <- min(mae_values)
mean_mae_zoa_conc <- mean(mae_values)
max_mae_zoa_conc <- max(mae_values)
sd_mae_zoa_conc <- sd(mae_values)


min_r2_zoa_conc <- min(r2_values)
mean_r2_zoa_conc <- mean(r2_values)
max_r2_zoa_conc <- max(r2_values)
sd_r2_zoa_conc <- sd(r2_values)



### ----------------------- Energy - heating load -------------------------
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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_zoa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_zoa
  
  # MAE hesaplama
  MAE_zoa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_zoa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_heat_mean_zoa <- mean(runtime)
runtime_heat_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_heat <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_heat[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_heat
min_values_zoa_heat <- sapply(sb_values, min)

iter_min_zoa_heat <- format(min(min_values_zoa_heat), scientific = TRUE, digits = 3)
iter_mean_zoa_heat <- format(mean(min_values_zoa_heat), scientific = TRUE, digits = 3)
iter_max_zoa_heat <- format(max(min_values_zoa_heat), scientific = TRUE, digits = 3)
iter_sd_zoa_heat <- format(sd(min_values_zoa_heat), scientific = TRUE, digits = 3)


min_rmse_zoa_heat <- min(rmse_values)
mean_rmse_zoa_heat <- mean(rmse_values)
max_rmse_zoa_heat <- max(rmse_values)
sd_rmse_zoa_heat <- sd(rmse_values)


min_mae_zoa_heat <- min(mae_values)
mean_mae_zoa_heat <- mean(mae_values)
max_mae_zoa_heat <- max(mae_values)
sd_mae_zoa_heat <- sd(mae_values)


min_r2_zoa_heat <- min(r2_values)
mean_r2_zoa_heat <- mean(r2_values)
max_r2_zoa_heat <- max(r2_values)
sd_r2_zoa_heat <- sd(r2_values)



### ----------------------- Energy - cooling load -------------------------
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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  
  # RMSE hesaplama
  RMSE_zoa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_zoa
  
  # MAE hesaplama
  MAE_zoa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_zoa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cool_mean_zoa <- mean(runtime)
runtime_cool_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_cool <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_cool[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_cool
min_values_zoa_cool <- sapply(sb_values, min)

iter_min_zoa_cool <- format(min(min_values_zoa_cool), scientific = TRUE, digits = 3)
iter_mean_zoa_cool <- format(mean(min_values_zoa_cool), scientific = TRUE, digits = 3)
iter_max_zoa_cool <- format(max(min_values_zoa_cool), scientific = TRUE, digits = 3)
iter_sd_zoa_cool <- format(sd(min_values_zoa_cool), scientific = TRUE, digits = 3)


min_rmse_zoa_cool <- min(rmse_values)
mean_rmse_zoa_cool <- mean(rmse_values)
max_rmse_zoa_cool <- max(rmse_values)
sd_rmse_zoa_cool <- sd(rmse_values)


min_mae_zoa_cool <- min(mae_values)
mean_mae_zoa_cool <- mean(mae_values)
max_mae_zoa_cool <- max(mae_values)
sd_mae_zoa_cool <- sd(mae_values)


min_r2_zoa_cool <- min(r2_values)
mean_r2_zoa_cool <- mean(r2_values)
max_r2_zoa_cool <- max(r2_values)
sd_r2_zoa_cool <- sd(r2_values)



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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,2:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,1]))
  
  # RMSE hesaplama
  RMSE_zoa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_zoa
  
  # MAE hesaplama
  MAE_zoa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_zoa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_auto_mean_zoa <- mean(runtime)
runtime_auto_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_mpg <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_mpg[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_mpg
min_values_zoa_mpg <- sapply(sb_values, min)

iter_min_zoa_mpg <- format(min(min_values_zoa_mpg), scientific = TRUE, digits = 3)
iter_mean_zoa_mpg <- format(mean(min_values_zoa_mpg), scientific = TRUE, digits = 3)
iter_max_zoa_mpg <- format(max(min_values_zoa_mpg), scientific = TRUE, digits = 3)
iter_sd_zoa_mpg <- format(sd(min_values_zoa_mpg), scientific = TRUE, digits = 3)



min_rmse_zoa_mpg <- min(rmse_values)
mean_rmse_zoa_mpg <- mean(rmse_values)
max_rmse_zoa_mpg <- max(rmse_values)
sd_rmse_zoa_mpg <- sd(rmse_values)


min_mae_zoa_mpg <- min(mae_values)
mean_mae_zoa_mpg <- mean(mae_values)
max_mae_zoa_mpg <- max(mae_values)
sd_mae_zoa_mpg <- sd(mae_values)


min_r2_zoa_mpg <- min(r2_values)
mean_r2_zoa_mpg <- mean(r2_values)
max_r2_zoa_mpg <- max(r2_values)
sd_r2_zoa_mpg <- sd(r2_values)


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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:6]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,7]))
  
  # RMSE hesaplama
  RMSE_zoa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_zoa
  
  # MAE hesaplama
  MAE_zoa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_zoa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_zoa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_house_mean_zoa <- mean(runtime)
runtime_house_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_house <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_house[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_house
min_values_zoa_house <- sapply(sb_values, min)

iter_min_zoa_house <- format(min(min_values_zoa_house), scientific = TRUE, digits = 3)
iter_mean_zoa_house <- format(mean(min_values_zoa_house), scientific = TRUE, digits = 3)
iter_max_zoa_house <- format(max(min_values_zoa_house), scientific = TRUE, digits = 3)
iter_sd_zoa_house <- format(sd(min_values_zoa_house), scientific = TRUE, digits = 3)



min_rmse_zoa_house <- min(rmse_values)
mean_rmse_zoa_house <- mean(rmse_values)
max_rmse_zoa_house <- max(rmse_values)
sd_rmse_zoa_house <- sd(rmse_values)


min_mae_zoa_house <- min(mae_values)
mean_mae_zoa_house <- mean(mae_values)
max_mae_zoa_house <- max(mae_values)
sd_mae_zoa_house <- sd(mae_values)


min_r2_zoa_house <- min(r2_values)
mean_r2_zoa_house <- mean(r2_values)
max_r2_zoa_house <- max(r2_values)
sd_r2_zoa_house <- sd(r2_values)


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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_zoa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_zoa
  
  # precision hesaplama
  precision_zoa <- TP / (TP + FP)
  precision_values[i] <- precision_zoa
  
  # recall hesaplama
  recall_zoa <- TP / (TP + FN)
  recall_values[i] <- recall_zoa
  
  # f1 score
  f1_score_zoa <- 2 * (precision_zoa * recall_zoa) / (precision_zoa + recall_zoa)
  f1_score_values[i] <- f1_score_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_breast_mean_zoa <- mean(runtime)
runtime_breast_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_cancer <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_cancer[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_cancer
min_values_zoa_cancer <- sapply(sb_values, min)

iter_min_zoa_cancer <- format(min(min_values_zoa_cancer), scientific = TRUE, digits = 3)
iter_mean_zoa_cancer <- format(mean(min_values_zoa_cancer), scientific = TRUE, digits = 3)
iter_max_zoa_cancer <- format(max(min_values_zoa_cancer), scientific = TRUE, digits = 3)
iter_sd_zoa_cancer <- format(sd(min_values_zoa_cancer), scientific = TRUE, digits = 3)


min_accuracy_zoa_cancer <- min(accuracy_values)
mean_accuracy_zoa_cancer <- mean(accuracy_values)
max_accuracy_zoa_cancer <- max(accuracy_values)
sd_accuracy_zoa_cancer <- sd(accuracy_values)


min_precision_zoa_cancer <- min(precision_values)
mean_precision_zoa_cancer <- mean(precision_values)
max_precision_zoa_cancer <- max(precision_values)
sd_precision_zoa_cancer <- sd(precision_values)


min_recall_zoa_cancer <- min(recall_values)
mean_recall_zoa_cancer <- mean(recall_values)
max_recall_zoa_cancer <- max(recall_values)
sd_recall_zoa_cancer <- sd(recall_values)

min_f1_score_zoa_cancer <- min(f1_score_values)
mean_f1_score_zoa_cancer <- mean(f1_score_values)
max_f1_score_zoa_cancer <- max(f1_score_values)
sd_f1_score_zoa_cancer <- sd(f1_score_values)


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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_zoa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_zoa
  
  # precision hesaplama
  precision_zoa <- TP / (TP + FP)
  precision_values[i] <- precision_zoa
  
  # recall hesaplama
  recall_zoa <- TP / (TP + FN)
  recall_values[i] <- recall_zoa
  
  # f1 score
  f1_score_zoa <- 2 * (precision_zoa * recall_zoa) / (precision_zoa + recall_zoa)
  f1_score_values[i] <- f1_score_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_tic_mean_zoa <- mean(runtime)
runtime_tic_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_tictac <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_tictac[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_tictac
min_values_zoa_tictac <- sapply(sb_values, min)

iter_min_zoa_tictac <- format(min(min_values_zoa_tictac), scientific = TRUE, digits = 3)
iter_mean_zoa_tictac <- format(mean(min_values_zoa_tictac), scientific = TRUE, digits = 3)
iter_max_zoa_tictac <- format(max(min_values_zoa_tictac), scientific = TRUE, digits = 3)
iter_sd_zoa_tictac <- format(sd(min_values_zoa_tictac), scientific = TRUE, digits = 3)

min_accuracy_zoa_tictac <- min(accuracy_values)
mean_accuracy_zoa_tictac <- mean(accuracy_values)
max_accuracy_zoa_tictac <- max(accuracy_values)
sd_accuracy_zoa_tictac <- sd(accuracy_values)

min_precision_zoa_tictac <- min(precision_values)
mean_precision_zoa_tictac <- mean(precision_values)
max_precision_zoa_tictac <- max(precision_values)
sd_precision_zoa_tictac <- sd(precision_values)

min_recall_zoa_tictac <- min(recall_values)
mean_recall_zoa_tictac <- mean(recall_values)
max_recall_zoa_tictac <- max(recall_values)
sd_recall_zoa_tictac <- sd(recall_values)

min_f1_score_zoa_tictac <- min(f1_score_values)
mean_f1_score_zoa_tictac <- mean(f1_score_values)
max_f1_score_zoa_tictac <- max(f1_score_values)
sd_f1_score_zoa_tictac <- sd(f1_score_values)


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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:14]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,15]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_zoa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_zoa
  
  # precision hesaplama
  precision_zoa <- TP / (TP + FP)
  precision_values[i] <- precision_zoa
  
  # recall hesaplama
  recall_zoa <- TP / (TP + FN)
  recall_values[i] <- recall_zoa
  
  # f1 score
  f1_score_zoa <- 2 * (precision_zoa * recall_zoa) / (precision_zoa + recall_zoa)
  f1_score_values[i] <- f1_score_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_aust_mean_zoa <- mean(runtime)
runtime_aust_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_australian <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_australian[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_australian
min_values_zoa_australian <- sapply(sb_values, min)

iter_min_zoa_australian <- format(min(min_values_zoa_australian), scientific = TRUE, digits = 3)
iter_mean_zoa_australian <- format(mean(min_values_zoa_australian), scientific = TRUE, digits = 3)
iter_max_zoa_australian <- format(max(min_values_zoa_australian), scientific = TRUE, digits = 3)
iter_sd_zoa_australian <- format(sd(min_values_zoa_australian), scientific = TRUE, digits = 3)

min_accuracy_zoa_australian <- min(accuracy_values)
mean_accuracy_zoa_australian <- mean(accuracy_values)
max_accuracy_zoa_australian <- max(accuracy_values)
sd_accuracy_zoa_australian <- sd(accuracy_values)

min_precision_zoa_australian <- min(precision_values)
mean_precision_zoa_australian <- mean(precision_values)
max_precision_zoa_australian <- max(precision_values)
sd_precision_zoa_australian <- sd(precision_values)

min_recall_zoa_australian <- min(recall_values)
mean_recall_zoa_australian <- mean(recall_values)
max_recall_zoa_australian <- max(recall_values)
sd_recall_zoa_australian <- sd(recall_values)

min_f1_score_zoa_australian <- min(f1_score_values)
mean_f1_score_zoa_australian <- mean(f1_score_values)
max_f1_score_zoa_australian <- max(f1_score_values)
sd_f1_score_zoa_australian <- sd(f1_score_values)



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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_zoa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_zoa
  
  # precision hesaplama
  precision_zoa <- TP / (TP + FP)
  precision_values[i] <- precision_zoa
  
  # recall hesaplama
  recall_zoa <- TP / (TP + FN)
  recall_values[i] <- recall_zoa
  
  # f1 score
  f1_score_zoa <- 2 * (precision_zoa * recall_zoa) / (precision_zoa + recall_zoa)
  f1_score_values[i] <- f1_score_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_bank_mean_zoa <- mean(runtime)
runtime_bank_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_bank <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_bank[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_bank
min_values_zoa_bank <- sapply(sb_values, min)

iter_min_zoa_bank <- format(min(min_values_zoa_bank), scientific = TRUE, digits = 3)
iter_mean_zoa_bank <- format(mean(min_values_zoa_bank), scientific = TRUE, digits = 3)
iter_max_zoa_bank <- format(max(min_values_zoa_bank), scientific = TRUE, digits = 3)
iter_sd_zoa_bank <- format(sd(min_values_zoa_bank), scientific = TRUE, digits = 3)

min_accuracy_zoa_bank <- min(accuracy_values)
mean_accuracy_zoa_bank <- mean(accuracy_values)
max_accuracy_zoa_bank <- max(accuracy_values)
sd_accuracy_zoa_bank <- sd(accuracy_values)

min_precision_zoa_bank <- min(precision_values)
mean_precision_zoa_bank <- mean(precision_values)
max_precision_zoa_bank <- max(precision_values)
sd_precision_zoa_bank <- sd(precision_values)

min_recall_zoa_bank <- min(recall_values)
mean_recall_zoa_bank <- mean(recall_values)
max_recall_zoa_bank <- max(recall_values)
sd_recall_zoa_bank <- sd(recall_values)

min_f1_score_zoa_bank <- min(f1_score_values)
mean_f1_score_zoa_bank <- mean(f1_score_values)
max_f1_score_zoa_bank <- max(f1_score_values)
sd_f1_score_zoa_bank <- sd(f1_score_values)



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
  
  # zoa ile YSA modelini eğitiyoruz
  result_zoa <- ZOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_zoa, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_zoa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_zoa
  
  # precision hesaplama
  precision_zoa <- TP / (TP + FP)
  precision_values[i] <- precision_zoa
  
  # recall hesaplama
  recall_zoa <- TP / (TP + FN)
  recall_values[i] <- recall_zoa
  
  # f1 score
  f1_score_zoa <- 2 * (precision_zoa * recall_zoa) / (precision_zoa + recall_zoa)
  f1_score_values[i] <- f1_score_zoa
  
  # zoa iterasyonlarındaki değerleri saklıyoruz
  zoa_curve_iter <- result_zoa$zoa_curve
  sb_values[[i]] <- zoa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_blood_mean_zoa <- mean(runtime)
runtime_blood_sd_zoa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_zoa_blood <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_zoa_blood[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_zoa_blood
min_values_zoa_blood <- sapply(sb_values, min)

iter_min_zoa_blood <- format(min(min_values_zoa_blood), scientific = TRUE, digits = 3)
iter_mean_zoa_blood <- format(mean(min_values_zoa_blood), scientific = TRUE, digits = 3)
iter_max_zoa_blood <- format(max(min_values_zoa_blood), scientific = TRUE, digits = 3)
iter_sd_zoa_blood <- format(sd(min_values_zoa_blood), scientific = TRUE, digits = 3)

min_accuracy_zoa_blood <- min(accuracy_values)
mean_accuracy_zoa_blood <- mean(accuracy_values)
max_accuracy_zoa_blood <- max(accuracy_values)
sd_accuracy_zoa_blood <- sd(accuracy_values)

min_precision_zoa_blood <- min(precision_values)
mean_precision_zoa_blood <- mean(precision_values)
max_precision_zoa_blood <- max(precision_values)
sd_precision_zoa_blood <- sd(precision_values)

min_recall_zoa_blood <- min(recall_values)
mean_recall_zoa_blood <- mean(recall_values)
max_recall_zoa_blood <- max(recall_values)
sd_recall_zoa_blood <- sd(recall_values)

min_f1_score_zoa_blood <- min(f1_score_values)
mean_f1_score_zoa_blood <- mean(f1_score_values)
max_f1_score_zoa_blood <- max(f1_score_values)
sd_f1_score_zoa_blood <- sd(f1_score_values)




runtime_concrete_mean_zoa
runtime_concrete_sd_zoa

runtime_heat_mean_zoa
runtime_heat_sd_zoa

runtime_cool_mean_zoa
runtime_cool_sd_zoa

runtime_auto_mean_zoa
runtime_auto_sd_zoa

runtime_house_mean_zoa
runtime_house_sd_zoa

runtime_sigmoid_mean_zoa
runtime_sigmoid_sd_zoa

runtime_cos_mean_zoa
runtime_cos_sd_zoa

runtime_sinus_mean_zoa
runtime_sinus_sd_zoa

runtime_breast_mean_zoa
runtime_breast_sd_zoa

runtime_tic_mean_zoa
runtime_tic_sd_zoa

runtime_aust_mean_zoa
runtime_aust_sd_zoa

runtime_bank_mean_zoa
runtime_bank_sd_zoa

runtime_blood_mean_zoa
runtime_blood_sd_zoa