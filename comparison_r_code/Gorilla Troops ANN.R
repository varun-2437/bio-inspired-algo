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

GTO <- function(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness) {
  # Initialize Silverback
  Silverback <- numeric(dimension)
  Silverback_Score <- Inf
  # Initialize the first random population of Gorilla
  X <- initialization(SearchAgents, dimension, upperbound, lowerbound)
  convergence_curve <- numeric(Max_iterations)
  
  Pop_Fit <- numeric(SearchAgents)
  
  for (i in 1:SearchAgents) {
    Pop_Fit[i] <- fitness(X[i,], X_train, Y_train, input_dim, hidden_dim, output_dim)
    if (Pop_Fit[i] < Silverback_Score) {
      Silverback_Score <- Pop_Fit[i]
      Silverback <- X[i,]
    }
  }
  
  GX <- X
  lb <- rep(lowerbound, dimension)
  ub <- rep(upperbound, dimension)
  
  # Controlling parameters
  p <- 0.03
  Beta <- 3
  w <- 0.8
  
  # Main loop
  for (It in 1:Max_iterations) {
    a <- (cos(2 * runif(1)) + 1) * (1 - It / Max_iterations)
    C <- a * (2 * runif(1) - 1)
    
    # Exploration:
    for (i in 1:SearchAgents) {
      if (runif(1) < p) {
        GX[i,] <- (ub - lb) * runif(dimension) + lb
      } else {
        if (runif(1) >= 0.5) {
          Z <- runif(dimension, -a, a)
          H <- Z * X[i,]
          GX[i,] <- (runif(1) - a) * X[sample(1:SearchAgents, 1),] + C * H
        } else {
          GX[i,] <- X[i,] - C * (C * (X[i,] - GX[sample(1:SearchAgents, 1),]) + runif(1) * (X[i,] - GX[sample(1:SearchAgents, 1),]))
        }
      }
    }
    
    GX <- boundaryCheck(GX, lowerbound, upperbound)
    
    # Group formation operation
    for (i in 1:SearchAgents) {
      New_Fit <- fitness(GX[i,], X_train, Y_train, input_dim, hidden_dim, output_dim)
      if (New_Fit < Pop_Fit[i]) {
        Pop_Fit[i] <- New_Fit
        X[i,] <- GX[i,]
      }
      if (New_Fit < Silverback_Score) {
        Silverback_Score <- New_Fit
        Silverback <- GX[i,]
      }
    }
    
    # Exploitation:
    for (i in 1:SearchAgents) {
      if (a >= w) {
        g <- 2^C
        delta <- (abs(colMeans(GX))^g)^(1 / g)
        GX[i,] <- C * delta * (X[i,] - Silverback) + X[i,]
      } else {
        h <- ifelse(runif(1) >= 0.5, rnorm(dimension), rnorm(1))
        r1 <- runif(1)
        GX[i,] <- Silverback - (Silverback * (2 * r1 - 1) - X[i,] * (2 * r1 - 1)) * (Beta * h)
      }
    }
    
    GX <- boundaryCheck(GX, lowerbound, upperbound)
    
    # Group formation operation
    for (i in 1:SearchAgents) {
      New_Fit <- fitness(GX[i,], X_train, Y_train, input_dim, hidden_dim, output_dim)
      if (New_Fit < Pop_Fit[i]) {
        Pop_Fit[i] <- New_Fit
        X[i,] <- GX[i,]
      }
      if (New_Fit < Silverback_Score) {
        Silverback_Score <- New_Fit
        Silverback <- GX[i,]
      }
    }
    
    convergence_curve[It] <- Silverback_Score
    cat("Iteration:", It, ", Best fitness:", convergence_curve[It], "\n")
  }
  
  return(list(Silverback_Score = Silverback_Score, Silverback = Silverback, convergence_curve = convergence_curve))
}

boundaryCheck <- function(X, lb, ub) {
  for (i in 1:nrow(X)) {
    FU <- X[i,] > ub
    FL <- X[i,] < lb
    X[i,] <- (X[i,] * !(FU + FL)) + ub * FU + lb * FL
  }
  return(X)
}

# Assuming you have a function 'initialization' to initialize the population
initialization <- function(SearchAgents, dimension, upperbound, lowerbound) {
  matrix(runif(SearchAgents * dimension, lowerbound, upperbound), nrow = SearchAgents, ncol = dimension)
}



# HO algoritmasını çalıştır
GTO_nn <- function(X_train, Y_train, hidden_dim, SearchAgents, Max_iterationsations, lowerbound = -1, upperbound = 1) {
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
  gto_result <- GTO(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterationsations, lowerbound, upperbound, num_params, fitness_function)
  
  # En iyi ağırlık ve bias değerleri
  best_params <- gto_result$Silverback
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
  
  result = list(weights = list(W1 = W1, W2 = W2), biases = list(b1 = b1, b2 = b2), gto_curve = gto_result$convergence_curve,
                RMSE = RMSE.data, MAE = MAE.data, MAPE = MAPE.data,
                MASE = MASE.data, RSQ = RSQ.data,
                normalization_data_X = X_train, data_X = data_X, data_y = data_y)
  class(result) <- 'GTO_nn'
  return(result)
}

predict.GTO_nn <- function(object, newdata) {
  
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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))

  
  # MAE hesaplama
  MAE_gto <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto

  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sigmoid_mean_gto <- mean(runtime)
runtime_sigmoid_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_sig <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_sig[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_sig
min_values_gto_sig <- sapply(sb_values, min)

iter_min_gto_sig <- format(min(min_values_gto_sig), scientific = TRUE, digits = 3)
iter_mean_gto_sig <- format(mean(min_values_gto_sig), scientific = TRUE, digits = 3)
iter_max_gto_sig <- format(max(min_values_gto_sig), scientific = TRUE, digits = 3)
iter_sd_gto_sig <- format(sd(min_values_gto_sig), scientific = TRUE, digits = 3)


min_mae_gto_sig <- min(mae_values)
mean_mae_gto_sig <- mean(mae_values)
max_mae_gto_sig <- max(mae_values)
sd_mae_gto_sig <- sd(mae_values)



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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_gto <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sinus_mean_gto <- mean(runtime)
runtime_sinus_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_sin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_sin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_sin
min_values_gto_sin <- sapply(sb_values, min)

iter_min_gto_sin <- format(min(min_values_gto_sin), scientific = TRUE, digits = 3)
iter_mean_gto_sin <- format(mean(min_values_gto_sin), scientific = TRUE, digits = 3)
iter_max_gto_sin <- format(max(min_values_gto_sin), scientific = TRUE, digits = 3)
iter_sd_gto_sin <- format(sd(min_values_gto_sin), scientific = TRUE, digits = 3)


min_mae_gto_sin <- min(mae_values)
mean_mae_gto_sin <- mean(mae_values)
max_mae_gto_sin <- max(mae_values)
sd_mae_gto_sin <- sd(mae_values)


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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_gto <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cos_mean_gto <- mean(runtime)
runtime_cos_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_cosin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_cosin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_cosin
min_values_gto_cosin <- sapply(sb_values, min)

iter_min_gto_cosin <- format(min(min_values_gto_cosin), scientific = TRUE, digits = 3)
iter_mean_gto_cosin <- format(mean(min_values_gto_cosin), scientific = TRUE, digits = 3)
iter_max_gto_cosin <- format(max(min_values_gto_cosin), scientific = TRUE, digits = 3)
iter_sd_gto_cosin <- format(sd(min_values_gto_cosin), scientific = TRUE, digits = 3)


min_mae_gto_cosin <- min(mae_values)
mean_mae_gto_cosin <- mean(mae_values)
max_mae_gto_cosin <- max(mae_values)
sd_mae_gto_cosin <- sd(mae_values)



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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_gto <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_gto
  
  # MAE hesaplama
  MAE_gto <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_gto <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_concrete_mean_gto <- mean(runtime)
runtime_concrete_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_conc <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_conc[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_conc
min_values_gto_conc <- sapply(sb_values, min)

iter_min_gto_conc <- format(min(min_values_gto_conc), scientific = TRUE, digits = 3)
iter_mean_gto_conc <- format(mean(min_values_gto_conc), scientific = TRUE, digits = 3)
iter_max_gto_conc <- format(max(min_values_gto_conc), scientific = TRUE, digits = 3)
iter_sd_gto_conc <- format(sd(min_values_gto_conc), scientific = TRUE, digits = 3)


min_rmse_gto_conc <- min(rmse_values)
mean_rmse_gto_conc <- mean(rmse_values)
max_rmse_gto_conc <- max(rmse_values)
sd_rmse_gto_conc <- sd(rmse_values)


min_mae_gto_conc <- min(mae_values)
mean_mae_gto_conc <- mean(mae_values)
max_mae_gto_conc <- max(mae_values)
sd_mae_gto_conc <- sd(mae_values)


min_r2_gto_conc <- min(r2_values)
mean_r2_gto_conc <- mean(r2_values)
max_r2_gto_conc <- max(r2_values)
sd_r2_gto_conc <- sd(r2_values)



## ----------------------------- Energy Heatng load----------------------
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
Max_iterationsations <- 250

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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_gto <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_gto
  
  # MAE hesaplama
  MAE_gto <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_gto <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_heat_mean_gto <- mean(runtime)
runtime_heat_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_heat <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_heat[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_heat
min_values_gto_heat <- sapply(sb_values, min)

iter_min_gto_heat <- format(min(min_values_gto_heat), scientific = TRUE, digits = 3)
iter_mean_gto_heat <- format(mean(min_values_gto_heat), scientific = TRUE, digits = 3)
iter_max_gto_heat <- format(max(min_values_gto_heat), scientific = TRUE, digits = 3)
iter_sd_gto_heat <- format(sd(min_values_gto_heat), scientific = TRUE, digits = 3)


min_rmse_gto_heat <- min(rmse_values)
mean_rmse_gto_heat <- mean(rmse_values)
max_rmse_gto_heat <- max(rmse_values)
sd_rmse_gto_heat <- sd(rmse_values)


min_mae_gto_heat <- min(mae_values)
mean_mae_gto_heat <- mean(mae_values)
max_mae_gto_heat <- max(mae_values)
sd_mae_gto_heat <- sd(mae_values)


min_r2_gto_heat <- min(r2_values)
mean_r2_gto_heat <- mean(r2_values)
max_r2_gto_heat <- max(r2_values)
sd_r2_gto_heat <- sd(r2_values)


## ----------------------------- Energy Cooling load----------------------
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
Max_iterationsations <- 250

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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  
  # RMSE hesaplama
  RMSE_gto <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_gto
  
  # MAE hesaplama
  MAE_gto <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_gto <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cool_mean_gto <- mean(runtime)
runtime_cool_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_cool <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_cool[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_cool
min_values_gto_cool <- sapply(sb_values, min)

iter_min_gto_cool <- format(min(min_values_gto_cool), scientific = TRUE, digits = 3)
iter_mean_gto_cool <- format(mean(min_values_gto_cool), scientific = TRUE, digits = 3)
iter_max_gto_cool <- format(max(min_values_gto_cool), scientific = TRUE, digits = 3)
iter_sd_gto_cool <- format(sd(min_values_gto_cool), scientific = TRUE, digits = 3)


min_rmse_gto_cool <- min(rmse_values)
mean_rmse_gto_cool <- mean(rmse_values)
max_rmse_gto_cool <- max(rmse_values)
sd_rmse_gto_cool <- sd(rmse_values)


min_mae_gto_cool <- min(mae_values)
mean_mae_gto_cool <- mean(mae_values)
max_mae_gto_cool <- max(mae_values)
sd_mae_gto_cool <- sd(mae_values)


min_r2_gto_cool <- min(r2_values)
mean_r2_gto_cool <- mean(r2_values)
max_r2_gto_cool <- max(r2_values)
sd_r2_gto_cool <- sd(r2_values)




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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,2:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,1]))
  
  # RMSE hesaplama
  RMSE_gto <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_gto
  
  # MAE hesaplama
  MAE_gto <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_gto <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_auto_mean_gto <- mean(runtime)
runtime_auto_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_mpg <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_mpg[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_mpg
min_values_gto_mpg <- sapply(sb_values, min)

iter_min_gto_mpg <- format(min(min_values_gto_mpg), scientific = TRUE, digits = 3)
iter_mean_gto_mpg <- format(mean(min_values_gto_mpg), scientific = TRUE, digits = 3)
iter_max_gto_mpg <- format(max(min_values_gto_mpg), scientific = TRUE, digits = 3)
iter_sd_gto_mpg <- format(sd(min_values_gto_mpg), scientific = TRUE, digits = 3)



min_rmse_gto_mpg <- min(rmse_values)
mean_rmse_gto_mpg <- mean(rmse_values)
max_rmse_gto_mpg <- max(rmse_values)
sd_rmse_gto_mpg <- sd(rmse_values)


min_mae_gto_mpg <- min(mae_values)
mean_mae_gto_mpg <- mean(mae_values)
max_mae_gto_mpg <- max(mae_values)
sd_mae_gto_mpg <- sd(mae_values)


min_r2_gto_mpg <- min(r2_values)
mean_r2_gto_mpg <- mean(r2_values)
max_r2_gto_mpg <- max(r2_values)
sd_r2_gto_mpg <- sd(r2_values)


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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:6]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,7]))
  
  # RMSE hesaplama
  RMSE_gto <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_gto
  
  # MAE hesaplama
  MAE_gto <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_gto
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_gto <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_house_mean_gto <- mean(runtime)
runtime_house_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_house <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_house[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_house
min_values_gto_house <- sapply(sb_values, min)

iter_min_gto_house <- format(min(min_values_gto_house), scientific = TRUE, digits = 3)
iter_mean_gto_house <- format(mean(min_values_gto_house), scientific = TRUE, digits = 3)
iter_max_gto_house <- format(max(min_values_gto_house), scientific = TRUE, digits = 3)
iter_sd_gto_house <- format(sd(min_values_gto_house), scientific = TRUE, digits = 3)



min_rmse_gto_house <- min(rmse_values)
mean_rmse_gto_house <- mean(rmse_values)
max_rmse_gto_house <- max(rmse_values)
sd_rmse_gto_house <- sd(rmse_values)


min_mae_gto_house <- min(mae_values)
mean_mae_gto_house <- mean(mae_values)
max_mae_gto_house <- max(mae_values)
sd_mae_gto_house <- sd(mae_values)


min_r2_gto_house <- min(r2_values)
mean_r2_gto_house <- mean(r2_values)
max_r2_gto_house <- max(r2_values)
sd_r2_gto_house <- sd(r2_values)

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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_gto <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_gto
  
  # precision hesaplama
  precision_gto <- TP / (TP + FP)
  precision_values[i] <- precision_gto
  
  # recall hesaplama
  recall_gto <- TP / (TP + FN)
  recall_values[i] <- recall_gto
  
  # f1 score
  f1_score_gto <- 2 * (precision_gto * recall_gto) / (precision_gto + recall_gto)
  f1_score_values[i] <- f1_score_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_breast_mean_gto <- mean(runtime)
runtime_breast_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_cancer <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_cancer[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_cancer
min_values_gto_cancer <- sapply(sb_values, min)

iter_min_gto_cancer <- format(min(min_values_gto_cancer), scientific = TRUE, digits = 3)
iter_mean_gto_cancer <- format(mean(min_values_gto_cancer), scientific = TRUE, digits = 3)
iter_max_gto_cancer <- format(max(min_values_gto_cancer), scientific = TRUE, digits = 3)
iter_sd_gto_cancer <- format(sd(min_values_gto_cancer), scientific = TRUE, digits = 3)


min_accuracy_gto_cancer <- min(accuracy_values)
mean_accuracy_gto_cancer <- mean(accuracy_values)
max_accuracy_gto_cancer <- max(accuracy_values)
sd_accuracy_gto_cancer <- sd(accuracy_values)


min_precision_gto_cancer <- min(precision_values)
mean_precision_gto_cancer <- mean(precision_values)
max_precision_gto_cancer <- max(precision_values)
sd_precision_gto_cancer <- sd(precision_values)


min_recall_gto_cancer <- min(recall_values)
mean_recall_gto_cancer <- mean(recall_values)
max_recall_gto_cancer <- max(recall_values)
sd_recall_gto_cancer <- sd(recall_values)

min_f1_score_gto_cancer <- min(f1_score_values)
mean_f1_score_gto_cancer <- mean(f1_score_values)
max_f1_score_gto_cancer <- max(f1_score_values)
sd_f1_score_gto_cancer <- sd(f1_score_values)


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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_gto <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_gto
  
  # precision hesaplama
  precision_gto <- TP / (TP + FP)
  precision_values[i] <- precision_gto
  
  # recall hesaplama
  recall_gto <- TP / (TP + FN)
  recall_values[i] <- recall_gto
  
  # f1 score
  f1_score_gto <- 2 * (precision_gto * recall_gto) / (precision_gto + recall_gto)
  f1_score_values[i] <- f1_score_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_tic_mean_gto <- mean(runtime)
runtime_tic_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_tictac <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_tictac[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_tictac
min_values_gto_tictac <- sapply(sb_values, min)

iter_min_gto_tictac <- format(min(min_values_gto_tictac), scientific = TRUE, digits = 3)
iter_mean_gto_tictac <- format(mean(min_values_gto_tictac), scientific = TRUE, digits = 3)
iter_max_gto_tictac <- format(max(min_values_gto_tictac), scientific = TRUE, digits = 3)
iter_sd_gto_tictac <- format(sd(min_values_gto_tictac), scientific = TRUE, digits = 3)

min_accuracy_gto_tictac <- min(accuracy_values)
mean_accuracy_gto_tictac <- mean(accuracy_values)
max_accuracy_gto_tictac <- max(accuracy_values)
sd_accuracy_gto_tictac <- sd(accuracy_values)

min_precision_gto_tictac <- min(precision_values)
mean_precision_gto_tictac <- mean(precision_values)
max_precision_gto_tictac <- max(precision_values)
sd_precision_gto_tictac <- sd(precision_values)

min_recall_gto_tictac <- min(recall_values)
mean_recall_gto_tictac <- mean(recall_values)
max_recall_gto_tictac <- max(recall_values)
sd_recall_gto_tictac <- sd(recall_values)

min_f1_score_gto_tictac <- min(f1_score_values)
mean_f1_score_gto_tictac <- mean(f1_score_values)
max_f1_score_gto_tictac <- max(f1_score_values)
sd_f1_score_gto_tictac <- sd(f1_score_values)



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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:14]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,15]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_gto <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_gto
  
  # precision hesaplama
  precision_gto <- TP / (TP + FP)
  precision_values[i] <- precision_gto
  
  # recall hesaplama
  recall_gto <- TP / (TP + FN)
  recall_values[i] <- recall_gto
  
  # f1 score
  f1_score_gto <- 2 * (precision_gto * recall_gto) / (precision_gto + recall_gto)
  f1_score_values[i] <- f1_score_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_aust_mean_gto <- mean(runtime)
runtime_aust_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_australian <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_australian[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_australian
min_values_gto_australian <- sapply(sb_values, min)

iter_min_gto_australian <- format(min(min_values_gto_australian), scientific = TRUE, digits = 3)
iter_mean_gto_australian <- format(mean(min_values_gto_australian), scientific = TRUE, digits = 3)
iter_max_gto_australian <- format(max(min_values_gto_australian), scientific = TRUE, digits = 3)
iter_sd_gto_australian <- format(sd(min_values_gto_australian), scientific = TRUE, digits = 3)

min_accuracy_gto_australian <- min(accuracy_values)
mean_accuracy_gto_australian <- mean(accuracy_values)
max_accuracy_gto_australian <- max(accuracy_values)
sd_accuracy_gto_australian <- sd(accuracy_values)

min_precision_gto_australian <- min(precision_values)
mean_precision_gto_australian <- mean(precision_values)
max_precision_gto_australian <- max(precision_values)
sd_precision_gto_australian <- sd(precision_values)

min_recall_gto_australian <- min(recall_values)
mean_recall_gto_australian <- mean(recall_values)
max_recall_gto_australian <- max(recall_values)
sd_recall_gto_australian <- sd(recall_values)

min_f1_score_gto_australian <- min(f1_score_values)
mean_f1_score_gto_australian <- mean(f1_score_values)
max_f1_score_gto_australian <- max(f1_score_values)
sd_f1_score_gto_australian <- sd(f1_score_values)



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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_gto <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_gto
  
  # precision hesaplama
  precision_gto <- TP / (TP + FP)
  precision_values[i] <- precision_gto
  
  # recall hesaplama
  recall_gto <- TP / (TP + FN)
  recall_values[i] <- recall_gto
  
  # f1 score
  f1_score_gto <- 2 * (precision_gto * recall_gto) / (precision_gto + recall_gto)
  f1_score_values[i] <- f1_score_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_bank_mean_gto <- mean(runtime)
runtime_bank_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_bank <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_bank[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_bank
min_values_gto_bank <- sapply(sb_values, min)

iter_min_gto_bank <- format(min(min_values_gto_bank), scientific = TRUE, digits = 3)
iter_mean_gto_bank <- format(mean(min_values_gto_bank), scientific = TRUE, digits = 3)
iter_max_gto_bank <- format(max(min_values_gto_bank), scientific = TRUE, digits = 3)
iter_sd_gto_bank <- format(sd(min_values_gto_bank), scientific = TRUE, digits = 3)

min_accuracy_gto_bank <- min(accuracy_values)
mean_accuracy_gto_bank <- mean(accuracy_values)
max_accuracy_gto_bank <- max(accuracy_values)
sd_accuracy_gto_bank <- sd(accuracy_values)

min_precision_gto_bank <- min(precision_values)
mean_precision_gto_bank <- mean(precision_values)
max_precision_gto_bank <- max(precision_values)
sd_precision_gto_bank <- sd(precision_values)

min_recall_gto_bank <- min(recall_values)
mean_recall_gto_bank <- mean(recall_values)
max_recall_gto_bank <- max(recall_values)
sd_recall_gto_bank <- sd(recall_values)

min_f1_score_gto_bank <- min(f1_score_values)
mean_f1_score_gto_bank <- mean(f1_score_values)
max_f1_score_gto_bank <- max(f1_score_values)
sd_f1_score_gto_bank <- sd(f1_score_values)



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
  
  # gto ile YSA modelini eğitiyoruz
  result_gto <- GTO_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_gto, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_gto <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_gto
  
  # precision hesaplama
  precision_gto <- TP / (TP + FP)
  precision_values[i] <- precision_gto
  
  # recall hesaplama
  recall_gto <- TP / (TP + FN)
  recall_values[i] <- recall_gto
  
  # f1 score
  f1_score_gto <- 2 * (precision_gto * recall_gto) / (precision_gto + recall_gto)
  f1_score_values[i] <- f1_score_gto
  
  # gto iterasyonlarındaki değerleri saklıyoruz
  gto_curve_iter <- result_gto$gto_curve
  sb_values[[i]] <- gto_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_blood_mean_gto <- mean(runtime)
runtime_blood_sd_gto<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_gto_blood <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_gto_blood[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_gto_blood
min_values_gto_blood <- sapply(sb_values, min)

iter_min_gto_blood <- format(min(min_values_gto_blood), scientific = TRUE, digits = 3)
iter_mean_gto_blood <- format(mean(min_values_gto_blood), scientific = TRUE, digits = 3)
iter_max_gto_blood <- format(max(min_values_gto_blood), scientific = TRUE, digits = 3)
iter_sd_gto_blood <- format(sd(min_values_gto_blood), scientific = TRUE, digits = 3)

min_accuracy_gto_blood <- min(accuracy_values)
mean_accuracy_gto_blood <- mean(accuracy_values)
max_accuracy_gto_blood <- max(accuracy_values)
sd_accuracy_gto_blood <- sd(accuracy_values)

min_precision_gto_blood <- min(precision_values)
mean_precision_gto_blood <- mean(precision_values)
max_precision_gto_blood <- max(precision_values)
sd_precision_gto_blood <- sd(precision_values)

min_recall_gto_blood <- min(recall_values)
mean_recall_gto_blood <- mean(recall_values)
max_recall_gto_blood <- max(recall_values)
sd_recall_gto_blood <- sd(recall_values)

min_f1_score_gto_blood <- min(f1_score_values)
mean_f1_score_gto_blood <- mean(f1_score_values)
max_f1_score_gto_blood <- max(f1_score_values)
sd_f1_score_gto_blood <- sd(f1_score_values)




runtime_concrete_mean_gto
runtime_concrete_sd_gto

runtime_heat_mean_gto
runtime_heat_sd_gto

runtime_cool_mean_gto
runtime_cool_sd_gto

runtime_auto_mean_gto
runtime_auto_sd_gto

runtime_house_mean_gto
runtime_house_sd_gto

runtime_sigmoid_mean_gto
runtime_sigmoid_sd_gto

runtime_cos_mean_gto
runtime_cos_sd_gto

runtime_sinus_mean_gto
runtime_sinus_sd_gto

runtime_breast_mean_gto
runtime_breast_sd_gto

runtime_tic_mean_gto
runtime_tic_sd_gto

runtime_aust_mean_gto
runtime_aust_sd_gto

runtime_bank_mean_gto
runtime_bank_sd_gto

runtime_blood_mean_gto
runtime_blood_sd_gto