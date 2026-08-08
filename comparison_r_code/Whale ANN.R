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

WOA <- function(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness) {
  # Initialize leader position and score
  Leader_pos <- rep(0, dimension)
  Leader_score <- Inf  # Use -Inf for maximization problems
  
  # Initialize the positions of search agents
  Positions <- initialization(SearchAgents, dimension, upperbound, lowerbound)
  Convergence_curve <- rep(0, Max_iterations)
  t <- 0  # Loop counter
  
  # Main loop
  while (t < Max_iterations) {
    for (i in 1:nrow(Positions)) {
      
      # Return back the search agents that go beyond the boundaries of the search space
      Flag4ub <- Positions[i, ] > upperbound
      Flag4lb <- Positions[i, ] < lowerbound
      Positions[i, ] <- (Positions[i, ] * (!(Flag4ub | Flag4lb))) + upperbound * Flag4ub + lowerbound * Flag4lb
      
      # Calculate objective function for each search agent
      fit <- fitness(Positions[i, ], X_train, Y_train, input_dim, hidden_dim, output_dim)
      
      # Update the leader
      if (fit < Leader_score) {  # Change this to > for maximization problem
        Leader_score <- fit  # Update alpha
        Leader_pos <- Positions[i, ]
      }
    }
    
    a <- 2 - t * (2 / Max_iterations)  # a decreases linearly from 2 to 0 in Eq. (2.3)
    
    # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
    a2 <- -1 + t * (-1 / Max_iterations)
    
    # Update the Position of search agents
    for (i in 1:nrow(Positions)) {
      r1 <- runif(1)  # r1 is a random number in [0,1]
      r2 <- runif(1)  # r2 is a random number in [0,1]
      
      A <- 2 * a * r1 - a  # Eq. (2.3)
      C <- 2 * r2  # Eq. (2.4)
      
      b <- 1  # Parameters in Eq. (2.5)
      l <- (a2 - 1) * runif(1) + 1  # Parameters in Eq. (2.5)
      
      p <- runif(1)  # p in Eq. (2.6)
      
      for (j in 1:ncol(Positions)) {
        if (p < 0.5) {
          if (abs(A) >= 1) {
            rand_leader_index <- sample(1:SearchAgents, 1)
            X_rand <- Positions[rand_leader_index, ]
            D_X_rand <- abs(C * X_rand[j] - Positions[i, j])  # Eq. (2.7)
            Positions[i, j] <- X_rand[j] - A * D_X_rand  # Eq. (2.8)
          } else {
            D_Leader <- abs(C * Leader_pos[j] - Positions[i, j])  # Eq. (2.1)
            Positions[i, j] <- Leader_pos[j] - A * D_Leader  # Eq. (2.2)
          }
        } else {
          distance2Leader <- abs(Leader_pos[j] - Positions[i, j])
          Positions[i, j] <- distance2Leader * exp(b * l) * cos(l * 2 * pi) + Leader_pos[j]  # Eq. (2.5)
        }
      }
    }
    
    t <- t + 1
    Convergence_curve[t] <- Leader_score
    
    cat("Iteration", t, ": Best Cost =", Convergence_curve[t], "\n")
  }
  
  return(list(Leader_score = Leader_score, Leader_pos = Leader_pos, Convergence_curve = Convergence_curve))
}

initialization <- function(SearchAgents, dimension, upperbound, lowerbound) {
  matrix(runif(SearchAgents * dimension, lowerbound, upperbound), nrow = SearchAgents, ncol = dimension)
}

# HO algoritmasını çalıştır
WOA_nn <- function(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations, lowerbound = -1, upperbound = 1) {
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
  woa_result <- WOA(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, num_params, fitness_function)
  
  # En iyi ağırlık ve bias değerleri
  best_params <- woa_result$Leader_pos
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
  
  result = list(weights = list(W1 = W1, W2 = W2), biases = list(b1 = b1, b2 = b2), woa_curve = woa_result$Convergence_curve,
                RMSE = RMSE.data, MAE = MAE.data, MAPE = MAPE.data,
                MASE = MASE.data, RSQ = RSQ.data,
                normalization_data_X = X_train, data_X = data_X, data_y = data_y)
  class(result) <- 'WOA_nn'
  return(result)
}

predict.WOA_nn <- function(object, newdata) {
  
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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  # MAE hesaplama
  MAE_woa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sigmoid_mean_woa <- mean(runtime)
runtime_sigmoid_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_sig <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_sig[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_sig
min_values_woa_sig <- sapply(sb_values, min)

iter_min_woa_sig <- format(min(min_values_woa_sig), scientific = TRUE, digits = 3)
iter_mean_woa_sig <- format(mean(min_values_woa_sig), scientific = TRUE, digits = 3)
iter_max_woa_sig <- format(max(min_values_woa_sig), scientific = TRUE, digits = 3)
iter_sd_woa_sig <- format(sd(min_values_woa_sig), scientific = TRUE, digits = 3)


min_mae_woa_sig <- min(mae_values)
mean_mae_woa_sig <- mean(mae_values)
max_mae_woa_sig <- max(mae_values)
sd_mae_woa_sig <- sd(mae_values)



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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_woa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sinus_mean_woa <- mean(runtime)
runtime_sinus_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_sin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_sin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_sin
min_values_woa_sin <- sapply(sb_values, min)

iter_min_woa_sin <- format(min(min_values_woa_sin), scientific = TRUE, digits = 3)
iter_mean_woa_sin <- format(mean(min_values_woa_sin), scientific = TRUE, digits = 3)
iter_max_woa_sin <- format(max(min_values_woa_sin), scientific = TRUE, digits = 3)
iter_sd_woa_sin <- format(sd(min_values_woa_sin), scientific = TRUE, digits = 3)


min_mae_woa_sin <- min(mae_values)
mean_mae_woa_sin <- mean(mae_values)
max_mae_woa_sin <- max(mae_values)
sd_mae_woa_sin <- sd(mae_values)


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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_woa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cos_mean_woa <- mean(runtime)
runtime_cos_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_cosin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_cosin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_cosin
min_values_woa_cosin <- sapply(sb_values, min)

iter_min_woa_cosin <- format(min(min_values_woa_cosin), scientific = TRUE, digits = 3)
iter_mean_woa_cosin <- format(mean(min_values_woa_cosin), scientific = TRUE, digits = 3)
iter_max_woa_cosin <- format(max(min_values_woa_cosin), scientific = TRUE, digits = 3)
iter_sd_woa_cosin <- format(sd(min_values_woa_cosin), scientific = TRUE, digits = 3)


min_mae_woa_cosin <- min(mae_values)
mean_mae_woa_cosin <- mean(mae_values)
max_mae_woa_cosin <- max(mae_values)
sd_mae_woa_cosin <- sd(mae_values)

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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_woa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_woa
  
  # MAE hesaplama
  MAE_woa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_woa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_concrete_mean_woa <- mean(runtime)
runtime_concrete_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_conc <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_conc[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_conc
min_values_woa_conc <- sapply(sb_values, min)

iter_min_woa_conc <- format(min(min_values_woa_conc), scientific = TRUE, digits = 3)
iter_mean_woa_conc <- format(mean(min_values_woa_conc), scientific = TRUE, digits = 3)
iter_max_woa_conc <- format(max(min_values_woa_conc), scientific = TRUE, digits = 3)
iter_sd_woa_conc <- format(sd(min_values_woa_conc), scientific = TRUE, digits = 3)


min_rmse_woa_conc <- min(rmse_values)
mean_rmse_woa_conc <- mean(rmse_values)
max_rmse_woa_conc <- max(rmse_values)
sd_rmse_woa_conc <- sd(rmse_values)


min_mae_woa_conc <- min(mae_values)
mean_mae_woa_conc <- mean(mae_values)
max_mae_woa_conc <- max(mae_values)
sd_mae_woa_conc <- sd(mae_values)


min_r2_woa_conc <- min(r2_values)
mean_r2_woa_conc <- mean(r2_values)
max_r2_woa_conc <- max(r2_values)
sd_r2_woa_conc <- sd(r2_values)


### ----------------------- Energy heating load ------------------------

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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_woa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_woa
  
  # MAE hesaplama
  MAE_woa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_woa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_heat_mean_woa <- mean(runtime)
runtime_heat_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_heat <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_heat[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_heat
min_values_woa_heat <- sapply(sb_values, min)

iter_min_woa_heat <- format(min(min_values_woa_heat), scientific = TRUE, digits = 3)
iter_mean_woa_heat <- format(mean(min_values_woa_heat), scientific = TRUE, digits = 3)
iter_max_woa_heat <- format(max(min_values_woa_heat), scientific = TRUE, digits = 3)
iter_sd_woa_heat <- format(sd(min_values_woa_heat), scientific = TRUE, digits = 3)


min_rmse_woa_heat <- min(rmse_values)
mean_rmse_woa_heat <- mean(rmse_values)
max_rmse_woa_heat <- max(rmse_values)
sd_rmse_woa_heat <- sd(rmse_values)


min_mae_woa_heat <- min(mae_values)
mean_mae_woa_heat <- mean(mae_values)
max_mae_woa_heat <- max(mae_values)
sd_mae_woa_heat <- sd(mae_values)


min_r2_woa_heat <- min(r2_values)
mean_r2_woa_heat <- mean(r2_values)
max_r2_woa_heat <- max(r2_values)
sd_r2_woa_heat <- sd(r2_values)


### ----------------------- Energy Cooling load ------------------------

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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  
  # RMSE hesaplama
  RMSE_woa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_woa
  
  # MAE hesaplama
  MAE_woa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_woa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cool_mean_woa <- mean(runtime)
runtime_cool_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_cool <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_cool[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_cool
min_values_woa_cool <- sapply(sb_values, min)

iter_min_woa_cool <- format(min(min_values_woa_cool), scientific = TRUE, digits = 3)
iter_mean_woa_cool <- format(mean(min_values_woa_cool), scientific = TRUE, digits = 3)
iter_max_woa_cool <- format(max(min_values_woa_cool), scientific = TRUE, digits = 3)
iter_sd_woa_cool <- format(sd(min_values_woa_cool), scientific = TRUE, digits = 3)


min_rmse_woa_cool <- min(rmse_values)
mean_rmse_woa_cool <- mean(rmse_values)
max_rmse_woa_cool <- max(rmse_values)
sd_rmse_woa_cool <- sd(rmse_values)


min_mae_woa_cool <- min(mae_values)
mean_mae_woa_cool <- mean(mae_values)
max_mae_woa_cool <- max(mae_values)
sd_mae_woa_cool <- sd(mae_values)


min_r2_woa_cool <- min(r2_values)
mean_r2_woa_cool <- mean(r2_values)
max_r2_woa_cool <- max(r2_values)
sd_r2_woa_cool <- sd(r2_values)



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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,2:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,1]))
  
  # RMSE hesaplama
  RMSE_woa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_woa
  
  # MAE hesaplama
  MAE_woa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_woa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_auto_mean_woa <- mean(runtime)
runtime_auto_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_mpg <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_mpg[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_mpg
min_values_woa_mpg <- sapply(sb_values, min)

iter_min_woa_mpg <- format(min(min_values_woa_mpg), scientific = TRUE, digits = 3)
iter_mean_woa_mpg <- format(mean(min_values_woa_mpg), scientific = TRUE, digits = 3)
iter_max_woa_mpg <- format(max(min_values_woa_mpg), scientific = TRUE, digits = 3)
iter_sd_woa_mpg <- format(sd(min_values_woa_mpg), scientific = TRUE, digits = 3)



min_rmse_woa_mpg <- min(rmse_values)
mean_rmse_woa_mpg <- mean(rmse_values)
max_rmse_woa_mpg <- max(rmse_values)
sd_rmse_woa_mpg <- sd(rmse_values)


min_mae_woa_mpg <- min(mae_values)
mean_mae_woa_mpg <- mean(mae_values)
max_mae_woa_mpg <- max(mae_values)
sd_mae_woa_mpg <- sd(mae_values)


min_r2_woa_mpg <- min(r2_values)
mean_r2_woa_mpg <- mean(r2_values)
max_r2_woa_mpg <- max(r2_values)
sd_r2_woa_mpg <- sd(r2_values)



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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:6]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,7]))
  
  # RMSE hesaplama
  RMSE_woa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_woa
  
  # MAE hesaplama
  MAE_woa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_woa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_woa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_house_mean_woa <- mean(runtime)
runtime_house_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_house <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_house[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_house
min_values_woa_house <- sapply(sb_values, min)

iter_min_woa_house <- format(min(min_values_woa_house), scientific = TRUE, digits = 3)
iter_mean_woa_house <- format(mean(min_values_woa_house), scientific = TRUE, digits = 3)
iter_max_woa_house <- format(max(min_values_woa_house), scientific = TRUE, digits = 3)
iter_sd_woa_house <- format(sd(min_values_woa_house), scientific = TRUE, digits = 3)



min_rmse_woa_house <- min(rmse_values)
mean_rmse_woa_house <- mean(rmse_values)
max_rmse_woa_house <- max(rmse_values)
sd_rmse_woa_house <- sd(rmse_values)


min_mae_woa_house <- min(mae_values)
mean_mae_woa_house <- mean(mae_values)
max_mae_woa_house <- max(mae_values)
sd_mae_woa_house <- sd(mae_values)


min_r2_woa_house <- min(r2_values)
mean_r2_woa_house <- mean(r2_values)
max_r2_woa_house <- max(r2_values)
sd_r2_woa_house <- sd(r2_values)


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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_woa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_woa
  
  # precision hesaplama
  precision_woa <- TP / (TP + FP)
  precision_values[i] <- precision_woa
  
  # recall hesaplama
  recall_woa <- TP / (TP + FN)
  recall_values[i] <- recall_woa
  
  # f1 score
  f1_score_woa <- 2 * (precision_woa * recall_woa) / (precision_woa + recall_woa)
  f1_score_values[i] <- f1_score_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_breast_mean_woa <- mean(runtime)
runtime_breast_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_cancer <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_cancer[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_cancer
min_values_woa_cancer <- sapply(sb_values, min)

iter_min_woa_cancer <- format(min(min_values_woa_cancer), scientific = TRUE, digits = 3)
iter_mean_woa_cancer <- format(mean(min_values_woa_cancer), scientific = TRUE, digits = 3)
iter_max_woa_cancer <- format(max(min_values_woa_cancer), scientific = TRUE, digits = 3)
iter_sd_woa_cancer <- format(sd(min_values_woa_cancer), scientific = TRUE, digits = 3)


min_accuracy_woa_cancer <- min(accuracy_values)
mean_accuracy_woa_cancer <- mean(accuracy_values)
max_accuracy_woa_cancer <- max(accuracy_values)
sd_accuracy_woa_cancer <- sd(accuracy_values)


min_precision_woa_cancer <- min(precision_values)
mean_precision_woa_cancer <- mean(precision_values)
max_precision_woa_cancer <- max(precision_values)
sd_precision_woa_cancer <- sd(precision_values)


min_recall_woa_cancer <- min(recall_values)
mean_recall_woa_cancer <- mean(recall_values)
max_recall_woa_cancer <- max(recall_values)
sd_recall_woa_cancer <- sd(recall_values)

min_f1_score_woa_cancer <- min(f1_score_values)
mean_f1_score_woa_cancer <- mean(f1_score_values)
max_f1_score_woa_cancer <- max(f1_score_values)
sd_f1_score_woa_cancer <- sd(f1_score_values)


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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_woa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_woa
  
  # precision hesaplama
  precision_woa <- TP / (TP + FP)
  precision_values[i] <- precision_woa
  
  # recall hesaplama
  recall_woa <- TP / (TP + FN)
  recall_values[i] <- recall_woa
  
  # f1 score
  f1_score_woa <- 2 * (precision_woa * recall_woa) / (precision_woa + recall_woa)
  f1_score_values[i] <- f1_score_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_tic_mean_woa <- mean(runtime)
runtime_tic_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_tictac <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_tictac[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_tictac
min_values_woa_tictac <- sapply(sb_values, min)

iter_min_woa_tictac <- format(min(min_values_woa_tictac), scientific = TRUE, digits = 3)
iter_mean_woa_tictac <- format(mean(min_values_woa_tictac), scientific = TRUE, digits = 3)
iter_max_woa_tictac <- format(max(min_values_woa_tictac), scientific = TRUE, digits = 3)
iter_sd_woa_tictac <- format(sd(min_values_woa_tictac), scientific = TRUE, digits = 3)

min_accuracy_woa_tictac <- min(accuracy_values)
mean_accuracy_woa_tictac <- mean(accuracy_values)
max_accuracy_woa_tictac <- max(accuracy_values)
sd_accuracy_woa_tictac <- sd(accuracy_values)

min_precision_woa_tictac <- min(precision_values)
mean_precision_woa_tictac <- mean(precision_values)
max_precision_woa_tictac <- max(precision_values)
sd_precision_woa_tictac <- sd(precision_values)

min_recall_woa_tictac <- min(recall_values)
mean_recall_woa_tictac <- mean(recall_values)
max_recall_woa_tictac <- max(recall_values)
sd_recall_woa_tictac <- sd(recall_values)

min_f1_score_woa_tictac <- min(f1_score_values)
mean_f1_score_woa_tictac <- mean(f1_score_values)
max_f1_score_woa_tictac <- max(f1_score_values)
sd_f1_score_woa_tictac <- sd(f1_score_values)


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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:14]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,15]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_woa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_woa
  
  # precision hesaplama
  precision_woa <- TP / (TP + FP)
  precision_values[i] <- precision_woa
  
  # recall hesaplama
  recall_woa <- TP / (TP + FN)
  recall_values[i] <- recall_woa
  
  # f1 score
  f1_score_woa <- 2 * (precision_woa * recall_woa) / (precision_woa + recall_woa)
  f1_score_values[i] <- f1_score_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_aust_mean_woa <- mean(runtime)
runtime_aust_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_australian <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_australian[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_australian
min_values_woa_australian <- sapply(sb_values, min)

iter_min_woa_australian <- format(min(min_values_woa_australian), scientific = TRUE, digits = 3)
iter_mean_woa_australian <- format(mean(min_values_woa_australian), scientific = TRUE, digits = 3)
iter_max_woa_australian <- format(max(min_values_woa_australian), scientific = TRUE, digits = 3)
iter_sd_woa_australian <- format(sd(min_values_woa_australian), scientific = TRUE, digits = 3)

min_accuracy_woa_australian <- min(accuracy_values)
mean_accuracy_woa_australian <- mean(accuracy_values)
max_accuracy_woa_australian <- max(accuracy_values)
sd_accuracy_woa_australian <- sd(accuracy_values)

min_precision_woa_australian <- min(precision_values)
mean_precision_woa_australian <- mean(precision_values)
max_precision_woa_australian <- max(precision_values)
sd_precision_woa_australian <- sd(precision_values)

min_recall_woa_australian <- min(recall_values)
mean_recall_woa_australian <- mean(recall_values)
max_recall_woa_australian <- max(recall_values)
sd_recall_woa_australian <- sd(recall_values)

min_f1_score_woa_australian <- min(f1_score_values)
mean_f1_score_woa_australian <- mean(f1_score_values)
max_f1_score_woa_australian <- max(f1_score_values)
sd_f1_score_woa_australian <- sd(f1_score_values)



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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_woa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_woa
  
  # precision hesaplama
  precision_woa <- TP / (TP + FP)
  precision_values[i] <- precision_woa
  
  # recall hesaplama
  recall_woa <- TP / (TP + FN)
  recall_values[i] <- recall_woa
  
  # f1 score
  f1_score_woa <- 2 * (precision_woa * recall_woa) / (precision_woa + recall_woa)
  f1_score_values[i] <- f1_score_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_bank_mean_woa <- mean(runtime)
runtime_bank_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_bank <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_bank[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_bank
min_values_woa_bank <- sapply(sb_values, min)

iter_min_woa_bank <- format(min(min_values_woa_bank), scientific = TRUE, digits = 3)
iter_mean_woa_bank <- format(mean(min_values_woa_bank), scientific = TRUE, digits = 3)
iter_max_woa_bank <- format(max(min_values_woa_bank), scientific = TRUE, digits = 3)
iter_sd_woa_bank <- format(sd(min_values_woa_bank), scientific = TRUE, digits = 3)

min_accuracy_woa_bank <- min(accuracy_values)
mean_accuracy_woa_bank <- mean(accuracy_values)
max_accuracy_woa_bank <- max(accuracy_values)
sd_accuracy_woa_bank <- sd(accuracy_values)

min_precision_woa_bank <- min(precision_values)
mean_precision_woa_bank <- mean(precision_values)
max_precision_woa_bank <- max(precision_values)
sd_precision_woa_bank <- sd(precision_values)

min_recall_woa_bank <- min(recall_values)
mean_recall_woa_bank <- mean(recall_values)
max_recall_woa_bank <- max(recall_values)
sd_recall_woa_bank <- sd(recall_values)

min_f1_score_woa_bank <- min(f1_score_values)
mean_f1_score_woa_bank <- mean(f1_score_values)
max_f1_score_woa_bank <- max(f1_score_values)
sd_f1_score_woa_bank <- sd(f1_score_values)



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
  
  # woa ile YSA modelini eğitiyoruz
  result_woa <- WOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_woa, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_woa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_woa
  
  # precision hesaplama
  precision_woa <- TP / (TP + FP)
  precision_values[i] <- precision_woa
  
  # recall hesaplama
  recall_woa <- TP / (TP + FN)
  recall_values[i] <- recall_woa
  
  # f1 score
  f1_score_woa <- 2 * (precision_woa * recall_woa) / (precision_woa + recall_woa)
  f1_score_values[i] <- f1_score_woa
  
  # woa iterasyonlarındaki değerleri saklıyoruz
  woa_curve_iter <- result_woa$woa_curve
  sb_values[[i]] <- woa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_blood_mean_woa <- mean(runtime)
runtime_blood_sd_woa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_woa_blood <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_woa_blood[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_woa_blood
min_values_woa_blood <- sapply(sb_values, min)

iter_min_woa_blood <- format(min(min_values_woa_blood), scientific = TRUE, digits = 3)
iter_mean_woa_blood <- format(mean(min_values_woa_blood), scientific = TRUE, digits = 3)
iter_max_woa_blood <- format(max(min_values_woa_blood), scientific = TRUE, digits = 3)
iter_sd_woa_blood <- format(sd(min_values_woa_blood), scientific = TRUE, digits = 3)

min_accuracy_woa_blood <- min(accuracy_values)
mean_accuracy_woa_blood <- mean(accuracy_values)
max_accuracy_woa_blood <- max(accuracy_values)
sd_accuracy_woa_blood <- sd(accuracy_values)

min_precision_woa_blood <- min(precision_values)
mean_precision_woa_blood <- mean(precision_values)
max_precision_woa_blood <- max(precision_values)
sd_precision_woa_blood <- sd(precision_values)

min_recall_woa_blood <- min(recall_values)
mean_recall_woa_blood <- mean(recall_values)
max_recall_woa_blood <- max(recall_values)
sd_recall_woa_blood <- sd(recall_values)

min_f1_score_woa_blood <- min(f1_score_values)
mean_f1_score_woa_blood <- mean(f1_score_values)
max_f1_score_woa_blood <- max(f1_score_values)
sd_f1_score_woa_blood <- sd(f1_score_values)




runtime_concrete_mean_woa
runtime_concrete_sd_woa

runtime_heat_mean_woa
runtime_heat_sd_woa

runtime_cool_mean_woa
runtime_cool_sd_woa

runtime_auto_mean_woa
runtime_auto_sd_woa

runtime_house_mean_woa
runtime_house_sd_woa

runtime_sigmoid_mean_woa
runtime_sigmoid_sd_woa

runtime_cos_mean_woa
runtime_cos_sd_woa

runtime_sinus_mean_woa
runtime_sinus_sd_woa

runtime_breast_mean_woa
runtime_breast_sd_woa

runtime_tic_mean_woa
runtime_tic_sd_woa

runtime_aust_mean_woa
runtime_aust_sd_woa

runtime_bank_mean_woa
runtime_bank_sd_woa

runtime_blood_mean_woa
runtime_blood_sd_woa