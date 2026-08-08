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

SSA <- function(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness) {
  if (length(upperbound) == 1) {
    upperbound <- rep(upperbound, dimension)
    lowerbound <- rep(lowerbound, dimension)
  }
  
  Convergence_curve <- numeric(Max_iterations)
  
  # Initialize the positions of salps
  SalpPositions <- initialization(SearchAgents, dimension, upperbound, lowerbound)
  FoodPosition <- numeric(dimension)
  FoodFitness <- Inf
  
  # Calculate the fitness of initial salps
  SalpFitness <- numeric(SearchAgents)
  for (i in 1:SearchAgents) {
    SalpFitness[i] <- fitness(SalpPositions[i,], X_train, Y_train, input_dim, hidden_dim, output_dim)
  }
  
  # Sort salps based on fitness
  sorted_indexes <- order(SalpFitness)
  Sorted_salps <- SalpPositions[sorted_indexes,]
  FoodPosition <- Sorted_salps[1,]
  FoodFitness <- SalpFitness[sorted_indexes[1]]
  
  # Main loop
  for (l in 1:Max_iterations) {
    c1 <- 2 * exp(-(4 * l / Max_iterations)^2)  # Eq. (3.2) in the paper
    
    for (i in 1:SearchAgents) {
      if (i <= SearchAgents / 2) {
        for (j in 1:dimension) {
          c2 <- runif(1)
          c3 <- runif(1)
          # Eq. (3.1) in the paper
          if (c3 < 0.5) {
            SalpPositions[i, j] <- FoodPosition[j] + c1 * ((upperbound[j] - lowerbound[j]) * c2 + lowerbound[j])
          } else {
            SalpPositions[i, j] <- FoodPosition[j] - c1 * ((upperbound[j] - lowerbound[j]) * c2 + lowerbound[j])
          }
        }
      } else {
        point1 <- SalpPositions[i-1, ]
        point2 <- SalpPositions[i, ]
        SalpPositions[i, ] <- (point2 + point1) / 2  # Eq. (3.4) in the paper
      }
    }
    
    # Boundary check
    for (i in 1:SearchAgents) {
      SalpPositions[i, ] <- pmax(pmin(SalpPositions[i, ], upperbound), lowerbound)
      
      SalpFitness[i] <- fitness(SalpPositions[i, ], X_train, Y_train, input_dim, hidden_dim, output_dim)
      
      if (SalpFitness[i] < FoodFitness) {
        FoodPosition <- SalpPositions[i, ]
        FoodFitness <- SalpFitness[i]
      }
    }
    
    Convergence_curve[l] <- FoodFitness
    cat("Iteration:", l, ", Best fitness:", Convergence_curve[l], "\n")
  }
  
  return(list(FoodFitness = FoodFitness, FoodPosition = FoodPosition, Convergence_curve = Convergence_curve))
}

# Auxiliary function for initialization (you need to define it as well)
initialization <- function(SearchAgents, dimension, upperbound, lowerbound) {
  SalpPositions <- matrix(runif(SearchAgents * dimension), nrow = SearchAgents, ncol = dimension)
  SalpPositions <- sweep(SalpPositions, 2, lowerbound, "*")
  SalpPositions <- sweep(SalpPositions, 2, upperbound - lowerbound, "*")
  return(SalpPositions)
}


# HO algoritmasını çalıştır
SSA_nn <- function(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations, lowerbound = -1, upperbound = 1) {
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
  ssa_result <- SSA(X_train, Y_train, input_dim, hidden_dim, output_dim, SearchAgents, Max_iterations, lowerbound, upperbound, num_params, fitness_function)
  
  # En iyi ağırlık ve bias değerleri
  best_params <- ssa_result$FoodPosition
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
  
  result = list(weights = list(W1 = W1, W2 = W2), biases = list(b1 = b1, b2 = b2), ssa_curve = ssa_result$Convergence_curve,
                RMSE = RMSE.data, MAE = MAE.data, MAPE = MAPE.data,
                MASE = MASE.data, RSQ = RSQ.data,
                normalization_data_X = X_train, data_X = data_X, data_y = data_y)
  class(result) <- 'SSA_nn'
  return(result)
}

predict.SSA_nn <- function(object, newdata) {
  
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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  # MAE hesaplama
  MAE_ssa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa

  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sigmoid_mean_ssa <- mean(runtime)
runtime_sigmoid_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_sig <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_sig[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_sig
min_values_ssa_sig <- sapply(sb_values, min)

iter_min_ssa_sig <- format(min(min_values_ssa_sig), scientific = TRUE, digits = 3)
iter_mean_ssa_sig <- format(mean(min_values_ssa_sig), scientific = TRUE, digits = 3)
iter_max_ssa_sig <- format(max(min_values_ssa_sig), scientific = TRUE, digits = 3)
iter_sd_ssa_sig <- format(sd(min_values_ssa_sig), scientific = TRUE, digits = 3)


min_mae_ssa_sig <- min(mae_values)
mean_mae_ssa_sig <- mean(mae_values)
max_mae_ssa_sig <- max(mae_values)
sd_mae_ssa_sig <- sd(mae_values)



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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_ssa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_sinus_mean_ssa <- mean(runtime)
runtime_sinus_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_sin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_sin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_sin
min_values_ssa_sin <- sapply(sb_values, min)

iter_min_ssa_sin <- format(min(min_values_ssa_sin), scientific = TRUE, digits = 3)
iter_mean_ssa_sin <- format(mean(min_values_ssa_sin), scientific = TRUE, digits = 3)
iter_max_ssa_sin <- format(max(min_values_ssa_sin), scientific = TRUE, digits = 3)
iter_sd_ssa_sin <- format(sd(min_values_ssa_sin), scientific = TRUE, digits = 3)


min_mae_ssa_sin <- min(mae_values)
mean_mae_ssa_sin <- mean(mae_values)
max_mae_ssa_sin <- max(mae_values)
sd_mae_ssa_sin <- sd(mae_values)


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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,2]))
  
  
  # MAE hesaplama
  MAE_ssa <- sum(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cos_mean_ssa <- mean(runtime)
runtime_cos_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_cosin <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_cosin[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_cosin
min_values_ssa_cosin <- sapply(sb_values, min)

iter_min_ssa_cosin <- format(min(min_values_ssa_cosin), scientific = TRUE, digits = 3)
iter_mean_ssa_cosin <- format(mean(min_values_ssa_cosin), scientific = TRUE, digits = 3)
iter_max_ssa_cosin <- format(max(min_values_ssa_cosin), scientific = TRUE, digits = 3)
iter_sd_ssa_cosin <- format(sd(min_values_ssa_cosin), scientific = TRUE, digits = 3)


min_mae_ssa_cosin <- min(mae_values)
mean_mae_ssa_cosin <- mean(mae_values)
max_mae_ssa_cosin <- max(mae_values)
sd_mae_ssa_cosin <- sd(mae_values)

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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_ssa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_ssa
  
  # MAE hesaplama
  MAE_ssa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_ssa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_concrete_mean_ssa <- mean(runtime)
runtime_concrete_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_conc <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_conc[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_conc
min_values_ssa_conc <- sapply(sb_values, min)

iter_min_ssa_conc <- format(min(min_values_ssa_conc), scientific = TRUE, digits = 3)
iter_mean_ssa_conc <- format(mean(min_values_ssa_conc), scientific = TRUE, digits = 3)
iter_max_ssa_conc <- format(max(min_values_ssa_conc), scientific = TRUE, digits = 3)
iter_sd_ssa_conc <- format(sd(min_values_ssa_conc), scientific = TRUE, digits = 3)


min_rmse_ssa_conc <- min(rmse_values)
mean_rmse_ssa_conc <- mean(rmse_values)
max_rmse_ssa_conc <- max(rmse_values)
sd_rmse_ssa_conc <- sd(rmse_values)


min_mae_ssa_conc <- min(mae_values)
mean_mae_ssa_conc <- mean(mae_values)
max_mae_ssa_conc <- max(mae_values)
sd_mae_ssa_conc <- sd(mae_values)


min_r2_ssa_conc <- min(r2_values)
mean_r2_ssa_conc <- mean(r2_values)
max_r2_ssa_conc <- max(r2_values)
sd_r2_ssa_conc <- sd(r2_values)



### ------------------- Energy - heating load ----------------------
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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,9]))
  
  # RMSE hesaplama
  RMSE_ssa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_ssa
  
  # MAE hesaplama
  MAE_ssa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_ssa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_heat_mean_ssa <- mean(runtime)
runtime_heat_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_heat <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_heat[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_heat
min_values_ssa_heat <- sapply(sb_values, min)

iter_min_ssa_heat <- format(min(min_values_ssa_heat), scientific = TRUE, digits = 3)
iter_mean_ssa_heat <- format(mean(min_values_ssa_heat), scientific = TRUE, digits = 3)
iter_max_ssa_heat <- format(max(min_values_ssa_heat), scientific = TRUE, digits = 3)
iter_sd_ssa_heat <- format(sd(min_values_ssa_heat), scientific = TRUE, digits = 3)


min_rmse_ssa_heat <- min(rmse_values)
mean_rmse_ssa_heat <- mean(rmse_values)
max_rmse_ssa_heat <- max(rmse_values)
sd_rmse_ssa_heat <- sd(rmse_values)


min_mae_ssa_heat <- min(mae_values)
mean_mae_ssa_heat <- mean(mae_values)
max_mae_ssa_heat <- max(mae_values)
sd_mae_ssa_heat <- sd(mae_values)


min_r2_ssa_heat <- min(r2_values)
mean_r2_ssa_heat <- mean(r2_values)
max_r2_ssa_heat <- max(r2_values)
sd_r2_ssa_heat <- sd(r2_values)




### ------------------- Energy - cooling load ----------------------
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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  
  # RMSE hesaplama
  RMSE_ssa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_ssa
  
  # MAE hesaplama
  MAE_ssa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_ssa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_cool_mean_ssa <- mean(runtime)
runtime_cool_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_cool <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_cool[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_cool
min_values_ssa_cool <- sapply(sb_values, min)

iter_min_ssa_cool <- format(min(min_values_ssa_cool), scientific = TRUE, digits = 3)
iter_mean_ssa_cool <- format(mean(min_values_ssa_cool), scientific = TRUE, digits = 3)
iter_max_ssa_cool <- format(max(min_values_ssa_cool), scientific = TRUE, digits = 3)
iter_sd_ssa_cool <- format(sd(min_values_ssa_cool), scientific = TRUE, digits = 3)


min_rmse_ssa_cool <- min(rmse_values)
mean_rmse_ssa_cool <- mean(rmse_values)
max_rmse_ssa_cool <- max(rmse_values)
sd_rmse_ssa_cool <- sd(rmse_values)


min_mae_ssa_cool <- min(mae_values)
mean_mae_ssa_cool <- mean(mae_values)
max_mae_ssa_cool <- max(mae_values)
sd_mae_ssa_cool <- sd(mae_values)


min_r2_ssa_cool <- min(r2_values)
mean_r2_ssa_cool <- mean(r2_values)
max_r2_ssa_cool <- max(r2_values)
sd_r2_ssa_cool <- sd(r2_values)





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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,2:8]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,1]))
  
  # RMSE hesaplama
  RMSE_ssa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_ssa
  
  # MAE hesaplama
  MAE_ssa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_ssa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_auto_mean_ssa <- mean(runtime)
runtime_auto_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_mpg <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_mpg[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_mpg
min_values_ssa_mpg <- sapply(sb_values, min)

iter_min_ssa_mpg <- format(min(min_values_ssa_mpg), scientific = TRUE, digits = 3)
iter_mean_ssa_mpg <- format(mean(min_values_ssa_mpg), scientific = TRUE, digits = 3)
iter_max_ssa_mpg <- format(max(min_values_ssa_mpg), scientific = TRUE, digits = 3)
iter_sd_ssa_mpg <- format(sd(min_values_ssa_mpg), scientific = TRUE, digits = 3)



min_rmse_ssa_mpg <- min(rmse_values)
mean_rmse_ssa_mpg <- mean(rmse_values)
max_rmse_ssa_mpg <- max(rmse_values)
sd_rmse_ssa_mpg <- sd(rmse_values)


min_mae_ssa_mpg <- min(mae_values)
mean_mae_ssa_mpg <- mean(mae_values)
max_mae_ssa_mpg <- max(mae_values)
sd_mae_ssa_mpg <- sd(mae_values)


min_r2_ssa_mpg <- min(r2_values)
mean_r2_ssa_mpg <- mean(r2_values)
max_r2_ssa_mpg <- max(r2_values)
sd_r2_ssa_mpg <- sd(r2_values)



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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:6]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,7]))
  
  # RMSE hesaplama
  RMSE_ssa <- sqrt(mean((predictions$test - predictions$yhattest)^2))
  rmse_values[i] <- RMSE_ssa
  
  # MAE hesaplama
  MAE_ssa <- mean(abs(predictions$test - predictions$yhattest))
  mae_values[i] <- MAE_ssa
  
  # R² hesaplama
  ss_total <- sum((predictions$test - mean(predictions$test))^2)
  ss_residual <- sum((predictions$test - predictions$yhattest)^2)
  R2_ssa <- 1 - (ss_residual / ss_total)
  r2_values[i] <- R2_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_house_mean_ssa <- mean(runtime)
runtime_house_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_house <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_house[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_house
min_values_ssa_house <- sapply(sb_values, min)

iter_min_ssa_house <- format(min(min_values_ssa_house), scientific = TRUE, digits = 3)
iter_mean_ssa_house <- format(mean(min_values_ssa_house), scientific = TRUE, digits = 3)
iter_max_ssa_house <- format(max(min_values_ssa_house), scientific = TRUE, digits = 3)
iter_sd_ssa_house <- format(sd(min_values_ssa_house), scientific = TRUE, digits = 3)



min_rmse_ssa_house <- min(rmse_values)
mean_rmse_ssa_house <- mean(rmse_values)
max_rmse_ssa_house <- max(rmse_values)
sd_rmse_ssa_house <- sd(rmse_values)


min_mae_ssa_house <- min(mae_values)
mean_mae_ssa_house <- mean(mae_values)
max_mae_ssa_house <- max(mae_values)
sd_mae_ssa_house <- sd(mae_values)


min_r2_ssa_house <- min(r2_values)
mean_r2_ssa_house <- mean(r2_values)
max_r2_ssa_house <- max(r2_values)
sd_r2_ssa_house <- sd(r2_values)

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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_ssa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_ssa
  
  # precision hesaplama
  precision_ssa <- TP / (TP + FP)
  precision_values[i] <- precision_ssa
  
  # recall hesaplama
  recall_ssa <- TP / (TP + FN)
  recall_values[i] <- recall_ssa
  
  # f1 score
  f1_score_ssa <- 2 * (precision_ssa * recall_ssa) / (precision_ssa + recall_ssa)
  f1_score_values[i] <- f1_score_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_breast_mean_ssa <- mean(runtime)
runtime_breast_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_cancer <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_cancer[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_cancer
min_values_ssa_cancer <- sapply(sb_values, min)

iter_min_ssa_cancer <- format(min(min_values_ssa_cancer), scientific = TRUE, digits = 3)
iter_mean_ssa_cancer <- format(mean(min_values_ssa_cancer), scientific = TRUE, digits = 3)
iter_max_ssa_cancer <- format(max(min_values_ssa_cancer), scientific = TRUE, digits = 3)
iter_sd_ssa_cancer <- format(sd(min_values_ssa_cancer), scientific = TRUE, digits = 3)


min_accuracy_ssa_cancer <- min(accuracy_values)
mean_accuracy_ssa_cancer <- mean(accuracy_values)
max_accuracy_ssa_cancer <- max(accuracy_values)
sd_accuracy_ssa_cancer <- sd(accuracy_values)


min_precision_ssa_cancer <- min(precision_values)
mean_precision_ssa_cancer <- mean(precision_values)
max_precision_ssa_cancer <- max(precision_values)
sd_precision_ssa_cancer <- sd(precision_values)


min_recall_ssa_cancer <- min(recall_values)
mean_recall_ssa_cancer <- mean(recall_values)
max_recall_ssa_cancer <- max(recall_values)
sd_recall_ssa_cancer <- sd(recall_values)

min_f1_score_ssa_cancer <- min(f1_score_values)
mean_f1_score_ssa_cancer <- mean(f1_score_values)
max_f1_score_ssa_cancer <- max(f1_score_values)
sd_f1_score_ssa_cancer <- sd(f1_score_values)


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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:9]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,10]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_ssa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_ssa
  
  # precision hesaplama
  precision_ssa <- TP / (TP + FP)
  precision_values[i] <- precision_ssa
  
  # recall hesaplama
  recall_ssa <- TP / (TP + FN)
  recall_values[i] <- recall_ssa
  
  # f1 score
  f1_score_ssa <- 2 * (precision_ssa * recall_ssa) / (precision_ssa + recall_ssa)
  f1_score_values[i] <- f1_score_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_tic_mean_ssa <- mean(runtime)
runtime_tic_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_tictac <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_tictac[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_tictac
min_values_ssa_tictac <- sapply(sb_values, min)

iter_min_ssa_tictac <- format(min(min_values_ssa_tictac), scientific = TRUE, digits = 3)
iter_mean_ssa_tictac <- format(mean(min_values_ssa_tictac), scientific = TRUE, digits = 3)
iter_max_ssa_tictac <- format(max(min_values_ssa_tictac), scientific = TRUE, digits = 3)
iter_sd_ssa_tictac <- format(sd(min_values_ssa_tictac), scientific = TRUE, digits = 3)

min_accuracy_ssa_tictac <- min(accuracy_values)
mean_accuracy_ssa_tictac <- mean(accuracy_values)
max_accuracy_ssa_tictac <- max(accuracy_values)
sd_accuracy_ssa_tictac <- sd(accuracy_values)

min_precision_ssa_tictac <- min(precision_values)
mean_precision_ssa_tictac <- mean(precision_values)
max_precision_ssa_tictac <- max(precision_values)
sd_precision_ssa_tictac <- sd(precision_values)

min_recall_ssa_tictac <- min(recall_values)
mean_recall_ssa_tictac <- mean(recall_values)
max_recall_ssa_tictac <- max(recall_values)
sd_recall_ssa_tictac <- sd(recall_values)

min_f1_score_ssa_tictac <- min(f1_score_values)
mean_f1_score_ssa_tictac <- mean(f1_score_values)
max_f1_score_ssa_tictac <- max(f1_score_values)
sd_f1_score_ssa_tictac <- sd(f1_score_values)


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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:14]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,15]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_ssa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_ssa
  
  # precision hesaplama
  precision_ssa <- TP / (TP + FP)
  precision_values[i] <- precision_ssa
  
  # recall hesaplama
  recall_ssa <- TP / (TP + FN)
  recall_values[i] <- recall_ssa
  
  # f1 score
  f1_score_ssa <- 2 * (precision_ssa * recall_ssa) / (precision_ssa + recall_ssa)
  f1_score_values[i] <- f1_score_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_aust_mean_ssa <- mean(runtime)
runtime_aust_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_australian <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_australian[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_australian
min_values_ssa_australian <- sapply(sb_values, min)

iter_min_ssa_australian <- format(min(min_values_ssa_australian), scientific = TRUE, digits = 3)
iter_mean_ssa_australian <- format(mean(min_values_ssa_australian), scientific = TRUE, digits = 3)
iter_max_ssa_australian <- format(max(min_values_ssa_australian), scientific = TRUE, digits = 3)
iter_sd_ssa_australian <- format(sd(min_values_ssa_australian), scientific = TRUE, digits = 3)

min_accuracy_ssa_australian <- min(accuracy_values)
mean_accuracy_ssa_australian <- mean(accuracy_values)
max_accuracy_ssa_australian <- max(accuracy_values)
sd_accuracy_ssa_australian <- sd(accuracy_values)

min_precision_ssa_australian <- min(precision_values)
mean_precision_ssa_australian <- mean(precision_values)
max_precision_ssa_australian <- max(precision_values)
sd_precision_ssa_australian <- sd(precision_values)

min_recall_ssa_australian <- min(recall_values)
mean_recall_ssa_australian <- mean(recall_values)
max_recall_ssa_australian <- max(recall_values)
sd_recall_ssa_australian <- sd(recall_values)

min_f1_score_ssa_australian <- min(f1_score_values)
mean_f1_score_ssa_australian <- mean(f1_score_values)
max_f1_score_ssa_australian <- max(f1_score_values)
sd_f1_score_ssa_australian <- sd(f1_score_values)



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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_ssa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_ssa
  
  # precision hesaplama
  precision_ssa <- TP / (TP + FP)
  precision_values[i] <- precision_ssa
  
  # recall hesaplama
  recall_ssa <- TP / (TP + FN)
  recall_values[i] <- recall_ssa
  
  # f1 score
  f1_score_ssa <- 2 * (precision_ssa * recall_ssa) / (precision_ssa + recall_ssa)
  f1_score_values[i] <- f1_score_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_bank_mean_ssa <- mean(runtime)
runtime_bank_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_bank <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_bank[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_bank
min_values_ssa_bank <- sapply(sb_values, min)

iter_min_ssa_bank <- format(min(min_values_ssa_bank), scientific = TRUE, digits = 3)
iter_mean_ssa_bank <- format(mean(min_values_ssa_bank), scientific = TRUE, digits = 3)
iter_max_ssa_bank <- format(max(min_values_ssa_bank), scientific = TRUE, digits = 3)
iter_sd_ssa_bank <- format(sd(min_values_ssa_bank), scientific = TRUE, digits = 3)

min_accuracy_ssa_bank <- min(accuracy_values)
mean_accuracy_ssa_bank <- mean(accuracy_values)
max_accuracy_ssa_bank <- max(accuracy_values)
sd_accuracy_ssa_bank <- sd(accuracy_values)

min_precision_ssa_bank <- min(precision_values)
mean_precision_ssa_bank <- mean(precision_values)
max_precision_ssa_bank <- max(precision_values)
sd_precision_ssa_bank <- sd(precision_values)

min_recall_ssa_bank <- min(recall_values)
mean_recall_ssa_bank <- mean(recall_values)
max_recall_ssa_bank <- max(recall_values)
sd_recall_ssa_bank <- sd(recall_values)

min_f1_score_ssa_bank <- min(f1_score_values)
mean_f1_score_ssa_bank <- mean(f1_score_values)
max_f1_score_ssa_bank <- max(f1_score_values)
sd_f1_score_ssa_bank <- sd(f1_score_values)



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
  
  # ssa ile YSA modelini eğitiyoruz
  result_ssa <- SSA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)
  
  # Test seti için tahminler
  yhattest <- predict(result_ssa, as.matrix(df_test[,1:4]))
  predictions <- as.data.frame(cbind(yhattest = yhattest$yhattest, test = df_test[,5]))
  predictions$yhattest <- ifelse(predictions$yhattest < 0.5, 0, 1)
  predictions
  
  
  # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN) hesaplama
  TP <- sum(predictions$test == 1 & predictions$yhattest == 1)  # Gerçek pozitif ve tahmin edilen pozitif
  FP <- sum(predictions$test == 0 & predictions$yhattest == 1)  # Gerçek negatif ama tahmin edilen pozitif
  TN <- sum(predictions$test == 0 & predictions$yhattest == 0)  # Gerçek negatif ve tahmin edilen negatif
  FN <- sum(predictions$test == 1 & predictions$yhattest == 0)  # Gerçek pozitif ama tahmin edilen negatif
  
  # Accuracy hesaplama
  accuracy_ssa <- (TP + TN) / (TP + TN + FP + FN)
  accuracy_values[i] <- accuracy_ssa
  
  # precision hesaplama
  precision_ssa <- TP / (TP + FP)
  precision_values[i] <- precision_ssa
  
  # recall hesaplama
  recall_ssa <- TP / (TP + FN)
  recall_values[i] <- recall_ssa
  
  # f1 score
  f1_score_ssa <- 2 * (precision_ssa * recall_ssa) / (precision_ssa + recall_ssa)
  f1_score_values[i] <- f1_score_ssa
  
  # ssa iterasyonlarındaki değerleri saklıyoruz
  ssa_curve_iter <- result_ssa$ssa_curve
  sb_values[[i]] <- ssa_curve_iter
  
  cat("experimental Results --------------------", i, "\n")
  end_time <- Sys.time()
  
  runtime[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
}

runtime_blood_mean_ssa <- mean(runtime)
runtime_blood_sd_ssa<- sd(runtime)


# Her bir sıradaki 1'lerin, 2'lerin, 3'lerin ortalamasını hesaplamak için
n <- length(sb_values[[1]])  # Her bir vektörün uzunluğunu alıyoruz (örneğin 3)
means_iter_ssa_blood <- numeric(n)  # Sonuçları depolayacağımız bir vektör oluşturuyoruz

for (i in 1:n) {
  # Tüm i. elemanları alıp ortalamasını hesaplıyoruz
  means_iter_ssa_blood[i] <- mean(sapply(sb_values, function(x) x[i]))
}

means_iter_ssa_blood
min_values_ssa_blood <- sapply(sb_values, min)

iter_min_ssa_blood <- format(min(min_values_ssa_blood), scientific = TRUE, digits = 3)
iter_mean_ssa_blood <- format(mean(min_values_ssa_blood), scientific = TRUE, digits = 3)
iter_max_ssa_blood <- format(max(min_values_ssa_blood), scientific = TRUE, digits = 3)
iter_sd_ssa_blood <- format(sd(min_values_ssa_blood), scientific = TRUE, digits = 3)

min_accuracy_ssa_blood <- min(accuracy_values)
mean_accuracy_ssa_blood <- mean(accuracy_values)
max_accuracy_ssa_blood <- max(accuracy_values)
sd_accuracy_ssa_blood <- sd(accuracy_values)

min_precision_ssa_blood <- min(precision_values)
mean_precision_ssa_blood <- mean(precision_values)
max_precision_ssa_blood <- max(precision_values)
sd_precision_ssa_blood <- sd(precision_values)

min_recall_ssa_blood <- min(recall_values)
mean_recall_ssa_blood <- mean(recall_values)
max_recall_ssa_blood <- max(recall_values)
sd_recall_ssa_blood <- sd(recall_values)

min_f1_score_ssa_blood <- min(f1_score_values)
mean_f1_score_ssa_blood <- mean(f1_score_values)
max_f1_score_ssa_blood <- max(f1_score_values)
sd_f1_score_ssa_blood <- sd(f1_score_values)




runtime_concrete_mean_ssa
runtime_concrete_sd_ssa

runtime_heat_mean_ssa
runtime_heat_sd_ssa

runtime_cool_mean_ssa
runtime_cool_sd_ssa

runtime_auto_mean_ssa
runtime_auto_sd_ssa

runtime_house_mean_ssa
runtime_house_sd_ssa

runtime_sigmoid_mean_ssa
runtime_sigmoid_sd_ssa

runtime_cos_mean_ssa
runtime_cos_sd_ssa

runtime_sinus_mean_ssa
runtime_sinus_sd_ssa

runtime_breast_mean_ssa
runtime_breast_sd_ssa

runtime_tic_mean_ssa
runtime_tic_sd_ssa

runtime_aust_mean_ssa
runtime_aust_sd_ssa

runtime_bank_mean_ssa
runtime_bank_sd_ssa

runtime_blood_mean_ssa
runtime_blood_sd_ssa