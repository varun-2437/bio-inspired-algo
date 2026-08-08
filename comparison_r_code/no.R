library(rsample)
library(yardstick)
no<-function(q,x){
  
  n<-length(x)
  x<-sort(x)
  s<-numeric()
  
  
  ifelse(n>2,  {for(g in 1:(n-2)){
    
    s[g]<-x[g+1]*(dbinom(g, size=n, prob=q)*(1-q)+dbinom(g+1, size=n, prob=q)*q)
    
  }
    
    sum(s,na.rm = T)
    t1<-(2*dbinom(0, size=n, prob=q)*q+dbinom(1, size=n, prob=q)*q)*x[1]
    
    t2<-(2*(1-q)*dbinom(n, size=n, prob=q)+dbinom(n-1, size=n, prob=q)*(1-q))*x[n]
    t3<-dbinom(0, size=n, prob=q)*(2-3*q)*x[2]-dbinom(0, size=n, prob=q)*(1-q)*x[3]-dbinom(n, size=n, prob=q)*q*x[n-2]+dbinom(n, size=n, prob=q)*(3*q-1)*x[n-1]
    
    quan<-sum(s,na.rm = T)+t1+t2+t3},
  
  ifelse(n==2,{quan <- (1-q)*x[1]+q*x[2]},quan<-x))
  
  return(quan)
}


no <- function(q, x) {
  n <- length(x)
  x <- sort(x)
  
  if (n > 2) {
    # dbinom değerlerini baştan hesapla
    db <- dbinom(0:n, size = n, prob = q)
    
    g <- 1:(n - 2)
    # vektörleştirilmiş hesaplama
    s <- x[g + 1] * (db[g + 1] * (1 - q) + db[g + 2] * q)
    
    t1 <- (2 * db[1] * q + db[2] * q) * x[1]
    t2 <- (2 * (1 - q) * db[n + 1] + db[n] * (1 - q)) * x[n]
    
    # index düzeltmeleri için dikkatli ol
    t3 <- db[1] * (2 - 3 * q) * x[2] - db[1] * (1 - q) * x[3] -
      db[n + 1] * q * x[n - 2] + db[n + 1] * (3 * q - 1) * x[n - 1]
    
    quan <- sum(s, na.rm = TRUE) + t1 + t2 + t3
    
  } else if (n == 2) {
    quan <- (1 - q) * x[1] + q * x[2]
  } else {
    quan <- x
  }
  
  return(quan)
}
