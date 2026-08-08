library(rsample)
library(yardstick)
hd <- function (x, probs = seq(0, 1, 0.25), se = FALSE, na.rm = FALSE, 
                names = TRUE, weights = FALSE){
  if (na.rm) {
    na <- is.na(x)
    if (any(na)) 
      x <- x[!na]
  }
  x <- sort(x, na.last = TRUE)
  n <- length(x)
  if (n < 2) 
    return(rep(NA, length(probs)))
  m <- n + 1
  ps <- probs[probs > 0 & probs < 1]
  qs <- 1 - ps
  a <- outer((0:n)/n, ps, function(x, p, m) pbeta(x, p * m, 
                                                  (1 - p) * m), m = m)
  w <- a[-1, ] - a[-m, ]
  r <- drop(x %*% w)
  rp <- range(probs)
  pp <- ps
  if (rp[1] == 0) {
    r <- c(x[1], r)
    pp <- c(0, pp)
  }
  if (rp[2] == 1) {
    r <- c(r, x[n])
    pp <- c(pp, 1)
  }
  r <- r[match(pp, probs)]
  if (names) 
    names(r) <- format(probs)
  if (weights) 
    attr(r, "weights") <- structure(w, dimnames = list(NULL, 
                                                       format(ps)))
  if (!se) 
    return(r)
  if (n < 3) 
    stop("must have n >= 3 to get standard errors")
  l <- n - 1
  a <- outer((0:l)/l, ps, function(x, p, m) pbeta(x, p * m, 
                                                  (1 - p) * m), m = m)
  w <- a[-1, ] - a[-n, ]
  storage.mode(x) <- "double"
  storage.mode(w) <- "double"
  nq <- length(ps)
  S <- matrix(.Fortran(F_jacklins, x, w, as.integer(n), as.integer(nq), 
                       res = double(n * nq))$res, ncol = nq)
  se <- l * sqrt(diag(var(S))/n)
  if (rp[1] == 0) 
    se <- c(NA, se)
  if (rp[2] == 1) 
    se <- c(se, NA)
  se <- se[match(pp, probs)]
  if (names) 
    names(se) <- names(r)
  attr(r, "se") <- se
  r
}
