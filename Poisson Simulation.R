## Simulate from Poisson using simulations from Exp(lambda) 
Poisson <- function(lambda=1){
  k <- 0
  u <- runif(1)
  y <- -(1/lambda)*log(1-u)
  sum <- y
  
  while(sum < 1){
    u <- runif(1)
    y <- -(1/lambda)*log(1-u)
    sum <- sum+y
    k <- k+1
  }
  x <- k
}

#For lambda = 2
pois_rep2 = replicate(100, Poisson(lambda=2))
hist(pois_rep2, freq=FALSE, main='lambda=2')
mean(pois_rep2)
var(pois_rep2)

