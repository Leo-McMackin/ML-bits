#Gibbs sampling using uninformative priors
gibbs1 = function(iters, y, mu=0, tau=1){
  alpha0 = .00001
  beta0 = 100 #var
  mu0 = 0
  tau0 = 100 #var
  
  x <- array(0, c(iters+1,2)) #array for storing algorithm output
  x[1, 1] = mu #Initial value for mu
  x[1, 2] = tau #Initial value for tau
  
  n = length(y)
  ybar = mean(y)
  
  for(t in 2:(iters+1)){
    x[t,1] = rnorm(1, (n*ybar*x[t-1,2] + mu0*tau0)/(n*x[t-1,2]+tau0), sqrt(1/(n*x[t-1,2]+tau0))) #Sample mu
    sn = sum((y-x[t,1])^2)
    x[t,2] = rgamma(1, alpha0+n/2)/(beta0+sn/2) #Sample tau
  }
  
  par(mfrow=c(1,2))
  
  plot(1:length(x[,1]),x[,1], type='l', lty=1,xlab='t', ylab='mu')
  plot(1:length(x[,2]),1/x[,2], type='l', lty=1, xlab='t', ylab='sigma^2')
  
  return(x)
  
}

y = rnorm(10, 1, sqrt(2))
g = gibbs1(1000, y)

