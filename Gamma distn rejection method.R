#First simulate from exponential
inv.exp <- function(n, lambda) {
  u <- runif(n)
  y <- -log(1-u)/lambda
  y
}

#Function to simulate from Gamma distribution
inv.gamma.int <- function (n, k, lambda) {
  x <- matrix (inv.exp (n=n*k, lambda=lambda), ncol=k)
  apply (x, 1, sum)
}

#Question 4: Write accept-reject function
Gamma = function(n,alpha,lambda){
  k <- floor(alpha)
  x <- alpha-k                          #Optimal x
  M <- dgamma(x,alpha,lambda) / dgamma(x,k,lambda-1)
  y <- vector(mode='numeric', length=n) #Initialise empty vector 
  accepted <- 0                         #Start count at 0     
  while (accepted<n) { 
    x <- inv.gamma.int(1,k,lambda-1)    #Simulate from envelope function
    f_x <- dgamma(x,alpha,lambda)       #Target density
    h_x <- dgamma(x,k,lambda-1)         #Envelope density
    accept_prob <- f_x / (M*h_x)        #Prob of accepting x 
    u <- runif(1)                       
    if (u<accept_prob) { 
      accepted <- accepted+1
      y[accepted] <- x
    }
  }
  return(x)
}


#Replicate 1000 times
z = replicate(1000, Gamma(1, 2.3, 2))
hist(z, freq=TRUE)
prob <- sum(z>4)/1000 
