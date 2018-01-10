
# Check fitted line for cubic fit
library(boot)
data(motor)
plot(motor$times,motor$accel)
mod <- lm(motor$accel ~ motor$times + I(motor$times^2) + I(motor$times^3))
lines(motor$times,mod$fitted.values,col='red')
#Cubic fit is poor

# Polyfit functions. Which degree fits best? 
lv_one_out <- function(y, xmat) { # xmat matrix containing x, x^2,..., x^p
  n <- length(y) # number of observations
  predy <- vector(length = n) # initilise the vector of the predicted y
  for(i in 1:n) {
    lmo <- lm(y[-i] ~ xmat[-i,,drop = FALSE]) # apply the regression to all the observation but i
    betahat <- as.vector(lmo$coef) # store the coefficients in the vector beta
    predy[i] <- betahat %*% c(1, xmat[i,]) # predict y_i
  }
  return(predy)
}

polyfit <- function(y, x, maxdeg) {
  n <- length(y)
  # create Xmat matrix containing x, x^2,..., x^maxdeg
  Xmat <- sweep(matrix(rep(x, maxdeg), nrow = n, ncol = maxdeg), 2, 1:maxdeg, '^')
  # create a list of class polyreg that will contain the output
  lmout <- list()
  class(lmout) <- 'polyreg'
  # fit different polynomial regressions, from degree 1 (linear regression) to degree maxdegree
  for(i in 1:maxdeg) { 
    lmo <- lm(y ~ Xmat[, 1:i, drop = FALSE])
    # add the results of the cross validation in a new tab of lmo
    lmo$cv.fitted.values <- lv_one_out(y, Xmat[, 1:i, drop = FALSE])
    lmout[[i]] <- lmo # store the results of this regression in a tab of lmout
  }
  # store also the data in the output
  lmout$x <- x
  lmout$y <- y
  return(lmout)
}

summary.polyreg <- function(lmo.obj) {
  maxdeg <- length(lmo.obj) - 2 # lmo.obj contains the outputs and x and y 
  n <- length(lmo.obj$y) 
  tbl <- matrix(nrow = maxdeg, ncol = 1) # table for the summary
  colnames(tbl) <- 'MSPE'
  for(i in 1:maxdeg) {
    curr.obj <- lmo.obj[[i]]
    errs <- lmo.obj$y - curr.obj$cv.fitted.values # absolute error in predicting y
    spe <- crossprod(errs, errs) # sum of squares of the vector errs
    tbl[i, 1] <- spe / n
  }
  cat('Mean squared predictions errors, by degree \n')
  print(tbl)
}


x <- motor$times
y <- motor$accel
maxdeg <- 10
lmo <- polyfit(y, x, maxdeg)
summary(lmo) # Degree 8 is best
plot(x, y)
lines(x, lmo[[8]]$fitted.values, col = 'red') # Better fit, still not great at the start



