library(Matrix)
library(ggplot2)
library(magrittr)
library(dplyr)

library(gramar)

# spatial domain (we choose a grid to make a nice image later)
# this generates a dataset of size 6400
xx <- seq(0, 1, length.out=50)
coords <- expand.grid(xx, xx) %>%
  #arrange(Var1, Var2) %>%
  as.matrix()
nn <- nrow(coords)
dd <- 2

axis_partition <- c(10, 10)
fixed_thresholds <- 1:dd %>% 
  lapply(function(i) gramar:::kthresholdscp(coords[,i], axis_partition[i])) 

# spatial data, exponential covariance
# phi=14, sigma^2=2
theta_warper <- matrix(1, ncol=1, nrow=3)
theta_warper[3,] <- 17

results <- gramar:::mgp_precision(coords, coords, fixed_thresholds, theta_warper, 0, 16)

image(results$H)
image(results$Ci_prd)

cholCitsqi <- t(chol(results$Ci_prd + 15 * diag(nn)))

cholCitsqi_cp <- cholCitsqi
#cholCitsqi_cp[46:90, 46:90] <- 2
image(cholCitsqi_cp)

image(dirCi)





ord <- results$linear_sort_map[,2]+1

x1 <- matrix(rnorm(nn), ncol=1)
x2 <- matrix(rnorm(nn), ncol=1)

prdCi <- results$Ci_prd[ord, ord]

t(x1) %*% prdCi %*% x2

m1deep:::mgp_qform(x1, x2, coords, fixed_thresholds, theta_warper, 0, 16)





ord <- results$linear_sort_map[1+results$linear_sort_map[,2],1]+1
H <- results$H#[ord, ord]
image(H)

Ri <- results$Ri#[ord, ord]
image(Ri)

system.time({
  c0 <- chol(Ci) })

system.time({
  c1 <- Cholesky(Ci) })

L <- as(cCi, "Matrix")
P <- as(cCi, "pMatrix")

image(Ci)
image(t(P) %*% L %*% t(L) %*% P)

image(solve( chol(Ci + 1e-6*diag(nn)) ))

image(t(diag(nn) - H) %*% Ri %*% (diag(nn) - H))
