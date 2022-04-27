rm(list=ls())

library(gramar)
library(tidyverse)
library(magrittr)
library(scico)

set.seed(1)

# generate covariates with correlations
p <- 15
Lx <- matrix(0, ncol=6, nrow=p)
diag(Lx) <- c(1, 0.7, .4, .5, .2, .1)
Lx[lower.tri(Lx)] <- runif(sum(lower.tri(Lx)), -1, 1)
C_x <- cov2cor(tcrossprod(Lx))

nr <- 2000
Xin <- mvtnorm::rmvnorm(nr, rep(0, p), C_x)
Xmed <- as.list(Xin %>% apply(2, median) %>% `[`(3:p))
listgrid <- list(
  seq(min(Xin[,1])-.1, max(Xin[,1])+.1, length.out=50),
  seq(min(Xin[,2])-.1, max(Xin[,2])+.1, length.out=50)
) %>% c(Xmed)
Xout <- expand.grid(listgrid)

colnames(Xin) <- colnames(Xout) <- paste0("X", 1:p)
X <- rbind(Xin, Xout) %>% as.matrix()

# generate output using GP, ARD kernel
theta_full <- rexp(p)
theta_full[theta_full < .2] <- 0
theta_true <- theta_full %>% c(2)
beta <- sample(c(0, -1, 1, 0), p, replace=T)

CC <- gramar:::gpkernel(X, theta_true)
LC <- t(chol(CC))
W <- LC %*% rnorm(nrow(X))

y <- X %*% beta + W + rnorm(nrow(X)) * 0.5
colnames(y) <- "y"

# insample
which_in <- 1:(nrow(Xin))
which_out <- setdiff(1:nrow(X), which_in)
yin <- y[which_in]
Xin <- X[which_in,]
Win <- W[which_in]

# outsample
yout <- y[which_out]
Xout <- X[which_out,]
Wout <- W[which_out]


# GP predictions

nrout <- nrow(Xout)
ixi <- 1:nr
ixo <- (nr+1):(nr+nrout)

KK <- function(x, y, theta=theta_true){
  return(gramar:::Correlationc(x, y, theta, F, F))
}

Xall <- rbind(Xin, Xout)
Call <- KK(Xall, Xall)

(Wpred <- KK(Xout, Xin) %*% solve(KK(Xin, Xin), W[1:nr,]))

dfplotc <- Xout[,1:2] %>% cbind(
  data.frame(wpred=Wpred))

gppred <- ggplot() + 
  geom_raster(data=dfplotc, aes(X1, X2, fill=wpred)) +
  scale_fill_viridis_c()



dataplot <- cbind(Xin, yin) %>% as.data.frame() %>% 
  ggplot(aes(x=X1, y=X2, color=yin)) +
  geom_point() +
  scale_color_viridis_c() +
  theme_minimal()

surfplot <- cbind(Xout, Wout) %>% as.data.frame() %>% 
  ggplot(aes(x=X1, y=X2, fill=Wout)) +
  geom_raster() +
  scale_fill_viridis_c() +
  theme_minimal()

set.seed(1)
gramarc_time <- system.time({
  gramarc_out <- gramar::gramar(y=yin, x=Xin, 
                                block_size = 30,
                                proj= "pca",
                                n_samples = 100,
                                n_burnin = 500,
                                n_thin = 1,
                                n_threads = 16,
                                verbose = 20,
                                #starting=list(tausq=.01),
                                debug = list(sample_beta=T, sample_tausq=T, 
                                             sample_theta=T, verbose=F, debug=F))
})



surfplotc <- cbind(gramarc_out$coordsdata, yin) %>% as.data.frame() %>% 
  ggplot(aes(x=PC1, y=PC2, color=yin)) +
  geom_point() +
  scale_color_viridis_c() +
  theme_minimal()


# in the order that was used in gramar
six <- gramarc_out$savedata$sort_ix
coordsin <- gramarc_out$coordsdata %>% as.matrix() %>% `[`(six,)
Xin <- X[six,]
Win <- gramarc_out$w_mcmc[six,]

xcentr <- matrix(gramarc_out$x_centers, nrow=1) %x% matrix(1, nrow=nrow(Xout))
coordsout <- (Xout-xcentr) %*% gramarc_out$proj_coeff
fixed_thresholds <- gramarc_out$savedata$fixed_thresholds

preddf <- cbind(yout, Wout, Xout, coordsout) %>% as.data.frame() %>% 
  arrange(PC1, PC2)
coordsout <- preddf %>% dplyr::select(PC1, PC2) %>% as.matrix()
Xout <- preddf %>% dplyr::select(contains("X")) %>% as.matrix()
yout <- preddf$yout
Wout <- preddf$Wout

preddf %>% ggplot(aes(X1, X2, fill=Wout, color=Wout)) +
  geom_raster() +
  scale_color_viridis_c() +
  scale_fill_viridis_c() +
  theme_minimal()

test_predict <- gramar:::gramar_wpredict_via_prec(Xin, coordsin, gramarc_out$indexing,
                                         Xout, coordsout, fixed_thresholds, 
                                         Win, gramarc_out$theta_mcmc,
                                         TRUE, TRUE)

newp <- test_predict$PP_all

CC <-  solve(test_predict$PP_all)
wcheck <- as.matrix(CC[ixo, ixi] %*% solve(CC[ixi, ixi], W[1:nr,]))




Wpred <- test_predict$w_predicted %>% apply(1, mean)
Wpredlow <- test_predict$w_predicted %>% apply(1, quantile, .05)
Wpredhigh <- test_predict$w_predicted %>% apply(1, quantile, .95)

dfplotc <- coordsout[,1:2] %>% cbind(
  data.frame(wpred=Wpred))

pcplot <- ggplot() + 
  geom_point(data=dfplotc, aes(PC1, PC2, color=wpred)) +
  scale_color_viridis_c()

dfplotx <- Xout[,1:2] %>% cbind(
  data.frame(wpred=Wpred))#,
             #wlow=Wpredlow,
             #whigh=Wpredhigh))

predplot <- ggplot() + 
  geom_raster(data=dfplotx, aes(X1, X2, fill=wpred)) +
  scale_color_viridis_c() +
  scale_fill_viridis_c() + 
  theme_minimal()


gridExtra::grid.arrange(dataplot, surfplot,
                        gppred, predplot, ncol=2)











wgramar <- test_predict$w_predicted %>% apply(1, mean)
ygramar <- Xout %*% beta + wgramar
cbind(wgramar, Wout) %>% cor


mliks <- glist %>% lapply(function(x) x$marglik) %>% abind::abind(along=3) %>% `[`(,1,)
colnames(mliks) <- paste0("model", 1:ncol(mliks)) 
mliks %<>% as.data.frame() %>% 
  mutate(ix=1:n()) %>% pivot_longer(cols=contains("model")) 
ggplot(mliks, aes(x=ix, y=value)) +
  geom_line(aes(color=name))

ww <- data.frame(ww=gramarc_out$w_mcmc %>% apply(1, mean))
latentplot <- cbind(X, y, ww) %>% as.data.frame() %>%
  ggplot(aes(x=X1, y=X2, color=ww)) +
  geom_point() +
  scale_color_viridis_c() +
  theme_minimal()

gridExtra::grid.arrange(dataplot, latentplot, ncol=2)

set.seed(1)
gramars_time <- system.time({
  gramars_out <- meshed:::gramar_pca(y=y, x=X, z=X,
                                     block_size = 50,
                                     n_samples = 10000,
                                     n_burnin = 0,
                                     n_threads = 16,
                                     verbose = 20,
                                     #starting=list(tausq=tsq),
                                     debug = list(sample_beta=T, sample_tausq=T, 
                                                  sample_theta=T, verbose=F, debug=F))
})



nr <- length(y)

yo <- y[gramar_out$savedata$sort_ix]

Ci <- gramar_out$Citsqi
Citsqi <- gramar_out$Citsqi2

cholCi <- t(chol(Ci))
ldetCi <- 2 * sum(log(diag(cholCi)))

cholCitsqi <- t(chol(Citsqi))
ldetCitsqi <- 2 * sum(log(diag(cholCitsqi)))


# find ldens via direct inversion
CC <- solve(Ci)
Ctsq <- CC + diag(nr) * tsq
cholCC <- t(chol(Ctsq))
ldetCC <- 2 * sum(log(diag(cholCC)))
(ldens_correct <- -0.5 * ldetCC - 0.5 * (t(yo) %*% solve(Ctsq, yo) )         )

# find ldens via woodbury and matrix determinant lemma
# determinant
ldet_Ctsq_i <- nr * log(tsq) - ldetCi + ldetCitsqi

solveCy <- solve(Citsqi, yo)

-0.5 * ldet_Ctsq_i - 0.5/tsq * 
  sum(yo^2) + 0.5/tsq^2 * crossprod(yo, solveCy)



gramar_out$ldens

# check ldens
thre <- seq(0,1,length.out=4) %>% head(-1) %>% tail(-1)

test <- gramar:::mgp_precision(X, X, list(thre, thre), rep(1, ncol(X)+1) %>% matrix(ncol=1),
                               0.1)








CC <- gramar:::gpkernel(X, theta)
LC <- t(chol(CC))
v <- rnorm(nrow(X))
W <- LC %*% v

ix1 <- 1:1000
ix2 <- 1001:1400

v1 <- v[ix1]
uu <- v[ix2]
#uu <- rnorm(100)



mychol <- '//[[Rcpp::depends(RcppArmadillo)
#include <RcppArmadillo.h>
//[[Rcpp::export]]
arma::mat achol(const arma::mat& x){
  return arma::chol(arma::symmatu(x), "lower");
}
'
Rcpp::sourceCpp(code=mychol)

Wc1 <- (LC %*% c(v1, uu))[ix2]
HH <- t(solve(CC[ix1, ix1], CC[ix1, ix2]))
RR <- CC[ix2, ix2] - HH %*% CC[ix1, ix2]
Wc2 <- HH %*% v1 + 
  achol(RR) %*% uu



Wc1 <- (LC %*% c(v1, uu))[ix1]
HH <- t(solve(CC[ix2, ix2], CC[ix2, ix1]))
RR <- CC[ix1, ix1] - HH %*% CC[ix2, ix1]
Wc2 <- HH %*% uu + 
  achol(RR) %*% v1





















