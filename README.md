## Graph Machine Regression

Graph Machine Regression (Gramar) is a method for fast Gaussian Process-based regression.
Sparsity in the GP is induced via Directed Acyclic Graphs rather than via small sets of inducing points.

Funding info: ERC 856506, NIH R01ES027498 and R01ES028804.

### Install from Github:
``` r
devtools::install_github("mkln/gramar")
library(gramar)
```

### Basics:
``` r
# your data:
# train data: yin (vector), Xin (matrix, scaled)
# test data: Xout (matrix)

gramar_out <- gramar(y=yin, x=Xin, verbose=5, n_threads=16)
gramar_predict <- predict(gramar_out, newx=Xout, n_threads=16)
```

### Example on Boston housing data:
``` r
library(MASS)
library(dplyr)
library(gramar)

set.seed(1)

sample_in <- sample(1:nrow(Boston), 450, replace=F)
sample_out <- (1:nrow(Boston)) %>% setdiff(sample_in)

BostonIn <- Boston[sample_in,]
BostonOut <- Boston[sample_out,]

# inputs

X_train <- BostonIn %>% 
  dplyr::select(crim, indus, nox, rm, age, ptratio, black, lstat)
X_train_s <- scale(X_train)
xcenters <- attr(X_train_s, "scaled:center")
xscales <- attr(X_train_s, "scaled:scale")

X_test <- BostonOut %>% 
  dplyr::select(crim, indus, nox, rm, age, ptratio, black, lstat)
X_test_s <- t(apply(X_test, 1, \(x) (x - xcenters)/xscales))

# output 
y_train_mean <- mean(BostonIn$medv)
y_train <- BostonIn$medv - y_train_mean

# run gramar
set.seed(1)
gramarc_time <- system.time({
  gramar_out <- gramar::gramar(y=y_train, x=X_train_s, 
                               verbose=5,
                               n_threads=16,
                               n_samples = 1500,
                               n_burnin = 500)})

# predict on test set
gramar_predict <- predict(gramar_out, newx=X_test_s, n_threads = 16)

# assess
yhat_gramar <- apply(gramar_predict$mu, 1, mean) + y_train_mean
sqrt(mean((BostonOut$medv - yhat_gramar)^2))

# make an image of the interaction between the first two predictors
xaxis <- seq(-2, 2, length.out=20)
X_test_image <- expand.grid(xaxis, xaxis) %>%
  cbind(0, 0, 0, 0, 0, 0) %>% as.matrix()
colnames(X_test_image) <- colnames(X_test_s)

gramar_predict_img <- predict(gramar_out, newx=X_test_image, n_threads = 16)

yhat_gramar_img <- gramar_predict_img$mu %>% apply(1, mean)

predictdf <- cbind(X_test_image, yhat_gramar_img) %>% as.data.frame()
ggplot(predictdf, aes(crim, indus, fill=yhat_gramar_img)) +
  geom_raster() +
  scale_fill_viridis_c() +
  theme_minimal()
```


### [Poster at ISBA](poster/poster3648.pdf) World Meeting in Montreal, Jun 29, 2022

 - [.indd file for Gramar](poster/poster3648.indd)
 - ["template" files including fonts](poster/template)

![](poster/posterPNG.png?raw=true)
