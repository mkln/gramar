predict.gramar <- function(object,
                             newx,
                             n_threads=4,
                             verbose=FALSE, ...){
  
  
  xcentr <- matrix(object$x_centers, nrow=1) %x% matrix(1, nrow=nrow(newx))
  coordsout <- (newx-xcentr) %*% object$proj_coeff
  fixed_thresholds <- object$savedata$fixed_thresholds
  
  preddf <- cbind(yout, Wout, Xout, coordsout) %>% as.data.frame() %>% 
    arrange(PC1, PC2)
  coordsout <- preddf %>% dplyr::select(PC1, PC2) %>% as.matrix()
  Xout_g <- preddf %>% dplyr::select(contains("X")) %>% as.matrix()
  yout_g <- preddf$yout
  Wout <- preddf$Wout
  
  preddf %>% ggplot(aes(X1, X2, fill=Wout, color=Wout)) +
    geom_raster() +
    scale_color_viridis_c() +
    scale_fill_viridis_c() +
    theme_minimal()
  
  coordsin <- object$coordsdata %>% as.matrix()
  
  test_predict0 <- gramar_wpredict_via_prec_part(object$savedata$x, coordsin, object$indexing,
                                                          newx, coordsout, fixed_thresholds, 
                                                          object$w_mcmc, object$theta_mcmc,
                                                          n_threads=n_threads,
                                                          FALSE, FALSE)
  
  mu <- newx %*% object$beta_mcmc + test_predict0$w_predicted
  noise_sd <- sqrt(t(matrix(1, ncol=nrow(newx)) %x% gramarc_out$tausq_mcmc))
  
  yhat <- mu + noise_sd*rnorm(prod(dim(noise_sd)), 0, 1)
  
  return(list(mu=mu, yhat=yhat))
}
