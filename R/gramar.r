gramar <- function(y, x, 
             axis_partition = NULL, 
             proj="pca",
             block_size = 30,
             n_samples = 1000,
             n_burnin = 100,
             n_thin = 1,
             n_threads = 4,
             verbose = 0,
             settings = list(adapting=TRUE),
             prior = list(beta=NULL, tausq=NULL, sigmasq = NULL,
                          phi=NULL, toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
             starting = list(beta=NULL, tausq=NULL, theta=NULL, 
                             mcmcsd=.1, mcmc_startfrom=0),
             debug = list(sample_beta=TRUE, sample_tausq=TRUE, 
                          sample_theta=TRUE, verbose=FALSE, debug=FALSE)
){
  
  # init
  if(verbose > 0){
    cat("Bayesian Graph Machine Regression fit via Markov chain Monte Carlo\n")
  }
  model_tag <- "Bayesian Meshed GP regression model\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n(Markov chain Monte Carlo)\n"
  
  set_default <- function(xx, default_val){
    return(if(is.null(xx)){
      default_val} else {
        xx
      })}
  
  # data management pt 1
  if(1){
    mcmc_keep <- n_samples
    mcmc_burn <- n_burnin
    mcmc_thin <- n_thin
    
    mcmc_adaptive    <- settings$adapting %>% set_default(TRUE)
    mcmc_verbose     <- debug$verbose %>% set_default(FALSE)
    mcmc_debug       <- debug$debug %>% set_default(FALSE)
    saving <- settings$saving %>% set_default(TRUE)
    
    p <- ncol(x)
    sumto1 <- function(x) x/sum(x)
    orthproj <- function(x) (diag(nrow(x)) - x %*% solve(crossprod(x), t(x)))
    
    proj_dim <- 2
    x_centers <- x %>% apply(2, mean)
    x_centered <- x %>% apply(2, function(x) x-mean(x))
    if(proj == "pca"){
      X_pca <- prcomp(x_centered)
      coords <- X_pca$x[,1:proj_dim] %>% as.matrix()
      
      proj_coeff <- X_pca$rotation[,1:2]
    } else {
      base1 <- runif(p) %>% sumto1 %>% matrix(ncol=1)
      base2 <- crossprod(orthproj(base1), ( runif(p) %>% sumto1 %>% matrix(ncol=1) ))
      
      proj_coeff <- cbind(base1, base2)
      coords <- x_centered %*% proj_coeff
    }
    
    coords %<>% as.matrix()
    
    dd             <- ncol(coords)
    
    
    # data management part 0 - reshape/rename
    if(is.null(dim(y))){
      y <- matrix(y, ncol=1)
      orig_y_colnames <- colnames(y) <- "Y_1"
    } else {
      if(is.null(colnames(y))){
        orig_y_colnames <- colnames(y) <- paste0('Y_', 1:ncol(y))
      } else {
        orig_y_colnames <- colnames(y)
        colnames(y)     <- paste0('Y_', 1:ncol(y))
      }
    }
    
    if(verbose == 0){
      mcmc_print_every <- 0
    } else {
      if(verbose <= 20){
        mcmc_tot <- mcmc_burn + mcmc_keep
        mcmc_print_every <- 1+round(mcmc_tot / verbose)
      } else {
        if(is.infinite(verbose)){
          mcmc_print_every <- 1
        } else {
          mcmc_print_every <- verbose
        }
      }
    }
    
    if(is.null(colnames(x))){
      orig_X_colnames <- colnames(x) <- paste0('X_', 1:ncol(x))
    } else {
      orig_X_colnames <- colnames(x)
      colnames(x)     <- paste0('X_', 1:ncol(x))
    }
    
    if(is.null(colnames(coords))){
      orig_coords_colnames <- colnames(coords) <- paste0('Var', 1:dd)
    } else {
      orig_coords_colnames <- colnames(coords)
      colnames(coords)     <- paste0('Var', 1:dd)
    }
    
    q <- 1
    k <- 1
    
    nr <- nrow(x)
    
    if(length(axis_partition) == 1){
      axis_partition <- rep(axis_partition, dd)
    }
    if(is.null(axis_partition)){
      axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
    }
    
    # what are we sampling
    sample_beta    <- debug$sample_beta %>% set_default(TRUE)
    sample_tausq   <- debug$sample_tausq %>% set_default(TRUE)
    sample_theta   <- debug$sample_theta %>% set_default(TRUE)

  }

  # data management pt 2
  if(1){
    if(any(is.na(y))){
      stop("Output variable contains NA values.")
    }
    if(any(is.na(x))){
      stop("Input variables contain NA values.")
    }
    if(length(axis_partition) < ncol(coords)){
      stop("Error: axis_partition not specified for all axes.")
    }
    
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, x) %>% 
      as.data.frame() %>%
      dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))

    absize <- round(nrow(simdata)/prod(axis_partition))
    
    # Domain partitioning and gibbs groups
    fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], axis_partition[i])) 
  }
  
  
  if(1){
    # prior and starting values for mcmc
    
    # sigmasq
    if(is.null(prior$sigmasq)){
      if(is.null(starting$sigmasq)){
        start_sigmasq <- 1
      } else {
        start_sigmasq <- starting$sigmasq
      }
      sigmasq_limits <- c(1e-4, 10)
    } else {
      sigmasq_limits <- prior$sigmasq
      if(is.null(starting$sigmasq)){
        start_sigmasq <- mean(sigmasq_limits)
      } else {
        start_sigmasq <- starting$sigmasq
      }
    }
    
    if(is.null(prior$phi)){
      phi_limits <- c(1e-5, 20)
    } else {
      phi_limits <- prior$phi
    }
    
    if(is.null(starting$phi)){
      start_phi <- mean(phi_limits)
    } else {
      start_phi <- starting$phi
    }
    
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(x)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2, 1)
    } else {
      tausq_ab <- prior$tausq
      if(length(tausq_ab) == 1){
        tausq_ab <- c(tausq_ab[1], 0)
      }
    }
    
    if(is.null(prior$sigmasq)){
      sigmasq_ab <- c(2, 1)
    } else {
      sigmasq_ab <- prior$sigmasq
    }
    
    btmlim <- prior$btmlim %>% set_default(1e-3)
    toplim <- prior$toplim %>% set_default(1e3)
    
    # starting values
    if(is.null(starting$beta)){
      start_beta   <- matrix(0, nrow=p, ncol=q)
    } else {
      start_beta   <- starting$beta
    }
    
    if(is.null(prior$set_unif_bounds)){
          theta_names <- c(paste0("phi", 1:p), "sigmasq")
          npar <- length(theta_names)
          
          start_theta <- matrix(0, ncol=k, nrow=npar) 
          set_unif_bounds <- matrix(0, nrow=npar, ncol=2)
          
          # phi
          set_unif_bounds[(1:p),] <- matrix(phi_limits, nrow=1) %x% matrix(1, nrow=p, ncol=1)
          start_theta[1:p,] <- start_phi
          
          # sigmasq
          set_unif_bounds[p+1,] <- c(btmlim, toplim)
          start_theta[p+1,] <- start_sigmasq
          
          set_unif_bounds <- 1:k %>% lapply(\(m) set_unif_bounds) %>%
            do.call(rbind, .)
          
          
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    # override defaults if starting values are provided
    if(!is.null(starting$theta)){
      start_theta <- starting$theta
    }
    
    n_par_each_process <- nrow(start_theta)
    if(is.null(starting$mcmcsd)){
      mcmc_mh_sd <- diag(k * n_par_each_process) * 0.2
    } else {
      if(length(starting$mcmcsd) == 1){
        mcmc_mh_sd <- diag(k * n_par_each_process) * starting$mcmcsd
      } else {
        mcmc_mh_sd <- starting$mcmcsd
      }
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- 1
    } else {
      start_tausq  <- starting$tausq
    }
    
    if(is.null(starting$mcmc_startfrom)){
      mcmc_startfrom <- 0
    } else {
      mcmc_startfrom <- starting$mcmc_startfrom
    }
    
  }
  
  
  y <- simdata %>% 
    dplyr::select(dplyr::contains("Y_")) %>% 
    as.matrix()
  colnames(y) <- orig_y_colnames
  
  x <- simdata %>% 
    dplyr::select(dplyr::contains("X_")) %>% 
    as.matrix()
  colnames(x) <- orig_X_colnames
  x[is.na(x)] <- 0 # NAs if added coords due to empty blocks
  
  na_which <- simdata$na_which

  
  coords <- simdata %>% 
    dplyr::select(dplyr::contains("Var")) %>% 
    as.matrix()
  sort_ix     <- simdata$ix
  
  
  coords_renamer <- colnames(coords)
  names(coords_renamer) <- orig_coords_colnames
  
  #return(list(simdata, coords, orig_coords_colnames))
  
  coordsdata <- simdata %>% 
    dplyr::select(2:(dd+1)) %>%
    dplyr::rename(!!!coords_renamer)
  
  
  if(verbose > 0){
    cat("Sending to MCMC.\n")
  }
  
  mcmc_run <- gramar:::gramar_mcmc_collapsed
  

  
  comp_time <- system.time({
      results <- mcmc_run(y, x, coords, 
                          
                          fixed_thresholds,
                              
                          
                              set_unif_bounds,
                              beta_Vi, 
                              
                          
                              sigmasq_ab,
                              tausq_ab,
                          
                              start_theta,
                              start_beta,
                              start_tausq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                          
                              mcmc_startfrom,
                              
                              n_threads,
                              
                              mcmc_adaptive, # adapting
                              
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_print_every, # print all iter
                          
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta,
                              sample_theta) 
    })
  
  if(saving){
    
    listN <- function(...){
      anonList <- list(...)
      names(anonList) <- as.character(substitute(list(...)))[-1]
      anonList
    }
    
    order_sort_ix <- order(sort_ix)
    saved <- listN(y[order_sort_ix], 
                   x[order_sort_ix,], 
                   coords[order_sort_ix,],

                   sort_ix,
      set_unif_bounds,
      beta_Vi, 
      
      tausq_ab,
      
      start_theta,
      start_beta,
      start_tausq,
      
      mcmc_mh_sd,
      
      mcmc_keep, mcmc_burn, 
      
      mcmc_startfrom,
      
      n_threads,
      
      mcmc_adaptive, # adapting
      
      mcmc_verbose, mcmc_debug, # verbose, debug
      mcmc_print_every, # print all iter
      # sampling of:
      # beta tausq sigmasq theta w
      sample_beta, sample_tausq, 
      sample_theta,
      fixed_thresholds)
  } else {
    saved <- "Model data not saved."
  }
  
  
  results$w_mcmc <- results$w_mcmc[order_sort_ix,]
  
  returning <- list(coordsdata = coordsdata[order_sort_ix,],
                    coordsdata_or = coordsdata,
                    proj_coeff = proj_coeff,
                    x_centers = x_centers,
                    fixed_thresholds = fixed_thresholds,
                    savedata = saved) %>% 
    c(results)
  
  class(returning) <- "gramar"
  
  return(returning) 
    
}


