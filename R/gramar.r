gramar <- function(y, x, k=NULL,
             family = "gaussian",
             axis_partition = NULL, 
             block_size = 30,
             grid_size=NULL,
             grid_custom = NULL,
             n_samples = 1000,
             n_burnin = 100,
             n_thin = 1,
             n_threads = 4,
             verbose = 0,
             predict_everywhere = FALSE,
             settings = list(adapting=TRUE,
                                ps=TRUE, saving=TRUE, low_mem=FALSE, hmc=4),
             prior = list(beta=NULL, tausq=NULL, sigmasq = NULL,
                          phi=NULL, a=NULL, nu = NULL,
                          toplim = NULL, btmlim = NULL, set_unif_bounds=NULL),
             starting = list(beta=NULL, tausq=NULL, theta=NULL, 
                             lambda=NULL, v=NULL,  a=NULL, nu = NULL,
                             mcmcsd=.05, mcmc_startfrom=0),
             debug = list(sample_beta=TRUE, sample_tausq=TRUE, 
                          sample_theta=TRUE, sample_w=TRUE,sample_lambda=TRUE,
                          verbose=FALSE, debug=FALSE)
){

  require(Matrix)
  
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
    
    which_hmc    <- settings$hmc %>% set_default(4)
    if(which_hmc > 4){
      warning("Invalid HMC algorithm choice. Choose settings$hmc in {1,2,3,4}")
      which_hmc <- 4
    }

    mcmc_adaptive    <- settings$adapting %>% set_default(TRUE)
    mcmc_verbose     <- debug$verbose %>% set_default(FALSE)
    mcmc_debug       <- debug$debug %>% set_default(FALSE)
    saving <- settings$saving %>% set_default(TRUE)
    low_mem <- settings$low_mem %>% set_default(FALSE)
    
    debugdag <- 1#debug$dag %>% set_default(1)
    
    proj_dim <- 2
    X_pca <- prcomp(x)
    coords <- X_pca$x[,1:proj_dim] %>% as.matrix()
    coords %<>% as.matrix()
    
    dd             <- ncol(coords)
    p              <- ncol(x)
    
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
        mcmc_tot <- mcmc_burn + mcmc_thin * mcmc_keep
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
    
    q              <- ncol(y)
    k              <- ifelse(is.null(k), q, k)
    
    # family id 
    family <- if(length(family)==1){rep(family, q)} else {family}
    
    if(all(family == "gaussian")){ 
      use_ps <- settings$ps %>% set_default(TRUE)
    } else {
      use_ps <- settings$ps %>% set_default(FALSE)
    }
    
    family_in <- data.frame(family=family)
    available_families <- data.frame(id=0:4, family=c("gaussian", "poisson", "binomial", "beta", "negbinomial"))
    
    suppressMessages(family_id <- family_in %>% 
                       left_join(available_families, by=c("family"="family")) %>% pull(.data$id))
    
    latent <- "gaussian"
    if(!(latent %in% c("gaussian"))){
      stop("Latent process not recognized. Choose 'gaussian'")
    }
    
    # for spatial data: matern or expon, for spacetime: gneiting 2002 
    #n_par_each_process <- ifelse(dd==2, 1, 3) 
    
    nr             <- nrow(x)
    
    if(length(axis_partition) == 1){
      axis_partition <- rep(axis_partition, dd)
    }
    if(is.null(axis_partition)){
      axis_partition <- rep(round((nr/block_size)^(1/dd)), dd)
    }
    
    # -- heuristics for gridded data --
    # if we observed all unique combinations of coordinates, this is how many rows we'd have
    heuristic_gridded <- prod( coords %>% apply(2, function(x) length(unique(x))) )
    # if data are not gridded, then the above should be MUCH larger than the number of rows
    # if we're not too far off maybe the dataset is actually gridded
    if(heuristic_gridded*0.5 < nrow(coords)){
      data_likely_gridded <- TRUE
    } else {
      data_likely_gridded <- FALSE
    }
    if(ncol(coords) == 3){
      # with time, check if there's equal spacing
      timepoints <- coords[,3] %>% unique()
      time_spacings <- timepoints %>% sort() %>% diff() %>% round(5) %>% unique()
      if(length(time_spacings) < .1 * length(timepoints)){
        data_likely_gridded <- TRUE
      }
    }
    
    use_cache <- TRUE 
    
    # what are we sampling
    sample_w       <- debug$sample_w %>% set_default(TRUE)
    sample_beta    <- debug$sample_beta %>% set_default(TRUE)
    sample_tausq   <- debug$sample_tausq %>% set_default(TRUE)
    sample_theta   <- debug$sample_theta %>% set_default(TRUE)
    sample_lambda  <- debug$sample_lambda %>% set_default(TRUE)

  }

  # data management pt 2
  if(1){
    yrownas <- apply(y, 1, function(i) ifelse(sum(is.na(i))==q, NA, 1))
    na_which <- ifelse(!is.na(yrownas), 1, NA)
    simdata <- data.frame(ix=1:nrow(coords)) %>% 
      cbind(coords, y, na_which, x) %>% 
      as.data.frame()
  
    if(length(axis_partition) < ncol(coords)){
      stop("Error: axis_partition not specified for all axes.")
    }
    
    absize <- round(nrow(simdata)/prod(axis_partition))

    
    simdata %<>% 
      dplyr::arrange(!!!rlang::syms(paste0("Var", 1:dd)))
    
    # Domain partitioning and gibbs groups
    fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], axis_partition[i])) 
  }
  
  
  if(1){
    # prior and starting values for mcmc
    
    # nu
    if(is.null(prior$nu)){
      if(is.null(starting$nu)){
        start_nu <- 1.1
      } else {
        start_nu <- starting$nu
      }
      nu_limits <- c(1,1.5)
    } else {
      nu_limits <- prior$nu
      if(is.null(starting$nu)){
        start_nu <- mean(nu_limits)
      } else {
        start_nu <- starting$nu
      }
    }
    
    
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
      phi_limits <- c(1e-5, 2)
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
      mcmc_mh_sd <- diag(k * n_par_each_process) * 0.05
    } else {
      if(length(starting$mcmcsd) == 1){
        mcmc_mh_sd <- diag(k * n_par_each_process) * starting$mcmcsd
      } else {
        mcmc_mh_sd <- starting$mcmcsd
      }
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- family %>% sapply(function(ff) if(ff == "gaussian"){.1} else {1})
    } else {
      start_tausq  <- starting$tausq
    }
    
    if(is.null(starting$lambda)){
      start_lambda <- matrix(0, nrow=q, ncol=k)
      diag(start_lambda) <- if(use_ps){10} else {1}
    } else {
      start_lambda <- starting$lambda
    }
    
    if(is.null(starting$lambda_mask)){
      if(k<=q){
        lambda_mask <- matrix(0, nrow=q, ncol=k)
        lambda_mask[lower.tri(lambda_mask)] <- 1
        diag(lambda_mask) <- 1 #*** 
      } else {
        stop("starting$lambda_mask needs to be specified")
      }
    } else {
      lambda_mask <- starting$lambda_mask
    }
    
    if(is.null(starting$mcmc_startfrom)){
      mcmc_startfrom <- 0
    } else {
      mcmc_startfrom <- starting$mcmc_startfrom
    }
    
    if(is.null(starting$v)){
      start_v <- matrix(0, nrow = nrow(simdata), ncol = k)
    } else {
      # this is used to restart MCMC
      # assumes the ordering and the sizing is correct, 
      # so no change is necessary and will be input directly to mcmc
      start_v <- starting$v #%>% matrix(ncol=q)
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
  
  mcmc_run <- gramar_mcmc_collapsed
  
  comp_time <- system.time({
      results <- mcmc_run(y, family_id, x, coords, 
                          
                          fixed_thresholds,
                          k,
                              
                          
                              set_unif_bounds,
                              beta_Vi, 
                              
                          
                              sigmasq_ab,
                              tausq_ab,
                          
                              start_v, 
                          
                              start_lambda,
                              lambda_mask,
                          
                              start_theta,
                          
                              start_beta,
                              start_tausq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                          
                              mcmc_startfrom,
                              
                              n_threads,
                              
                              which_hmc,
                              mcmc_adaptive, # adapting
                              
                              use_cache,
                          
                              use_ps,
                              
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_print_every, # print all iter
                              low_mem,
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta, sample_tausq, 
                              sample_lambda,
                              sample_theta, 
                              sample_w) 
    })
  
  if(saving){
    
    listN <- function(...){
      anonList <- list(...)
      names(anonList) <- as.character(substitute(list(...)))[-1]
      anonList
    }
    
    saved <- listN(y, x, coords, k,
                    fixed_thresholds,
                   family,
      
                   
      set_unif_bounds,
      beta_Vi, 
      
      tausq_ab,
      
      start_v, 
      
      start_lambda,
      lambda_mask,
      
      start_theta,
      start_beta,
      start_tausq,
      
      mcmc_mh_sd,
      
      mcmc_keep, mcmc_burn, mcmc_thin,
      
      mcmc_startfrom,
      
      n_threads,
      
      mcmc_adaptive, # adapting
      
      
      use_ps,
      
      mcmc_verbose, mcmc_debug, # verbose, debug
      mcmc_print_every, # print all iter
      # sampling of:
      # beta tausq sigmasq theta w
      sample_beta, sample_tausq, 
      sample_lambda,
      sample_theta, sample_w,
      fixed_thresholds)
  } else {
    saved <- "Model data not saved."
  }
  
  returning <- list(coordsdata = coordsdata,
                    savedata = saved) %>% 
    c(results)
  
  class(returning) <- "spmeshed"
  
  return(returning) 
    
}


