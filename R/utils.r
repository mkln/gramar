


tessellation_axis_parallel_fix <- function(coordsmat, thresholds, n_threads){
  
  blocks_by_coord <- apply(part_axis_parallel_fixed(coordsmat, thresholds, n_threads), 2, factor)
  colnames(blocks_by_coord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_coord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
  result <- cbind(coordsmat, blockdf)# %>% 
  #mutate(color = (L1+L2) %% 2)
  
  if(ncol(coordsmat)==2){
    result <- cbind(coordsmat, blockdf) #%>% 
    #mutate(color = ((L1-1)*2+(L2-1)) %% 4)
    return(result)
  } else {
    result <- cbind(coordsmat, blockdf) #%>% 
    #mutate(color = 4*(L3 %% 2) + (((L1-1)*2+(L2-1)) %% 4))
    return(result)
  }
}


mesh_graph_build <- function(coords_blocking, Mv, verbose=TRUE, n_threads=1, debugdag=1){
  cbl <- coords_blocking %>% dplyr::select(-dplyr::contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% 
      dplyr::group_by(.data$L1, .data$L2, .data$L3, .data$block) %>% 
      dplyr::summarize(na_which = sum(.data$na_which, na.rm=TRUE)/dplyr::n())#, color=unique(color))
  } else {
    cbl %<>% 
      dplyr::group_by(.data$L1, .data$L2, .data$block) %>% 
      dplyr::summarize(na_which = sum(.data$na_which, na.rm=TRUE)/dplyr::n())#, color=unique(color))
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  
  dag_both_axes <- TRUE
  if(debugdag==1){
    graphed <- mesh_graph_cpp(blocks_descr, Mv, verbose, dag_both_axes, n_threads)
  } else {
    graphed <- mesh_graph_cpp3(blocks_descr)
  }
  #
  
  
  block_ct_obs <- coords_blocking %>% 
    dplyr::group_by(.data$block) %>% 
    dplyr::summarise(block_ct_obs = sum(.data$na_which, na.rm=TRUE)) %>% 
    dplyr::arrange(.data$block) %$% 
    block_ct_obs# %>% `[`(order(block_names))
  
  graph_blanketed <- blanket(graphed$parents, graphed$children, graphed$names, block_ct_obs)
  groups <- coloring(graph_blanketed, graphed$names, block_ct_obs)
  
  blocks_descr %<>% 
    as.data.frame() %>% 
    dplyr::arrange(.data$block) %>% 
    cbind(groups) 
  groups <- blocks_descr$groups#[order(blocks_descr$block)]
  groups[groups == -1] <- max(groups)+1
  
  return(list(parents = graphed$parents,
              children = graphed$children,
              names = graphed$names,
              groups = groups))
}




