rm(list=ls())

# install the meshed package -- requires compilation
#install.packages("R_Code_Presentations/Michele_Peruzzi/meshed_0.21.04.29.tar.gz", repos=NULL, type="source")

library(gramar)
library(tidyverse)
library(magrittr)
library(scico)
library(readxl)

set.seed(20210429)

codebook <- readxl::read_excel("~/gramar-article/ExposomeDataChallenge2021/codebook.xlsx")
load("~/gramar-article/ExposomeDataChallenge2021/exposome_v2_NA.RData")

chemicals <- codebook %>% filter(domain == "Chemicals",
                                 period == "Pregnancy") %$% variable_name %>%
  intersect(colnames(exposomeNA))

exposome_chemicals <- exposomeNA[,chemicals]

gp_inputs <- exposome_chemicals %>% 
  dplyr::select(where(is.numeric)) %>%
  apply(2, scale) %>% as.data.frame() %>%
  bind_cols(exposomeNA %>% dplyr::select(ID), .) %>%
  left_join(covariatesNA %>% dplyr::select(ID, e3_gac_None, h_age_None, hs_child_age_None))

colnames(gp_inputs)[-1] <- paste0("X_", colnames(gp_inputs)[-1])

# covariates with missing data
Ytildes <- covariatesNA %>% 
  dplyr::select(ID, 
                #hs_wgtgain_None, 
                h_mbmi_None#,  
                #hs_c_height_None, 
                #hs_c_weight_None
  )
colnames(Ytildes)[-1] <- paste0("Yt_", colnames(Ytildes)[-1])

# covariates & confounders
covariatesNA_sub <- covariatesNA %>%
  dplyr::filter(complete.cases(h_edumc_None, h_native_None, h_age_None, h_parity_None)) %>%
  dplyr::select(-h_mbmi_None, 
                -e3_gac_None, 
                
                -h_age_None, 
                -hs_child_age_None,
                -hs_wgtgain_None, 
                -hs_c_height_None, 
                -hs_c_weight_None) %>%
  mutate(e3_sex_None = 1*(e3_sex_None == "female"))
covariatesNA_sub <- model.matrix(~. -1, data=covariatesNA_sub) %>% as.data.frame() %>%
  dplyr::select(-h_cohort6) 
colnames(covariatesNA_sub)[-1] <- paste0("Z_", colnames(covariatesNA_sub)[-1])

# outcomes
phenotypeNA_sub <- phenotypeNA %>% 
  dplyr::select(ID, 
                e3_bw,
                hs_zbmi_who#,
                #hs_bmi_c_cat
  ) #%>% 
#mutate(overweight = 1*(hs_bmi_c_cat %in% c(3,4))) %>% 
#dplyr::select(-hs_bmi_c_cat)
colnames(phenotypeNA_sub)[-1] <- paste0("Y_", colnames(phenotypeNA_sub)[-1])

dataf <- covariatesNA_sub %>% 
  left_join(phenotypeNA_sub) %>% 
  left_join(Ytildes) %>%
  left_join(gp_inputs)


Z <- dataf %>% 
  dplyr::select(contains("Z_", ignore.case=F)) %>%
  as.matrix() 
# gelman : http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
#Z[,c("Z_e3_gac_None", "Z_h_age_None", "Z_hs_child_age_None")] %<>%
#  apply(2, function(x) (x-mean(x))/(2*sd(x))) 
#Z <- matrix(1, ncol=1, nrow=nrow(dataf))

X <- dataf %>% 
  dplyr::select(contains("X_", ignore.case=F)) %>% 
  as.matrix() %>% apply(2, function(x) scale(x) %>% as.numeric())

Y <- dataf %>% 
  dplyr::select(contains("Yt_", ignore.case=F),
                contains("Y_", ignore.case=F)) %>% 
  as.matrix() 
# types
data_types <- c("gaussian", "gaussian", "gaussian")

# center gaussian outcomes
Y[,data_types == "gaussian"] <- Y[,data_types=="gaussian"] %>% 
  apply(2, function(x) (x-mean(x, na.rm=T))/sd(x,na.rm=T))

Y <- Y[,1,drop=F]

s <- 1


set.seed(s)
y_dim <- prod(dim(Y))
which_na <- sample(1:y_dim, round(y_dim/10))
Y_full <- Y
Y[which_na] <- NA



# # # # # # # # # # # # # # # # # # # # # #
#           run gramar                    #
# # # # # # # # # # # # # # # # # # # # # #

meshed_time <- system.time({
  meshed_out <- gramar::gramar(y=Y, x=X, k=1,
                               block_size = 40,
                               n_samples = 10,
                               n_burnin = 10,
                               n_thin = 1,
                               n_threads = 16,
                               verbose = Inf,
                               family = data_types[1],
                               debug = list(sample_beta=T, sample_tausq=T, 
                                            sample_theta=T, sample_w=T, sample_lambda=T,
                                            verbose=T, debug=T))
})
