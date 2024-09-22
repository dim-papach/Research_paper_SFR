library(tidyverse)
library(ggfortify)

# open outer_join.ecsv
dat <- read.table("outer_join.fits", header = TRUE, sep = "\t")
str(dat)
summary(dat)

# pca analysis

## 1. Data preparation: select numeric columns

num_cols <- select_if(dat, is.numeric)
char_cols <- select_if(dat, is.character)

num_cols %>% nrow 

## 2. Center and scale
num_cols <- scale(num_cols, center = TRUE, scale = TRUE)
## 3. remove columns with E_ in the column name
num_cols <- as_tibble(num_cols) %>% select(-matches("^E_"))

## 3. NA and NaN = 0
num_cols[is.na(num_cols)] <- 0

#pca
pca_result <- prcomp(num_cols, center = TRUE, scale. = TRUE)

# biplot with autoplot
autoplot(pca_result, data = dat,color = "logM_HEC" )#,loadings = TRUE, loadings.label = TRUE, loadings.label.size = 3)


