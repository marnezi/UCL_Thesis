rm(list=ls())
library(data.table)
library(lubridate)
library(quantmod)
library(parallel)
library(ggplot2)
library(MASS)
library(plotly)
library(MVN)
library(MASS)
library(tseries)
library(igraph)
library(ggraph)
library(scales)
library(fitdistrplus)

gc()#clean unnecessary data 

# Read data ---------------------------------------------------------------

#Choose 20 stocks from the S&P500 index 
fav_stocks <- c( "JPM" ,"XOM", "CVX", "PG" , "WMT" ,"PEP", "GE" , "C" ,  "GS" , "BA" , "T" ,  "VZ" , "COP" ,"DUK" ,"SO" , "NEE" ,"KMB" ,"HD" ,"LOW" ,"TGT", "NSC" ,"LUV", "APA")

start_date <- "2003-06-01"
end_date <- "2007-01-01"

# download each ticket time series 
getSymbols(fav_stocks, src = "yahoo", from = start_date, to = end_date)

# extract closing prices
closing_prices_list <- mclapply(fav_stocks, function(symbol) data.table(data.frame(Cl(get(symbol))),keep.rownames = T), mc.cores = 1)

#merge all together
closing_prices_df <- Reduce(function(x, y) merge(x, y, by = "rn", all = TRUE), closing_prices_list)
colnames(closing_prices_df) <- gsub("\\.Close", "", colnames(closing_prices_df))
setnames(closing_prices_df,'rn','Date')
summary(closing_prices_df)

closing_prices_df_plot <- melt(closing_prices_df,id.vars = 'Date')
closing_prices_df_plot[, Date := as.Date(Date)]
 
closing_prices_df_plot[, variable := as.factor(variable)]

# transform the prices into log returns

closing_prices_logr <- melt(closing_prices_df,id.vars = 'Date')

closing_prices_logr[,value_shifted:= shift(value), by = variable]
closing_prices_logr[,log_return:= log(value)-log(value_shifted), by = variable]

closing_prices_logr <- dcast(closing_prices_logr,formula = Date ~ variable, value.var = 'log_return')
closing_prices_logr <- na.omit(closing_prices_logr)
closing_prices_logr_qq <- melt(closing_prices_logr,id.vars = 'Date')

# check stationarity 

closing_prices_df_plot <- melt(closing_prices_logr,id.vars = 'Date')
closing_prices_df_plot[, Date := as.Date(Date)]

closing_prices_df_plot[, variable := as.factor(variable)]


pallete_col <- colorRampPalette(c("navy", "orange"))(23)

p <- ggplot(closing_prices_df_plot, aes(x = Date, y = value, color = variable, group = variable, text = variable)) +
  geom_line(size = 0.3) +
  scale_x_date(date_breaks = "6 months", date_labels = "%Y-%m") +  
  labs(
    x = "Date",
    y = "Standardized log-returns",
    color = "Ticket") +theme_minimal()+scale_color_manual(values = pallete_col)+  theme(
      legend.position = "top",
      legend.direction = "horizontal",
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 12),
      legend.key.size = unit(1.2, "lines"), 
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      axis.title.x = element_blank()
    ) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))
p


# Evaluate marginals ------------------------------------------------------

# Q-Q- plots 

ggplot(closing_prices_logr_qq[variable %in% c(fav_stocks[20:22])], aes(sample = value)) +
  stat_qq() +
  stat_qq_line(color = "blue") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = "Q-Q Plots of Log Returns by Stock",
       x = "Theoretical Quantiles", y = "Sample Quantiles")

shapiro_results_pvalue <- sapply(closing_prices_logr[,-1], function(x) {shapiro.test(x)$p.value})
shapiro_results_W <- sapply(closing_prices_logr[,-1], function(x) {shapiro.test(x)$statistic})
shapiro_results <- data.table(data.frame(shapiro_results_pvalue),keep.rownames = T)
shapiro_results$W <- shapiro_results_W
setnames(shapiro_results,c('ticket','p-value','W'))
shapiro_results


# Test dependence structure using copulas  --------------------------------

library(VineCopula)

#transform to uniform - ties are set to the average 

rank2uniform <- closing_prices_logr[,-1][, lapply(.SD, function(col) {
  rank(col, ties.method = "average") / (length(col) + 1)
})]


final_dt <- data.table()
pairs <- combn(ncol(rank2uniform), 2)

k=1

for (k in seq_len(ncol(pairs))) {
  print(k)
  i <- pairs[1, k]
  j <- pairs[2, k]
  
  fit <- BiCopSelect(rank2uniform[[i]], rank2uniform[[j]],
                     familyset = 0:5, selectioncrit = "AIC",indeptest = TRUE, level = 0.05)
  print(fit$family)
  out_cop <- data.table(var_i = colnames(rank2uniform)[i],var_j = colnames(rank2uniform)[j],
                        AIC   = fit$AIC,BIC   = fit$BIC,Copula = fit$familyname)
  
  final_dt <- rbind(final_dt,out_cop)
  
}

#saveRDS(final_dt,'final_dt.rds')
final_dt <- readRDS('final_dt.rds')

perc <- final_dt[, .N, by = Copula]
perc[Copula == 'Gumbel']$N <- perc[Copula == 'Gumbel']$N +perc[Copula == 'Survival Gumbel']$N 
perc <- perc[Copula != 'Survival Gumbel']

perc$percentage <- 100*perc$N/sum( perc$N)
perc

final_dt[, .N, by = Copula][, perc := 100 * N / sum(N)]

# visualise some of the pairs 

#Gaussian case 
k = 1 
i <- pairs[1, k]
j <- pairs[2, k]
fit <- BiCopSelect(rank2uniform[[i]], rank2uniform[[j]], familyset = 0:5, selectioncrit = "AIC",indeptest = TRUE, level = 0.05)

rho_from_tau <- BiCopTau2Par(family = 1, tau = fit$tau)
u <- BiCopSim(1000, family = 1, par = rho_from_tau)


plot1 <- ggplot()+geom_point(aes(JPM,XOM),rank2uniform[,c('JPM','XOM')])+theme_minimal()+
  geom_point(aes(u[,1],u[,2]), color = 'orange')+
  ggtitle('Gaussian Copula')

#Frank case 
final_dt

k = 2 
i <- pairs[1, k]
j <- pairs[2, k]
fit <- BiCopSelect(rank2uniform[[i]], rank2uniform[[j]], familyset = 0:5, selectioncrit = "AIC",indeptest = TRUE, level = 0.05)

u <- BiCopSim(1000, family = 1, par = fit$par)


plot2 <- ggplot()+geom_point(aes(JPM,CVX),rank2uniform[,c('JPM','CVX')])+theme_minimal()+
  geom_point(aes(u[,1],u[,2]), color = 'orange')+
  ggtitle('Frank Copula')
plot2

#t case 
final_dt

k = 3
i <- pairs[1, k]
j <- pairs[2, k]
fit <- BiCopSelect(rank2uniform[[i]], rank2uniform[[j]], familyset = 0:5, selectioncrit = "AIC",indeptest = TRUE, level = 0.05)

u <- BiCopSim(1000, family = 2,  par = fit$par, par2 = fit$par2)

plot3 <- ggplot()+geom_point(aes(JPM,PG),rank2uniform[,c('JPM','PG')])+theme_minimal()+
  geom_point(aes(u[,1],u[,2]), color = 'orange')+
  ggtitle('t-Copula')
plot3

ggpubr::ggarrange(plot1,plot2,plot3,ncol = 3,nrow =1 )


# Assume normality - apply double fourier estimator -----------------------
# configuration settings
etta_parameter <-  0.5
mass_parameter <- 1
nodes_number <- uniqueN(colnames(closing_prices_df[,-1]))
# covariance_matrix <- solve(laplacian_matrix + mass_parameter*diag(nodes_number))

#generate multiple samples from Sigma
set.seed(123) 
x_samples <- as.matrix(closing_prices_logr[, -1, with = FALSE])  # wide returns table
x_samples <- scale(x_samples, center = TRUE, scale = FALSE)      # ensure mean 0
dim(x_samples)

y_samples = mvrnorm(mu = rep(0, nodes_number),Sigma= etta_parameter*diag(nodes_number), n = dim(x_samples)[1])
dim(y_samples)

# Define phi_t function
phi_t <- function(t, y_samples, x_samples) {
  phi <- mean(exp(1i * rowSums(y_samples * (x_samples + t))))
  return(phi)
}

laplacian_etta_hat <- matrix(0, nrow = nodes_number, ncol = nodes_number)
e <- diag(1, nodes_number)

# Estimate phi_t for diagonal points
phi_t_values <- sapply(1:nodes_number, function(i) phi_t(e[i,], y_samples, x_samples))
phi_t_0 <- phi_t(rep(0, nodes_number), y_samples, x_samples)  # phi_t(0)

start.time <- Sys.time()
# Estimate the diagonal and off-diagonal entries in the Laplacian matrix
for (i in 1:nodes_number) {
  print(paste("Running iteration i =", i))
  for (j in 1:nodes_number) {
    print(paste("Running iteration j =", j))
    if (i == j) {
      laplacian_etta_hat[i, i] <- -2 * log(abs(phi_t_values[i])) + 2 * log(abs(phi_t_0))
    } else {
      term_1 <- (e[i, ] + e[j, ]) / sqrt(2)
      laplacian_etta_hat[j, i] <- laplacian_etta_hat[i, j] <- -2 * log(abs(phi_t(term_1, y_samples, x_samples))) + log(abs(phi_t_values[i])) + log(abs(phi_t_values[j]))
    }
  }
}
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# View the result
print(laplacian_etta_hat)
eigen(laplacian_etta_hat)$values
min(eigen(laplacian_etta_hat)$values)

#Compute the precision matrix
precision_matrix_hat = etta_parameter**2*(solve(laplacian_etta_hat+mass_parameter*diag(nodes_number))) - etta_parameter*diag(nodes_number)

#Compute the weights from the Laplacian
adjacency_matrix = -laplacian_etta_hat
min(adjacency_matrix)

adjacency_matrix <- adjacency_matrix +abs(min(adjacency_matrix))
min(adjacency_matrix)

diag(adjacency_matrix) <- 0

weights_matrix = copy(adjacency_matrix)

range(weights_matrix)

degree_matrix = matrix(0, nrow = dim(closing_prices_df[,-1])[2], ncol = dim(closing_prices_df[,-1])[2])
diag(degree_matrix) <- diag(laplacian_etta_hat)
degree_matrix

#average connectivity 
average_connectiivty <- mean(degree_matrix)

#fiedler   
fielder <- eigen(laplacian_etta_hat)$values[order(eigen(laplacian_etta_hat)$values)][2]
fielder;average_connectiivty


