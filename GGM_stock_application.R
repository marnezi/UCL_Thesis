rm(list=ls())
library(data.table)
library(lubridate)
library(quantmod)
library(parallel)
library(ggplot2)
library(MASS)
library(plotly)
library(MVN)
library(tseries)
library(igraph)
library(ggraph)

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

# Plot with explicit group
closing_prices_df_plot[, value_plot := fifelse(variable == "C", value / 10, value)] # distorts the plots

ggplot(closing_prices_df_plot, aes(Date, value_plot, color = variable, group = variable)) +
  geom_line(size = 0.7, alpha = 0.9) +
  scale_x_date(date_labels = "%Y", date_breaks = "1 year") +
  scale_y_continuous(
    labels = scales::dollar_format(prefix = "$"),
    sec.axis = sec_axis(~ . * 10, name = "Closing price for C ($)")
  ) +
  labs(
    title = "Stock Closing Prices (2003–2007)",
    subtitle = "Selected S&P 500 Companies",
    x = "Date", y = "Closing price ($)", color = "Ticker"
  ) +
  theme_minimal(base_size = 14)


cor(closing_prices_df[,-1])

# Create interactive plot
p <- ggplot(closing_prices_df_plot, aes(x = Date, y = value, color = variable, group = variable, text = variable)) +
  geom_line(size = 0.7, alpha = 0.9) +
  labs(
    title = "Interactive Stock Closing Prices (2015–2019)",
    x = "Date",
    y = "Closing Price (USD)",
    color = "Ticker"
  ) +
  theme_minimal()

# Convert to interactive plotly object
interactive_plot <- ggplotly(p, tooltip = c("x", "y", "text"))

# Display the plot
interactive_plot

# transform the prices into log returns

closing_prices_logr <- melt(closing_prices_df,id.vars = 'Date')

closing_prices_logr[,value_shifted:= shift(value), by = variable]
closing_prices_logr[,log_return:= log(value)-log(value_shifted), by = variable]

closing_prices_logr <- dcast(closing_prices_logr,formula = Date ~ variable, value.var = 'log_return')
closing_prices_logr <- na.omit(closing_prices_logr)
closing_prices_logr_qq <- melt(closing_prices_logr,id.vars = 'Date')

# Q-Q- plots 

ggplot(closing_prices_logr_qq[variable %in% c(fav_stocks[20:22])], aes(sample = value)) +
  stat_qq() +
  stat_qq_line(color = "blue") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = "Q-Q Plots of Log Returns by Stock",
       x = "Theoretical Quantiles", y = "Sample Quantiles")

shapiro_results <- sapply(closing_prices_logr[,-1], function(x) {shapiro.test(x)$p.value})
shapiro_results[1]

#test stationarity 

#plot the log returns 
closing_prices_df_plot <- melt(closing_prices_logr,id.vars = 'Date')
closing_prices_df_plot[, Date := as.Date(Date)]

closing_prices_df_plot[, variable := as.factor(variable)]


ggplot(closing_prices_df_plot, aes(Date, value, color = variable, group = variable)) +
  geom_line(size = 0.7, alpha = 0.9) +
  scale_x_date(date_labels = "%Y", date_breaks = "1 year") +
  scale_y_continuous(
    labels = scales::dollar_format(prefix = "$"),
    sec.axis = sec_axis(~ . * 10, name = "Closing price for C ($)")
  ) +
  labs(
    title = "Stock Closing Prices (2003–2007)",
    subtitle = "Selected S&P 500 Companies",
    x = "Date", y = "Closing price ($)", color = "Ticker"
  ) +
  theme_minimal(base_size = 14)


# Assume normality - apply double fourier estimator -----------------------
# configuration settings
etta_parameter <-  0.5
mass_parameter <- 1
nodes_number <- uniqueN(colnames(closing_prices_df[,-1]))
# covariance_matrix <- solve(laplacian_matrix + mass_parameter*diag(nodes_number))

#generate mutliple samples from Sigma
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
fielder

# plot


