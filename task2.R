rm(list=ls())
library(data.table)
library(ggplot2)
library(readxl)
library(lubridate)

# Read data  --------------------------------------------------------------

rainfall_dt <- fread('df_rain_historical.csv')

rainfall_dt[,Date:= as.Date(Date,format = '%Y-%m-%d')]

rainfall_dt[,year:=year(Date)]
rainfall_dt[,month:=month(Date)]
rainfall_dt[,day:=day(Date)]

# keep only period between December to March
rainfall_dt <- rainfall_dt[month %in% c(12,1,2,3)]

#create season
rainfall_dt[, season_year := fifelse(month %in% 1:3, year, year+1)]

#delete day Feb
rainfall_dt <- rainfall_dt[!(month == 2 & day == 29)]
rainfall_dt <- rainfall_dt[!(month == 3 & day %in% seq(2,31))]

full_period <- length(seq( as.Date('2021-12-01',format = '%Y-%m-%d'), as.Date('2022-03-01',format = '%Y-%m-%d'), by = 'day'))
full_period

#delete missing periods
rainfall_dt[season_year==2022]
rainfall_dt[,periodsN:= .N, by = season_year ]

#check
unique(rainfall_dt[,.(periodsN,season_year)])
which(unique(rainfall_dt[,.(periodsN,season_year)])$periodsN!=full_period)

#delete the first season 
rainfall_dt <- rainfall_dt[season_year!=1950]


#Create a new dataframe df_cum_rain by regrouping by season and by computing the cumulative rainfall over the risk period.

rainfall_dt[,cum_rain:= cumsum(`daily rain`), by = season_year]

# risk period cumulative rain 
df_cum_rain <- unique(rainfall_dt[,.(season_year,cum_rain)])
df_cum_rain

#df_historical_payouts with column hist_payouts

# 
# The client is looking to be covered against excess of rain.
# 
# The client wishes to receive a payout under the following conditions:
#   
# Depends on the cumulative rainfall (mm) over the season.
# Linear payout starting at a deductible of 650 mm and limit of 750 mm with a payout of AUD 500,000.
# It means that
# 
# If the cumulative rainfall is below 650 mm, the client receives nothing.
# If the cumulative rainfall is above 750 mm, the client receives the max payout of AUD 500,000 no matter the loss.
# If the cumulative rainfall is between 650 and 750 mm, the payout is linear, slope of AUD 5,000 per mm. 
# For example, if the cumulative rainfall is 700 mm, the payout would be AUD 250,000.

#

df_cum_rain[, c('nothing', 'linear', 'cap'):= 0]
df_cum_rain[,hist_payouts:=ifelse(any(cum_rain>750), 500000,0), by = season_year]

top_cap <- 750
min_cap <- 650

df_cum_rain[,hist_payouts:=ifelse(any(max(cum_rain)>=650 & max(cum_rain)<=750), 
                                  500000*(top_cap-max(cum_rain))/(top_cap-min_cap),hist_payouts), by = season_year]

plot(df_cum_rain$hist_payouts)

df_cum_rain[,max_period:= max(cum_rain), by = season_year]
df_cum_rain[,min_period:= min(cum_rain), by = season_year]

unique(df_cum_rain[,.(min_period,max_period,season_year,hist_payouts)])

#burning cost 
burning_cost <- mean(unique(df_cum_rain[,.(min_period,max_period,season_year,hist_payouts)]$hist_payouts))

df_cum_rain


burning_cost

#weights
forecast_types <- fread('df_year_type.csv')
setnames(forecast_types,'season','season_year')

df_cum_rain <- merge(df_cum_rain,forecast_types, by = 'season_year')
setnames(df_cum_rain,'year_type','next_year_forecast')

#payouts expectations
df_cum_rain[next_year_forecast == 'dry', e_payouts:= mean(hist_payouts)]
df_cum_rain[next_year_forecast == 'rainy', e_payouts:= mean(hist_payouts)]
df_cum_rain[next_year_forecast == 'neutral', e_payouts:= mean(hist_payouts)]
df_cum_rain

unique(df_cum_rain[,.(next_year_forecast,e_payouts)])

#next_year_forecast = {"dry": 0.58, "neutral": 0.37, "rainy": 0.05}

37377.09*0.58 +51097.03*0.05
