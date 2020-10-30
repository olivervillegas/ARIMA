from stock import stock

GOOGLE = stock('GOOGL.csv', "ARIMA: Google", 0.988114105)
APPLE = stock('AAPL.csv', "ARIMA: Apple", 0.988114105)
AMAZON = stock('AMZN.csv', "ARIMA: Amazon", 0.988114105)
OREILLY = stock('ORLY.csv', "ARIMA: O'Reilly Automotive", 0.988114105)

GENMILLS = stock('GNM.csv', "ARIMA: General Mills", 0.988114105)
KELLOGG = stock('KLLG.csv', "ARIMA: Kellogg", 0.988114105)
CITIGROUP = stock('CITI.csv', "ARIMA: Citigroup", 0.988114105)
DANONE = stock('DNN.csv', "ARIMA: Danone", 0.988114105)

TRANSOCEAN = stock('RIG.csv', "ARIMA: Transocean", 0.988114105)
MURPHYOIL = stock('MUR.csv', "ARIMA: Murphy Oil", 0.988114105)
IMAX = stock('IMAX.csv', "ARIMA: IMAX", 0.988114105)
FLUOR = stock('FLR.csv', "ARIMA: Fluor", 0.988114105)

#GOOGLE.test_stationarity()
#GOOGLE.model_and_forecast()

#APPLE.test_stationarity()
#APPLE.model_and_forecast()

#AMAZON.test_stationarity()
#AMAZON.model_and_forecast()

#OREILLY.test_stationarity()
#OREILLY.model_and_forecast()


#GENMILLS.test_stationarity()
#GENMILLS.model_and_forecast()

#CITIGROUP.test_stationarity()
#CITIGROUP.model_and_forecast()

#DANONE.test_stationarity()
#DANONE.model_and_forecast()

#KELLOGG.test_stationarity()
#KELLOGG.model_and_forecast()


#TRANSOCEAN.test_stationarity()
#TRANSOCEAN.model_and_forecast()

#FLUOR.test_stationarity()
#FLUOR.model_and_forecast()

#MURPHYOIL.test_stationarity()
#MURPHYOIL.model_and_forecast()

IMAX.test_stationarity()
IMAX.model_and_forecast()