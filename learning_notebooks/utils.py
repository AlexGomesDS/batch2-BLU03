import pandas as pd

# TODO: send this to utils.py
def load_airline_data():
    airlines = pd.read_csv('../data/international-airline-passengers.csv',
                           index_col='Month')[:-1]

    airlines.columns = ['passengers_thousands']
    airlines = airlines['passengers_thousands']
    airlines.index = pd.to_datetime(airlines.index)

    return airlines.asfreq('M', method='ffill')


def predict_next_period(model, airlines, n_periods, number_of_periods_ahead):

    X_train, y_train, X_last_period = prepare_for_prediction(airlines.iloc[0:n_periods],
                                                             number_of_periods_ahead)

    model.fit(X_train, y_train)

    next_period_index = [y_train.index.max() + pd.DateOffset(months=number_of_periods_ahead)]
    next_period_prediction_values = model.predict(X_last_period.values.reshape(1, -1))
    next_period_prediction = pd.Series(next_period_prediction_values, next_period_index)
    return next_period_prediction


