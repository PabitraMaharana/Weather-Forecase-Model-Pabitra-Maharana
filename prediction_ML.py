import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

weather = pd.read_csv("weather.csv", index_col="DATE")
null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]
valid_columns = weather.columns[null_pct < 0.05]

weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()

weather.index = pd.to_datetime(weather.index)
weather.index.year.value_counts().sort_index()

def backtest(weather, model, predictors, target, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i+step), :]

        model.fit(train[predictors], train[target])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test[target], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)

def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col, target):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

def expand_mean(df):
    return df.expanding(1).mean()
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]

# Snow
weather["target"] = weather.shift(-1)["snow"]
weather = weather.ffill()
rr = Ridge(alpha=0.1)
predictions_snow = backtest(weather, rr, predictors, "target")
predictions_snow.sort_values("diff", ascending=False)

# Precipitation
weather["target"] = weather.shift(-1)["prcp"]
weather = weather.ffill()
rr = Ridge(alpha=0.1)
predictions_prcp = backtest(weather, rr, predictors, "target")
predictions_prcp.sort_values("diff", ascending=False)

# Maximum Temperature
weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()
rr = Ridge(alpha=0.1)
predictions_tmax = backtest(weather, rr, predictors, "target")
predictions_tmax.sort_values("diff", ascending=False)

# Minimum Temperature
weather["target"] = weather.shift(-1)["tmin"]
weather = weather.ffill()
rr = Ridge(alpha=0.1)
predictions_tmin = backtest(weather, rr, predictors, "target")
predictions_tmin.sort_values("diff", ascending=False)

def plot_actual_vs_prediction(predictions, target_name, subplot_num):
    plt.subplot(subplot_num)
    plt.plot(predictions.index, predictions['actual'], label='Actual', color='blue')
    plt.plot(predictions.index, predictions['prediction'], label='Prediction', color='red')
    plt.xlabel('Date')
    plt.ylabel(target_name)
    plt.legend()
    plt.title(f'Actual vs. Prediction for {target_name}')
    
def scatter_actual_vs_prediction(predictions, target_name, subplot_num):
    plt.subplot(subplot_num)
    plt.scatter(predictions['actual'], predictions['prediction'], color='blue', label='Actual vs. Prediction')
    plt.xlabel('Actual ' + target_name)
    plt.ylabel('Predicted ' + target_name)
    plt.legend()
    plt.title(f'Scatter Plot for {target_name}')

print("Snow (SNOW) Predictions:")
print(predictions_snow)
print("Precipitation (PRCP) Predictions:")
print(predictions_prcp)
print("Maximum Temperature (TMAX) Predictions:")
print(predictions_tmax)
print("Minimum Temperature (TMIN) Predictions:")
print(predictions_tmin)

plt.figure(figsize=(12, 8))
plot_actual_vs_prediction(predictions_snow, 'Snow (SNOW)', 221)
plot_actual_vs_prediction(predictions_prcp, 'Precipitation (PRCP)', 222)
plot_actual_vs_prediction(predictions_tmax, 'Maximum Temperature (TMAX)', 223)
plot_actual_vs_prediction(predictions_tmin, 'Minimum Temperature (TMIN)', 224)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
scatter_actual_vs_prediction(predictions_snow, 'Snow (SNOW)', 221)
scatter_actual_vs_prediction(predictions_prcp, 'Precipitation (PRCP)', 222)
scatter_actual_vs_prediction(predictions_tmax, 'Maximum Temperature (TMAX)', 223)
scatter_actual_vs_prediction(predictions_tmin, 'Minimum Temperature (TMIN)', 224)
plt.tight_layout()
plt.show()

