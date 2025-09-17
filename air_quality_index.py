import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("city_day.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Keep only important columns
df = df[["City", "Date", "AQI"]]

# Drop missing AQI
df = df.dropna(subset=["AQI"])

# Select multiple cities
cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]
df_cities = df[df["City"].isin(cities)]

plt.figure(figsize=(12,6))
sns.lineplot(x="Date", y="AQI", hue="City", data=df_cities)
plt.title("AQI Trends (Delhi vs Mumbai vs Chennai vs Kolkata vs Bengaluru)")
plt.xlabel("Year")
plt.ylabel("AQI")
plt.legend(title="City")
plt.savefig("AQI_trends.png")
plt.show()

# Seasonal AQI comparison
def get_season(date):
    month = date.month
    if month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8,9]:
        return "Monsoon"
    else:
        return "Winter"

df_cities.loc[:, 'Season'] = df_cities['Date'].apply(get_season)
seasonal_avg = df_cities.groupby(['City', 'Season'])['AQI'].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(x='City', y='AQI', hue='Season', data=seasonal_avg)
plt.title("Average AQI by Season for Selected Cities")
plt.ylabel("Average AQI")
plt.savefig("seasonal_AQI_comparison.png")
plt.show()

# City ranking by average AQI
city_avg = df_cities.groupby('City')['AQI'].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8,6))
sns.barplot(x='AQI', y='City', data=city_avg, color='skyblue')
plt.title("City Ranking by Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.savefig("city_AQI_ranking.png")
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

forecast_results = {}
mae_results = {}
rmse_results = {}
unhealthy_days = {}
very_unhealthy_days = {}

for city in cities:
    city_df = df_cities[df_cities["City"]==city].set_index("Date").asfreq("D")
    city_df = city_df.ffill()
    
    # Count unhealthy and very unhealthy days
    unhealthy_days[city] = (city_df["AQI"] > 100).sum()
    very_unhealthy_days[city] = (city_df["AQI"] > 200).sum()
    
    # AQI level alerts on historical plot
    plt.figure(figsize=(12,5))
    plt.plot(city_df.index, city_df["AQI"], label=f"{city} Historical")
    
    # Annotate AQI > 100 and > 200
    unhealthy = city_df[city_df["AQI"] > 100]
    very_unhealthy = city_df[city_df["AQI"] > 200]
    plt.scatter(unhealthy.index, unhealthy["AQI"], color='orange', label="Unhealthy AQI > 100", s=20)
    plt.scatter(very_unhealthy.index, very_unhealthy["AQI"], color='red', label="Very Unhealthy AQI > 200", s=20)
    
    # Split last 30 days for test
    train = city_df.iloc[:-30]
    test = city_df.iloc[-30:]
    
    # Fit SARIMA model on train data
    # Seasonal order chosen as (1,1,1,12) to capture yearly seasonality
    model = SARIMAX(train["AQI"], order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Forecast 30 days with confidence intervals
    forecast_obj = model_fit.get_forecast(steps=30)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)
    forecast_results[city] = (forecast, conf_int)
    
    # Plot forecast with confidence intervals
    plt.plot(pd.date_range(train.index[-1], periods=31, freq="D")[1:], forecast, label=f"{city} Forecast", color="red")
    plt.fill_between(pd.date_range(train.index[-1], periods=31, freq="D")[1:], conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.3, label="95% CI")
    plt.legend()
    plt.title(f"{city} AQI Historical and SARIMA Forecast with Alerts")
    plt.savefig(f"{city}_AQI_forecast_with_alerts.png")
    plt.show()
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(test["AQI"], forecast)
    rmse = np.sqrt(np.mean((test["AQI"] - forecast) ** 2))
    mae_results[city] = mae
    rmse_results[city] = rmse

# Display forecast evaluation results
eval_df = pd.DataFrame({
    'City': list(mae_results.keys()),
    'MAE': list(mae_results.values()),
    'RMSE': list(rmse_results.values())
})

print("Forecast Evaluation Metrics (Last 30 Days):")
print(eval_df)

plt.figure(figsize=(8,6))
sns.barplot(x='City', y='MAE', data=eval_df)
plt.title("MAE of SARIMA Forecast for Each City")
plt.savefig("forecast_MAE_comparison.png")
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x='City', y='RMSE', data=eval_df)
plt.title("RMSE of SARIMA Forecast for Each City")
plt.savefig("forecast_RMSE_comparison.png")
plt.show()

# Correlation analysis with weather features (placeholder)
# Check if weather features exist
weather_features = [col for col in df.columns if col not in ['City', 'Date', 'AQI']]
if len(weather_features) > 0:
    corr_results = {}
    for city in cities:
        city_data = df[(df["City"]==city) & df[weather_features].notnull().all(axis=1)]
        if not city_data.empty:
            corr = city_data[['AQI'] + weather_features].corr()['AQI'].drop('AQI')
            corr_results[city] = corr
    # Plot correlations
    for city, corr in corr_results.items():
        plt.figure(figsize=(8,6))
        corr.plot(kind='bar')
        plt.title(f"AQI Correlation with Weather Features for {city}")
        plt.ylabel("Correlation Coefficient")
        plt.ylim(-1,1)
        plt.savefig(f"{city}_AQI_weather_correlation.png")
        plt.show()
else:
    print("No weather features available for correlation analysis.")

plt.figure(figsize=(12,6))
for city in cities:
    forecast, conf_int = forecast_results[city]
    dates = pd.date_range(df_cities["Date"].max(), periods=30, freq="D")
    plt.plot(dates, forecast, label=city)
    plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.3)

plt.title("30-Day AQI SARIMA Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Forecasted AQI")
plt.legend()
plt.savefig("AQI_forecast_comparison.png")
plt.show()

# Generate summary CSV
summary_list = []
for city in cities:
    avg_aqi = city_avg.loc[city_avg['City']==city, 'AQI'].values[0]
    seasonal_avgs = seasonal_avg[seasonal_avg['City']==city].set_index('Season')['AQI'].to_dict()
    unhealthy_count = unhealthy_days.get(city, 0)
    very_unhealthy_count = very_unhealthy_days.get(city, 0)
    mae = mae_results.get(city, np.nan)
    rmse = rmse_results.get(city, np.nan)
    
    summary_list.append({
        'City': city,
        'Average_AQI': avg_aqi,
        'Summer_AQI': seasonal_avgs.get('Summer', np.nan),
        'Monsoon_AQI': seasonal_avgs.get('Monsoon', np.nan),
        'Winter_AQI': seasonal_avgs.get('Winter', np.nan),
        'Unhealthy_Days_>100': unhealthy_count,
        'Very_Unhealthy_Days_>200': very_unhealthy_count,
        'MAE': mae,
        'RMSE': rmse
    })

summary_df = pd.DataFrame(summary_list)
summary_df.to_csv("city_AQI_summary.csv", index=False)