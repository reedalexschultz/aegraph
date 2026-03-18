from aegraph import AEGraph
import numpy as np
import pandas as pd

gini_csv = pd.read_csv("API_SI.POV.GINI_DS2_en_csv_v2_35.csv")

countries_mean_gini = {}

for country in gini_csv["Country Name"]:

    country_df = gini_csv[gini_csv["Country Name"] == country]
    country_df = country_df.iloc[:, 4:].dropna(axis=1)

    data = country_df.iloc[0].astype(float).dropna()
    years = data.index.astype(float)

    # Only plot countries with sufficient data
    if len(data) > 5:
        countries_mean_gini[country] = np.mean(data)
        
# Example with gradient coloring (from red to green)
s = pd.Series(countries_mean_gini).sort_values(ascending=False).head(30)

y_positions = np.arange(len(s))
widths = s.values
labels = s.index.tolist()

(
    AEGraph(width=960, height=2000, compwidth=2160, compheight=3840,
            drop_shadow=True, cinematic_effects=True, comp_name="Gini_Barh_Gradient", fps=60)
    .barh(y_positions, widths, animate=5.0, bar_duration=2, alpha=0.8)  # smooth overlapping!
    .gradient("red", "p_green")  # Apply color gradient from high to low inequality
    .set_xlabel("Mean Gini Coefficient")
    .set_ylabel("Country")
    .set_title("Top 15 Countries by Mean Gini - Color Gradient")
    .set_xticks()
    .set_yticks(positions=y_positions, labels=labels)
    .grid(show=True, color="gray", alpha=0.2)
    .save()
    .render()
)