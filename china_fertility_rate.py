from aegraph import AEGraph
import pandas as pd
import numpy as np

df = pd.read_csv("all_countries_fertility.csv")
def get_country(idx):
    returndf = df.iloc[idx,4:-3]
    returndf.name = "Fertility Rate"
    returndf = pd.DataFrame(returndf)
    return returndf
    
america = get_country(251)
japan = get_country(119)
china = get_country(40)

animation_time = 3

(
AEGraph(
    # width=1400,
    # height=900,
    width=900,
    height=900,
    compwidth=1080+300,
    compheight=1920+300,
    drop_shadow=True,
    comp_name="Fertility Rates Over Time",
    fps=60,
    show_all_points=True,
    # animate_opacity=False,
    # animate_axes=False,
    bg_color=[25, 25, 25],
    ui_color=[212, 218, 223],
    full_bg = True,
    cinematic_effects = True,
    distress_texture = 1,
    ease_speed = 0,
    ease_influence = 60,
    font_scale = 1
).plot(list((america.index.astype(int))),america["Fertility Rate"], color =[41, 145, 248], animate = animation_time, label = "United States", drop_shadow = True)
.plot(list((japan.index.astype(int))),japan["Fertility Rate"], color =[7,190,60], animate = 4, linestyle = "--", label = "Japan", drop_shadow = True, delay = 3, dash_size = 1.5, ease_influence = 30)
.plot(list((china.index.astype(int))),china["Fertility Rate"], color = [204,67,67], animate = 4, linestyle = "--", label = "China", drop_shadow = True, delay = 3, dash_size = 1.5, ease_influence = 30)
# .plot(list((china.index.astype(int))),[2.1 for i in range(len(list((china.index.astype(int)))))], color = "red", animate = 0.1) # line of the DEATH ZONE
.set_xticks(list(np.arange(1960,2030,10)))
.set_ylim(0,7.5)
.set_yticks(list(np.arange(0,8)))
.set_title("Fertility Rates Over Time")
.set_subtitle("Source: Worldbank.org", color=[134, 137, 140])
.set_xlabel("Year")
.set_ylabel("Fertility Rate")
.grid(color = [118, 135, 152], alpha = 0.3, hide_horizontal = True, linestyle = "--", dash_size = 0.5)
.add_legend(style='line_style')
.render()
)