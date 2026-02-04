from aegraph import AEGraph
import numpy as np

chi_data = np.random.chisquare(df=4, size=4000)
chi_data2 = np.random.chisquare(df=15, size=4000)
chi_data3 = np.random.chisquare(df=8, size=4000)

# AEGraph plot
plot = (
    AEGraph(comp_name="Chi-Square Histogram", 
            width=720, 
            height=720,
            comp_height=1080,
            comp_width=1920,
            drop_shadow=True,
            cinematic_effects=False,
            fps=60)
    .set_title("Chi-Square Distribution (df=4)")
    .set_xlabel("Value")
    .set_ylabel("Frequency")
    .set_xlim(0, 20)
    .set_ylim(0, 0.25)   # Adjust depending on density
    .grid(show=True, color="gray", alpha=0.3)
)

plot.histogram(
    chi_data, bins=100, color="p_purple", alpha=0.6,
    label="Chi-Square (df=4)", density=True,
    animate=4.0, bar_anim_times=0.50
)
plot.histogram(
    chi_data2, bins=100, color="p_red", alpha=0.6,
    label="Chi-Square (df=15)", density=True,
    animate=4.0, bar_anim_times=0.50
)
plot.histogram(
    chi_data3, bins=100, color="p_blue", alpha=0.6,
    label="Chi-Square (df=8)", density=True,
    animate=4.0, bar_anim_times=0.50
)

plot.set_xticks().set_yticks()
plot.render()
