from cycler import cycler
from matplotlib import font_manager
import seaborn.objects as so


def set():
    font_path = "../../assets/FiraCode-Regular.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    theme = {
        "axes.facecolor": "#f8f5f0",
        "axes.prop_cycle": cycler(
            color=[
                "#325d88",
                "#8e8c84",
                "#93c54b",
                "#29abe0",
                "#f47c3c",
                "#d9534f",
            ]
        ),
        "font.family": "sans-serif",
        "font.sans-serif": prop.get_name(),
    }
    so.Plot.config.theme.update(theme)
