from matplotlib import font_manager
from matplotlib import style
import seaborn.objects as so


def set_theme():
    so.Plot.config.theme.update(style.library["Solarize_Light2"])
    font_path = "../../assets/FiraCode-Regular.ttf"  # TODO include in library
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    so.Plot.config.theme.update(
        {"font.family": "sans-serif", "font.sans-serif": prop.get_name()}
    )
