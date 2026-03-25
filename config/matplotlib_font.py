import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from pathlib import Path


for font_path in [
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]:
    p = Path(font_path)
    if p.exists():
        try:
            font_manager.fontManager.addfont(str(p))
        except Exception:
            pass

# 这里把英文/数字优先放在前面；中文回退到 AR PL UMing CN
plt.rcParams["font.family"] = ["Times New Roman", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
# 数学文本也尽量贴近 Times 风格
plt.rcParams["mathtext.fontset"] = "stix"
# 避免对数刻度中的负号使用 Unicode 字符，修复 10^-1 渲染异常
plt.rcParams["axes.unicode_minus"] = False
