# User study data and plotting scripts

Final plotting inputs and accompanying scripts for Figures 5 and 6 of the IHF user study. Files are organized by the manuscript panel numbers. Original source files were not modified.

## Contents

| Directory | Panel |
| --- | --- |
| [Fig5/a_posture_confusion](Fig5/a_posture_confusion) | Fig. 5a, posture recognition |
| [Fig5/b_emoji_confusion](Fig5/b_emoji_confusion) | Fig. 5b, emoji recognition |
| [Fig5/c_paired_recognition](Fig5/c_paired_recognition) | Fig. 5c, paired recognition accuracies |
| [Fig5/d_localization](Fig5/d_localization) | Fig. 5d, localization heatmaps and errors |
| [Fig6/a_recognition_localization](Fig6/a_recognition_localization) | Fig. 6a, recognition and localization |
| [Fig6/b_identity_mismatch](Fig6/b_identity_mismatch) | Fig. 6b, identity-mismatched recognition |
| [Fig6/c_spatial_mismatch](Fig6/c_spatial_mismatch) | Fig. 6c, spatially mismatched localization |
| [Fig6/d_questionnaire](Fig6/d_questionnaire) | Fig. 6d, questionnaire ratings |

## Use

Use Python 3.10 or newer. Install dependencies with `python -m pip install -r requirements.txt`. Open a panel folder and run the command in its README. Fig. 5d and Fig. 6a have separate folders for their component plots. Scripts export individual panels, not a fully assembled manuscript figure.

Only files directly read for plotting are included. Historical exports and intermediate processing versions are omitted. Existing filenames are preserved, including historical panel letters. Existing numerical data, filtering, category mappings and statistical implementations are retained. Script changes are limited to local input/output paths, loading the same emoji image assets from PNG files, and a portable font fallback for the questionnaire plot. Some scripts retain Windows font preferences and use a fallback font elsewhere, so typography can differ between computers. The localization heatmap scripts write to their own `output/` folders and replace previous exports there when rerun.

Participant codes are used throughout. The questionnaire tables include only the columns used to plot participant groups and scores. Unused detailed clinical information and free-text responses are excluded. The package does not include participant photographs or contact details.

## 中文说明

本目录只收录 Fig. 5 和 Fig. 6 最终绘图脚本直接读取的数据，以及对应绘图脚本。不加入原始实验导出、历史版本或中间处理表，不改变数据数值或原脚本的统计方法。原始文件保持不变。

按正文子图编号查找文件。旧文件名中的 panel_d、panel_e、panel_f 不代表当前正文编号。每个绘图目录的 README 提供运行命令。定位图所需的三个 Emoji 图片从原网页中原样提取，脚本改为读取本地 PNG。

问卷公开表仅保留绘图所需的参与者编号、分组和五项评分，未使用的详细临床资料及自由文本不上传。保留字段的数值和行顺序不变。
