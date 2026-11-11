# emgteach（LHE7904 实时肌电采集与三指标改进版）

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)

本项目基于开源表面肌电教学平台 [aagisto-maker/emgteach](https://github.com/aagisto-maker/emgteach.git)（基线版本 `v3.11.0`）二次开发，针对 **LHE7904 + nRF54L15（UART 1,000,000 baud）** 采集链路扩展了实时四图趋势显示与时频域核心指标计算。

## 主要功能

1. **多硬件后端支持**
   - **LHE7904 (UART, 1,000,000 baud)**：支持 1000 / 2000 / 4000 SPS 采样率及可编程增益（PGA 1–24），实时接收 `JustFloat` 帧数据。
   - 完整保留原项目的 **BITalino (Bluetooth)** 与 **Arduino + MyoWare 2.0 (USB)** 采集后端。
2. **Acquisition 页面四图实时显示**
   - **图 1 — Raw EMG (`mV`)**：连续显示原始肌电波形（默认显示最近 5 秒，支持周期自适应缩放与幅值适配）。
   - **图 2 — iEMG (`µV·s`) / Envelope (`mV`)**：默认显示每秒更新的积分肌电趋势（`iEMG`），可在工具栏一键切换为原版 5 Hz 平滑包络（`Envelope`）。
   - **图 3 — MPF (`Hz`)**：每秒更新的平均功率频率趋势。
   - **图 4 — MF (`Hz`)**：每秒更新的中位频率趋势。
   - 三张指标图默认显示最近 **60 秒**滑窗趋势，标题栏实时展示最新计算值；丢样或无效窗口自动标记为 `Invalid`（`NaN` 断线），绝不用 `0` 冒充有效结果。
3. **实时滤波与数据保存**
   - 针对 LHE7904 提供可配置的软件带通滤波（默认 `10–500 Hz`，可选 `20–450 Hz`）与单频 `50 Hz` 工频陷波（默认不级联谐波陷波以保留有效肌电频谱），严格校验奈奎斯特准则（$0 < f_{\text{low}} < f_{\text{high}} < f_s / 2$）。
   - 原始未滤波波形保存至标准 **EDF+** 文件（`.edf`），每秒实时三指标同步持久化至同名旁路文件 `<文件名>.edf.metrics.csv`，录制结束后支持四图同步回看。

## 环境要求与安装

支持 **Python 3.10, 3.11, 3.12**（Linux / Windows / macOS）：

```bash
git clone https://github.com/aagisto-maker/emgteach.git
cd emgteach
uv sync
```

（或使用 `pip install -e ".[dev]"` 安装。）

## 快速使用

### 1. 启动上位机界面

```bash
uv run python -m emgteach
```

在 **Acquisition** 页面顶部选择 `LHE7904 (UART, 1,000,000 baud)`，选择对应串口（如 `/dev/ttyUSB0` 或 `COMx`）、采样率（默认 `4000 Hz`）及滤波配置，点击 **Connect** 后再点击 **Start recording** 即可开始实时采集与四图监测。

### 2. 指标准确性与录制文件校验

```bash
# 运行内置理论信号与 4000 SPS 实时三指标校验
uv run python validation/validate_metrics.py

# 对实机录制的 EDF 及旁路 .metrics.csv 进行指标核验（以 100 Hz 正弦信号为例）
uv run python validation/validate_metrics.py --edf <录制文件.edf> --expect-freq 100 --expect-amp 0.5
```

## 架构与测试

本项目包含 Qt-free analytic core (io, dsp, fatigue, mvc, apda, coactivation, phases, force_velocity, fv_rehearsal, profiles, selection, exports, reports, i18n, modes) 与 PySide6 桌面界面层，全套测试共 **1981 tests**：

```bash
uv run pytest
```

## 致谢与开源协议

- 上游原项目：[aagisto-maker/emgteach](https://github.com/aagisto-maker/emgteach.git)（作者：Álvaro Agis-Torres 等，Universidad Complutense de Madrid，DOI: [10.5281/zenodo.21002297](https://doi.org/10.5281/zenodo.21002297)）。
- 开源许可证：[GNU General Public License v3.0 (GPL-3.0)](LICENSE)。
