# WAVNormalizer (批次音訊音量標準化與優化工具) 🎚️🔊

專業級免 VST 伴奏與音訊批次最佳化工具。自動評估音質風險、將單聲道聲音自動拓寬為立體聲、套用平滑 Opto 風格壓縮，並透過 FFmpeg 嚴格執行 EBU R128 loudnorm 輸出廣播級標準（-14 LUFS / -1 dBTP / 24-bit WAV）。

## 🌟 核心功能 / Features

1. 🔍 **音質風險評估**：計算頻譜平衡度、高頻穩定性與 roll-off，輸出音質評估報告（支援 `warn` 與 `skip`）。
2. 🎧 **單聲道立體聲拓寬**：全通濾波相位去相關拓寬，150 Hz 以下低頻維持置中，保證完全單聲道相容（無相位抵消失真）。
3. 🎛️ **Opto 光電風格壓縮**：使用 Spotify Pedalboard 壓縮器（3.5:1 / 10 ms attack / 650 ms release）平滑動態。
4. 📊 **精準雙 Pass Loudnorm**：正規化至 -14 LUFS / -1 dBTP 輸出標準 24-bit WAV。

## 📂 檔案結構 / File Structure

```text
WAVnormalizer/
├── WAVnormalizer.py    # 核心處理原始碼
├── build_exe.py        # PyInstaller 打包腳本
├── requirements.txt    # 相依套件清單
├── icon.ico            # 應用程式圖示
└── README.md           # 專案說明文件
```

## 🚀 快速使用 / Quick Start

### 1. 安裝相依套件
```bash
pip install -r requirements.txt
```

### 2. 執行處理
```bash
# 預設維持與輸入檔案相同格式（如 .m4a 預設以 24-bit ALAC 無損封裝，避免二次壓縮損失音質）
python WAVnormalizer.py "路徑/至/音訊.m4a" "路徑/至/資料夾"

# 強制輸出為 24-bit WAV 格式
python WAVnormalizer.py "路徑/至/音訊.m4a" --format wav

# 輸出為 M4A 格式並指定為 AAC 320k 高音質有損壓縮
python WAVnormalizer.py "路徑/至/音訊.m4a" --format m4a --m4a-codec aac
```

## 📄 授權 / License

MIT License
