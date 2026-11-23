# WAVNormalizer

WAVNormalizer 會把多首歌調到接近的音量，並將結果另存成新檔。

## 在 Windows 使用

1. 把一個或多個音檔或資料夾拖到 `WAVNormalizer.exe` 圖示上。也可以先雙擊程式，再把檔案拖進開啟的視窗並按 Enter。直接按 Enter 會使用程式所在的資料夾。資料夾只會掃描第一層。一起拖入的檔案若格式不支援，程式會顯示檔名並略過。
2. 選擇是否把單聲道或接近單聲道的歌曲轉成立體聲。直接按 Enter 會轉換，選 `2` 會保留原有聲道。原本有立體聲效果的歌曲會保留原本的聲道配置。
3. 選擇儲存格式。`1` 沿用原副檔名，也是直接按 Enter 的預設值。`2` 是 WAV，`3` 是無損 M4A，`4` 是 FLAC。
4. 如果程式發現某首歌的聲音可能不清楚，會問你是否繼續。直接按 Enter 會繼續，選 `2` 會略過該首歌。

處理結束後，視窗會顯示完成、略過、失敗的首數，以及新檔所在的資料夾。新檔放在各自原檔旁邊，檔名會加上 `_enhanced`，例如 `歌曲.m4a` 會儲存為 `歌曲_enhanced.m4a`。成功時會取代同名的新檔。重新掃描時會略過這些已輸出的檔案。看完結果後按 Enter 關閉視窗。

## 聲音會怎麼變

程式通常只會把整首歌的音量一起調高或調低。如果調整後有突然過大的聲音，才會用快速限幅壓低這些峰值。程式不使用等化器，也不做其他動態壓縮。靜音、無法量測或無法同時達到音量和峰值要求的歌曲會回報失敗。

| 支援的副檔名 | 選擇沿用原格式時的編碼 |
| --- | --- |
| `.wav` | 24-bit PCM，無損 |
| `.aif`、`.aiff` | 24-bit PCM，無損 |
| `.flac` | FLAC，無損 |
| `.m4a` | 24-bit ALAC，無損 |
| `.aac` | AAC，有損 |
| `.mp3` | MP3，有損 |
| `.ogg` | Vorbis，有損 |
| `.opus` | Opus，有損 |
| `.wma` | Windows Media Audio 2，有損 |

有損格式會重新編碼。想避免再次有損編碼時，可選 WAV、M4A 或 FLAC。原檔的歌曲標籤、中繼資料和封面圖不會跟著存到新檔。

## 開發與打包

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python WAVnormalizer.py
python -m pytest -q
```

`WAVnormalizer.py` 包含互動流程和音訊處理，測試位於 `tests/test_wavnormalizer.py`。程式用 FFmpeg 的 `ebur128` 量測輸入和每次編碼後的輸出。輸出以 `-14 LUFS` 為目標，容許 `±0.3 LU`，並將真峰值控制在 `-1 dBTP`（容許超出 `0.1 dB`）。最多編碼六次。

在 Windows 執行 `python build_exe.py`，可用 PyInstaller 建立含主控台視窗的 `dist/WAVNormalizer.exe`。打包腳本會附上 FFmpeg 和 `icon.ico`。

推送到 `main` 時，GitHub Actions 會執行測試、打包並試跑 Windows 執行檔。全部通過後才會遞增版號的最後一位，並在 GitHub Releases 發布 `WAVNormalizer.exe`。重跑同一提交會沿用原版號。

## 授權

MIT License
