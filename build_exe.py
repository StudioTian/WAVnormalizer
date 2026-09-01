from __future__ import annotations

import shutil
import sys
from pathlib import Path

import imageio_ffmpeg
import PyInstaller.__main__


ROOT = Path(__file__).resolve().parent
ENTRY_POINT = ROOT / "WAVnormalizer.py"
ICON = ROOT / "icon.ico"
APP_NAME = "WAVNormalizer"


def _clean_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return
    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
    elif resolved.exists():
        resolved.unlink(missing_ok=True)


def _find_ffmpeg_binary() -> Path:
    # 1. Check local directory
    for candidate_name in ("ffmpeg.exe", "ffmpeg"):
        local_candidate = ROOT / candidate_name
        if local_candidate.is_file():
            return local_candidate.resolve()

    # 2. Check imageio-ffmpeg bundled in venv
    try:
        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        if ffmpeg_exe.is_file():
            return ffmpeg_exe
    except Exception:
        pass

    # 3. Check system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return Path(system_ffmpeg).resolve()

    raise FileNotFoundError("找不到 ffmpeg 執行檔（請確認 imageio-ffmpeg 套件或本地 ffmpeg 是否存在）。")


def main() -> int:
    if not ENTRY_POINT.is_file():
        raise FileNotFoundError(f"找不到程式入口點：{ENTRY_POINT}")
    if not ICON.is_file():
        raise FileNotFoundError(f"找不到圖示檔案：{ICON}")

    ffmpeg_binary = _find_ffmpeg_binary()
    print(f"打包來源：{ENTRY_POINT.name}")
    print(f"使用圖示：{ICON.name}")
    print(f"包含 FFmpeg：{ffmpeg_binary}")

    _clean_path(ROOT / "build")
    _clean_path(ROOT / f"{APP_NAME}.spec")

    separator = ";" if sys.platform == "win32" else ":"
    PyInstaller.__main__.run(
        [
            str(ENTRY_POINT),
            "--name",
            APP_NAME,
            "--onefile",
            "--console",
            "--noconfirm",
            "--icon",
            str(ICON),
            "--add-binary",
            f"{ffmpeg_binary}{separator}.",
            "--collect-all",
            "pedalboard",
            "--distpath",
            str(ROOT / "dist"),
            "--workpath",
            str(ROOT / "build"),
            "--specpath",
            str(ROOT),
        ]
    )

    executable = ROOT / "dist" / f"{APP_NAME}.exe"
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller 未產生執行檔：{executable}")

    print(f"\n[成功] 已建置完成：{executable} ({executable.stat().st_size / 1024 / 1024:.1f} MiB)")
    _clean_path(ROOT / "build")
    _clean_path(ROOT / f"{APP_NAME}.spec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
