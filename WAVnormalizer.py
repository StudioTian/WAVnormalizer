from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from pedalboard import Compressor, Pedalboard
from pedalboard.io import AudioFile
from scipy import signal


SUPPORTED_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
OUTPUT_SUFFIX = "_enhanced"
EPSILON = 1.0e-12


@dataclass(frozen=True)
class QualityReport:
    artifact_score: float
    metrics: Dict[str, float]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ProcessResult:
    input_path: Path
    output_path: Optional[Path]
    quality_report: QualityReport
    widened: bool
    skipped: bool


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, EPSILON) / max(denominator, EPSILON))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _as_channel_first(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.ndim != 2:
        raise ValueError("音訊資料必須是一維或二維陣列。")
    if audio.shape[0] not in (1, 2):
        raise ValueError(f"目前只支援單聲道或雙聲道，收到 {audio.shape[0]} 聲道。")
    if audio.shape[1] == 0:
        raise ValueError("音檔沒有可處理的取樣。")
    if not np.all(np.isfinite(audio)):
        raise ValueError("音檔包含 NaN 或無限值。")
    return audio


def _load_with_pedalboard(path: Path) -> Tuple[np.ndarray, int]:
    with AudioFile(str(path), "r") as audio_file:
        sample_rate = int(audio_file.samplerate)
        audio = audio_file.read(audio_file.frames)
    return _as_channel_first(audio), sample_rate


def load_audio(path: Path, ffmpeg_path: Optional[Path] = None) -> Tuple[np.ndarray, int]:
    try:
        return _load_with_pedalboard(path)
    except (OSError, RuntimeError) as original_error:
        if ffmpeg_path is None:
            raise original_error

    # Pedalboard intentionally supports a smaller codec set than ffmpeg. Decode
    # uncommon containers to a temporary floating-point WAV, then keep the same
    # downstream DSP path.
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="wavnormalizer_decode_", suffix=".wav", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        temporary_path.unlink(missing_ok=True)
        _run_ffmpeg(
            [
                str(ffmpeg_path),
                "-hide_banner",
                "-nostats",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "pcm_f32le",
                str(temporary_path),
            ]
        )
        return _load_with_pedalboard(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def is_effectively_mono(audio: np.ndarray) -> bool:
    audio = _as_channel_first(audio)
    if audio.shape[0] == 1:
        return True

    left = audio[0].astype(np.float64, copy=False)
    right = audio[1].astype(np.float64, copy=False)
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    side_to_mid_db = _db_ratio(_rms(side), _rms(mid))

    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    correlation = 1.0 if denominator <= EPSILON else float(
        np.dot(left_centered, right_centered) / denominator
    )
    return side_to_mid_db <= -45.0 or (
        correlation >= 0.99995 and side_to_mid_db <= -35.0
    )


def _analysis_excerpt(mono: np.ndarray, sample_rate: int, seconds: float = 45.0) -> np.ndarray:
    limit = max(1, int(sample_rate * seconds))
    if mono.size <= limit:
        return mono
    start = (mono.size - limit) // 2
    return mono[start : start + limit]


def _stft(samples: np.ndarray, sample_rate: int, n_fft: int, hop_length: int) -> np.ndarray:
    _, _, matrix = signal.stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    return np.asarray(matrix, dtype=np.complex128)


def analyze_quality(audio: np.ndarray, sample_rate: int) -> QualityReport:
    """Return a conservative heuristic score for separation/musical-noise artifacts.

    This is a warning system, not a perceptual proof. A warning requires both a
    high-frequency energy deficit and unstable frame-to-frame high-frequency
    behaviour, which intentionally favours false negatives over false positives.
    """

    audio = _as_channel_first(audio)
    mono = np.mean(audio, axis=0, dtype=np.float64)
    mono = _analysis_excerpt(mono, sample_rate)
    peak = float(np.max(np.abs(mono)))
    overall_rms = _rms(mono)

    if peak <= 1.0e-7 or overall_rms <= 1.0e-8:
        return QualityReport(
            artifact_score=100.0,
            metrics={"rms_dbfs": -160.0},
            reasons=("音檔幾乎沒有可分析的訊號",),
        )

    mono = mono / peak
    n_fft = 2048
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.size))

    magnitude = np.abs(_stft(mono, sample_rate, n_fft, 512))
    power = np.square(magnitude)
    frame_energy = np.sum(power, axis=0)
    active = frame_energy >= max(float(np.max(frame_energy)) * 1.0e-4, EPSILON)
    if not np.any(active):
        active = np.ones(frame_energy.shape, dtype=bool)

    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0
    high_cutoff = min(8000.0, max(3000.0, nyquist * 0.58))
    high_bins = frequencies >= high_cutoff
    if not np.any(high_bins):
        high_bins = frequencies >= nyquist * 0.5

    total_energy = float(np.sum(power[:, active]))
    high_energy = float(np.sum(power[high_bins][:, active]))
    high_frequency_ratio = high_energy / max(total_energy, EPSILON)

    normalized_spectrum = magnitude / np.maximum(
        np.sum(magnitude, axis=0, keepdims=True), EPSILON
    )
    flux = np.sqrt(
        np.sum(np.square(np.diff(normalized_spectrum, axis=1)), axis=0)
    )
    spectral_flux = float(np.median(flux)) if flux.size else 0.0

    high_magnitude = magnitude[high_bins]
    normalized_high = high_magnitude / np.maximum(
        np.sum(high_magnitude, axis=0, keepdims=True), EPSILON
    )
    high_flux_values = np.sqrt(
        np.sum(np.square(np.diff(normalized_high, axis=1)), axis=0)
    )
    high_flux = (
        float(np.percentile(high_flux_values, 75.0))
        if high_flux_values.size
        else 0.0
    )

    high_envelope = np.sqrt(np.mean(np.square(high_magnitude), axis=0) + EPSILON)
    high_envelope_db = 20.0 * np.log10(high_envelope + EPSILON)
    active_pairs = active[1:] & active[:-1]
    high_jumps = np.abs(np.diff(high_envelope_db))
    high_jitter_db = float(
        np.percentile(high_jumps[active_pairs], 75.0)
        if np.any(active_pairs)
        else np.percentile(high_jumps, 75.0)
    )

    flatness_power = np.square(magnitude) + EPSILON
    flatness = np.exp(np.mean(np.log(flatness_power), axis=0)) / np.maximum(
        np.mean(flatness_power, axis=0), EPSILON
    )
    spectral_flatness = float(np.median(flatness[active]))
    cumulative_spectrum = np.cumsum(magnitude, axis=0)
    rolloff_threshold = 0.90 * cumulative_spectrum[-1]
    rolloff_indices = np.argmax(
        cumulative_spectrum >= rolloff_threshold[np.newaxis, :], axis=0
    )
    rolloff = frequencies[rolloff_indices]
    rolloff_ratio = float(np.median(rolloff[active]) / max(nyquist, 1.0))

    # Inter-channel phase coherence is reported for real stereo files. It only
    # has a small, gated influence on the score because naturally wide mixes
    # can also have low coherence.
    high_phase_coherence = 1.0
    high_phase_jitter_rad = 0.0
    if audio.shape[0] == 2 and not is_effectively_mono(audio):
        left_excerpt = _analysis_excerpt(audio[0].astype(np.float64), sample_rate)
        right_excerpt = _analysis_excerpt(audio[1].astype(np.float64), sample_rate)
        left_stft = _stft(left_excerpt, sample_rate, n_fft, 512)
        right_stft = _stft(right_excerpt, sample_rate, n_fft, 512)
        cross_spectrum = left_stft[high_bins] * np.conj(right_stft[high_bins])
        left_power = np.abs(left_stft[high_bins]) ** 2
        right_power = np.abs(right_stft[high_bins]) ** 2
        coherence_denominator = math.sqrt(
            float(np.sum(left_power)) * float(np.sum(right_power))
        )
        if coherence_denominator > EPSILON:
            high_phase_coherence = float(
                np.clip(abs(np.sum(cross_spectrum)) / coherence_denominator, 0.0, 1.0)
            )
        phase_difference = np.angle(cross_spectrum)
        phase_jumps = np.abs(
            np.angle(np.exp(1j * np.diff(phase_difference, axis=1)))
        )
        phase_weights = np.sqrt(left_power[:, 1:] * right_power[:, 1:])
        valid_phase = phase_weights >= max(float(np.max(phase_weights)) * 1.0e-4, EPSILON)
        if np.any(valid_phase):
            high_phase_jitter_rad = float(np.median(phase_jumps[valid_phase]))

    # A low-pass/energy-loss score and an instability score must agree before
    # the default warning threshold (65) is crossed.
    energy_loss = 0.55 * _clip01((0.012 - high_frequency_ratio) / 0.012)
    energy_loss += 0.45 * _clip01((0.52 - rolloff_ratio) / 0.22)
    instability = 0.55 * _clip01((high_flux - 0.10) / 0.30)
    instability += 0.45 * _clip01((high_jitter_db - 3.0) / 8.0)
    texture_anomaly = _clip01((0.0025 - spectral_flatness) / 0.0025)
    phase_anomaly = _clip01((0.45 - high_phase_coherence) / 0.45)
    artifact_score = 100.0 * (
        0.50 * energy_loss
        + 0.40 * instability
        + 0.05 * texture_anomaly
        + 0.05 * energy_loss * phase_anomaly
    )

    reasons: List[str] = []
    if energy_loss >= 0.65:
        reasons.append("高頻能量或頻寬有明顯斷層")
    if instability >= 0.55:
        reasons.append("高頻能量呈現不自然的逐幀抖動")
    if spectral_flatness <= 0.0005 and energy_loss >= 0.45 and instability >= 0.35:
        reasons.append("頻譜過度稀疏，可能存在音樂噪聲")
    if phase_anomaly >= 0.7 and energy_loss >= 0.55:
        reasons.append("高頻左右聲道的相位一致性偏低")

    metrics = {
        "rms_dbfs": 20.0 * math.log10(max(overall_rms, EPSILON)),
        "spectral_flatness": spectral_flatness,
        "spectral_flux": spectral_flux,
        "high_frequency_ratio": high_frequency_ratio,
        "high_frequency_flux": high_flux,
        "high_frequency_jitter_db": high_jitter_db,
        "high_frequency_phase_coherence": high_phase_coherence,
        "high_frequency_phase_jitter_rad": high_phase_jitter_rad,
        "rolloff_ratio": rolloff_ratio,
    }
    return QualityReport(
        artifact_score=float(np.clip(artifact_score, 0.0, 100.0)),
        metrics=metrics,
        reasons=tuple(reasons),
    )


def _allpass_cascade(
    samples: np.ndarray,
    sample_rate: int,
    center_frequencies: Sequence[float],
    radii: Sequence[float],
) -> np.ndarray:
    output = np.asarray(samples, dtype=np.float64)
    for center_frequency, radius in zip(center_frequencies, radii):
        omega = 2.0 * math.pi * center_frequency / sample_rate
        coefficient = -2.0 * radius * math.cos(omega)
        denominator = np.array([1.0, coefficient, radius * radius])
        numerator = denominator[::-1]
        output = signal.lfilter(numerator, denominator, output)
    return output


def widen_mono(
    mono: np.ndarray,
    sample_rate: int,
    width: float = 0.35,
    mono_bass_hz: float = 150.0,
    seed: int = 42,
) -> np.ndarray:
    """Create stereo with decorrelated all-pass networks and exact M/S collapse."""

    if not 0.0 <= width <= 1.0:
        raise ValueError("width 必須介於 0 與 1。")
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)
    nyquist = sample_rate / 2.0
    if not 0.0 < mono_bass_hz < nyquist * 0.8:
        raise ValueError("mono_bass_hz 必須低於 Nyquist 頻率。")

    highpass = signal.butter(
        4, mono_bass_hz, btype="highpass", fs=sample_rate, output="sos"
    )
    high_band = signal.sosfilt(highpass, mono)

    base_centers = np.array([310.0, 690.0, 1450.0, 3050.0, 6200.0])
    base_centers = base_centers[base_centers < nyquist * 0.86]
    if base_centers.size < 2:
        base_centers = np.linspace(mono_bass_hz * 1.5, nyquist * 0.75, 2)

    rng = np.random.default_rng(seed)
    centers_a = base_centers * rng.uniform(0.90, 1.10, base_centers.size)
    centers_b = base_centers * rng.uniform(0.78, 1.22, base_centers.size)
    centers_a = np.clip(centers_a, mono_bass_hz * 1.15, nyquist * 0.90)
    centers_b = np.clip(centers_b, mono_bass_hz * 1.15, nyquist * 0.90)
    radii_a = rng.uniform(0.58, 0.82, base_centers.size)
    radii_b = rng.uniform(0.58, 0.82, base_centers.size)

    decorrelated_a = _allpass_cascade(high_band, sample_rate, centers_a, radii_a)
    decorrelated_b = _allpass_cascade(high_band, sample_rate, centers_b, radii_b)
    raw_side = 0.5 * (decorrelated_a - decorrelated_b)

    raw_side_rms = _rms(raw_side)
    target_side_rms = _rms(high_band) * width
    if raw_side_rms > EPSILON:
        side = raw_side * (target_side_rms / raw_side_rms)
    else:
        side = np.zeros_like(mono)

    left = mono + side
    right = mono - side
    stereo = np.stack((left, right)).astype(np.float32)

    # Keep a safe floating-point headroom before compression. The same scale is
    # applied to Mid and Side, so mono compatibility remains exact.
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.98:
        stereo *= 0.98 / peak
    return stereo


def apply_opto_compression(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -18.0,
    ratio: float = 3.5,
    attack_ms: float = 10.0,
    release_ms: float = 650.0,
) -> np.ndarray:
    audio = _as_channel_first(audio)
    compressor = Compressor(
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    board = Pedalboard([compressor])
    result = board(audio, sample_rate, reset=True)
    return _as_channel_first(result)


def _resolve_executable(candidate: str) -> Optional[Path]:
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_file():
        return candidate_path.resolve()
    resolved = shutil.which(candidate)
    return Path(resolved).resolve() if resolved else None


def find_ffmpeg(explicit_path: Optional[str] = None) -> Path:
    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    if explicit_path:
        resolved = _resolve_executable(explicit_path)
        if resolved:
            return resolved
        raise FileNotFoundError(f"找不到指定的 ffmpeg：{explicit_path}")

    environment_path = os.environ.get("FFMPEG_BINARY")
    if environment_path:
        resolved = _resolve_executable(environment_path)
        if resolved:
            return resolved

    roots: List[Path] = []
    if getattr(sys, "frozen", False):
        roots.extend(
            [
                Path(sys.executable).resolve().parent,
                Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)),
            ]
        )
    roots.extend([Path(__file__).resolve().parent, Path.cwd()])

    for root in roots:
        for relative_path in (
            Path(executable_name),
            Path("ffmpeg") / executable_name,
            Path("ffmpeg") / "bin" / executable_name,
        ):
            candidate = root / relative_path
            if candidate.is_file():
                return candidate.resolve()
        # Single-file builds embed imageio-ffmpeg under its upstream filename,
        # such as ffmpeg-win-x86_64-v7.1.exe.
        pattern = "ffmpeg*.exe" if os.name == "nt" else "ffmpeg*"
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_file():
                return candidate.resolve()

    resolved = _resolve_executable("ffmpeg")
    if resolved:
        return resolved
    raise FileNotFoundError(
        "找不到 ffmpeg。請安裝 ffmpeg、放到 PATH，或使用 --ffmpeg 指定執行檔。"
    )


def _subprocess_startupinfo() -> Optional[subprocess.STARTUPINFO]:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startupinfo


def _run_ffmpeg(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        startupinfo=_subprocess_startupinfo(),
    )


def _extract_loudnorm_measurements(stderr: str) -> Dict[str, str]:
    matches = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    if not matches:
        raise ValueError("ffmpeg 未回傳 loudnorm 測量資料。")
    data = json.loads(matches[-1])
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if any(key not in data for key in required):
        raise ValueError("ffmpeg loudnorm 測量資料不完整。")
    for key in required:
        if not math.isfinite(float(data[key])):
            raise ValueError(f"ffmpeg loudnorm 測量值無效：{key}={data[key]}")
    return {key: str(data[key]) for key in required}


def normalize_loudness(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path,
    sample_rate: int,
    target_lufs: float = -14.0,
    true_peak_db: float = -1.0,
    loudness_range: float = 11.0,
) -> None:
    common_filter = f"I={target_lufs}:TP={true_peak_db}:LRA={loudness_range}"
    first_pass = _run_ffmpeg(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_path),
            "-af",
            f"loudnorm={common_filter}:print_format=json",
            "-f",
            "null",
            os.devnull,
        ]
    )
    measured = _extract_loudnorm_measurements(first_pass.stderr)
    second_filter = (
        f"loudnorm={common_filter}"
        f":measured_I={measured['input_i']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}"
        ":linear=true:print_format=summary"
    )
    _run_ffmpeg(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-nostats",
            "-y",
            "-i",
            str(input_path),
            "-map_metadata",
            "-1",
            "-vn",
            "-sn",
            "-dn",
            "-af",
            second_filter,
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ]
    )


def _quality_message(report: QualityReport) -> str:
    reasons = "、".join(report.reasons) if report.reasons else "未發現明確單一特徵"
    return (
        f"品質風險分數 {report.artifact_score:.1f}/100；{reasons}。"
        "此結果是啟發式警示，請以實際試聽確認。"
    )


def process_file(
    input_path: Path,
    ffmpeg_path: Path,
    output_path: Optional[Path] = None,
    quality_threshold: float = 65.0,
    quality_action: str = "warn",
    width: float = 0.35,
    mono_bass_hz: float = 150.0,
    seed: int = 42,
    target_lufs: float = -14.0,
    true_peak_db: float = -1.0,
) -> ProcessResult:
    input_path = input_path.resolve()
    output_path = (output_path or input_path.with_name(input_path.stem + OUTPUT_SUFFIX + ".wav")).resolve()
    print(f"\n[處理] {input_path.name}")
    audio, sample_rate = load_audio(input_path, ffmpeg_path)
    print(f"  格式：{sample_rate} Hz / {audio.shape[0]} 聲道")

    report = analyze_quality(audio, sample_rate)
    print(f"  分析：{_quality_message(report)}")
    suspicious = report.artifact_score >= quality_threshold
    if suspicious:
        print("  [警告] 音檔音質不佳（可能存在水下聲或高頻劣化）")
        if quality_action == "skip":
            print("  [略過] 已依 --quality-action skip 停止此檔案的處理。")
            return ProcessResult(input_path, None, report, False, True)

    widened = is_effectively_mono(audio)
    if widened:
        mono = np.mean(audio, axis=0, dtype=np.float64)
        audio = widen_mono(mono, sample_rate, width, mono_bass_hz, seed)
        print(f"  拓寬：已套用全通去相關（寬度 {width:.2f}，{mono_bass_hz:.0f} Hz 單聲道低頻）")
    else:
        print("  拓寬：原檔已有有效立體聲，保持原音場。")

    audio = apply_opto_compression(audio, sample_rate)
    print("  壓縮：已套用 Opto 風格參數（3.5:1 / 10 ms / 650 ms）。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_input: Optional[Path] = None
    temporary_output: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="wavnormalizer_dsp_", suffix=".wav", delete=False
        ) as handle:
            temporary_input = Path(handle.name)
        sf.write(str(temporary_input), audio.T, sample_rate, subtype="FLOAT")

        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}_", suffix=".wav", dir=str(output_path.parent), delete=False
        ) as handle:
            temporary_output = Path(handle.name)
        temporary_output.unlink(missing_ok=True)

        normalize_loudness(
            temporary_input,
            temporary_output,
            ffmpeg_path,
            sample_rate,
            target_lufs,
            true_peak_db,
        )
        os.replace(temporary_output, output_path)
        temporary_output = None
    finally:
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

    print(f"  [完成] {output_path.name}（{target_lufs:g} LUFS / {true_peak_db:g} dBTP / 24-bit WAV）")
    return ProcessResult(input_path, output_path, report, widened, False)


def _application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def discover_audio_files(paths: Iterable[str]) -> List[Path]:
    raw_paths = list(paths)
    candidates = [Path(path).expanduser() for path in raw_paths]
    if not candidates:
        candidates = [_application_directory()]

    discovered: List[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            discovered.extend(
                path
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(candidate)
        else:
            print(f"[警告] 找不到或不支援：{candidate}")

    unique: List[Path] = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        if path.stem.endswith(OUTPUT_SUFFIX) or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自動檢測伴奏品質、拓寬單聲道、平滑動態並正規化為直播用 WAV。"
    )
    parser.add_argument("paths", nargs="*", help="音檔或資料夾；省略時處理腳本所在資料夾。")
    parser.add_argument("--ffmpeg", help="ffmpeg 執行檔路徑。")
    parser.add_argument(
        "--quality-action",
        choices=("warn", "skip"),
        default="warn",
        help="品質疑慮時警告後繼續，或略過該檔（預設：warn）。",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=65.0,
        help="品質警告門檻 0–100，越低越敏感（預設：65）。",
    )
    parser.add_argument("--width", type=float, default=0.35, help="單聲道拓寬強度 0–1。")
    parser.add_argument(
        "--mono-bass-hz", type=float, default=150.0, help="Side 高通截止頻率（預設：150 Hz）。"
    )
    parser.add_argument("--seed", type=int, default=42, help="相位網路的可重現亂數種子。")
    parser.add_argument("--target-lufs", type=float, default=-14.0, help="目標整體響度。")
    parser.add_argument("--true-peak", type=float, default=-1.0, help="True Peak 上限 dBTP。")
    return parser


def _configure_windows_output() -> None:
    # When stdout is captured or piped, Windows may otherwise encode Traditional
    # Chinese logs with the legacy code page while the receiver expects UTF-8.
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_windows_output()
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.quality_threshold <= 100.0:
        print("[錯誤] --quality-threshold 必須介於 0 與 100。")
        return 2
    if not 0.0 <= args.width <= 1.0:
        print("[錯誤] --width 必須介於 0 與 1。")
        return 2

    files = discover_audio_files(args.paths)
    if not files:
        print("[錯誤] 找不到可處理的音檔。")
        return 1

    try:
        ffmpeg_path = find_ffmpeg(args.ffmpeg)
    except FileNotFoundError as error:
        print(f"[錯誤] {error}")
        return 2

    print(f"WAVNormalizer：找到 {len(files)} 個音檔")
    print(f"ffmpeg：{ffmpeg_path}")
    failures = 0
    skipped = 0
    for input_path in files:
        try:
            result = process_file(
                input_path=input_path,
                ffmpeg_path=ffmpeg_path,
                quality_threshold=args.quality_threshold,
                quality_action=args.quality_action,
                width=args.width,
                mono_bass_hz=args.mono_bass_hz,
                seed=args.seed,
                target_lufs=args.target_lufs,
                true_peak_db=args.true_peak,
            )
            skipped += int(result.skipped)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            failures += 1
            print(f"  [失敗] {input_path.name}：{error}")
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                tail = "\n".join(error.stderr.strip().splitlines()[-8:])
                print(tail)

    completed = len(files) - failures - skipped
    print(f"\n批次結束：完成 {completed}、略過 {skipped}、失敗 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
