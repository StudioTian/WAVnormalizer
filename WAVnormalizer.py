from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
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
TARGET_LUFS = -14.0
TRUE_PEAK_DB = -1.0
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
    except (ValueError, OSError, RuntimeError) as original_error:
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

    # Scale Mid and Side together to preserve their balance when folded to mono.
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.98:
        stereo *= 0.98 / peak
    return stereo


def _resolve_executable(candidate: str) -> Optional[Path]:
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_file():
        return candidate_path.resolve()
    resolved = shutil.which(candidate)
    return Path(resolved).resolve() if resolved else None


def find_ffmpeg() -> Path:
    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
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

    try:
        import imageio_ffmpeg

        imageio_bin = _resolve_executable(imageio_ffmpeg.get_ffmpeg_exe())
        if imageio_bin:
            return imageio_bin
    except Exception:
        pass

    raise FileNotFoundError("找不到處理音檔所需的 FFmpeg。請確認程式檔案是否完整。")


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


def _parse_ebur128_summary(stderr: str) -> Tuple[float, float]:
    _, heading, summary = stderr.rpartition("Summary:")
    measurements = re.search(
        r"Integrated loudness:\s*I:\s*(\S+)\s+LUFS\b.*?"
        r"True peak:\s*Peak:\s*(\S+)\s+dBFS\b",
        summary,
        flags=re.DOTALL,
    )
    if not heading or measurements is None:
        raise ValueError("無法讀取音量與真峰值量測結果。")
    integrated_lufs, peak_db = map(float, measurements.groups())
    if not math.isfinite(integrated_lufs) or integrated_lufs <= -70.0 or not math.isfinite(peak_db):
        raise ValueError("音檔是靜音，或聲音太短或太小，無法測量音量。")
    return integrated_lufs, peak_db


def _audio_codec_args_for_output(output_path: Path) -> List[str]:
    suffix = output_path.suffix.lower()
    if suffix == ".m4a":
        return ["-c:a", "alac", "-sample_fmt", "s32p"]
    elif suffix == ".aac":
        return ["-c:a", "aac", "-b:a", "320k"]
    elif suffix == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", "320k"]
    elif suffix == ".flac":
        return ["-c:a", "flac"]
    elif suffix in (".aif", ".aiff"):
        return ["-c:a", "pcm_s24be"]
    elif suffix == ".ogg":
        return ["-c:a", "libvorbis", "-q:a", "8"]
    elif suffix == ".opus":
        return ["-c:a", "libopus", "-b:a", "192k"]
    elif suffix == ".wma":
        return ["-c:a", "wmav2", "-b:a", "192k"]
    return ["-c:a", "pcm_s24le"]


def _measure_loudness(input_path: Path, ffmpeg_path: Path) -> Tuple[float, float]:
    result = _run_ffmpeg(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_path),
            "-af",
            "ebur128=peak=true:framelog=quiet",
            "-f",
            "null",
            os.devnull,
        ]
    )
    return _parse_ebur128_summary(result.stderr)


def _encode_audio(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path,
    sample_rate: int,
    audio_filter: str,
) -> None:
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
            audio_filter,
            "-ar",
            str(sample_rate),
            *_audio_codec_args_for_output(output_path),
            str(output_path),
        ]
    )


def normalize_loudness(
    input_path: Path,
    output_path: Path,
    ffmpeg_path: Path,
    sample_rate: int,
    input_lufs: float,
    input_peak_db: float,
) -> bool:
    gain_db = TARGET_LUFS - input_lufs
    use_limiter = input_peak_db + gain_db > TRUE_PEAK_DB
    ceiling_db = TRUE_PEAK_DB
    previous = None
    max_attempts = 6
    for _ in range(max_attempts):
        audio_filter = f"volume={gain_db:.8f}dB:precision=double"
        if use_limiter:
            ceiling = 10 ** (ceiling_db / 20.0)
            audio_filter += (
                f",aresample={sample_rate * 4}:osf=dbl"
                f",alimiter=limit={ceiling:.8f}:attack=1:release=50"
                ":level=0:level_in=1:level_out=1:latency=1"
                f",aresample={sample_rate}:osf=dbl"
            )
        _encode_audio(input_path, output_path, ffmpeg_path, sample_rate, audio_filter)
        output_lufs, output_peak_db = _measure_loudness(output_path, ffmpeg_path)
        lufs_error = TARGET_LUFS - output_lufs
        peak_ok = output_peak_db <= TRUE_PEAK_DB + 0.1
        if TARGET_LUFS - 0.3 <= output_lufs <= TARGET_LUFS + 0.3 and peak_ok:
            return use_limiter

        if not use_limiter:
            use_limiter = output_peak_db + lufs_error > TRUE_PEAK_DB + 0.1
            gain_db += lufs_error
            continue

        # Relative to the ceiling, the previous render estimates how much extra gain survives limiting.
        drive = gain_db - ceiling_db
        relative_lufs = output_lufs - ceiling_db
        slope = 1.0
        if previous is not None:
            previous_drive, previous_lufs = previous
            drive_change = drive - previous_drive
            if abs(drive_change) >= 0.1:
                slope = min(1.0, max(0.25, (relative_lufs - previous_lufs) / drive_change))
        previous = drive, relative_lufs
        ceiling_change = 0.0
        if not peak_ok:
            next_ceiling_db = max(-24.0, ceiling_db - (output_peak_db - TRUE_PEAK_DB) - 0.05)
            ceiling_change = next_ceiling_db - ceiling_db
            ceiling_db = next_ceiling_db
        gain_db += ceiling_change + (lufs_error - ceiling_change) / slope

    raise ValueError(
        f"試了 {max_attempts} 次，仍無法將音量調到目標並控制突發的大音量。"
    )


def process_file(
    input_path: Path,
    ffmpeg_path: Path,
    output_path: Optional[Path] = None,
    output_format: str = "same",
    widen: bool = True,
    confirm_quality: Optional[Callable[[], bool]] = None,
) -> ProcessResult:
    input_path = input_path.resolve()
    if output_path is None:
        if output_format == "same":
            target_suffix = input_path.suffix.lower()
        else:
            target_suffix = f".{output_format.lower().lstrip('.')}"
        output_path = input_path.with_name(input_path.stem + OUTPUT_SUFFIX + target_suffix)
    else:
        output_path = output_path.resolve()
    print(f"\n[處理] {input_path.name}")
    audio, sample_rate = load_audio(input_path, ffmpeg_path)
    report = analyze_quality(audio, sample_rate)
    widened = widen and is_effectively_mono(audio)
    if widened:
        mono = np.mean(audio, axis=0, dtype=np.float64)
        audio = widen_mono(mono, sample_rate)

    temporary_input: Optional[Path] = None
    temporary_output: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="wavnormalizer_dsp_", suffix=".wav", delete=False
        ) as handle:
            temporary_input = Path(handle.name)
        sf.write(str(temporary_input), audio.T, sample_rate, subtype="FLOAT")
        input_lufs, input_peak_db = _measure_loudness(temporary_input, ffmpeg_path)
        if report.artifact_score >= 65.0:
            print("  這首歌的聲音可能不太清楚，建議先試聽。")
            if confirm_quality is not None and not confirm_quality():
                print("  [略過] 這首歌未另存新檔。")
                return ProcessResult(input_path, None, report, False, True)
        if widened:
            print("  已將單聲道轉成立體聲。")
        elif not widen:
            print("  保留原有聲道。")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}_", suffix=output_path.suffix, dir=str(output_path.parent), delete=False
        ) as handle:
            temporary_output = Path(handle.name)
        temporary_output.unlink(missing_ok=True)

        limited = normalize_loudness(
            temporary_input, temporary_output, ffmpeg_path, sample_rate, input_lufs, input_peak_db
        )
        os.replace(temporary_output, output_path)
        temporary_output = None
    finally:
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

    print("  已控制突然過大的聲音。" if limited else "  音量只做了整首一致的調整。")
    print(f"  [完成] 已儲存 {output_path.name}")
    return ProcessResult(input_path, output_path, report, widened, False)


def _application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def discover_audio_files(paths: Iterable[str]) -> List[Path]:
    discovered: List[Path] = []
    for path_text in list(paths) or [str(_application_directory())]:
        if not path_text:
            print("已略過空白路徑。")
            continue
        try:
            candidate = Path(path_text).expanduser()
            if candidate.is_dir():
                discovered.extend(
                    path
                    for path in sorted(candidate.iterdir())
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                continue
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered.append(candidate)
                continue
        except (OSError, RuntimeError, ValueError):
            pass
        print(f"已略過 {path_text}（找不到、無法開啟或格式不支援）。")

    unique: List[Path] = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        if path.stem.endswith(OUTPUT_SUFFIX) or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _parse_dragged_paths(text: str) -> List[str]:
    paths = shlex.shlex(text, posix=True)
    paths.whitespace_split = True
    paths.quotes = '"'
    paths.escape = ""
    paths.commenters = ""
    return list(paths)


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
    interactive = sys.stdin is not None and sys.stdin.isatty()

    def ask(prompt: str, choices: Sequence[str] = (), default: str = "") -> str:
        nonlocal interactive
        while interactive:
            try:
                answer = input(prompt).strip()
            except (EOFError, OSError):
                interactive = False
                break
            if not answer:
                return default
            if not choices or answer in choices:
                return answer
            print("請輸入畫面上的其中一個選項，再按 Enter。")
        return default

    try:
        _configure_windows_output()
        paths = list(sys.argv[1:] if argv is None else argv)
        files = discover_audio_files(paths) if paths else []
        if paths and not files and not interactive:
            print("找不到可處理的音檔。")
            return 1
        while not files:
            answer = ask("請把音檔或資料夾拖到這裡，再按 Enter。直接按 Enter 會使用程式所在的資料夾。\n> ")
            try:
                paths = _parse_dragged_paths(answer)
            except ValueError:
                print("讀不到這些路徑，請把檔案重新拖進來。")
                continue
            files = discover_audio_files(paths)
            if not files:
                print("找不到可處理的音檔。")
                if not interactive:
                    return 1

        print(f"找到 {len(files)} 首歌。")
        widen = ask(
            "要把單聲道歌曲轉成立體聲嗎？已有立體聲效果的歌曲會保留原有聲道。\n"
            "1. 要（直接按 Enter）\n2. 不要\n> ",
            ("1", "2"),
            "1",
        ) == "1"
        formats = {"1": "same", "2": "wav", "3": "m4a", "4": "flac"}
        format_choice = ask(
            "要儲存成哪種格式？\n1. 和原檔相同（直接按 Enter）\n"
            "2. WAV\n3. M4A（無損）\n4. FLAC\n> ",
            tuple(formats),
            "1",
        )

        def confirm_quality() -> bool:
            return ask("要繼續處理這首歌嗎？\n1. 繼續（直接按 Enter）\n2. 略過\n> ", ("1", "2"), "1") == "1"

        ffmpeg_path = find_ffmpeg()
        completed = skipped = failures = 0
        output_folders = set()
        for path in files:
            try:
                result = process_file(
                    path,
                    ffmpeg_path,
                    output_format=formats[format_choice],
                    widen=widen,
                    confirm_quality=confirm_quality,
                )
                if result.skipped:
                    skipped += 1
                elif result.output_path is not None:
                    completed += 1
                    output_folders.add(result.output_path.parent)
            except subprocess.CalledProcessError:
                failures += 1
                print(f"  [失敗] {path.name} 無法讀取或儲存。請確認音檔能正常播放，且資料夾可寫入。")
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                print(f"  [失敗] {path.name}。{error}")

        print(f"\n處理結束。完成 {completed} 首，略過 {skipped} 首，失敗 {failures} 首。")
        for folder in sorted(output_folders):
            print(f"新檔已儲存到 {folder}")
        return 1 if failures else 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"處理中斷。{error}")
        return 1
    finally:
        try:
            ask("按 Enter 關閉視窗。")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
