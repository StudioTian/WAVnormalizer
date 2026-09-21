import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

import WAVnormalizer as enhancer


SAMPLE_RATE = 48_000


def sine(frequency: float, seconds: float = 2.0, amplitude: float = 0.12) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def band_energy(samples: np.ndarray, low: float, high: float) -> float:
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
    selected = (frequencies >= low) & (frequencies < high)
    return float(np.sum(np.abs(spectrum[selected]) ** 2))


def test_identical_stereo_is_effectively_mono() -> None:
    mono = sine(440.0)
    assert enhancer.is_effectively_mono(mono)
    assert enhancer.is_effectively_mono(np.stack((mono, mono)))


def test_real_stereo_is_not_effectively_mono() -> None:
    left = sine(440.0)
    right = sine(660.0)
    assert not enhancer.is_effectively_mono(np.stack((left, right)))


def test_widening_has_exact_mono_collapse_and_mono_bass() -> None:
    mono = sine(60.0) + sine(1_200.0)
    widened = enhancer.widen_mono(mono, SAMPLE_RATE, width=0.35, seed=7)

    collapsed = 0.5 * (widened[0] + widened[1])
    np.testing.assert_allclose(collapsed, mono, atol=2.0e-7)

    side = 0.5 * (widened[0] - widened[1])
    low_energy = band_energy(side, 35.0, 100.0)
    high_energy = band_energy(side, 900.0, 1_500.0)
    assert high_energy > low_energy * 100.0


def test_widening_is_reproducible() -> None:
    mono = sine(997.0)
    first = enhancer.widen_mono(mono, SAMPLE_RATE, seed=123)
    second = enhancer.widen_mono(mono, SAMPLE_RATE, seed=123)
    np.testing.assert_array_equal(first, second)


def test_quality_analysis_returns_finite_metrics() -> None:
    audio = np.stack((sine(440.0) + sine(5_000.0), sine(660.0) + sine(7_000.0)))
    report = enhancer.analyze_quality(audio, SAMPLE_RATE)
    assert 0.0 <= report.artifact_score <= 100.0
    assert report.metrics
    assert all(np.isfinite(value) for value in report.metrics.values())


def test_quality_analysis_flags_severe_fragmented_high_band() -> None:
    rng = np.random.default_rng(12)
    sample_count = SAMPLE_RATE * 4
    low_band = signal.sosfilt(
        signal.butter(10, 3_200, fs=SAMPLE_RATE, output="sos"),
        rng.normal(0.0, 0.09, sample_count),
    )
    artifact = np.zeros(sample_count)
    block_size = 2_048
    block_time = np.arange(block_size) / SAMPLE_RATE
    for block_index, start in enumerate(range(0, sample_count, block_size)):
        stop = min(start + block_size, sample_count)
        amplitude = 0.0015 if block_index % 2 else 0.000001
        frequency = rng.uniform(9_000.0, 19_000.0)
        artifact[start:stop] = amplitude * np.sin(
            2.0 * np.pi * frequency * block_time[: stop - start]
        )

    report = enhancer.analyze_quality((low_band + artifact).astype(np.float32), SAMPLE_RATE)
    assert report.artifact_score >= 65.0
    assert any("高頻" in reason for reason in report.reasons)


def test_opto_compression_preserves_shape_and_finite_values() -> None:
    audio = np.stack((sine(440.0, amplitude=0.8), sine(660.0, amplitude=0.8)))
    compressed = enhancer.apply_opto_compression(audio, SAMPLE_RATE)
    assert compressed.shape == audio.shape
    assert np.all(np.isfinite(compressed))


def test_audio_file_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    mono = sine(440.0, seconds=0.25)
    sf.write(source, mono, SAMPLE_RATE, subtype="PCM_24")
    loaded, sample_rate = enhancer.load_audio(source)
    assert sample_rate == SAMPLE_RATE
    assert loaded.shape == (1, mono.size)
    np.testing.assert_allclose(loaded[0], mono, atol=1.0e-6)


def test_audio_loader_uses_ffmpeg_fallback_on_value_error(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.m4a"
    expected = sine(440.0, seconds=0.1)[np.newaxis, :]
    commands = []

    def fake_load(path):
        if path == source:
            raise ValueError('Failed to open audio file: file "source.m4a" does not seem to contain audio data in a known or supported format.')
        return expected, SAMPLE_RATE

    def fake_run(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(enhancer, "_load_with_pedalboard", fake_load)
    monkeypatch.setattr(enhancer, "_run_ffmpeg", fake_run)
    loaded, sample_rate = enhancer.load_audio(source, Path("ffmpeg.exe"))

    np.testing.assert_array_equal(loaded, expected)
    assert sample_rate == SAMPLE_RATE
    assert len(commands) == 1
    assert "pcm_f32le" in commands[0]


def test_extract_loudnorm_measurements() -> None:
    payload = {
        "input_i": "-20.10",
        "input_tp": "-3.20",
        "input_lra": "4.50",
        "input_thresh": "-30.20",
        "output_i": "-14.00",
        "output_tp": "-1.00",
        "output_lra": "4.40",
        "output_thresh": "-24.10",
        "normalization_type": "dynamic",
        "target_offset": "0.10",
    }
    stderr = "ffmpeg preamble\n" + json.dumps(payload, indent=4) + "\n"
    measured = enhancer._extract_loudnorm_measurements(stderr)
    assert measured["input_i"] == "-20.10"
    assert measured["target_offset"] == "0.10"


def test_normalize_loudness_codec_selection(monkeypatch, tmp_path: Path) -> None:
    calls = []
    measurement = {
        "input_i": "-20.10",
        "input_tp": "-3.20",
        "input_lra": "4.50",
        "input_thresh": "-30.20",
        "target_offset": "0.10",
    }

    def fake_run(command):
        calls.append(command)
        stderr = json.dumps(measurement) if len(calls) % 2 == 1 else "done"
        return subprocess.CompletedProcess(command, 0, "", stderr)

    monkeypatch.setattr(enhancer, "_run_ffmpeg", fake_run)

    # Test WAV output uses pcm_s24le
    enhancer.normalize_loudness(
        tmp_path / "input.wav",
        tmp_path / "output.wav",
        Path("ffmpeg.exe"),
        SAMPLE_RATE,
    )
    assert "pcm_s24le" in calls[1]

    # Test M4A output defaults to alac (lossless)
    enhancer.normalize_loudness(
        tmp_path / "input.wav",
        tmp_path / "output.m4a",
        Path("ffmpeg.exe"),
        SAMPLE_RATE,
    )
    assert "alac" in calls[3]

    # Test M4A output with aac codec option
    enhancer.normalize_loudness(
        tmp_path / "input.wav",
        tmp_path / "output.m4a",
        Path("ffmpeg.exe"),
        SAMPLE_RATE,
        m4a_codec="aac",
    )
    assert "aac" in calls[5]


def test_discovery_ignores_its_own_outputs(tmp_path: Path) -> None:
    (tmp_path / "song.wav").touch()
    (tmp_path / "song_enhanced.wav").touch()
    (tmp_path / "song.m4a").touch()
    (tmp_path / "song_enhanced.m4a").touch()
    (tmp_path / "notes.txt").touch()
    discovered = enhancer.discover_audio_files([str(tmp_path)])
    assert (tmp_path / "song.wav").resolve() in discovered
    assert (tmp_path / "song.m4a").resolve() in discovered
    assert (tmp_path / "song_enhanced.wav").resolve() not in discovered
    assert (tmp_path / "song_enhanced.m4a").resolve() not in discovered


def test_end_to_end_m4a_processing(tmp_path: Path) -> None:
    ffmpeg_path = enhancer.find_ffmpeg()
    assert ffmpeg_path is not None

    # Generate a small 1-second m4a file
    m4a_input = tmp_path / "test.m4a"
    subprocess.run(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-nostats",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.0",
            "-c:a",
            "aac",
            str(m4a_input),
        ],
        check=True,
    )

    # Process with default (which is now 'same' -> outputs ALAC .m4a)
    res_default = enhancer.process_file(
        input_path=m4a_input,
        ffmpeg_path=ffmpeg_path,
    )
    assert res_default.output_path is not None
    assert res_default.output_path.suffix.lower() == ".m4a"
    assert res_default.output_path.exists()
    assert res_default.output_path.stat().st_size > 0

    # Probe to ensure it's ALAC (lossless, no secondary lossy compression)
    probe = subprocess.run(
        [str(ffmpeg_path), "-i", str(res_default.output_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert "alac" in probe.stderr.lower()

    # Process to WAV explicitly
    res_wav = enhancer.process_file(
        input_path=m4a_input,
        ffmpeg_path=ffmpeg_path,
        output_format="wav",
    )
    assert res_wav.output_path is not None
    assert res_wav.output_path.suffix.lower() == ".wav"
    assert res_wav.output_path.exists()
    assert res_wav.output_path.stat().st_size > 0

    # Process to M4A with AAC explicitly
    res_m4a_aac = enhancer.process_file(
        input_path=m4a_input,
        ffmpeg_path=ffmpeg_path,
        output_format="m4a",
        m4a_codec="aac",
    )
    assert res_m4a_aac.output_path is not None
    assert res_m4a_aac.output_path.suffix.lower() == ".m4a"
    assert res_m4a_aac.output_path.exists()
