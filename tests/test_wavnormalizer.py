import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

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


def test_compressor_symbols_are_removed() -> None:
    for name in ("apply_opto_compression", "Compressor", "Pedalboard", "GAIN_ONLY_EXTENSIONS"):
        assert not hasattr(enhancer, name)


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


def test_parse_final_ebur128_summary() -> None:
    stderr = (
        "Summary:\nIntegrated loudness:\n  I: -70.0 LUFS\nTrue peak:\n  Peak: -inf dBFS\n"
        "Summary:\nIntegrated loudness:\n  I: -20.1 LUFS\n  Threshold: -30.1 LUFS\n"
        "Sample peak:\n  Peak: -4.3 dBFS\nTrue peak:\n  Peak: -3.2 dBFS\n"
    )
    assert enhancer._parse_ebur128_summary(stderr) == (-20.1, -3.2)


@pytest.mark.parametrize(
    ("integrated", "peak"), [("-70.0", "-48.1"), ("-inf", "-2.0"), ("-20.0", "-inf"), ("-20.0", "")],
)
def test_reject_unmeasurable_ebur128_summary(integrated: str, peak: str) -> None:
    stderr = f"Summary:\nIntegrated loudness:\n  I: {integrated} LUFS\nTrue peak:\n  Peak: {peak} dBFS\n"
    with pytest.raises(ValueError, match="量測|測量"):
        enhancer._parse_ebur128_summary(stderr)


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


def write_encoded_audio(
    output_path: Path, samples: np.ndarray, ffmpeg: Path, codec: str, sample_rate: int = SAMPLE_RATE
) -> None:
    source = output_path.with_name(f"{output_path.stem}_fixture.wav")
    sf.write(source, samples, SAMPLE_RATE, subtype="FLOAT")
    enhancer._run_ffmpeg(
        [
            str(ffmpeg),
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-ar",
            str(sample_rate),
            "-c:a",
            codec,
            str(output_path),
        ]
    )


@pytest.mark.parametrize(
    ("suffix", "input_codec", "output_codec", "sample_rate"),
    [
        (".aac", "aac", "aac", SAMPLE_RATE),
        (".aif", "pcm_s24be", "pcm_s24be", SAMPLE_RATE),
        (".aiff", "pcm_s24be", "pcm_s24be", SAMPLE_RATE),
        (".flac", "flac", "flac", SAMPLE_RATE),
        (".m4a", "aac", "alac", 44_100),
        (".mp3", "libmp3lame", "mp3", SAMPLE_RATE),
        (".ogg", "libvorbis", "vorbis", SAMPLE_RATE),
        (".opus", "libopus", "opus", SAMPLE_RATE),
        (".wav", "pcm_s24le", "pcm_s24le", SAMPLE_RATE),
        (".wma", "wmav2", "wmav2", SAMPLE_RATE),
    ],
)
def test_same_format_output(
    tmp_path: Path, capsys, suffix: str, input_codec: str, output_codec: str, sample_rate: int
) -> None:
    ffmpeg = enhancer.find_ffmpeg()
    source = tmp_path / f"test{suffix}"
    write_encoded_audio(source, sine(440.0), ffmpeg, input_codec, sample_rate)

    result = enhancer.process_file(source, ffmpeg)
    output = tmp_path / f"test_enhanced{suffix}"
    assert result.output_path == output
    assert result.widened
    assert output.stat().st_size > 0
    probe = enhancer._run_ffmpeg(
        [str(ffmpeg), "-i", str(output), "-f", "null", os.devnull]
    )
    assert f"Audio: {output_codec}" in probe.stderr
    if output_codec == "alac":
        assert "24 bit" in probe.stderr
    check_output_loudness(output, ffmpeg)
    assert "音量只做了整首一致的調整" in capsys.readouterr().out
    audio, output_rate = enhancer.load_audio(output, ffmpeg)
    assert audio.shape[0] == 2
    assert output_rate == sample_rate


def check_output_loudness(output: Path, ffmpeg: Path) -> None:
    integrated_lufs, peak_db = enhancer._measure_loudness(output, ffmpeg)
    assert -14.3 <= integrated_lufs <= -13.7
    assert peak_db <= -0.9


def test_wide_dynamics_matches_independent_ebur128_scan(tmp_path: Path) -> None:
    ffmpeg = enhancer.find_ffmpeg()
    time = np.arange(15 * SAMPLE_RATE) / SAMPLE_RATE
    envelope = 0.004 + 0.5 * np.sin(2 * np.pi * time / 15) ** 8
    noise = np.random.default_rng(42).standard_normal(time.size)
    wave = envelope * (np.sin(2 * np.pi * 330 * time) + 0.2 * noise)
    source = tmp_path / "wide-dynamics.wav"
    sf.write(source, np.column_stack((wave, wave)), SAMPLE_RATE, subtype="FLOAT")

    result = enhancer.process_file(source, ffmpeg, widen=False)
    scan = enhancer._run_ffmpeg([
        str(ffmpeg), "-hide_banner", "-nostats", "-i", str(result.output_path),
        "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", os.devnull,
    ])
    summary = scan.stderr.rsplit("Summary:", 1)[1]
    readings = dict(line.strip().split(":", 1) for line in summary.splitlines() if ":" in line)
    assert -14.3 <= float(readings["I"].split()[0]) <= -13.7
    assert float(readings["Peak"].split()[0]) <= -0.9


def test_static_gain_preserves_dynamics(tmp_path: Path, monkeypatch) -> None:
    ffmpeg = enhancer.find_ffmpeg()
    source = tmp_path / "dynamics.M4A"
    samples = np.concatenate((sine(440.0, amplitude=0.04), sine(440.0, amplitude=0.24)))
    write_encoded_audio(source, samples, ffmpeg, "alac")
    filters = []
    encode = enhancer._encode_audio

    def record_encode(input_path, output_path, ffmpeg_path, sample_rate, audio_filter):
        filters.append(audio_filter)
        return encode(input_path, output_path, ffmpeg_path, sample_rate, audio_filter)

    monkeypatch.setattr(enhancer, "_encode_audio", record_encode)
    result = enhancer.process_file(source, ffmpeg, widen=False)
    assert not result.widened
    assert result.output_path == tmp_path / "dynamics_enhanced.m4a"
    before, _ = enhancer.load_audio(source, ffmpeg)
    after, _ = enhancer.load_audio(result.output_path, ffmpeg)
    assert before.shape == after.shape == (1, samples.size)
    assert len(filters) == 1
    gain = float(filters[0].removeprefix("volume=").removesuffix("dB:precision=double"))
    mask = np.abs(before) > 0.01
    np.testing.assert_allclose(after[mask] / before[mask], 10 ** (gain / 20), atol=2.0e-5)
    check_output_loudness(result.output_path, ffmpeg)


@pytest.mark.parametrize(
    ("suffix", "input_codec"),
    [(".wav", "pcm_s24le"), (".m4a", "alac"), (".opus", "libopus")],
)
def test_limiter_matches_loudness_and_controls_peak(
    tmp_path: Path, capsys, suffix: str, input_codec: str
) -> None:
    ffmpeg = enhancer.find_ffmpeg()
    left = sine(440.0, seconds=4.0, amplitude=0.065)
    right = sine(660.0, seconds=4.0, amplitude=0.055)
    start = SAMPLE_RATE
    burst = sine(950.0, seconds=0.035, amplitude=0.88)
    left[start : start + burst.size] += burst
    source = tmp_path / f"dynamics{suffix}"
    write_encoded_audio(source, np.column_stack((left, right)), ffmpeg, input_codec)
    input_lufs, input_peak_db = enhancer._measure_loudness(source, ffmpeg)
    target_gain = -14.0 - input_lufs
    assert input_peak_db + target_gain > 0.0

    result = enhancer.process_file(source, ffmpeg)
    assert not result.widened
    assert result.output_path == tmp_path / f"dynamics_enhanced{suffix}"
    check_output_loudness(result.output_path, ffmpeg)
    assert "已控制突然過大的聲音" in capsys.readouterr().out
    if suffix == ".wav":
        decoded, _ = sf.read(result.output_path)
        # A peak on the left also reduces the right at the same point.
        peak_region = slice(start + 100, start + burst.size - 100)
        baseline = slice(0, SAMPLE_RATE // 2)
        assert decoded.shape == (left.size, 2)
        peak_gain = np.linalg.norm(decoded[peak_region, 1]) / np.linalg.norm(right[peak_region])
        baseline_gain = np.linalg.norm(decoded[baseline, 1]) / np.linalg.norm(right[baseline])
        assert peak_gain < 0.85 * baseline_gain


def test_unmeasurable_input_leaves_no_output(tmp_path: Path) -> None:
    ffmpeg = enhancer.find_ffmpeg()
    source = tmp_path / "silent.wav"
    sf.write(source, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError, match="靜音"):
        enhancer.process_file(source, ffmpeg)
    assert list(tmp_path.iterdir()) == [source]


def test_encoding_peak_can_trigger_limiter(monkeypatch) -> None:
    filters = []
    peaks = iter([-0.7, -1.0])
    monkeypatch.setattr(enhancer, "_encode_audio", lambda *args: filters.append(args[-1]))
    monkeypatch.setattr(enhancer, "_measure_loudness", lambda *args: (-14.0, next(peaks)))

    assert enhancer.normalize_loudness(
        Path("in.wav"), Path("out.opus"), Path("ffmpeg.exe"), SAMPLE_RATE, -20.0, -7.2,
    )
    assert len(filters) == 2
    assert filters[0] == "volume=6.00000000dB:precision=double"
    assert ",alimiter=" in filters[1]


def test_retries_are_bounded_and_keep_previous_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.opus"
    output = tmp_path / "source_enhanced.opus"
    output.write_bytes(b"previous result")
    filters = []

    def fake_ffmpeg(command):
        audio_filter = command[command.index("-af") + 1]
        assert output.read_bytes() == b"previous result"
        if audio_filter == "ebur128=peak=true:framelog=quiet":
            assert command[-3:] == ["-f", "null", os.devnull]
            stderr = "Summary:\nIntegrated loudness:\n  I: -20.0 LUFS\nTrue peak:\n  Peak: -0.6 dBFS\n"
        else:
            assert ",aresample=192000:osf=dbl,alimiter=" in audio_filter
            assert ":attack=1:release=50:level=0:level_in=1:level_out=1:latency=1" in audio_filter
            assert audio_filter.endswith(",aresample=48000:osf=dbl")
            filters.append(audio_filter)
            Path(command[-1]).write_bytes(b"encoded result")
            stderr = ""
        return subprocess.CompletedProcess(command, 0, "", stderr)

    monkeypatch.setattr(enhancer, "load_audio", lambda *args: (sine(440.0)[None, :], SAMPLE_RATE))
    monkeypatch.setattr(enhancer, "_run_ffmpeg", fake_ffmpeg)
    with pytest.raises(ValueError, match="試了 6 次"):
        enhancer.process_file(source, Path("ffmpeg.exe"), widen=False)
    assert len(set(filters)) == len(filters) == 6
    assert output.read_bytes() == b"previous result"
    assert list(tmp_path.iterdir()) == [output]


def mock_console(monkeypatch, answers, interactive=True):
    prompts = []
    answers = iter(answers)

    def fake_input(prompt):
        prompts.append(prompt)
        answer = next(answers)
        if isinstance(answer, EOFError):
            raise answer
        return answer

    monkeypatch.setattr(enhancer.sys, "stdin", SimpleNamespace(isatty=lambda: interactive))
    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


@pytest.mark.parametrize("mode", ["enter", "eof", "noninteractive"])
def test_interactive_defaults(tmp_path: Path, monkeypatch, capsys, mode: str) -> None:
    source = tmp_path / "song.wav"
    source.touch()
    monkeypatch.setattr(enhancer, "_application_directory", lambda: tmp_path)
    monkeypatch.setattr(enhancer, "find_ffmpeg", lambda: Path("ffmpeg"))
    answers = {"enter": [""] * 5, "eof": [EOFError()], "noninteractive": []}
    prompts = mock_console(monkeypatch, answers[mode], interactive=mode != "noninteractive")
    calls = []

    def fake_process(path, ffmpeg_path, *, output_format, widen, confirm_quality):
        calls.append((path, output_format, widen, confirm_quality()))
        report = enhancer.QualityReport(90.0, {}, ())
        return enhancer.ProcessResult(path, tmp_path / "song_enhanced.wav", report, True, False)

    monkeypatch.setattr(enhancer, "process_file", fake_process)
    assert enhancer.main([]) == 0
    assert calls == [(source, "same", True, True)]
    assert len(prompts) == {"enter": 5, "eof": 1, "noninteractive": 0}[mode]
    if mode == "enter":
        assert prompts[-1] == "按 Enter 關閉視窗。"
    assert f"新檔已儲存到 {tmp_path}" in capsys.readouterr().out


def test_interactive_drag_retry_menu_and_quality_skip(tmp_path: Path, monkeypatch, capsys) -> None:
    first = tmp_path / "第一 首歌.wav"
    second = tmp_path / "第二 首歌.wav"
    for path in [first, second]:
        sf.write(path, sine(440.0), SAMPLE_RATE)
    invalid = tmp_path / "附註.txt"
    invalid.touch()
    reports = iter([enhancer.QualityReport(90.0, {}, ()), enhancer.QualityReport(0.0, {}, ())])
    monkeypatch.setattr(enhancer, "analyze_quality", lambda *args: next(reports))
    prompts = mock_console(
        monkeypatch,
        ['"未結束', "missing.wav", f'"{first}" "{invalid}" "" "{second}"', "3", "2", "9", "3", "2", ""],
    )

    assert enhancer.main([]) == 0
    assert len(prompts) == 9
    assert not (tmp_path / "第一 首歌_enhanced.m4a").exists()
    output = tmp_path / "第二 首歌_enhanced.m4a"
    audio, _ = enhancer.load_audio(output, enhancer.find_ffmpeg())
    assert audio.shape[0] == 1
    log = capsys.readouterr().out
    assert f"已略過 {invalid}" in log
    assert "已略過空白路徑" in log
    assert "找到 2 首歌" in log
    assert "完成 1 首，略過 1 首，失敗 0 首" in log


def test_windows_dragged_paths_preserve_backslashes() -> None:
    assert enhancer._parse_dragged_paths(
        r'"C:\音樂 資料\第一首.wav" D:\歌曲\第二首.opus "\\server\音 樂\\"'
    ) == [r"C:\音樂 資料\第一首.wav", r"D:\歌曲\第二首.opus", "\\\\server\\音 樂\\\\"]


@pytest.mark.parametrize("error", [FileNotFoundError("找不到 FFmpeg"), KeyError("unexpected")])
def test_dragged_argv_and_errors_still_pause(tmp_path: Path, monkeypatch, capsys, error: Exception) -> None:
    source = tmp_path / "音 樂.wav"
    source.touch()
    prompts = mock_console(monkeypatch, ["", "", ""])

    def fail_to_find_ffmpeg():
        raise error

    monkeypatch.setattr(enhancer, "find_ffmpeg", fail_to_find_ffmpeg)
    assert enhancer.main([str(source)]) != 0
    assert len(prompts) == 3
    assert prompts[-1] == "按 Enter 關閉視窗。"
    assert "處理中斷" in capsys.readouterr().out
