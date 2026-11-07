"""Render a truthful, sampled playback of an existing benchmark (no rerun).

Uses the system OpenCV H.264 encoder:
  PYTHONNOUSERSITE=1 /usr/bin/python3 experiments/routing_mission_video.py
Sources remain read-only. This module lives outside the frozen study package.
"""

import argparse
import hashlib
import json
import math
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BG = "#101923"
PANEL = "#1b2938"
WHITE = "#edf3f9"
MUTED = "#a6b7c9"
CYAN = "#52e4df"
ORANGE = "#ffbc70"
RED = "#ff8795"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SIZE = (1280, 900)
FPS = 12
SPEED = 6
INTRO = 5
OUTRO = 8


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(None)
def font(size):
    return ImageFont.truetype(FONT, size, index=1)


def label(draw, xy, text, size=20, color=WHITE):
    draw.text(xy, text, font=font(size), fill=color)


def panel(draw, box):
    draw.rounded_rectangle(box, radius=12, fill=PANEL)


def model(event):
    return event["action"]["config_id"].removeprefix("CFG_")


def completed(events, t):
    return [e for e in events if e["completion_time_s"] <= t]


def reports_at(result, t):
    reports = [r for r in result["report_quality"]["reports"] if r["completion_time_s"] <= t]
    return len({r["target_id"] for r in reports if r["target_id"]}), sum(
        r["target_id"] is None for r in reports
    )


class Playback:
    def __init__(self, live, study):
        self.live = live
        self.study = study
        self.mission = live / "mission"
        self.context = study / "clear/comparison/user/policy_07"
        self.events = read(self.mission / "events.json")
        self.adaptive = read(self.context / "events.json")
        self.result = read(self.mission / "results.json")
        self.adaptive_result = read(self.context / "results.json")
        self.scenario = read(self.mission / "scenario.json")
        self.validation = read(live / "live_validation.json")
        assert self.validation["status"] == "completed"
        assert self.validation["pose_agreement"]["within_tolerance"]
        assert self.result["backend"]["backend"] == "live_gazebo"
        assert self.adaptive_result["routing_only_audit"]["same_recording"]
        self.duration = max(self.result["final_time_s"], self.adaptive_result["final_time_s"])
        self.frames = []
        for event in self.events:
            path = self.mission / f"observation_{event['observation_id']:05d}.npz"
            with np.load(path, allow_pickle=False) as data:
                rgb, mask = data["rgb"].copy(), data["prediction_mask"].copy()
            assert rgb.shape == (720, 960, 3) and mask.shape == rgb.shape[:2]
            raw = Image.fromarray(rgb).resize((768, 576))
            highlighted = rgb.copy()
            highlighted[mask] = (0.35 * highlighted[mask] + 0.65 * np.array([82, 228, 223])).astype(
                np.uint8
            )
            overlay = Image.fromarray(highlighted).resize((768, 576))
            od = ImageDraw.Draw(overlay)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            for x, y, w, h, area in stats[1:count]:
                if area:
                    od.rectangle(
                        (x * 0.8 - 5, y * 0.8 - 5, (x + w) * 0.8 + 5, (y + h) * 0.8 + 5),
                        outline=CYAN,
                        width=2,
                    )
            self.frames.append((raw, overlay, path))

    def title(self, subtitle):
        im = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(im)
        label(d, (24, 19), "DRONE ROUTING BENCHMARK", 28)
        label(d, (24, 62), subtitle, 20, MUTED)
        return im, d

    def intro(self):
        im, d = self.title("실제 Gazebo 실행 기록 · 모델 선택이 미션 결과에 미치는 영향")
        panel(d, (24, 125, 1256, 315))
        label(d, (48, 147), "USER → INTENT → MODEL ROUTING → MISSION SCORE", 27, CYAN)
        label(d, (48, 201), "“사람 수색 · 400초 이내 · 첫 유효 보고 300초 이내”", 29)
        label(d, (48, 252), "보고 재현율 ≥ 50%   |   오탐 보고 ≤ 2   |   잔여 배터리 ≥ 20%", 25)
        for y, heading, lines in [
            (
                350,
                "01   물리 시뮬레이터 + 실제 신경망 추론",
                ["왼쪽: Gazebo 카메라와 CABiNet 실제 추론. 정답 마스크는 표시하지 않음."],
            ),
            (
                470,
                "02   동일 비행 조건에서 라우팅 비교",
                ["오른쪽: 동일 기준 기록으로 평가한 context 라우터. 실제 모델 예측 캐시 사용."],
            ),
            (
                590,
                "03   실험 기록 그대로 재생",
                ["5초마다 저장된 관측을 6배속 재생. 관측 사이 화면은 유지하며 보간하지 않음."],
            ),
        ]:
            label(d, (48, y), heading, 26, CYAN)
            label(d, (48, y + 45), lines[0], 22)
        label(
            d,
            (48, 742),
            "실제 비행 요약을 바탕으로 구성한 시나리오 · 단순화된 지형과 정적 사람 메시",
            22,
            MUTED,
        )
        label(
            d,
            (48, 784),
            "연속 화면 녹화 아님 · 에너지/미션 지연은 설정값 · 실측 추론시간은 별도 로그",
            22,
            MUTED,
        )
        return im

    def route(self, d, index):
        panel(d, (816, 164, 1256, 348))
        label(d, (834, 176), "고정 경로 / 실제 관측 위치", 20)

        def xy(p):
            return (1036 + p[0] * 1.8, 265 - p[1] * 1.25)

        world = self.scenario["world"]
        route = [world["initial_position_m"]] + [w["position_m"] for w in world["waypoints"]]
        d.line([xy(p) for p in route], fill="#546679", width=4)
        trail = [xy(e["capture_body_pose"]["position_m"]) for e in self.events[: index + 1]]
        if len(trail) > 1:
            d.line(trail, fill=CYAN, width=3)
        x, y = trail[-1]
        d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=CYAN)
        z = self.events[index]["capture_body_pose"]["position_m"][2]
        label(d, (834, 312), f"관측 고도 {z:.1f} m   |   경로 개입 없음", 18, MUTED)

    def policy(self, d, t, events, result, y, adaptive=False):
        panel(d, (816, y, 1256, y + 186))
        started = [e for e in events if e["capture_time_s"] <= t]
        done = completed(events, t)
        current = started[-1]
        busy = current["completion_time_s"] > t
        finished = t >= result["final_time_s"]
        heading = "CONTEXT / 기준 기록 재생" if adaptive else "CABiNet 고정 / 실제 실행"
        label(d, (834, y + 10), heading, 21, ORANGE if adaptive else CYAN)
        status = "처리 중" if busy else "다음 관측 대기"
        if finished:
            status = "성공" if result["mission_success"] else "실패"
        label(d, (834, y + 46), f"{model(current)}  ·  {status}", 20)
        battery = (
            current["runtime_at_capture"]["battery_frac"] if busy else current["battery_after"]
        )
        if finished:
            battery = result["final_battery_frac"]
        changes = sum(model(a) != model(b) for a, b in pairwise(started))
        label(
            d,
            (834, y + 79),
            f"완료 {len(done):02d}회   전환 {changes}회   배터리 {battery:.1%}",
            19,
        )
        correct, false = reports_at(result, t)
        label(d, (834, y + 111), f"평가: 유효 보고 {correct}/3명 · 오탐 보고 {false}건", 19)
        if adaptive:
            contrast = current["router_decision"]["rgb_contrast"]
            label(d, (834, y + 145), f"입력: intent + 영상(대비 {contrast:.3f}) + 자원", 17, MUTED)
        else:
            label(d, (834, y + 145), "청록: 모델 예측 영역 / 박스는 가시성 보조", 17, MUTED)

    def frame(self, t):
        index = max(i for i, e in enumerate(self.events) if e["capture_time_s"] <= t)
        event = self.events[index]
        ready = event["completion_time_s"] <= t
        im, d = self.title("실행 기록 재생  |  5초 간격 관측(0.2 Hz)  |  6배속  |  CLEAR 환경")
        label(d, (995, 23), f"SIM {t:06.1f} s", 27, CYAN)
        panel(d, (24, 104, 1256, 148))
        label(
            d,
            (40, 111),
            "INTENT  사람 수색 · 400초 / 첫 보고 300초 · 보고 재현율 ≥50% · 오탐 ≤2 · 배터리 ≥20%",
            20,
        )
        im.paste(self.frames[index][1 if ready else 0], (24, 164))
        d.rectangle((24, 164, 792, 201), fill=BG)
        label(
            d,
            (36, 167),
            f"GAZEBO RGB  |  관측 #{event['observation_id']:02d}"
            f"  |  촬영 t={event['capture_time_s']:.0f}s",
            20,
        )
        d.rectangle((24, 700, 792, 740), fill=BG)
        if ready:
            s = (
                f"결과 t={event['completion_time_s']:.2f}s"
                f"  ·  예측 영역 {event['score']['predicted_components']}개"
            )
        else:
            s = "추론 처리 중: 이 관측의 예측 결과는 아직 표시하지 않음"
        label(d, (36, 705), s, 19, CYAN if ready else ORANGE)
        self.route(d, index)
        self.policy(d, t, self.events, self.result, 362)
        self.policy(d, t, self.adaptive, self.adaptive_result, 562, True)
        label(d, (24, 747), "카메라: 실제 실행 / context: 동일 기준 비행 기록으로 평가", 19, MUTED)
        label(
            d,
            (24, 775),
            "배터리는 마지막 로그값 · 지연/에너지는 설정값 · 보고 정오 판정은 평가자 전용",
            17,
            MUTED,
        )
        d.rounded_rectangle((24, 824, 1256, 834), radius=5, fill=PANEL)
        d.rounded_rectangle(
            (24, 824, 24 + 1232 * min(t / self.duration, 1), 834), radius=5, fill=CYAN
        )
        for report in self.result["report_quality"]["reports"]:
            if report["completion_time_s"] <= t:
                x = 24 + 1232 * report["completion_time_s"] / self.duration
                color = CYAN if report["target_id"] else RED
                d.ellipse((x - 5, 822, x + 5, 836), fill=color)
        label(d, (24, 847), "0 s", 17, MUTED)
        label(d, (330, 847), "● 유효 보고", 17, CYAN)
        label(d, (450, 847), "● 오탐 보고", 17, RED)
        label(d, (570, 847), "(시간축의 평가 이벤트)", 17, MUTED)
        label(d, (1140, 847), f"{self.duration:.2f} s", 17, MUTED)
        return im

    def outro(self):
        im, d = self.title(
            "미션 결과  |  단순 탐지 정확도뿐 아니라 보고·시간·자원 조건을 함께 평가"
        )
        for x, name, result, color in [
            (24, "CABiNet 고정 / 실제 실행", self.result, CYAN),
            (652, "Context / 기준 기록 재생", self.adaptive_result, ORANGE),
        ]:
            panel(d, (x, 135, x + 604, 540))
            label(d, (x + 26, 157), name, 28, color)
            label(
                d,
                (x + 26, 214),
                "MISSION SUCCESS" if result["mission_success"] else "MISSION FAILED",
                33,
                color,
            )
            quality = result["report_quality"]
            lines = [
                f"보고 재현율  {quality['report_recall']:.1%}",
                f"오탐 보고  {quality['false_reports']}건",
                f"최종 배터리  {result['final_battery_frac']:.1%}",
                f"추론 {result['processed_observations']}회"
                f" / 모델 전환 {result['config_switches']}회",
            ]
            for n, line in enumerate(lines):
                label(d, (x + 26, 288 + n * 51), line, 27)
        label(
            d,
            (48, 580),
            "이 사례에서는 고정 CABiNet이 성공했고, context 라우터는 실패했습니다.",
            28,
        )
        label(
            d,
            (48, 631),
            "라우터가 모델을 바꾼다는 사실만으로 미션 성능이 좋아지지는 않습니다.",
            26,
            CYAN,
        )
        label(
            d,
            (48, 705),
            "정식 정책 비교는 동일 기준 관측을 사용. 실제 실행은 기준 경로와 0.013 mm 이내 일치.",
            21,
            MUTED,
        )
        label(
            d,
            (48, 747),
            "실제 비행 요약 기반 시나리오이며 실제 현장 영상/GPS 궤적 재생은 아닙니다.",
            21,
            MUTED,
        )
        label(
            d,
            (48, 789),
            "이 영상은 CLEAR / user intent 한 사례입니다. 전체 실험으로 일반화하지 않습니다.",
            21,
            MUTED,
        )
        return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", type=Path, default=ROOT / "results/routing_mission_live_cabinet_20261016"
    )
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/routing_mission_field_20261016"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/routing_mission_video_20261016"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    playback = Playback(args.live, args.study)
    path = args.output / "benchmark_playback.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("H.264 encoder unavailable")
    intro, outro = playback.intro(), playback.outro()
    timeline = []
    frames = 0

    def emit(im):
        nonlocal frames
        writer.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        frames += 1

    try:
        for _ in range(INTRO * FPS):
            emit(intro)
        for i in range(math.ceil(playback.duration / SPEED * FPS) + 1):
            t = min(i * SPEED / FPS, playback.duration)
            emit(playback.frame(t))
            timeline.append({"video_frame": frames - 1, "mission_time_s": t})
        for _ in range(OUTRO * FPS):
            emit(outro)
    finally:
        writer.release()
    for name, im in [("intro", intro), ("preview", playback.frame(225.5)), ("result", outro)]:
        im.save(args.output / f"{name}.jpg", quality=92)
    # Decode every frame, not just the container header.
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (SIZE[1], SIZE[0], 3)
        decoded += 1
    cap.release()
    assert decoded == frames, (decoded, frames)
    sources = [
        args.live / "live_validation.json",
        args.live / "protocol.json",
        args.study / "lock.json",
    ]
    for folder in [playback.mission, playback.context]:
        sources += [
            folder / name
            for name in ["events.json", "results.json", "scenario.json", "contract.json"]
        ]
    sources += [entry[2] for entry in playback.frames]
    provenance = {
        "kind": "sampled_execution_log_playback_not_screen_recording",
        "source_files": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "script_sha256": digest(Path(__file__)),
        "video_sha256": digest(path),
        "codec": "H264/avc1",
        "fps": FPS,
        "size": SIZE,
        "frames": frames,
        "decoded_frames": decoded,
        "video_duration_s": frames / FPS,
        "playback_speed": SPEED,
        "observation_interval_s": 5,
        "frame_interpolation": False,
        "prediction_overlay": "only after logged completion for that capture; no GT overlay",
        "battery": "last logged state, configured cost accounting",
        "context": "paired canonical replay with precomputed real neural predictions",
        "timeline": timeline,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            {"path": str(path), "frames": frames, "duration_s": frames / FPS, "decoded": decoded}
        )
    )


if __name__ == "__main__":
    main()
