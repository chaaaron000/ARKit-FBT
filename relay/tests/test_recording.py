import numpy as np

from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.pipeline import Processor, Settings
from arkit_fbt_relay.protocol import BodyFrame, Pose, build_frame
from arkit_fbt_relay.recording import Recorder, analyze, read_records

CAM = Pose(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))


def test_record_roundtrip_and_analyze(tmp_path):
    path = tmp_path / "a.afbtrec"
    rec = Recorder(path)
    rng = np.random.default_rng(0)
    packets = []
    for i in range(120):
        joints = synth_joints(i / 60, noise=0.005, rng=rng) if i % 10 else None
        data = build_frame(i + 1, i / 60, CAM, joints)
        packets.append(data)
        rec.write(100.0 + i / 60 + 0.001 * (i % 3), data)
    rec.close()

    got = list(read_records(path))
    assert [d for _, d in got] == packets

    report = analyze(path)
    assert "패킷 120개" in report
    assert "hips_joint" in report
    assert "좌우 규약 학습값: +" in report  # 가짜 데이터는 left = 사람 왼쪽


def test_prediction_leads_motion():
    def run(pred):
        # 기본값과 무관하게, 지연이 뚜렷한 필터 설정으로 고정한다
        p = Processor(Settings(min_cutoff=1.5, beta=0.7, prediction_ms=pred))
        for i in range(60):
            t = i / 60
            j = synth_joints(0.0)
            j = {n: Pose(v.pos + np.array([1.0 * t, 0, 0]), v.rot) for n, v in j.items()}  # 1 m/s
            out = p.process(BodyFrame(i, t, True, 1.0, CAM, None, j))
        return next(tr.pos[0] for tr in out.trackers if tr.name == "hip")

    truth = synth_joints(0.0)["hips_joint"].pos[0] + 59 / 60
    lag_none = truth - run(0.0)
    lag_pred = truth - run(60.0)
    assert lag_none > 0.01  # 필터 때문에 뒤처진다
    assert abs(lag_pred) < lag_none  # 예측이 지연을 줄인다
