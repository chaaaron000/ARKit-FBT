# ARKit FBT

iPhone(ARKit 바디 트래킹)으로 VRChat 풀트래킹을 하는 프로젝트입니다. OSC 트래커를 씁니다.

```
[iPhone, 삼각대에 고정]  ── UDP 39570 ──▶  [PC 중계 프로그램]  ── OSC 9000 ──▶  [VRChat]
  ARKit 관절 21개                          보정 · 필터 · 좌표 변환
```

- **iPhone 앱** (`ios/`): ARKit으로 관절을 추적해서 PC로 보냅니다. A12 칩 이상(iPhone XS/XR 이후)이 필요합니다.
- **PC 중계 프로그램** (`relay/`, Python): 받은 관절 데이터를 VRChat OSC 트래커(골반, 발, 무릎 등)로 바꿔 보냅니다.

## 동작 원리

| 문제 | 해결 |
|---|---|
| ARKit 월드 좌표가 미끄러짐(드리프트) | 폰은 고정되어 있다고 보고, 카메라 자세를 기준으로 되돌립니다. |
| 뼈대가 카메라 쪽으로 통째로 기울어짐 (실측 5~12°) | 몸 기울기를 약 8초 평균으로 계속 추적해서 바로 세웁니다. |
| 헤드셋을 쓰면 ARKit 머리가 앞으로 쏠림 | VRChat에 보내는 머리 위치는 수평은 가슴 기준, 높이는 머리 기준으로 만듭니다. |
| 카메라 깊이(앞뒤) 방향 잡음이 큼 | 골반 기준 상대 위치를 필터링해서 몸 전체 출렁임을 없애고, 깊이 방향만 더 강하게 거릅니다. |
| 헤드셋·컨트롤러 때문에 관절 추적이 자주 끊김 (실측 21% 프레임) | 끊긴 관절은 마지막 정상 위치를 유지합니다. 추정값은 크게 흔들려서 쓰지 않습니다. |
| ARKit 발 각도가 엉터리 (발끝이 아래를 향함) | 발·무릎 회전은 몸 방향(yaw)만 따르게 합니다. |
| 폰 공간과 VR 공간의 위치 차이 | 머리 위치를 `/tracking/trackers/head/position`으로 보내면 VRChat이 **매 프레임** 아바타 머리에 맞춰 줍니다. |
| 방향(yaw) 차이 | 시작할 때 카메라를 정면으로 보고 서 있으면 자동으로 1회 정렬합니다. 이후 **두 손을 머리 위로 1.5초** 들면 다시 정렬합니다. |
| 떨림 | One Euro 필터. 창에서 실시간으로 조절할 수 있습니다. |
| 순간적으로 튀는 값 | 3프레임 중앙값과 관절 최대 속도(3m/s) 제한을 씁니다. |

## 1. PC 중계 프로그램 실행

[uv](https://docs.astral.sh/uv/)가 필요합니다. **`run_relay.bat`을 더블클릭**하거나, 다음처럼 실행합니다.

```powershell
cd relay
uv run arkit-fbt-relay
```

- 처음 실행할 때 **Windows 방화벽 창이 뜨면 "개인 네트워크" 허용**을 눌러야 iPhone 데이터를 받을 수 있습니다.
- 창 왼쪽 위 "이 PC 주소"를 iPhone 앱에 입력합니다.
- 옵션: `--headless`(콘솔 모드), `--dry-run`(VRChat으로 보내지 않음), `--record`(시작하자마자 녹화), `--help`

### 튜닝
| 슬라이더 | 효과 |
|---|---|
| 위치/회전 안정도 | 낮을수록 가만히 있을 때 떨림이 줄지만 반응이 느려집니다. |
| 위치/회전 반응성 | 높을수록 빠른 동작에서 지연이 줄지만 떨림이 늘어납니다. |
| 예측 (ms) | 움직이는 방향으로 미리 내다봐서 지연을 줄입니다. 30~60ms부터 시험해 보세요. 너무 크면 동작이 튀어나갑니다. |
| 튐 제거 속도 | 이보다 빠른 순간 이동은 오인식으로 보고 무시합니다. |

### 녹화와 분석
창의 **● 녹화 시작** 버튼을 누르면 iPhone에서 오는 원본 데이터가 `relay/recordings/`에 저장됩니다. 녹화 파일로 실제 떨림, 지연 변동, 관절 인식률을 확인하고, 같은 동작을 반복 재생하면서 튜닝할 수 있습니다.

```powershell
cd relay
uv run python -m arkit_fbt_relay.recording analyze recordings\<파일>.afbtrec   # 분석 리포트
uv run python -m arkit_fbt_relay.recording replay  recordings\<파일>.afbtrec   # 중계 프로그램으로 다시 재생
```

## 2. iPhone 앱 설치

Mac이 없어도 됩니다. GitHub Actions가 빌드합니다.

1. [Releases의 `ios-latest`](../../releases/tag/ios-latest)에서 `ARKitFBT.ipa`를 받습니다.
2. iPhone을 USB로 연결하고 **Sideloadly**로 설치합니다(무료 Apple ID 사용).
3. iPhone 설정:
   - 설정 → 개인정보 보호 및 보안 → **개발자 모드** 켜기 → 재부팅
   - 설정 → 일반 → VPN 및 기기 관리 → 개발자 프로필 **신뢰**
4. 앱 실행 → **카메라**와 **로컬 네트워크** 권한 허용 → PC 주소 입력

무료 Apple ID로 설치한 앱은 **7일마다 Sideloadly로 다시 설치**해야 합니다.

## 3. VRChat에서 사용

1. VRChat을 VR 모드(SteamVR)로 실행하고 OSC를 켭니다 (액션 메뉴 → Options → OSC → Enabled).
2. iPhone을 허리 높이에 고정하고, 2.5~3m 떨어져서 전신이 화면에 들어오게 섭니다.
3. 카메라를 정면으로 보고 1~2초 서 있으면 자동으로 방향이 정렬됩니다.
4. VRChat에서 풀트래킹 캘리브레이션을 합니다.
   - 디버그할 때는 트래커 표시 모델을 "Tracker: System"으로 두면 캘리브레이션 후에도 트래커가 보입니다.
5. 방향이 틀어지면 카메라를 보고 **두 손을 머리 위로 1.5초** 들어 다시 정렬합니다.

처음에는 **골반 + 양발(3개)**만 쓰는 걸 추천합니다. VRChat 문서도 정확도가 완벽하지 않으면 트래커가 적을수록 IK가 더 잘 보정한다고 안내합니다.

## 개발

```powershell
cd relay
uv run pytest                                          # 테스트
uv run python -m arkit_fbt_relay.fake_phone --seconds 30  # 가짜 iPhone (중계 프로그램 실행 중일 때)
```

- 패킷 형식: `ios/ARKitFBT/WireProtocol.swift` 주석 ↔ `relay/src/arkit_fbt_relay/protocol.py` (둘을 함께 고쳐야 합니다)
- iOS 프로젝트는 XcodeGen(`ios/project.yml`)으로 생성합니다. `ios/` 아래를 고쳐서 main에 push하면 `.github/workflows/ios.yml`이 IPA를 빌드해 `ios-latest` 릴리스에 올립니다.
