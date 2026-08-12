# Kronos 개념 표현·조정 실험 — 작업 인수인계

새 세션은 이 파일을 먼저 읽는다. **무엇을 하는 프로젝트인지, 지금 어디까지 왔는지, 어떻게 실행하는지, 이미 밟은 지뢰가 무엇인지**를 담는다.

---

## 1. 무엇을 하는가

TSFM인 **Kronos**에 **Moment Steering 논문**의 표현 분석·개입 방법론을 적용해 두 가지를 검증한다.

1. "선형 증가 모멘텀" 개념이 Kronos latent space에 **선형 표현**으로 존재하는가 (linear probing + LDR)
2. 존재한다면 `h_i ← h_i + λS_i` 개입만으로 출력을 모멘텀 방향으로 **유도**할 수 있는가 (concept steering)

### 읽어야 할 문서 (우선순위 순)

| 파일 | 내용 |
|---|---|
| `experiment/0812_1차실험_중간보고_stage0-4.md` | **연구 결과 보고서.** Stage 0~4의 모든 측정 수치와 4개 결론. 결과를 알고 싶으면 이것부터 |
| `experiment/0812_1차실험_구현단계.md` | 단계별 구현 계획 + 결정 기록 |
| `experiment/0812 _1차실험_구현명세.md` | 사용자가 준 원 구현 명세 (7절 구성) |
| `experiment/0812 _1차실험_전체설계.md` | 사용자가 준 원 전체 설계 (Step 1~5) |
| `paper/Kronos_MOMENT_Steering_2409.12915v5.pdf` | 방법론 원논문 |
| `paper/Kronos_ A Foundation Model...pdf` | 대상 모델 논문 |

참조 구현체는 `Kronos/`(모델)와 `representations-in-tsfms/`(steertool)에 있다. 둘 다 읽기 전용으로 참조만 한다.

---

## 2. 지금 어디까지 왔는가

| Stage | OU (주 데이터셋) | RW (robustness) |
|---|---|---|
| 0 환경·스펙 실측 | 완료 — Kronos-base 채택 | — |
| 1 합성 데이터셋 | 완료 | 완료 |
| 2 활성화 추출 | 완료 | 완료 (**scratch, 세션 종료 시 소멸**) |
| 3 Fisher probe / LDR | 완료 | 완료 |
| 4 Steering vector | 완료 | 완료 |
| 5 개입 추론 | **완료** — 9팔 × 8λ = 72조합 + REF | **REF만 완료.** 개입 팔 미실행 |
| 6 평가·시각화 | **완료** | 미실행 |

**다음 할 일은 RW 개입 팔이다** (6절 참조).

### 재현된 결론 (OU·RW 양쪽)

이 넷은 두 데이터셋에서 모두 확인됐다. 가장 신뢰할 수 있는 결과다.

1. **개념은 선형 표현으로 존재한다.** held-out LDR vs null(레이블 셔플) 비율 — OU **987배**, RW **632배**
2. **PCA로는 이 개념을 볼 수 없다.** 상위 2 PC가 분산의 48~80%를 담지만 LDA 방향은 그 안에 **0.01~0.02%**
3. **mean ≈ median** (코사인 0.978 / 0.986). 논문의 절제 실험 1이 예측한 대로.
4. **LDA 방향 ⊥ 평균차 방향** (코사인 0.040 / 0.022). LDA 는 논문 방법의 변종이 아니라 다른 개입이다.
5. **layer 0 은 OOD 에 취약하다.** BSQ 이산 임베딩 격자라 연속 섭동이 갈 곳이 없다.

### 데이터셋에 따라 갈리는 결론 — 주의할 것

**명세 6절의 causal frontier 가설은 데이터 형태에 의존한다.**

| | OU (매끈한 램프) | RW (랜덤워크+drift) |
|---|---|---|
| LDR 정점 | t ≈ 259~350 | **t = 511** |
| `causal_frontier_holds` | False | **True** |

RW 에서는 성립한다. OU 의 trend 는 거의 결정론적 램프라 어느 구간만 봐도 탐지되고, 시퀀스 전체 z-score 가 램프를 중앙정렬시켜 정보가 양 끝에 대칭으로 실린다. **OU 단독으로 가설을 반증했다고 결론지으면 안 된다.**

### Stage 5/6 결과 (OU)

- **논문 방법이 작동한다.** A/B/C 모두 λ_rel=0.25 에서 **32/32 양의 기울기** (p=4.7e-10), coherence 위반 0%.
- **작동 방식은 "덧셈"이 아니라 "덮어쓰기"다.** 기준선 std 0.023 → 개입 후 0.001 (26배 붕괴). 원래 상승하던 표본까지 끌어내려 전부 작은 양수로 수렴시킨다.
- **single-token 이 all-tokens 와 동등하다 — 논문과 반대.** 논문은 인코더에서 single-token 실패를 보고했으나, decoder-only 는 마지막 토큰이 곧 출력 경로라 동일하게 작동한다.
- **얕은 층 개입은 역방향으로 조정한다.** F(layer {1,2})는 λ=0.35 에서 **0/32** (p=4.7e-10). 그런데 layer 1 은 전역 LDR 최대(29.27) 지점이다.
- **LDA 방향은 무력하다.** G(읽기량 정합) 분산비 0.951 로 모델을 거의 건드리지 못한다. H(동일 노름, 읽기량 26배)도 모멘텀 방향으로는 못 민다.
- **`I_ctrl_median_vector`(명세 7절의 vector 축약형)가 가장 빠르다.** λ=0.05 에서 이미 65.6%. 대조군으로 넣었는데 최고 성능이었다.
- **행동이 표현 도달보다 먼저 포화한다.** λ=0.25 는 표현을 두 클래스 사이 골짜기(간격의 54%)에 놓는데 행동은 거기서 이미 100%. λ=0.5 에서 표현이 trend 평균에 도착해도 행동은 나아지지 않고 OOD 만 악화된다.

### 인과 해석에 대한 유보 — 가장 중요한 미해결 쟁점

목표 분포 대조군(진짜 trend 입력을 개입 없이 통과)을 돌린 결과:

| | slope | close_drift | 양수 |
|---|---|---|---|
| OU REF (trend 입력) | −0.0656 | −3.11 | **0/32** |
| RW REF (trend 입력) | −0.0113 | −0.71 | 53.1% |
| 기준선 (base 입력) | −0.0011 | +0.05 | 46.9% |
| 조정된 base (OU, λ=0.25) | +0.0038 | +0.86 | **100%** |

- OU 의 극적 반전(0/32)은 **매끈한 램프가 OOD 였던 인공물**이다. RW 에서는 사라진다.
- 그러나 RW 에서도 모델은 진짜 추세 입력에 **중립**(53.1%)이다. **모델은 모멘텀을 외삽하지 않는다.**
- KS 검정: 조정된 출력이 목표 분포와 닮은 (팔, λ) 조합이 **하나도 없다** (모두 p ≈ 1e-18).

**따라서 "steering 이 작동하면 → 모델이 그 개념을 기저 기작으로 쓴다"는 추론이 성립하지 않는다.** 개입은 모델이 자연적으로 하지 않는 행동을 주입한다. 분산 26배 붕괴 + 골짜기 착지와 합치면, **저밀도 영역에서의 정형화된 출력**이라는 해석과 일관된다.

단, RW 개입 팔이 아직 없어 이 결론은 OU 기준이다. RW 에서 steering 이 재현되는지가 확정의 관건이다.

---

## 3. 코드 구조

```
colab_control.py                 Colab 셀에 붙여넣는 실행 제어 (v3-multistage)
experiment/code/
  kexp/                          실험 패키지
    config.py                    모든 하이퍼파라미터 단일 소스 + 해시
    paths.py                     Colab/로컬 경로 해석, Kronos sys.path 부트스트랩
    kronos_loader.py             모델 로드, 스펙 실측, 전처리 재현
    hooks.py                     ActivationRecorder(추출) / steering(개입)
    synth.py                     Stage 1 micro-path → OHLC 집계
    activations.py               Stage 2 레이어별 memmap 입출력
    ldr.py                       Stage 3 closed-form Fisher/LDA, 평활, 구간 통계
    steering_vec.py              Stage 4 S_i 구성, λ 캘리브레이션, OOD 진단
    intervene.py                 Stage 5 Steerer 훅 (위치 정합 포함)
  stages/stage{0..6}_*.py        독립 실행 엔트리포인트
```

**패키지 이름이 `kexp`인 이유**: `experiment/code`를 `sys.path`에 올리므로 하위 패키지가 `model`이면 Kronos의 `model` 패키지와 충돌한다. `code`도 표준 라이브러리 모듈명이라 최상위 노출 금지.

### 주요 CLI 인자

```
stage0  --compare                        mini/small/base 스펙 비교만
stage1  --force --noise {ou,rw,both}
stage2  --model --noise --batch-size --force --smoke --smoke-n
stage3  --model --noise --chunk --smoke --no-class-stats
stage4  --model --noise --smoke --ood-n --ood-threshold --pca-n
stage5  --model --noise --n-eval --seed --smoke --smoke-n
        --arms {all,paper,ours} --force --pred-len --sample-count
        --reference-trend        trend 클래스를 개입 없이 통과 (목표 분포)
stage6  --model --noise
```

`--reference-trend` 는 `--arms` 를 덮어쓴다. 목표 분포는 REF 팔 하나만 돌린다.

### 산출물 파일

```
Drive results/<tag>/<noise>_<model>/
  ldr.npz                  Stage 3: LDR, w_last, h_norm
  class_stats_layer{i}.npz Stage 3: 위치별 median/mean (Stage 4 입력)
  stage3_summary.json      Stage 3: 구간별 LDR, 정점 위치, LDA vs PCA
  steering.npz             Stage 4: S 벡터/행렬, OOD, lambda 표
  stage4_summary.json      Stage 4: 노름, 코사인, OOD
  pca_subset.npz           Stage 4: 활성화 부분집합 (Stage 6 기하 검증용, 36MB)
  steer/results.json       Stage 5: 조합별 지표 + slopes (증분 저장)
  stage6_summary.json/.csv Stage 6: 팔별 판정
```

### 저장 위치 규칙

| 위치 | 내용 | 세션 종료 시 |
|---|---|---|
| **Drive** `/content/drive/MyDrive/Kronos_probing/` | 데이터셋, results, figs | 보존 |
| **scratch** `/content/kexp_scratch/` | 활성화 39GB | **소멸** (재생성 = Stage 2, 약 9분) |

Stage 3·4만 활성화를 읽는다. **Stage 5·6은 Drive만으로 동작한다.**

---

## 4. 어떻게 실행하는가

### 역할 분담 (사용자와의 합의)

- **나**: 코드 작성 + 로컬 검증 + 로컬 커밋. **푸시는 하지 않는다** — 명령어만 제시한다.
- **사용자**: `git push`, Colab에서 `colab_control.py` 붙여넣기 실행, 출력 전문을 붙여넣기.
- **그림 확인**: 사용자가 `experiment/confirm_data/`에 내려받아 올려주면 내가 Read로 본다. (이 디렉토리는 gitignore 대상)

### Colab

`colab_control.py` 전체를 셀에 붙여넣고 상단 값만 수정한다.

```python
STAGE = [1, 2, 3]              # int, list, "1,2,3" 모두 가능
EXTRA_ARGS = {1: "--force"}    # 문자열이면 전 stage, dict면 stage별
```

첫 줄에 `[colab_control v3-multistage]` 배너가 찍힌다. **안 찍히면 사용자가 옛 셀을 붙여넣은 것이다.**

푸시하지 않은 stage 스크립트는 Colab에 존재하지 않는다(`git reset --hard`로 repo를 되돌리므로). 반드시 푸시 먼저.

### 로컬 검증 (중요)

**Colab에 넘기기 전에 반드시 로컬에서 돌려본다.** 전용 venv가 있다.

```bash
~/.venvs/kronos/bin/python ...
```

torch(CPU/MPS), numpy, pandas, scipy, matplotlib, scikit-learn, pypdf 설치됨. Kronos-mini/small 가중치도 받아져 있고 HuggingFace 네트워크 접근이 된다.

검증 순서: **로컬(Kronos-small, 축소 표본) → Colab 스모크(base, `--smoke`) → Colab 본 실행**.

로컬 프록시 모델 선택:
- **Kronos-small 우선** — base와 토크나이저·max_context가 같아 AR 롤링 경로가 동일. L=8/d=512만 다름.
- mini는 `max_context=2048`이라 AR이 다른 분기(append)를 탄다. 쓰려면 512를 강제할 것.

로컬 산출물은 `experiment/_out`, `experiment/_scratch`(둘 다 gitignore). 테스트 후 지운다.

**주의**: Bash 도구가 2분에 타임아웃된다. 긴 실행은 `run_in_background: true`를 쓰거나 `until grep -q DONE ...; do sleep 10; done` 패턴으로 대기한다.

---

## 5. 이미 밟은 지뢰 (재발 방지)

실제로 발생했고 시간을 잃은 것들이다.

| 사고 | 원인 | 대응 (코드에 반영됨) |
|---|---|---|
| 본 실행이 통째로 건너뛰어짐 | 스모크와 본 실행이 같은 출력 디렉토리 사용 | 스모크 경로에 `_smoke` 접미사 |
| 활성화 39GB 불필요 재계산(9분) | 재개 판정에 **전체** config 해시 사용 → Stage 5의 λ 격자만 바꿔도 무효화 | `activation_hash()`로 모델·데이터·dtype만 해싱 |
| 낡은 데이터로 만든 활성화 재사용 위험 | 생성 로직이 바뀌어도 config 값은 그대로 | Stage 1이 `data_fingerprint`를 남기고 Stage 2가 대조 |
| Stage 5 결과에 스모크 값 혼입 | 스모크가 `results.json`에 썼고 이후 파일 분리 | 항목마다 `run_settings` 기록 + 이어받을 때 불일치 폐기 |
| 첫 토큰이 최고 분리도로 나옴 (있을 수 없는 결과) | OU를 영 초기조건으로 시작해 모든 표본이 같은 지점에서 출발 | burn-in `10/θ` 스텝 폐기 |
| coherence 위반 96.88% (개입 없는 기준선에서) | 정규화 공간에서 검사. Kronos는 O/H/L/C를 **채널별 독립** z-score | 역정규화 후 검사 |
| 위치별 argmax가 노이즈 스파이크를 고름 | 512개 위치에서 raw argmax | 25점 이동평균 + 구간(early/mid/late/last) 평균 |
| steering 훅 테스트가 "효과 없음"으로 나옴 | recorder 훅을 Steerer보다 **먼저** 등록해 수정 전 출력을 캡처 | 테스트 시 훅 등록 순서 주의 |
| Stage 6 이 "모든 팔 효과 없음"으로 오판 | 기준선과의 **대응표본 평균 비교**를 주 검정으로 썼다. 이 개입은 평균을 옮기는 게 아니라 분포를 붕괴시키므로 쌍별 차이가 17/32 로 갈린다 | 주 검정을 **결과 부호의 이항검정**으로 교체 (32/32, p=5e-10) |
| 목표 분포 없이 인과 결론을 낼 뻔함 | "steering 이 작동한다"만으로는 개념 설치인지 저밀도 영역의 정형화인지 구분 불가 | `--reference-trend` 대조군 추가 |

### 수치·해석상 주의

- **기울기는 정규화 공간, coherence는 역정규화 공간**에서 잰다.
- LDR의 분산 분모는 **ddof=0** (steertool/논문 구현체와 일치시킴).
- λ는 절대값이 아니라 **상대 강도** `λ_rel = λ_i‖S_i‖/‖h_i‖`. base는 `‖h‖`가 레이어별 7.7배 변한다.
- `pred_len`이 다르면 기울기 스케일이 달라 **비교 불가**. 한 실행 안에서 섞지 말 것.
- **small 결과로 base 실험을 바꾸지 않는다.** 사용자가 명시적으로 요구한 원칙. 로컬(small)은 코드 버그 검출용이고, 과학적 결정은 base 실측으로만 한다.

---

## 6. 다음 할 일

### 1순위 — RW 개입 팔 (미실행)

핵심 미해결 질문: **OU 에서 관측된 "steering 100% 상승"이 RW 에서도 재현되는가?**

- 재현되면 → 데이터 형태와 무관한 모델 성질. "모델이 자연적으로 안 하는 행동을 주입한다"는 결론이 확정된다.
- 재현 안 되면 → OU 의 steering 효과가 그 인공물에 묶여 있었다는 뜻. 결론을 대폭 수정해야 한다.

논문 재현 팔부터 (A/B/C = 22조합, L4 에서 약 35분):

```python
STAGE = [5, 6]
EXTRA_ARGS = {5: "--noise rw --arms paper", 6: "--noise rw"}
```

REF 는 이미 RW 결과 파일에 있으므로 Stage 6 이 KS 비교까지 바로 낸다.
활성화 39GB 는 필요 없다 (Stage 5·6 은 Drive 만 읽는다).

**참고**: RW 는 OU 보다 안전 λ 범위가 좁다. 레이어별 OOD 임계 이하 최대 λ_rel 이
`[0.05, 0.1, 0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15]` 이고
`‖S‖/‖h‖` 도 0.66~0.78 로 OU(0.44~0.60)보다 크다. λ=0.15~0.25 구간이 관건일 가능성이 높다.

### 2순위

- RW 개선안 팔 (`--arms ours`) — 1순위 결과를 보고 판단
- 중간보고서(`experiment/0812_1차실험_중간보고_stage0-4.md`)를 Stage 5/6 및 RW 결과까지 반영해 확장. 현재 제목은 stage0-4 이고 §6 이 "실행 중"으로 남아 있다.
- OU 의 trend 클래스가 OOD 였다는 점을 고려하면, snr 을 낮춘 더 현실적인 OU 변형도 후보다.

---

## 7. 사용자 작업 방식

- 한국어로 소통한다.
- **결정이 필요하면 한 번에 답하지 말고, 필요한 정보를 먼저 요구하고 각 정보를 본 뒤 논리적으로 판단**하기를 원한다.
- 원논문의 방법을 **유지**하면서 데이터에서 발견한 개선안을 **추가**하는 방식을 선호한다. 논문 방법을 대체하지 않는다.
- 근거 없는 추론을 경계한다. 실제로 "small 결과로 base를 바꾸는 건 위험하지 않냐", "첫 토큰이 최고 분리도라는 건 말이 안 된다" 같은 지적으로 두 번 방향을 바로잡았다 — 둘 다 옳았다.
- 그림·수치는 직접 확인시켜 주기를 원한다.
