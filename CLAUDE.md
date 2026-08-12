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

| Stage | 상태 | 산출물 |
|---|---|---|
| 0 환경·스펙 실측 | 완료 | 모델 3종 스펙 확정, Kronos-base 채택 |
| 1 합성 데이터셋 | 완료 | OU/RW 각 2048×2 표본, Drive |
| 2 활성화 추출 | 완료 | `[12, 2048, 512, 832]` fp16 ×2 클래스, **scratch(39GB, 세션 종료 시 소멸)** |
| 3 Fisher probe / LDR | 완료 | LDR 히트맵·프로파일·PCA·LDA 뷰, `ldr.npz`, `class_stats_layer*.npz` |
| 4 Steering vector | 완료 | `steering.npz`, OOD 진단, `pca_subset.npz`(36MB, Stage 6용) |
| **5 개입 추론** | **실행 중** | 72조합 중 일부. L4 GPU에서 재개, `steer/results.json`에 증분 저장 |
| 6 평가·시각화 | **미작성** | slope vs λ 곡선, 붕괴 지표, 개입 전/후 PCA |

### Stage 4까지의 핵심 결론

1. **개념은 선형 표현으로 존재한다.** held-out LDR 23.76 vs null(레이블 셔플) 0.024 → **987배**
2. **causal accumulation 정상.** 첫 토큰이 전 레이어에서 최저(5.6~6.4)
3. **명세의 causal frontier 가설은 반증됨.** LDR 정점이 t≈259~350이고 마지막 토큰이 전역 최대가 아니다. 단 깊이에 따른 국소화는 관측됨(layer 4→11에서 마지막 토큰 LDR 17.83→23.52 단조 증가)
4. **PCA로는 이 개념을 볼 수 없다.** 상위 2 PC가 분산의 48~80%를 담지만 LDA 방향은 그 안에 **0.01%**

### Stage 5 예비 결과 (A/B 팔 일부)

양의 기울기 비율이 λ_rel 0→1에서 **59.4% → 100%** 로 단조 증가. λ≥0.35에서 포화. coherence 위반 0%(붕괴 없음). `slope_std`가 22배 감소 — 개입이 예측을 거의 결정론적으로 만든다(예상 밖 현상, Stage 6에서 다룰 것).

별도 스모크에서 **G(LDA 방향, 개념축 이동량 정합)가 출력을 전혀 바꾸지 못했다.** 유지되면 "탐침이 개념을 **읽는** 방향 ≠ 모델이 **쓰는** 방향"이라는 결과가 된다.

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
  stages/stage{0..5}_*.py        독립 실행 엔트리포인트
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

### 수치·해석상 주의

- **기울기는 정규화 공간, coherence는 역정규화 공간**에서 잰다.
- LDR의 분산 분모는 **ddof=0** (steertool/논문 구현체와 일치시킴).
- λ는 절대값이 아니라 **상대 강도** `λ_rel = λ_i‖S_i‖/‖h_i‖`. base는 `‖h‖`가 레이어별 7.7배 변한다.
- `pred_len`이 다르면 기울기 스케일이 달라 **비교 불가**. 한 실행 안에서 섞지 말 것.
- **small 결과로 base 실험을 바꾸지 않는다.** 사용자가 명시적으로 요구한 원칙. 로컬(small)은 코드 버그 검출용이고, 과학적 결정은 base 실측으로만 한다.

---

## 6. 다음 할 일

1. **Stage 5 완료 대기.** L4에서 재개 중. 증분 저장·이어받기가 되므로 중단/재시작 자유.
2. **Stage 6 작성** (미착수). 설계 Step 5 기준:
   - slope vs λ 곡선 (팔별)
   - 붕괴 지표 (coherence 위반, 예측 변동성, `slope_std` 감소 현상)
   - **개입 전/후 PCA 이동** — `pca_subset.npz`(Drive)를 쓰면 활성화 39GB 없이 가능. 개입 전 데이터로 fit하고 개입 후를 transform할 것
   - 논문 재현 팔(A/B/C) vs 데이터 기반 개선안(D~G) 비교표
3. **미검증 항목**: RW(랜덤워크) 데이터셋으로 Stage 2 이후 robustness 실행. Stage 1은 이미 생성돼 Drive에 있다.

---

## 7. 사용자 작업 방식

- 한국어로 소통한다.
- **결정이 필요하면 한 번에 답하지 말고, 필요한 정보를 먼저 요구하고 각 정보를 본 뒤 논리적으로 판단**하기를 원한다.
- 원논문의 방법을 **유지**하면서 데이터에서 발견한 개선안을 **추가**하는 방식을 선호한다. 논문 방법을 대체하지 않는다.
- 근거 없는 추론을 경계한다. 실제로 "small 결과로 base를 바꾸는 건 위험하지 않냐", "첫 토큰이 최고 분리도라는 건 말이 안 된다" 같은 지적으로 두 번 방향을 바로잡았다 — 둘 다 옳았다.
- 그림·수치는 직접 확인시켜 주기를 원한다.
