# Sooyeon Lee — data, models, experiments.

> building useful things from imperfect data.

AI Engineer at **LabQ**, collaborating with manufacturing companies on **Manufacturing AX** initiatives.

Most of the work below is small-sample regression on plant sensor and historian data — predicting or diagnosing a quality index (COOH, isomer ratio, NMR byproduct ratio) for continuous chemical/polymer processes, where the real constraint is a few hundred labeled rows rather than a leaderboard metric. The rest spans text-based prediction (cafeteria menu choice) and a full-stack product built solo (senior companion app).

---

## Stack

`Python` · `TypeScript` · `pandas` · `scikit-learn` · `XGBoost` · `statsmodels` · `Hugging Face` · `OpenAI API` · `React` · `FastAPI` · `PostgreSQL` · `Jupyter` · `Git`

## Selected work

### 폴리머 품종 전환 구간의 품질 지표 예측
연속 중합 공정에서 품종을 바꾸는 전환 구간에 한정해, 공정 historian 태그만으로 말단 카복실기 농도(COOH)를 예측하는 회귀 프로젝트. 전환 이벤트 97건에서 윈도 정의·태그 선정부터 baseline 모델 선정까지 담당하고, 이후 운영 루프 설계를 넘긴 뒤 후속 담당자에게 인계했다.

- **접근**: 전환 종료 후 안정화 시간을 품질 기준(직전 품종 μ±2σ 이탈/다음 품종 복귀)으로 재산출(24h/48h 잠정값 → 10h/12h)하고, 47개 후보 태그를 COOH 관련성·SNR·다중공선성·품종 구분력으로 점수화해 11개로 축소
- **검증**: 설비 증설 이후 322행을 이벤트 그룹 분할로 학습 — 행 단위 무작위 분할이 보류 세트를 낙관적으로 만든다는 것(이벤트 분할 시 Test MAE 20.6→83.1)을 측정해 기본값에서 배제
- **결과**: 튜닝한 Random Forest, 58 features로 Test MAE 20.44 eq/ton · MAPE 6.52% · R² 0.933 — 300행 규모에서 부스팅 계열(Validation/Train 비 30~50배) 대비 RF(1.8배)의 분산이 작아 채택
- **스택**: Python, pandas, scikit-learn (Random Forest), Optuna

<details>
<summary><b>English</b></summary>

#### Predicting a quality index across polymer grade transitions
A regression project predicting carboxyl end-group concentration (COOH) during grade changeovers in a continuous polymerization process, using plant historian tags only. Scope covered window definition, tag selection, and baseline model selection across 97 transition events, with the operating-loop design handed to the next owner.

- **Approach**: Re-derived the post-transition stabilization window from a quality criterion (departure from the previous grade's μ±2σ band, return into the next grade's), cutting the provisional 24h/48h windows to 10h/12h, and scored 47 candidate tags on COOH relevance, SNR, multicollinearity, and grade separability down to 11
- **Validation**: Trained on 322 rows after a reactor expansion with an event-grouped split — measured that a row-level random split makes the holdout optimistic (Test MAE 20.6 → 83.1 under an event split), and excluded it from the default
- **Result**: A tuned Random Forest on 58 features reaches Test MAE 20.44 eq/ton, MAPE 6.52%, R² 0.933 — chosen over boosting models (30–50× validation/train ratio) for its much lower variance (1.8×) at this sample size
- **Stack**: Python, pandas, scikit-learn (Random Forest), Optuna

</details>

🔗 [polymer_grade_transition_predictor](https://github.com/jaynee07/polymer_grade_transition_predictor)

### 촉매 반응기 품질 지표 예측과 반응 온도 추천
연속 수소화 반응 공정에서 공정 센서만으로 품질 지표(이성질체 비율)를 예측하고, 목표치에 맞춘 반응 온도 조정량을 추천하는 과제. 11개 버전까지 진행된 모델을 인수해 진단하고 고도화를 시도한 뒤, 되지 않는 이유를 측정해 보고하고 과제 범위를 "예측 정확도 개선"에서 "운전 의사결정 지원"으로 이동시켜 종료했다.

- **접근**: 인수한 모델의 버전별 성능 기록을 처음부터 재검증해 문서상 최고 모델(v7)이 실제 최고가 아님(v8)을 확인하고, 피처·구조·앙상블·데이터 증량 등 6가지 고도화를 시도
- **진단**: oracle 앙상블로 구조의 상한을 측정하고, 변곡점(24주→35주)처럼 근거 없이 기록된 값들을 재산출 — 유일하게 채택된 개선은 구간별 가중 라우팅
- **한계**: 동일 조작에도 품질이 0.2~0.7%p로 흩어지는 공정 고유 변동이 목표 오차(0.1%p)보다 커서, 예측 정확도 대신 온도 추천으로 범위를 이동해 종료
- **스택**: Python, pandas, scikit-learn, statsmodels

<details>
<summary><b>English</b></summary>

#### Predicting a reactor quality index, and recommending the temperature to hit it
A continuous hydrogenation process whose quality index (isomer ratio) is confirmed only by a once-a-day lab analysis of material that passed the reactor hours earlier. Took over an inherited model at version 11, diagnosed what it actually was, attempted to improve it, measured why that could not be done, and closed the project by moving its scope from prediction accuracy to temperature-recommendation support.

- **Approach**: Re-validated the inherited model's version history from scratch, found the documented best model (v7) was not actually best (v8), and tried six further improvements across features, structure, ensembling, and more data
- **Diagnosis**: Measured the ensemble's ceiling with an oracle, and re-derived undocumented values like the regime inflection point (24 weeks → 35 weeks) — the only adopted improvement was per-segment weighted routing
- **Limitation**: Identical operator actions produce quality outcomes that scatter 0.2–0.7 percentage points — more than the stated target error (0.1pp) — so the project closed by shifting scope from prediction accuracy to temperature recommendation
- **Stack**: Python, pandas, scikit-learn, statsmodels

</details>

🔗 [reactor_quality_advisor](https://github.com/jaynee07/reactor_quality_advisor)

### 시니어 AI 말동무
혼자 사는 시니어가 배우자·자녀·손주·친구 중 대화 상대를 골라 음성으로 일상을 나누는 웹앱. 낙상, 거동 불가, 자해처럼 당장 도움이 필요한 표현이 감지되면 보호자에게 메일로 알린다. AI는 상담사나 보호자를 대신하지 않고, 평소엔 관계에 맞는 말투로 대화를 이어가다가 위험 신호가 나오면 대화를 멈추고 사람에게 연결하는 쪽으로 동작한다.

- **접근**: 위급 감지는 모델 판단에만 맡기지 않고 키워드 규칙을 우선 적용 — 같은 세션에서 같은 유형은 한 번만 알림. 관계별 페르소나 프롬프트와 안전 정책을 시스템 프롬프트로 분리 설계하고, 대화 원문은 세션 종료 24시간 후 삭제하되 한 줄 요약만 남겨 다음 대화와 자연스럽게 이어지게 함
- **범위**: 아이디어부터 끝까지 전 과정을 수행한 개인 POC(기여도 100%) — 문제 정의, UX 설계, React/FastAPI 풀스택 구현, PostgreSQL 데이터 모델, 위급 감지·알림 정책, 문서화까지 포함
- **한계**: 실제 시니어 대상 사용자 테스트나 현장 실증은 진행하지 않음 — 사용성, 위급 감지 정확도, 정서적 고립감 완화 효과는 추후 검증이 필요한 가설
- **스택**: React 18, TypeScript, Vite, Tailwind CSS, FastAPI, SQLAlchemy 2(async), PostgreSQL, OpenAI, Naver Clova Voice, Web Speech API

<details>
<summary><b>English</b></summary>

#### Senior AI Companion
A web app for older adults living alone to share daily life by voice with a chosen conversation partner — spouse, child, grandchild, or friend. If the conversation includes an urgent phrase like a fall, inability to move, or self-harm, it detects the type and emails a configured caregiver. The AI doesn't replace a counselor or caregiver; ordinary conversation continues in a voice fitting the relationship, and when words sound dangerous it stops the chat and moves toward connecting the user with a person.

- **Approach**: Urgent-phrase detection doesn't rely on the model alone — keyword rules run first, and the same type is only alerted once per session. Relationship personas and safety rules are designed as separate system-prompt layers, and raw conversation text is deleted 24 hours after a session ends, leaving only a one-line summary so later conversations can refer back naturally
- **Scope**: A personal POC taken from idea to a working system (100% contribution) — problem definition, UX design, full-stack React/FastAPI implementation, PostgreSQL data model, urgent-detection/alerting policy, and documentation
- **Limitation**: No user test with older adults and no field pilot — usability, detection accuracy, and the hoped-for reduction in emotional isolation remain hypotheses to verify later
- **Stack**: React 18, TypeScript, Vite, Tailwind CSS, FastAPI, SQLAlchemy 2 (async), PostgreSQL, OpenAI, Naver Clova Voice, Web Speech API

</details>

🔗 [senior-ai-companion](https://github.com/jaynee07/senior-ai-companion)

### 코폴리머 공정 부산물 비율 예측
연속 중합 공정의 실시간 제어 로그만으로, 하루 뒤에 나오는 분광 분석의 부산물 비율을 미리 추정하는 회귀 프로젝트.

- **접근**: 도메인 경험칙에 앵커링한 4단 잔차 회귀로 설계 — `pred = 베이스라인(품종) + 품종 오프셋 + 원료 조성 잔차 + 캘리브레이션`
- **검증**: 품종 층화 25% 보류 + 반복 5-fold RepeatedKFold(50회)
- **결과**: 약 56건 중 42건 보류 데이터 기준 평균 MAE 0.13 wt%, 샘플당 중앙값 오차 비율 약 1.3%
- **스택**: Python, scikit-learn (Ridge, Random Forest, Gradient Boosting), pandas, numpy

<details>
<summary><b>English</b></summary>

#### Predicting a byproduct ratio in a copolymer process
A regression project that estimates, from real-time process-control logs alone, the byproduct ratio that lab spectroscopy will report a day later.

- **Approach**: A four-stage residual regression anchored on a domain rule of thumb — `pred = baseline(grade) + grade offset + material residual + calibration`
- **Validation**: Stratified 25% holdout by grade plus a repeated 5-fold RepeatedKFold (50 repeats)
- **Result**: Mean MAE of 0.13 wt% and median per-sample error ratio of about 1.3% on a 42-sample holdout out of ~56 total
- **Stack**: Python, scikit-learn (Ridge, Random Forest, Gradient Boosting), pandas, numpy

</details>

🔗 [nmr_predictor](https://github.com/jaynee07/nmr_predictor)

### 구내식당 중식 선택 비율 예측
메뉴 텍스트만으로 그날 한식 vs 일품 중 어느 쪽을 더 많이 선택할지 비율을 예측하는 프로젝트. 식수(인원) 자체는 요일·근무 인원에 좌우되므로 제외하고, 메뉴가 바꾸는 부분인 "두 코너 간 선택 비율"만 타깃으로 설계했다.

- **접근**: 메뉴를 키워드(면류/국·탕·찌개/튀김/매운맛 등)로 벡터화하고, 한식−일품의 키워드 차이를 Bradley–Terry 이항 GLM으로 학습 — 매일을 "한식 vs 일품 대결"로 모델링
- **검증**: 미래를 보지 않는 TimeSeriesSplit 5-fold + 식수 가중 MAE(WMAE)로 평가
- **결과**: 베타회귀·XGBoost·MLP·BGE-M3 임베딩 모델과 비교 실험한 결과, 표본이 작은(~170일) 상황에서 단순 선형 모델인 Bradley–Terry GLM이 가장 낮은 오차(WMAE 0.066)를 기록
- **스택**: Python, pandas, statsmodels(GLM), XGBoost, scikit-learn, BAAI/bge-m3

<details>
<summary><b>English</b></summary>

#### Predicting Cafeteria Lunch Choice Share
Predicts, from menu text alone, what share of diners will choose the **Korean set** vs. the **single-plate special** on a given day. Total headcount is driven by weekday/staffing and left out of scope — the target is only the part the menu actually moves: how the choice splits between the two counters.

- **Approach**: Menus are vectorized into keyword groups (noodles, soups/stews, fried food, spicy dishes, etc.); the Korean−special keyword difference feeds a Bradley–Terry binomial GLM, treating each day as a head-to-head "Korean vs. special" contest
- **Validation**: 5-fold `TimeSeriesSplit` (never trains on the future) scored with headcount-weighted MAE (WMAE)
- **Result**: Benchmarked against beta regression, XGBoost, MLP, and BGE-M3 sentence embeddings — with a small sample (~170 days), the simple linear Bradley–Terry GLM won out with the lowest error (WMAE 0.066)
- **Stack**: Python, pandas, statsmodels (GLM), XGBoost, scikit-learn, BAAI/bge-m3

</details>

🔗 [lunch_prediction](https://github.com/jaynee07/lunch_prediction)
