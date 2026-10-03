# Sooyeon Lee — data, models, experiments.

> building useful things from imperfect data.

AI Engineer at **LabQ**, collaborating with manufacturing companies on **Manufacturing AX** initiatives.

Interested in small-data modeling, interpretable ML, and text-based prediction.

---

## Stack

`Python` · `TypeScript` · `pandas` · `scikit-learn` · `XGBoost` · `statsmodels` · `Hugging Face` · `OpenAI API` · `React` · `FastAPI` · `PostgreSQL` · `Jupyter` · `Git`

## Selected work

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

### 시니어 AI 말동무
혼자 사는 시니어가 배우자·자녀·손주·친구 중 대화 상대를 골라 음성으로 일상을 나누는 웹앱. 낙상, 거동 불가, 자해처럼 당장 도움이 필요한 표현이 감지되면 보호자에게 메일로 알린다. AI는 상담사나 보호자를 대신하지 않고, 평소엔 관계에 맞는 말투로 대화를 이어가다가 위험 신호가 나오면 대화를 멈추고 사람에게 연결하는 쪽으로 동작한다.

- **접근**: 위급 감지는 모델 판단에만 맡기지 않고 키워드 규칙을 우선 적용 — 같은 세션에서 같은 유형은 한 번만 알림. 관계별 페르소나 프롬프트와 안전 정책을 시스템 프롬프트로 분리 설계하고, 대화 원문은 세션 종료 24시간 후 삭제하되 한 줄 요약만 남겨 다음 대화와 자연스럽게 이어지게 함
- **범위**: 아이디어부터정을 수행한 개인 POC(기여도 100%) — 문제 정의, UX 설계, React/FastAPI 풀스택 구현, PostgreSQL
데이터 모델, 위급 감지·  지 포함
- **한계**: 실제 시니어 대상 사용자 테스트나 현장 실증은 진행하지 않음 —
사용성, 위급 감지 정확도추후 검증이 필요한 가설
- **스택**: React 18, TypeScript, Vite, Tailwind CSS, FastAPI, SQLAlchemy
2(async), PostgreSQL, Opeb Speech API

<details>
<summary><b>English</b></summary>

#### Senior AI Companion
A web app for older adulout daily life by voicewith a chosen partner — spouse, child, grandchild, or friend. If theconversation includes anall, inability to move, or self-harm, it detects the type and emails a configured caregiver. Thdoesn't replace a counseary conversation itcontinues in a voice fitting the relationship, and when words sound dangerous it stops the ccting the user with aperson.                                                                    
- **Approach**: Urgent-phrase detection doesn't rely on the model alone —  keyword rules run first,ted only once per session. Relationship personas and safety rules are designed as separate prompt     layers folded into the sare deleted 24 hours after a session ends, leaving only a one-line summary so later conversations can refer back naturally
- **Scope**: A personal POC taken from idea to a working system (100% contribution) — problem l-stack React/FastAPIimplementation, PostgreSQL data model, urgent-detection/alerting/privapolicy, and documentatio
- **Limitation**: No user test with older adults and no field pilot — usability, detection accmotional isolation remainhypotheses to verify later
- **Stack**: React 18, TCSS, FastAPI, SQLAlchemy 2 (async), PostgreSQL, OpenAI, Naver Clova Voice, Web Speech API

</details>

🔗 [senior-ai-companion](httenior-ai-companion)
