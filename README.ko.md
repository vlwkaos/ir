# ir

[ENG](README.md) | [한국어](README.ko.md)

마크다운 지식베이스를 위한 로컬 시맨틱 검색 엔진. [qmd](https://github.com/tobi/qmd)의 Rust 포트, 세 가지 핵심 차이점:

- **컬렉션별 SQLite** — 각 컬렉션이 독립 파일; 공유 전역 인덱스 없음
- **퍼시스턴트 데몬** — 모델이 쿼리 사이에 메모리에 상주; 첫 검색 시 자동 시작
- **이중 LLM 캐시** — 확장기 출력과 재순위 점수 영속화; 반복 쿼리는 즉각 반환

4개 BEIR 데이터셋 기준 검색 품질 측정; 재순위화로 순수 벡터 대비 최대 +14.5% nDCG@10.

<details>
<summary><strong>기능</strong></summary>

- **하이브리드 검색** — BM25 탐색 → 점수 융합 (0.80·벡터 + 0.20·BM25) → LLM 재순위화
- **쿼리 확장** — 확장기 모델 존재 시 lex/vec/hyde 타입 서브쿼리 생성
- **강신호 단축** — BM25 최고점 ≥ 0.75 AND 차이 ≥ 0.10이면 즉시 반환
- **데몬 모드** — 쿼리 사이에 모델 상주; 첫 검색 시 자동 시작
- **이중 LLM 캐시** — 확장기 출력 전역 캐시; 재순위 점수 컬렉션별 캐시
- **컬렉션별 SQLite** — 독립 WAL 저널, 격리 백업, 컬렉션 간 경합 없음
- **내용 주소 저장** — SHA-256으로 동일 파일 중복 제거
- **FTS5 인젝션 안전** — 모든 사용자 입력 FTS5 쿼리 생성 전 이스케이프
- **Metal GPU** — macOS에서 기본적으로 전 레이어 Metal 오프로드; `IR_GPU_LAYERS=N`으로 조정
- **자동 다운로드** — 첫 사용 시 HuggingFace Hub에서 모델 자동 다운로드

</details>

## 설치

**Homebrew (macOS):**

```bash
brew tap vlwkaos/tap
brew install ir
```

**소스에서 빌드:**

```bash
cargo install --path .
```

Rust 1.80 이상 필요. macOS에서 llama.cpp가 Metal과 자동 링크됩니다.

## 빠른 시작

```bash
ir collection add notes ~/notes   # 컬렉션 등록
ir update notes                   # 파일 스캔 → 텍스트 추출 → FTS5 인덱스 구축 (BM25)
ir embed notes                    # 텍스트 청킹 → 임베딩 모델 실행 → 벡터 저장 (벡터 + 하이브리드 검색)
ir search "러스트 메모리 안전성"  # 검색 (데몬 자동 시작)
```

`ir update`는 빠릅니다 (모델 불필요, 순수 텍스트 처리). `ir embed`는 첫 실행 시 느리지만 (청크별 모델 추론), 이후에는 변경된 내용만 재임베딩합니다. BM25 검색은 `update`만으로 동작하며, 벡터 및 하이브리드 검색은 `embed`가 필요합니다.

**한국어/일본어/중국어 컬렉션:**

```bash
ir preprocessor install ko        # lindera-tokenize 다운로드 및 "ko" 등록
                                  # 설치 후 컬렉션 바인딩 피커 표시

ir collection add wiki ~/wiki     # 컬렉션이 없는 경우 추가
ir preprocessor bind ko wiki      # "ko"를 컬렉션에 연결하고 재인덱싱

ir search "서울 지하철" -c wiki
```

전처리기 없이는 "이스탄불의", "検索エンジン" 같은 교착어가 하나의 FTS 토큰으로 처리되어 형태소 단위 검색이 불가합니다.

<details>
<summary><strong>모델</strong></summary>

모델은 첫 사용 시 HuggingFace Hub에서 자동으로 다운로드되어 `~/.cache/huggingface/`에 캐시됩니다. 별도 설정 불필요.

| 모델 | HF 저장소 | 필요 기능 |
|---|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | `ggml-org/embeddinggemma-300M-GGUF` | `ir embed`, 벡터 검색, 하이브리드 |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | `unsloth/Qwen3.5-0.8B-GGUF` | 통합 확장+재순위 (선택) |
| [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | `unsloth/Qwen3.5-2B-GGUF` | 통합 확장+재순위 (선택) |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` | 재순위화 전용 (선택) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | `tobil/qmd-query-expansion-1.7B` | 쿼리 확장 전용 (선택) |
| [BGE-M3 568M](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) | `ggml-org/bge-m3-Q8_0-GGUF` | 한국어 임베딩 대안 (선택) |

BM25 검색은 모델 없이 동작합니다. `IR_QWEN_MODEL`이 설정되거나 `~/local-models/`에 Qwen3.5 GGUF가 있으면 확장기와 재순위기를 대체합니다.

**로컬 모델:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_QWEN_MODEL="$HOME/local-models/Qwen3.5-2B-Q4_K_M.gguf"   # 통합
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

탐색 순서: 환경변수 → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/` → HF Hub 자동 다운로드.

`IR_*_MODEL` 환경변수는 `.gguf` 파일 경로, 모델이 포함된 디렉터리 경로, 또는 HuggingFace 레포 ID(`owner/name`)를 허용합니다. 인식되지 않는 값은 기본 모델을 조용히 로드하는 대신 즉시 오류를 출력합니다.

**GPU:**

```bash
IR_GPU_LAYERS=0 ir search "쿼리"    # CPU 강제
IR_GPU_LAYERS=32 ir search "쿼리"   # 부분 오프로드
```

</details>

<details>
<summary><strong>한국어 임베딩 모델</strong></summary>

기본 EmbeddingGemma (300M, 768d)로 하이브리드+재순위 시 MIRACL-Korean nDCG@10 = 0.8411.
한국어 특화 dense retrieval이 필요하면 BGE-M3를 대체 모델로 사용할 수 있습니다.

| | EmbeddingGemma | BGE-M3 |
|---|---|---|
| 파라미터 | ~150M | ~570M |
| 차원 | 768 | 1024 |
| GGUF (Q8_0) | ~300MB | ~600MB |
| 자동 감지 | 파일명 "embeddinggemma" | 파일명 "bge-m3" |

```bash
# HuggingFace에서 자동 다운로드
export IR_EMBEDDING_MODEL="ggml-org/bge-m3-Q8_0-GGUF"

# 또는 로컬 파일 (파일명에 "bge-m3" 포함 필수)
export IR_EMBEDDING_MODEL="$HOME/local-models/bge-m3-Q8_0.gguf"

# 기존 컬렉션 재임베딩 (차원 자동 변환)
ir embed <collection> --force
```

파일명에 "bge-m3"가 포함되면 CLS 풀링 및 쿼리 프리픽스가 자동 적용됩니다.
모델 변경 후 `ir embed --force`를 실행하면 벡터 테이블 차원이 자동으로 조정됩니다.

**KURE-v1 (실험적):** MTEB-ko Recall@1 = 0.5264 (dense only). BGE-M3 기반이지만 GGUF 변환이 검증되지 않았습니다. llama.cpp의 `convert_hf_to_gguf.py`로 직접 변환이 필요합니다.

**참고:** 한국어 쿼리 확장(expander)은 비권장 -- 영어 SFT 모델이라 MIRACL-Korean에서 -0.4% 성능 저하.

</details>

<details>
<summary><strong>사용법</strong></summary>

**컬렉션:**

```bash
ir collection add notes ~/notes
ir collection add code  ~/code
ir collection ls
ir collection rm notes
ir status                    # 컬렉션별 인덱스 상태
```

**인덱싱 및 임베딩:**

```bash
ir update                    # 모든 컬렉션 인덱싱
ir update notes              # 특정 컬렉션
ir update notes --force      # 전체 재인덱싱

ir embed                     # 미임베딩 문서 임베딩
ir embed notes --force       # 전체 재임베딩
```

**검색:**

```bash
ir search "러스트 메모리 안전성"
ir search "sqlite 아키텍처"    --mode bm25
ir search "비동기 패턴"        --mode vector
ir search "에러 처리"          --mode hybrid -c notes --min-score 0.4

# 출력 형식
ir search "소유권" --json
ir search "소유권" --md
ir search "소유권" --files       # 경로만
ir search "소유권" --full        # 결과에 문서 전문 포함
ir search "소유권" --chunk       # 가장 관련성 높은 청크 텍스트 포함 (벡터 결과)
```

**문서 조회:**

```bash
ir get "2026/Daily/04/2026-04-07.md"            # 컬렉션 상대 경로
ir get "Notes/2026/Daily/04/2026-04-07.md"      # 볼트 루트 경로 (컬렉션 디렉토리명 접두사 자동 제거)
ir get "2026-04-07" -c periodic                  # 부분 일치, 컬렉션 지정
ir get "some/path.md" --json                     # JSON으로 전체 메타데이터 출력
ir get "some/path.md" --section "설치"           # 해당 헤딩 섹션만 추출 (대소문자 무관)
ir get "some/path.md" --max-chars 3000           # 앞 3000자만 반환
ir get "some/path.md" --offset 1000 --max-chars 2000  # 1000~3000번째 문자

ir multi-get "file1.md" "file2.md" "file3.md"   # 일괄 조회
ir multi-get "file1.md" "file2.md" --json        # {found: [...], not_found: [...]}
ir multi-get "file1.md" "file2.md" --files       # 찾은 경로만 출력
ir multi-get "file1.md" "file2.md" --max-chars 2000  # 각 문서 잘라서 반환
```

경로 매칭 순서: 정확 일치 → 접미 일치(`%/path`) → 부분 문자열. 볼트 루트 경로(첫 번째 구성 요소가 컬렉션 디렉토리명과 일치하는 경우)는 일반 매칭 전에 먼저 처리됩니다.

**데몬:**

```bash
ir daemon start              # 시작 (첫 검색 시 자동 시작)
ir daemon stop
ir daemon status
```

데몬은 모델을 메모리에 유지합니다. Unix 소켓을 통한 후속 쿼리는 모델 로딩을 건너뜁니다 (약 30ms 응답).

</details>

<details>
<summary><strong>MCP 서버 — Claude Desktop / Claude Code</strong></summary>

`ir mcp`는 Model Context Protocol 서버를 실행하여 Claude가 인덱싱된 문서를 직접 검색할 수 있게 합니다.

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ir": {
      "command": "ir",
      "args": ["mcp"]
    }
  }
}
```

**Claude Code** (프로젝트 루트의 `.mcp.json` 또는 `~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "ir": {
      "command": "ir",
      "args": ["mcp"]
    }
  }
}
```

다섯 가지 도구가 제공됩니다:

| 도구 | 설명 |
|------|------|
| `search` | 하이브리드 BM25+벡터 검색. 경로, 제목, 점수, 스니펫 반환. `mode`, `limit`, `min_score`, `collections`, `full`(전문 포함), `include_chunk`(청크 텍스트 포함) 파라미터 지원. |
| `get` | 경로로 문서 조회 (정확 → 접미 → 부분 일치). `collections`, `section`(헤딩 텍스트, 대소문자 무관), `offset`(문자 오프셋), `max_chars`(잘라내기) 파라미터 지원. |
| `multi_get` | 문서 일괄 조회. `paths[]`, `collections`, `max_chars`(각 문서 잘라내기) 파라미터. `found`와 `not_found` 반환. |
| `status` | 인덱스 상태 — 컬렉션 이름, 문서 수, DB 크기, 데몬 상태. |
| `update` | 파일 변경 후 컬렉션 재인덱싱. `collection`과 `force` 파라미터 지원. |

**HTTP 모드** (원격 접속 또는 멀티 클라이언트):

```bash
ir mcp --http 3620    # 전체 인터페이스, 포트 3620
```

클라이언트를 `http://<host>:3620/mcp`로 설정합니다. 첫 검색 도구 호출 시 데몬이 자동 시작됩니다.

> **보안 주의:** HTTP 모드는 인증 없이 전체 인터페이스에 바인딩됩니다. 신뢰할 수 있는 네트워크에서만 노출하세요. `update` 도구는 재인덱싱을 유발할 수 있으므로 로컬 쓰기 권한 서비스로 취급하세요.

</details>

<details>
<summary><strong>전처리기 — 한국어 / 일본어 / 중국어</strong></summary>

전처리기는 BM25 인덱싱 전에 텍스트를 형태소 분석합니다. 전처리기 없이는 교착어 형식("이스탄불의", "東京都")이 하나의 FTS 토큰으로 처리되어 형태소 단위 쿼리와 매칭되지 않습니다. 인덱싱 시와 쿼리 시 동일한 전처리기가 적용됩니다.

**한국어 (lindera, Mode::Decompose):**

```bash
ir preprocessor install ko          # lindera-tokenize 다운로드 후 "ko"로 등록
                                    # 설치 후 컬렉션 바인딩 피커 표시
ir collection add wiki ~/wiki       # 컬렉션 추가 (아직 없는 경우)
ir preprocessor bind ko wiki        # "ko"를 컬렉션에 연결하고 재인덱싱
ir search "서울 지하철" -c wiki
```

`ir preprocessor install ko`는 GitHub 릴리즈에서 미리 빌드된 바이너리를 다운로드합니다. mecab-ko-dic 사전이 내장되어 별도 시스템 의존성이나 Rust 툴체인이 필요 없습니다.

저장소에서 직접 빌드:

```bash
cd preprocessors/ko/lindera-tokenize && cargo build --release
ir preprocessor add ko ./target/release/lindera-tokenize
```

**다른 언어:**

```bash
ir preprocessor install ja    # 일본어 (lindera)
ir preprocessor install zh    # 중국어 (바이그램 토크나이저)
```

**관리:**

```bash
ir preprocessor list
ir preprocessor remove ko
```

프로토콜은 stdin/stdout 라인 단위: UTF-8 한 줄 입력, 토큰화된 한 줄 출력, 프로세스는 라인 간 유지. 이 프로토콜을 따르는 실행 파일은 모두 등록 가능.

lindera 처리 속도: M-시리즈 Mac 기준 약 5,600 문서/초 · 1.8 MB/초. 시작 시간 거의 없음 (Rust 바이너리, 내장 사전).

**한국어 BM25 벤치마크** (MIRACL-Korean, 쿼리 213개):

| 전처리기 | nDCG@10 | 비고 |
|---|---|---|
| 없음 | 0.0009 | 교착어 토큰 매칭 불가 |
| lindera | 0.0460 | 형태소 분석으로 50배 향상 |
| lindera hybrid+rerank | **0.8411** | 2,835 패시지 기준 거의 최고 성능 |

복합어 분해 벤치마크 (복합어 내 구성 요소를 타겟으로 한 쿼리 50개):

| 전처리기 | nDCG@10 | 비고 |
|---|---|---|
| 없음 | 0.0000 | FTS 인덱스에 구성 요소 없음 |
| lindera | **0.6326** | Mode::Decompose로 복합어 분해 |

상세 결과 및 근거: [research/experiment.md](research/experiment.md)

</details>

<details>
<summary><strong>검색 파이프라인</strong></summary>

```
쿼리
  │
  ├─ BM25 탐색 ──► 점수 ≥ 0.75 AND 차이 ≥ 0.10? ──► 즉시 반환
  │
  ├─ 확장기 있음:  확장 → lex/vec/hyde 서브쿼리 → RRF 융합
  ├─ 확장기 없음:  BM25 + 벡터 → 점수 융합 (0.80·벡터 + 0.20·BM25)
  │
  └─ 재순위기: 최종 = 0.40·융합 + 0.60·P(관련)
```

확장기와 재순위기 출력은 SQLite에 캐시됩니다. 반복 쿼리는 LLM 추론을 건너뜁니다.

</details>

<details>
<summary><strong>벤치마크 — BEIR (4개 데이터셋, nDCG@10)</strong></summary>

EmbeddingGemma 300M 임베딩 + qmd-expander-1.7B + Qwen3-Reranker-0.6B.

| 데이터셋 | BM25 | 벡터 | 하이브리드 | +재순위 | LLM 향상 |
|---|---|---|---|---|---|
| NFCorpus (323q) | 0.2046 | 0.3898 | 0.3954 | **0.4001** | +1.2% |
| SciFact (300q) | 0.0500 | 0.7847 | 0.7873 | **0.7797** | −1.0% |
| FiQA (648q) | 0.0298 | 0.4324 | 0.4266 | **0.4567** | +7.1% |
| ArguAna (1406q) | 0.0012 | 0.4264 | 0.4263 | **0.4879** | +14.5% |

BM25 융합은 어느 데이터셋에서도 순수 벡터 대비 통계적으로 유의미한 향상 없음 (paired t-test). 재순위 향상은 대화형/논증 검색 작업에서 가장 큼.

재현 방법: [research/experiment.md](research/experiment.md)

</details>

<details>
<summary><strong>qmd와 비교</strong></summary>

ir은 [qmd](https://github.com/tobi/qmd)의 Rust 포트로, 다른 저장소 모델과 퍼시스턴트 데몬을 갖춤.

| | qmd | ir |
|---|---|---|
| 저장소 | 모든 컬렉션에 단일 SQLite | 컬렉션별 SQLite — `rm name.sqlite`로 삭제 |
| 동시 쓰기 | 공유 WAL 저널 | 컬렉션별 독립 WAL |
| sqlite-vec | 동적 로드 `.so` | 정적 컴파일 |
| 프로세스 모델 | 쿼리마다 스폰 | 데몬이 모델 유지 |
| LLM 캐시 | 재순위 점수 (컬렉션별) | 재순위 점수 + 확장기 출력 (전역) |
| 품질 (NFCorpus nDCG@10) | 미공개 | 0.4001 |

**성능** (macOS M4 Max, 동일 모델·쿼리):

| | ir | qmd | 비율 |
|---|---:|---:|---|
| **콜드** (캐시 없음) | 3.0s | 9.5s | **3×** |
| **웜** (데몬 + 캐시 활성) | 30ms | 840ms | **28×** |

콜드 차이: ir은 재순위 후보 최대 20개 vs qmd 40개. 웜 차이: qmd는 쿼리마다 ~800ms 프로세스 스폰 + JS 런타임 부담; ir 데몬 왕복은 30ms (임베딩 + kNN만).

</details>

<details>
<summary><strong>개발</strong></summary>

```bash
cargo build                  # 디버그 빌드
cargo build --release        # 릴리즈 빌드
cargo test                   # 단위 테스트 (모델 불필요)
cargo test -- --ignored      # 모델 의존 테스트 (모델 필요)
cargo run --bin eval -- --data test-data/nfcorpus --mode all
```

</details>

<details>
<summary><strong>스키마</strong></summary>

각 컬렉션 데이터베이스 (`~/.config/ir/collections/<name>.sqlite`):

```
content          — 해시 → 전체 텍스트 (내용 주소)
documents        — 경로, 제목, 해시, 활성 플래그
documents_fts    — FTS5 가상 테이블 (porter 토크나이저)
vectors_vec      — sqlite-vec kNN (768차원 코사인, EmbeddingGemma 형식)
content_vectors  — 청크 메타데이터 (해시, 순번, 위치, 모델)
llm_cache        — 재순위 점수 캐시 (sha256(모델+쿼리+문서) → 점수)
meta             — 컬렉션 메타데이터 (이름, 스키마 버전)
```

전역 캐시 (`~/.config/ir/expander_cache.sqlite`):

```
expander_cache   — sha256(모델+쿼리) → JSON Vec<SubQuery>
```

트리거가 insert/update/delete 시 `documents_fts`를 `documents`와 동기화합니다.

</details>
