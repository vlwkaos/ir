# ir

[ENG](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

마크다운 지식베이스와 코드 저장소를 함께 검색하는 로컬 시맨틱 검색 엔진. [qmd](https://github.com/tobi/qmd)의 Rust 포트, 세 가지 핵심 차이점:

- **컬렉션별 SQLite** — 각 컬렉션이 독립 파일; 공유 전역 인덱스 없음
- **퍼시스턴트 데몬** — 모델이 쿼리 사이에 메모리에 상주; 첫 검색 시 자동 시작
  차가운 첫 쿼리는 데몬이 뒤에서 예열되는 동안 BM25 결과를 먼저 돌려줄 수 있다.
- **이중 LLM 캐시** — 확장기 출력과 재순위 점수 영속화; 반복 쿼리는 즉각 반환
- **연결 검색** — 선택적으로 코드 단위와 명시적 마크다운/주석 링크를 함께 반환

4개 BEIR 데이터셋 기준 검색 품질 측정; 재순위화로 순수 벡터 대비 최대 +14.5% nDCG@10.

<details>
<summary><strong>기능</strong></summary>

- **하이브리드 검색** — BM25 탐색 → 점수 융합 (0.80·벡터 + 0.20·BM25) → LLM 재순위화
- **쿼리 확장** — 확장기 모델 존재 시 lex/vec/hyde 타입 서브쿼리 생성
- **강신호 단축** — BM25 최고점 ≥ 0.75 AND 차이 ≥ 0.10이면 즉시 반환
- **데몬 모드** — 쿼리 사이에 모델 상주; 첫 검색 시 자동 시작
  cold start라도 첫 useful BM25 결과를 막지 않는다.
- **이중 LLM 캐시** — 확장기 출력 전역 캐시; 재순위 점수 컬렉션별 캐시
- **컬렉션별 SQLite** — 독립 WAL 저널, 격리 백업, 컬렉션 간 경합 없음
- **내용 주소 저장** — SHA-256으로 동일 파일 중복 제거
- **FTS5 인젝션 안전** — 모든 사용자 입력 FTS5 쿼리 생성 전 이스케이프
- **GPU 가속** — macOS에서는 Metal 기본 사용, Linux에서는 CUDA/ROCm/Vulkan 선택 가능 (feature 플래그); `IR_GPU_LAYERS=N`으로 조정
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

Rust 1.80 이상 필요. macOS에서 llama.cpp가 Metal과 자동 링크됩니다. Linux에서는 `--features llama-cuda`, `llama-rocm`, `llama-vulkan` 중 하나를 지정해 GPU를 활성화할 수 있습니다.

## 빠른 시작

```bash
ir collection add notes ~/notes   # 컬렉션 등록
ir update notes                   # 파일 스캔 → 텍스트 추출 → FTS5 인덱스 구축 (BM25)
ir embed notes                    # 텍스트 청킹 → 임베딩 모델 실행 → 벡터 저장 (벡터 + 하이브리드 검색)
ir search "러스트 메모리 안전성"  # 검색 (데몬 자동 시작)
```

`ir update`는 빠릅니다 (모델 불필요, 순수 텍스트 처리). `ir embed`는 첫 실행 시 느리지만 (청크별 모델 추론), 이후에는 변경된 내용만 재임베딩합니다. BM25 검색은 `update`만으로 동작하며, 벡터 및 하이브리드 검색은 `embed`가 필요합니다.

## 코드 + 지식 연결 검색

기존 마크다운 전용 동작은 그대로 유지됩니다. 코드와 지식을 함께 인덱싱하려면 mixed 프리셋을 사용합니다:

```bash
ir collection add project . --preset mixed
ir update project
ir embed project
ir search "reranking을 왜 건너뛰는가" -c project --chunk --related 3 --json
```

`--preset mixed`는 마크다운과 주요 코드 확장자(Rust, Python, JS/TS, Go, Java, C/C++, C#, Ruby, PHP, Swift, Kotlin, Scala, shell, Lua, Dart, Elixir, Erlang, F#, Clojure)를 포함하고 일반적인 빌드/벤더 디렉터리를 제외합니다. 코드 심볼 추출은 best-effort이며, 인식되지 않는 형태도 파일 단위로 인덱싱됩니다.

연결은 추론하지 않고 명시적 구조에서만 만듭니다: `[[wikilink]]`, 로컬 마크다운 링크, frontmatter `related:`, 그리고 주석/본문의 `[concept-slug]`. 줄 번호는 표시용 힌트일 뿐이며, 검색 결과는 인덱싱 당시의 단위 텍스트와 단위 텍스트 해시인 `indexed_hash`를 함께 반환합니다. 자세한 에이전트 규칙은 [docs/linked-retrieval-agent.md](docs/linked-retrieval-agent.md)를 참고하세요.

**한국어/일본어/중국어 컬렉션:**

```bash
ir preprocessor install ko        # 공식 lindera CLI + ko-dic 다운로드, "ko" 등록
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

BM25 검색은 모델 없이 동작합니다. 기본 tier-2 경로는 전용 확장기 + 재순위기입니다. `IR_COMBINED_MODEL`은 통합 모델 실험이나 테스트를 할 때만 명시적으로 사용합니다.

**로컬 모델:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

통합 모드는 명시적으로만 사용합니다:

```bash
export IR_COMBINED_MODEL="$HOME/local-models/Qwen3.5-2B-Q4_K_M.gguf"   # 테스트 / 실험 전용
```

탐색 순서: 환경변수 → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → `~/.cache/qmd/models/` → HF Hub 자동 다운로드.

`IR_*_MODEL` 환경변수는 `.gguf` 파일 경로, 모델이 포함된 디렉터리 경로, 또는 HuggingFace 레포 ID(`owner/name`)를 허용합니다. 인식되지 않는 값은 기본 모델을 조용히 로드하는 대신 즉시 오류를 출력합니다.

**설정 디렉터리:**

```bash
export IR_CONFIG_DIR="~/vault/.config/ir"   # 다른 기기에서도 동일하게 사용 가능
```

`IR_CONFIG_DIR`은 설정 파일, 컬렉션 DB, 데몬 파일이 저장되는 디렉터리를 지정합니다. `~` 및 `$VAR` 확장을 지원하여 여러 기기에 동기화되는 MCP 설정에서도 안전하게 사용할 수 있습니다. 우선순위: `IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir` (deprecated) → `~/.config/ir`.

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
ir search "소유권" --chunk       # 가장 관련성 높은 청크/단위 텍스트 포함
ir search "소유권" --related 3   # 명시적 1-hop 연결 컨텍스트 포함 (최대 20)
ir search "소유권" --quiet       # stderr 억제 (진행 표시, 로그) — 스크립팅용

# 필터 (-f/--filter, 반복 가능; 모든 조건 AND)
ir search "설계" -f "modified_at>=2026-01-01"
ir search "설계" -f "meta.tags=rust"
ir search "설계" -f "path~notes/"
ir search "설계" -f "modified_at>=2025-01-01" -f "meta.author=vlwkaos"
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

**필터 문법 (`-f/--filter`):**

각 조건은 `FIELD OP VALUE` 형식입니다. 여러 `-f`는 AND로 결합됩니다.

| 필드 | 설명 |
|------|------|
| `path` | 문서 경로 (컬렉션 루트 기준) |
| `modified_at` | 파일 수정 시간 (UTC RFC3339) |
| `created_at` | 파일 생성 시간 (UTC RFC3339) |
| `meta.<name>` | 프론트매터 필드 (예: `meta.tags`, `meta.author`) |

| 연산자 | 의미 |
|--------|------|
| `=` / `!=` | 동일 / 다름 (대소문자 구분) |
| `>` / `>=` / `<` / `<=` | 사전식 비교 (날짜는 UTC RFC3339로 정규화) |
| `~` / `!~` | 포함 / 미포함 (대소문자 무관) |

날짜 값(`modified_at`, `created_at`, `meta.date`)은 UTC RFC3339로 정규화됩니다 (`YYYY-MM-DD` → `YYYY-MM-DDT00:00:00Z`). 배열 프론트매터 필드(태그 등)는 **어느 한** 요소가 조건을 만족하면 일치로 처리됩니다 — `!=`도 마찬가지입니다. `["rust", "go"]`로 태그된 문서는 `"go"`가 조건을 만족하므로 `meta.tags!=rust`에 매칭됩니다. `meta.*` 절은 메타데이터 행이 없는 문서에서 항상 실패합니다.

> **참고:** 이번 릴리즈 이후 첫 사용 시 DB 스키마가 버전 2로 업그레이드됩니다. 기존 프론트매터에서 `document_metadata`를 채우는 일회성 작업이 실행되며, 10,000개 미만 문서 기준 1초 이내에 완료됩니다.

**데몬:**

```bash
ir daemon start              # 시작 (첫 검색 시 자동 시작)
ir daemon stop
ir daemon status
```

데몬은 모델을 메모리에 유지합니다. Unix 소켓을 통한 후속 쿼리는 모델 로딩을 건너뜁니다 (약 30ms 응답).

</details>

<details>
<summary><strong>증분 인덱싱</strong></summary>

IR은 SHA-256 해싱을 사용한 콘텐츠 주소 저장소를 통해 변경된 파일만 효율적으로 처리합니다.

**작동 방식:**

- **변경 감지**: 파일을 해시(SHA-256)하여 저장된 해시와 비교
- **스마트 업데이트**: 수정되거나 새로운 파일만 재처리
- **삭제 처리**: 제거된 파일은 비활성으로 표시 (소프트 삭제)
- **중복 제거**: 컬렉션 내 동일한 콘텐츠는 저장소 공유

**인덱스 작업:**

```bash
# 일반 증분 업데이트 (기본값)
ir update                    # 모든 컬렉션
ir update notes              # 특정 컬렉션

# 처음부터 전체 재인덱싱 강제
ir update notes --force      # 전체 인덱스 재구축

# 변경 사항 확인 (요약 확인)
ir update notes
# 출력: "2 added, 1 updated, 0 deactivated"
```

**임베딩 작업:**

```bash
# 증분 임베딩 (새로운/변경된 문서만)
ir embed                     # 미임베딩 콘텐츠 임베딩
ir embed notes               # 특정 컬렉션

# 전체 재임베딩 강제
ir embed notes --force       # 모든 벡터 재계산
```

**성능 특성:**

- 초기 인덱싱: 빠름 (모델 없음, 순수 텍스트 추출)
- 증분 업데이트: 변경된 파일만 처리
- 해시 비교: 수천 개 파일도 즉시 처리
- 임베딩: 첫 실행은 느림, 증분 업데이트는 빠름

**예제 워크플로우:**

```bash
# 월요일: 초기 설정
ir collection add notes ~/notes
ir update notes              # 500개 파일 인덱싱
ir embed notes               # 500개 임베딩 계산 (느림)

# 화요일: 3개 파일 추가, 2개 수정
ir update notes              # 출력: "3 added, 2 updated, 0 deactivated"
ir embed notes               # 5개 문서만 임베딩 (빠름)

# 수요일: 1개 파일 삭제
ir update notes              # 출력: "0 added, 0 updated, 1 deactivated"
# 삭제에는 임베딩 불필요
```

증분 방식으로 인해 성능 저하 없이 `ir update`를 자주 실행할 수 있습니다 — 변경된 콘텐츠만 처리됩니다.

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
| `search` | 하이브리드 BM25+벡터 검색. 경로, 제목, 점수, 스니펫 반환. `mode`, `limit`, `min_score`, `collections`, `full`(전문 포함), `include_chunk`(청크/단위 텍스트 포함), `include_related`, `related_limit`(최대 20), `filter`(`{field, op, value}` 객체 배열, AND 결합) 파라미터 지원. |
| `get` | 경로로 문서 조회 (정확 → 접미 → 부분 일치). `collections`, `section`(헤딩 텍스트, 대소문자 무관), `offset`(문자 오프셋), `max_chars`(잘라내기) 파라미터 지원. |
| `multi_get` | 문서 일괄 조회. `paths[]`, `collections`, `max_chars`(각 문서 잘라내기) 파라미터. `found`와 `not_found` 반환. |
| `status` | 인덱스 상태 — 컬렉션 이름, 문서 수, DB 크기, 데몬 상태. |
| `update` | 파일 변경 후 컬렉션 재인덱싱. `collection`과 `force` 파라미터 지원. |

`filter` 배열 예시: `{"field": "modified_at", "op": ">=", "value": "2024-01-01"}`. 필드: `path`, `modified_at`, `created_at`, `meta.<이름>`. 연산자: `=`, `!=`, `>`, `>=`, `<`, `<=`, `~`(포함), `!~`(미포함).

**HTTP 모드** (원격 접속 또는 멀티 클라이언트):

```bash
ir mcp --http 3620                              # 전체 인터페이스, 포트 3620
ir mcp --http 3620 --cors '*'                   # 모든 브라우저 출처 허용 (와일드카드)
ir mcp --http 3620 --cors 'https://app.example.com'  # 특정 출처만 허용
```

클라이언트를 `http://<host>:3620/mcp`로 설정합니다. 첫 검색 도구 호출 시 데몬이 자동 시작됩니다.

`--cors`는 브라우저 클라이언트(웹 앱, Claude.ai 웹)가 연결할 수 있도록 `Access-Control-Allow-Origin`을 설정합니다. `--cors '*'`는 rmcp의 DNS 리바인딩 호스트 검사도 비활성화하므로 신뢰할 수 있는 네트워크에서만 사용하세요. `--cors` 없이 실행 시 CORS 헤더가 설정되지 않습니다(curl/CLI 클라이언트에는 영향 없음).

> **보안 주의:** HTTP 모드는 인증 없이 전체 인터페이스에 바인딩됩니다. 신뢰할 수 있는 네트워크에서만 노출하세요. `update` 도구는 재인덱싱을 유발할 수 있으므로 로컬 쓰기 권한 서비스로 취급하세요.

</details>

<details>
<summary><strong>전처리기 — 한국어 / 일본어 / 중국어</strong></summary>

전처리기는 BM25 인덱싱 전에 텍스트를 형태소 분석합니다. 전처리기 없이는 교착어 형식("이스탄불의", "東京都")이 하나의 FTS 토큰으로 처리되어 형태소 단위 쿼리와 매칭되지 않습니다. 인덱싱 시와 쿼리 시 동일한 전처리기가 적용됩니다.

**한국어 (lindera, Mode::Decompose):**

```bash
ir preprocessor install ko          # 공식 lindera CLI + ko-dic 다운로드, "ko" 등록
                                    # 설치 후 컬렉션 바인딩 피커 표시
ir collection add wiki ~/wiki       # 컬렉션 추가 (아직 없는 경우)
ir preprocessor bind ko wiki        # "ko"를 컬렉션에 연결하고 재인덱싱
ir search "서울 지하철" -c wiki
```

`ir preprocessor install ko`는 lindera 공식 GitHub 릴리즈에서 lindera CLI 바이너리와 ko-dic 사전을 다운로드합니다. 지원 플랫폼: **macOS** (arm64, x86\_64) 및 **Linux** (x86\_64, aarch64). 별도 시스템 의존성이나 Rust 툴체인이 필요 없습니다. 설치 시 컬렉션 바인딩 피커가 표시됩니다.
내장 `ko` alias를 바인딩하면 해당 컬렉션에 현재 한국어 routing 기본값도 함께 기록됩니다:

```yaml
routing:
  fused_strong_product: 0.05
```

이 값은 bind 시점에 써 넣는 기본값이며, 숨겨진 런타임 special case가 아닙니다. 이미 `routing:`을 직접 설정한 경우에는 그 명시적 설정이 우선합니다.

**컬렉션별 routing override** (`config.yml`, 선택):

```yaml
collections:
  - name: wiki-ko
    path: ~/wiki
    preprocessor: [ko]
    routing:
      fused_strong_product: 0.05
```

특정 컬렉션에만 BM25/fused strong-signal threshold를 다르게 적용할 때 사용합니다. 지원 필드:

- `fused_strong_floor`
- `fused_strong_product`
- `bm25_strong_floor`
- `bm25_strong_gap`

override는 검색에 포함된 모든 컬렉션이 같은 값을 명시했을 때만 적용됩니다. 서로 다른 override가 섞인 multi-collection 검색은 전역 기본 threshold로 되돌아갑니다.

**일본어:**

```bash
ir preprocessor install ja    # 일본어 (lindera + ipadic)
```

**중국어 (lindera + jieba, Mode::Decompose):**

```bash
ir preprocessor install zh          # lindera CLI + jieba 사전 다운로드, "zh" 등록
ir collection add notes ~/notes     # 컬렉션 추가
ir preprocessor bind zh notes       # "zh"를 컬렉션에 연결하고 재인덱싱
ir search "机器学习" -c notes
```

`ir preprocessor install zh`는 lindera 공식 GitHub 릴리즈에서 lindera CLI 바이너리와 jieba 분절 사전을 다운로드합니다 (`ko`/`ja`와 동일 바이너리, 다른 사전). 지원 플랫폼: **macOS** (arm64, x86\_64) 및 **Linux** (x86\_64, aarch64).

`ko`와 달리 `zh` 바인딩 시 routing override가 자동 적용되지 않습니다. 전역 strong-signal threshold가 적용됩니다. 컬렉션별 `routing:` 블록을 직접 추가하세요.

**제한사항:** 불용어 필터링 없음. 기능어(的、了、在、是 등)가 인덱스 항목으로 남습니다. BM25 정밀도가 기능어 중심 쿼리에서 낮을 수 있으며, hybrid+rerank로 보완됩니다.

**관리:**

```bash
ir preprocessor list
ir preprocessor remove ko
```

프로토콜은 stdin/stdout 라인 단위: UTF-8 한 줄 입력, 토큰화된 한 줄 출력 (모든 토큰이 필터링된 경우 출력 없음), 프로세스는 라인 간 유지. ASCII 단일 단어 줄은 변경 없이 통과시켜야 함 — `ir`가 출력 없는 줄을 감지하기 위해 내부 sentinel 토큰을 사용. 이 프로토콜을 따르는 실행 파일은 모두 등록 가능.

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
`scripts/bench.sh <dataset>`는 모드별 표(`bm25`, `vector`, `hybrid`)를 출력하고 전체 JSON 결과를 `logs/results/<dataset>/`에 캐시합니다.
BM25가 아닌 벤치마크 실행에서는 래퍼가 tier-2를 전용 확장기 + 재순위기 경로로 고정하고 점수 계산 전에 벤치마크 데몬을 다시 시작합니다. 따라서 로컬 Qwen 통합 GGUF나 이전 데몬 상태 때문에 벤치마크 파이프라인이 조용히 바뀌지 않습니다.
머신이 prepare/index/embed 이후 점수 계산 전에 종료되면, 같은 `scripts/bench.sh <dataset>` 명령을 다시 실행했을 때 처음부터 다시 만들지 않고 준비된 컬렉션에서 재개합니다. 점수 계산도 자동으로 재개됩니다. `scripts/beir-eval.py run --output ...`는 쿼리별 진행 상태를 `<output>.partial/`에 저장하고, 다시 실행하면 그 지점부터 이어서 계산합니다.
대규모 코퍼스는 `--size N --seed N`으로 샘플링한 풀에서 벤치마크할 수 있습니다. MIRACL-Ko는 현재 연구 기본값으로 `scripts/bench.sh miracl-ko --size 50000`을 사용합니다. `10000` docs는 풀 크기 연구상 최소 안정 샘플이지만, 점수가 너무 높아 세밀한 순위 비교에는 포화되는 경향이 있기 때문입니다.
유지보수용 단축 진입점은 `scripts/research-harness.sh`입니다. baseline 고정, signal 수집, threshold sweep, 그리고 shortlist threshold의 holdout 자동 검증 흐름까지 감싸며, threshold 경계 근처를 촘촘히 보려면 `validate-thresholds --products ...`로 정확한 fused 값을 직접 검증할 수 있습니다. 자세한 절차는 [research/experiment.md](research/experiment.md)에 정리되어 있습니다.
현재 연구 방향은 다음과 같습니다.

- `fiqa`는 현재 fused threshold를 유지
- 한국어 threshold 연구는 `miracl-ko --size 50000` 기준으로 진행
- learned router보다 먼저 stricter fused gating을 검증
- router는 holdout에서 단순 threshold gating보다 분명히 나아지기 전까지는 오프라인 연구로만 유지

Tier-2 router 연구는 이 흐름과 분리해 둡니다. 먼저 `bash scripts/signal-sweep.sh --dataset miracl-ko --size 50000 --pools 3 --tier1`로 router용 signal을 수집한 뒤, `bash scripts/router-data.sh ko`로 한국어 전용 `smoltrain` 번들을 만드세요. holdout 평가는 기본 런타임을 바꾸지 않고 `scripts/router-bench.py`로 오프라인에서 진행합니다.
macOS에서는 `scripts/bench.sh`가 기본적으로 안전 감시기와 함께 실행됩니다. 속도를 위해 Metal은 유지하되, 시스템 여유 메모리가 너무 낮아지거나 swapout이 시작되거나 `ir`가 CPU fallback처럼 과도한 CPU를 몇 차례 연속 사용하면 벤치마크를 중단합니다. `IR_BENCH_MIN_FREE_PCT`, `IR_BENCH_MAX_IR_CPU_PCT`, `IR_BENCH_CPU_STRIKES`로 조정할 수 있고, `IR_BENCH_GUARD=0`으로 끌 수 있습니다.

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
