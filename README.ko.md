# ir

[![crates.io](https://img.shields.io/crates/v/ir-search.svg)](https://crates.io/crates/ir-search)
[![CI - check](https://github.com/vlwkaos/ir/actions/workflows/ci.yml/badge.svg)](https://github.com/vlwkaos/ir/actions/workflows/ci.yml)
[![CI - evaluation](https://github.com/vlwkaos/ir/actions/workflows/ci-evaluation.yml/badge.svg)](https://github.com/vlwkaos/ir/actions/workflows/ci-evaluation.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ENG](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

여러 언어가 섞인 마크다운 지식베이스를 위한 로컬 검색. BM25 + 벡터 + LLM 재순위화를 전부 로컬 머신에서 수행 — 컬렉션당 SQLite 파일 하나, 퍼시스턴트 데몬이 모델을 메모리에 상주시키고, 모든 LLM 출력은 캐시된다. 선택적으로 노트 옆의 소스 코드를 보조 근거로 색인한다.

```bash
brew install vlwkaos/tap/ir          # macOS
cargo install ir-search              # 모든 플랫폼 (바이너리 이름: ir)
```

```bash
ir collection add notes ~/notes      # 컬렉션 등록
ir sync notes                        # 텍스트 인덱스 + 벡터 임베딩
ir search "러스트 메모리 안전성"     # 검색 (데몬 자동 시작)
```

BM25 검색은 모델 없이 동작한다. 벡터/하이브리드 검색은 첫 사용 시 HuggingFace에서 모델을 자동 다운로드한다. 소스 빌드는 Rust 1.80+ 필요; macOS에서는 Metal이 자동 링크되고, Linux GPU 백엔드는 opt-in (`--features llama-cuda|llama-rocm|llama-vulkan`).

## 검색 동작 방식 — 어려운 쿼리에만 비용을 낸다

3개 티어. 각 티어는 **이전 티어가 확실하지 않을 때만** 실행되므로, 대부분의 쿼리는 티어 0 또는 1에서 반환되고 LLM을 건드리지 않는다. 하나의 웜 데몬이 모든 모델을 메모리에 유지하고, 모든 LLM 출력은 SQLite에 캐시된다.

```
query
  │
  ▼  TIER 0 · no model · instant
  BM25 (+ doc-graph neighbours, + documents that matching code links to)
  clear keyword winner? ─── yes ──► results
  │
  ▼  TIER 1 · embedding model · ~50 ms
  BM25  ∥  embed query ─► vector kNN (HNSW)          run in parallel
  fuse, choosing per query:
    a top BM25 passage holds every query word ─► vector + BM25 boost for exact matches
    answers are in another script than the query ─► vector only (cross-language)
    otherwise ─► 0.8 · vector + 0.2 · BM25
  confident? ─── yes ──► results
  │
  ▼  TIER 2 · reranker 0.6B · ~2–5 s
  rerank the top 100 ─► results
```

쿼리마다 융합 방식을 고르므로 키워드 조회는 정확하게 유지되고, 질문은 다른 언어로 쓰인 답도 찾는다. 영어 질문이 한국어 노트와 일치하고, 그 반대도 마찬가지다.

전처리기는 한국어·일본어·중국어 BM25 검색의 성패를 가르는 단계다: 색인 텍스트와 쿼리를 동일하게 형태소 단위로 토큰화하며, 없으면 CJK BM25 점수가 0에 가깝다.

**콜드 vs 웜** (M4 Max): 첫 쿼리는 데몬이 모델을 로드하는 ~3.0초, 이후 모든 쿼리는 ~30ms 왕복 — 콜드 스타트 중에도 BM25는 즉시 응답한다. 파이프라인 설정은 [고급 설정](#고급-설정)에서 컬렉션별로 조정할 수 있다.

## 측정된 품질 (nDCG@10)

각 열은 **쿼리가 그 단계에서 멈췄을 때**의 점수다. `BM25`와 `Vector`는 단일 신호 기준선이고, `Hybrid`는 티어 1이며, `+ Rerank`는 상위 100개에 0.6B 재순위기를 더한다(티어 2). LLM 확장기는 **기본 비활성화**.

| 코퍼스 | BM25 | Vector | Hybrid | + Rerank |
|---|---|---|---|---|
| NFCorpus (en, 3.6k 문서, 323 쿼리) | 0.31 | 0.39 | 0.39 | 0.40 |
| FiQA (en, 57.6k 문서, 648 쿼리) | 0.24 | 0.40 | 0.40 | **0.44** |
| MIRACL-ko 50k (ko, 213 쿼리) | 0.73 | — | 0.92 | **0.96** |
| Allganize RAG-eval-KO (ko, 1.4k 페이지, 298 쿼리) | 0.70 | — | 0.69 | 0.72 |
| 한/일/중/영 혼합, 교차 언어 질문 (19k 패시지, 200 쿼리) | ≈ 0 | 0.78 | **0.78** | **0.80** |
| **중앙값 지연시간** (웜, M-시리즈) | ~1 ms | ~50 ms | ~50–280 ms | ~2–5 s |

교차 언어 질문은 한국어·일본어·중국어 패시지에 대한 영어 질문과 영어 패시지에 대한 한국어 질문을 한 코퍼스(MIRACL)에 섞은 것이다. 이전의 고정 `0.8 · vector + 0.2 · BM25` 융합은 여기서 **0.21**(재순위기 포함 0.71)이었다. BM25가 같은 언어의 근접 오답을 정답 위로 끌어올렸기 때문이다. 위의 단일 언어 결과는 쿼리별 융합으로 바뀌지 않았다(±0.003 이내). 티어 0은 **doc-graph 확장**도 실행하며, 희소 코퍼스(NFCorpus)에서 ≈ +0.02, 밀집 코퍼스(FiQA)에서는 ~중립이다. 한국어 BM25는 `ko` 전처리기가 필요하다 (없으면 ≈ 0). 한국어 `Vector`는 여기서 재측정하지 않았다(—). 방법과 데이터: [research/adr-0002-per-query-fusion.md](research/adr-0002-per-query-fusion.md).

## 버전 한눈에 보기

- **≤ 0.15** — 코어 파이프라인, 데몬, MCP, CJK 전처리기.
- **0.16** — `ir sync` (인덱스 + 임베딩 단일 명령); 삭제된 파일은 제거되고 이동된 콘텐츠는 캐시된 벡터를 재사용.
- **0.17** — opt-in 연구 경로: 문서 그래프와 HNSW ANN 인덱스.
- **0.18** — 그 경로들이 기본값이 된다 (HNSW ANN, 티어-0 그래프 확장, 재순위 윈도우 100); LLM 쿼리 확장기는 호출 에이전트로 이관. [ADR-0001](research/adr-0001-default-retrieval-pipeline.md).
- **다음** — 여러 언어가 섞인 컬렉션을 위한 쿼리별 하이브리드 융합, 병렬 티어 1, 코드 도메인, 다시 쓴 `--help`. [ADR-0002](research/adr-0002-per-query-fusion.md). 기존 컬렉션은 재색인 없이 그대로 동작한다.

## 문서

<details>
<summary><strong>모델</strong></summary>

모델은 첫 사용 시 HuggingFace Hub에서 자동 다운로드된다 (캐시: `~/.cache/huggingface/`). `HF_HUB_OFFLINE=1`로 다운로드를 비활성화한다.

| 모델 | 필요 기능 |
|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | 벡터 / 하이브리드 검색 |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | 재순위화 (선택) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | 쿼리 확장 (선택) |
| [BGE-M3](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) | 한국어 최적화 임베딩 대안 |

**로컬 모델 / 오버라이드:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

`IR_*_MODEL`은 `.gguf` 파일 경로, 알려진 모델이 들어 있는 디렉터리, 또는 HuggingFace 레포 ID를 허용한다. 탐색 순서: 환경변수 → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → HF Hub. `IR_COMBINED_MODEL`(확장+재순위 단일 모델)은 실험 전용 opt-in. 임베딩 모델을 바꾸면 `ir embed --force`가 필요하다.

**설정 디렉터리:**

```bash
export IR_CONFIG_DIR="~/vault/.config/ir"   # 이식 가능; ~ 및 $VAR 확장 지원
```

우선순위: `IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir` (deprecated) → `~/.config/ir`.

**GPU:** `IR_GPU_LAYERS=0`은 CPU 강제, `IR_GPU_LAYERS=N`은 부분 오프로드.

</details>

<details>
<summary><strong>한국어 임베딩 (BGE-M3)</strong></summary>

기본 EmbeddingGemma로도 한국어 하이브리드+재순위 성능은 충분히 높지만, 한국어 특화 dense retrieval이 필요하면 BGE-M3를 대체 임베더로 사용할 수 있다.

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

파일명에 "bge-m3"가 포함되면 CLS 풀링과 쿼리 프리픽스가 자동 적용되고, `ir embed --force` 시 벡터 테이블 차원도 자동 조정된다.

**참고:** 한국어 쿼리 확장(expander)은 비권장 — 영어 SFT 모델이라 MIRACL-Korean에서 오히려 성능이 소폭 저하된다. KURE-v1은 GGUF 변환이 검증되지 않아 실험적 (llama.cpp `convert_hf_to_gguf.py`로 직접 변환 필요).

</details>

<details>
<summary><strong>사용법</strong></summary>

**컬렉션 및 인덱싱:**

```bash
ir collection add notes ~/notes
ir collection ls
ir collection rm notes
ir status                    # 컬렉션별 인덱스 상태

ir sync [notes] [--force]    # 텍스트 인덱스 + 임베딩 (기본 유지보수 명령)
ir update [notes] [--force]  # 텍스트 인덱스만 — 빠름, 모델 불필요
ir embed [notes] [--force]   # 벡터 복구 / 재임베딩
```

인덱싱은 증분·내용 주소 방식(SHA-256)이다. 변경된 파일만 재처리하고, 동일 콘텐츠는 중복 제거하며, 삭제된 파일은 제거하고, 이동/복원된 콘텐츠는 재추론 없이 캐시된 벡터를 재사용한다.

`ir update`는 컬렉션 경로, 스캔된 파일 수, 아직 벡터가 없는 문서 수를 함께 출력한다.

```
updating 'notes' (/Users/me/notes)…
  321 scanned: 1 added, 0 updated, 0 deactivated
  1 pending embedding (run `ir embed` or `ir sync`)
```

`0 scanned`는 경로나 glob에 맞는 파일이 없다는 뜻이다. `ir update --force`는 텍스트 인덱스를 다시 만들면서 컬렉션에 저장된 벡터를 지우므로, `ir embed`나 `ir sync`를 실행하기 전까지 모든 문서가 임베딩 대기 상태가 된다.

컬렉션의 전처리기를 시작할 수 없거나 실행 중에 멈추면, `ir update`는 경고를 출력하고 원문 텍스트로 인덱싱한다. 따라서 모든 파일이 계속 검색된다. 전처리기가 다시 동작하면 다음 `ir update`가 그 파일들을 다시 인덱싱한다. 여러 `ir` 프로세스(CLI 실행, MCP 서버)가 같은 컬렉션을 동시에 업데이트해도 된다. `ir sync`와 `ir embed`는 다른 프로세스가 같은 컬렉션을 임베딩하는 동안 기다린다.

**검색:**

```bash
ir search "러스트 메모리 안전성"                    # 하이브리드 (기본)
ir search "sqlite 아키텍처" --mode bm25            # 모델 불필요
ir search "비동기 패턴" --mode vector
ir search "에러 처리" -c notes --min-score 0.4

ir search "소유권" --json | --md | --files | --full | --chunk | --quiet
ir search "설계" -f "modified_at>=2026-01-01" -f "meta.tags=rust"
ir search "token refresh" --aux                  # 코드 심볼도 문서 뒤에 함께 반환
```

쿼리와 문서의 언어가 달라도 된다. 영어 질문이 `hybrid`와 `vector` 모드에서 한국어 노트를 찾는다. `bm25`는 단어만 일치시킨다.

필터 절(`-f`, 반복 가능, AND 결합): 필드 `path`, `modified_at`, `created_at`, `meta.<name>`; 연산자 `=` `!=` `>` `>=` `<` `<=` `~` `!~`. 날짜는 UTC RFC3339로 정규화된다. 배열 프론트매터 필드는 **어느 한** 요소라도 조건을 만족하면 일치로 처리된다 (`!=` 포함).

**문서 조회:**

```bash
ir get "2026/Daily/2026-04-07.md"              # 정확 → 접미 → 부분 일치
ir get "2026-04-07" -c periodic --section "Log" --max-chars 3000
ir multi-get "a.md" "b.md" --json               # {found, not_found}
```

**데몬:**

```bash
ir daemon start|stop|status   # 첫 검색 시 자동 시작
```

웜 쿼리는 Unix 소켓 왕복 ~30ms. 콜드 스타트에서는 모델이 백그라운드에서 로드되는 동안 첫 쿼리가 BM25 결과를 즉시 반환할 수 있다.

`ir`을 업그레이드하면, 다음 검색이 이전 바이너리로 시작된 데몬을 자동으로 교체한다. 그 쿼리는 콜드 스타트와 같은 방식으로 응답한다. Homebrew와 `cargo install`은 `ir`이 실행 중이어도 업그레이드한다. 바이너리를 직접 복사한다면 덮어쓰지 말고 파일을 교체한다(`mv` 또는 `install`). Linux는 실행 중인 프로그램에 쓰기를 거부한다("Text file busy"). `--mode bm25` 검색은 데몬을 시작하거나 호출하지 않는다. 데몬이 120초 안에 응답하지 않으면 검색은 기다리기를 멈추고 이미 가진 결과를 반환한다.

</details>

<details>
<summary><strong>한국어 / 일본어 / 중국어 전처리기</strong></summary>

CJK 텍스트는 BM25 이전에 형태소 토큰화가 필요하다 — 없으면 교착어 형태가 형태소 단위 쿼리와 전혀 매칭되지 않는다 (한국어 BM25가 ~0.00에서 실용 수준으로 상승). 인덱싱 시와 쿼리 시 동일한 전처리기가 적용된다.

```bash
ir preprocessor install ko    # lindera + ko-dic (공식 바이너리; macOS/Linux)
ir preprocessor install ja    # lindera + ipadic
ir preprocessor install zh    # lindera + jieba
ir preprocessor bind ko wiki  # 컬렉션에 연결하고 재인덱싱
```

`ko` 바인딩 시 측정 기반 한국어 라우팅 기본값(`fused_strong_product: 0.05`)도 해당 컬렉션에 기록된다; 명시적 `routing:` 설정이 항상 우선한다. 컬렉션별 라우팅 오버라이드(`fused_strong_floor/product`, `bm25_strong_floor/gap`)는 `config.yml`에 두며, 검색에 포함된 모든 컬렉션이 같은 값일 때만 적용된다.

어떤 실행 파일이든 전처리기가 될 수 있다: stdin으로 UTF-8 라인 입력 → stdout으로 0 또는 1개의 토큰화된 라인 출력, 라인 간 프로세스 유지, ASCII 단일 단어 라인은 변경 없이 통과. lindera 처리량: M-시리즈 기준 한국어 문서 ~5,600개/초.

**효과** (MIRACL-Korean):

| 전처리기 | BM25 nDCG@10 |
|---|---|
| 없음 | 0.00 |
| lindera (`ko`) | 0.73 (50k 문서 샘플) |

</details>

<details>
<summary><strong>코드 도메인 — 소스 코드를 보조 근거로</strong></summary>

컬렉션 문서 옆의 소스 파일을 함께 색인합니다. 코드 파일은 검색 결과로 나오지 않습니다. 대신 심볼과 doc comment가 그 심볼을 언급한 문서의 순위를 올리거나 결과에 추가합니다.

```bash
ir collection add notes ~/project --glob 'docs/**/*.md' --code-glob 'src/**/*.rs' --code-glob 'src/**/*.py'
ir update notes
ir search "drop outdated values after a migration" --json   # 해당 함수를 언급한 문서를 찾습니다
```

```yaml
collections:
  - name: notes
    path: ~/project
    auxiliary:
      code:
        globs: ["src/**/*.rs", "src/**/*.py"]
        excludes: ["src/generated/**"]   # 선택
```

- 지원 형식: `.rs`, `.py`, `.js`/`.jsx`/`.mjs`/`.cjs`, `.ts`/`.tsx`, `.go` (tree-sitter). 함수, 메서드, 타입, trait/interface, 상수, 모듈을 doc comment와 함께 색인합니다. 함수 본문은 색인하지 않습니다.
- 문서가 심볼 이름을 쓰면 연결됩니다. `snake_case`, `camelCase`, 두 단어 이상의 `PascalCase`는 어디서든, 다른 이름은 백틱 안에서만 연결됩니다. 파일 경로(`src/db/schema.rs` 또는 `db/schema.rs`처럼 유일한 접미사)를 쓰면 파일과 연결됩니다.
- 보조 근거가 있는 결과는 `--json` 출력에 `evidence` 필드가 붙습니다(예: `["code:src/cache.rs:invalidate_entry"]`). 코드 도메인이 없으면 결과와 출력은 이전과 같습니다.
- 컬렉션에서 `auxiliary:`를 지우면 다음 `ir update`에서 코드 색인이 삭제됩니다.
- `ir status`, MCP `status`, MCP 연결 안내문은 코드가 있는 컬렉션에만 `aux: code N`을 표시합니다. 소스 파일이 있는 폴더를 `--code-glob` 없이 `ir collection add`로 추가하면, 추가할 설정을 담은 안내를 한 번 출력합니다.
- 문서가 이름을 쓰지 않는 코드는 명시적 링크로 연결합니다.
  - 문서 frontmatter: `code: [src/keys.rs:rotate_all, src/ledger.rs, parse_config]` (파일 경로나 유일한 접미사, `path:symbol`, 심볼 이름)
  - 코드 주석: `// knowledge: design/ledger.md` 또는 `// [[Ledger design]]` (문서 경로나 제목). 파일 전체에 적용됩니다.
- `ir update`는 아무것도 가리키지 않는 명시적 링크마다 경고합니다(예: 이름 변경 후). 낡은 링크를 바로 고칠 수 있습니다.
- 코드 일치는 쿼리 단어의 절반 이상을 포함할 때만 근거로 씁니다.
- `ir search --aux`(MCP: `aux: ["all"]`, 도메인을 지정하려면 `--aux=code` / `["code"]`)는 일치한 심볼과 파일 자체도 **모든 문서 뒤에** 반환합니다. 결과에는 `"kind": "code"`, 소스 파일 `path`, `code:<path>:<line>` 형식의 `doc_id`가 붙고, snippet은 시그니처와 doc comment입니다. 기본값은 꺼짐입니다. 코드는 보조 정보로 남습니다. 필터(`-f`)는 문서에만 적용됩니다.

</details>

<details>
<summary><strong>MCP 서버 — Claude Desktop / Claude Code</strong></summary>

```json
{ "mcpServers": { "ir": { "command": "ir", "args": ["mcp"] } } }
```

도구: `search` (`mode`, `limit`, `min_score`, `collections`, `filter`, `aux` 지원), `get`, `multi_get`, `status`, `update`.

`update`는 텍스트 전용이다. BM25 인덱스만 갱신하고 임베딩은 계산하지 않는다. 결과에는 컬렉션마다 `path`, `scanned`, `added`, `updated`, `deactivated`, `pending_embeddings`, `notes`가 들어 있다. 대기 중인 문서를 임베딩하려면 터미널에서 `ir embed`나 `ir sync`를 실행한다. `force: true`는 그때까지 컬렉션의 기존 벡터를 지운다.

`update` 결과의 `notes`와 `search` 결과 뒤에 붙는 두 번째 `notes:` 블록에는 `ir`가 터미널에 출력하는 경고가 담긴다. 예를 들어 시작하지 못한 전처리기, 인덱싱되지 않은 컬렉션, 모델 로딩 중이라 BM25 결과만 반환한 경우다. 알릴 내용이 없으면 비어 있거나 생략된다.

원격/멀티 클라이언트용 HTTP 모드:

```bash
ir mcp --http 3620 [--cors '*' | --cors 'https://app.example.com']
```

> HTTP 모드는 인증 없이 전체 인터페이스에 바인딩된다 — 신뢰할 수 있는 네트워크에서만 사용할 것.

</details>

<details>
<summary><strong>벤치마크 및 재현</strong></summary>

위의 모든 수치는 동봉된 하네스로 재현 가능하다:

```bash
scripts/bench.sh nfcorpus            # 모드별 전체 표, git 해시별 캐시
scripts/bench.sh miracl-ko --size 50000 --seed 42
bash scripts/preship.sh              # 픽스처 기반 안정성 / 속도 / 품질 게이트
```

실행은 재개 가능하고 (쿼리별 진행 상태가 크래시를 견딤) macOS에서는 메모리 감시기가 보호한다. 과거 BEIR 결과 (이전 파이프라인 설정): ArguAna에서 재순위화가 순수 벡터 대비 최대 +14.5% nDCG@10; 영어 코퍼스에서 융합 단독은 순수 벡터보다 유의미하게 낫지 않았다 — tier-2의 가치는 재순위기에 있다.

HNSW 인덱스(usearch)는 정확 검색과 top-10 일치율 99.2%, 동일한 nDCG@10을 보였다(MIRACL-ko 50k). 혼합 언어 세트는 MIRACL로 만들었다. 방법, 데이터, 스크립트는 [ADR-0002](research/adr-0002-per-query-fusion.md)에 있다.

</details>

<details>
<summary><strong>qmd와 비교</strong></summary>

ir은 [qmd](https://github.com/tobi/qmd)의 Rust 포트로, 저장소 모델이 다르고 퍼시스턴트 데몬을 갖췄다.

| | qmd | ir |
|---|---|---|
| 저장소 | 단일 SQLite | 컬렉션별 SQLite (`rm name.sqlite`로 삭제) |
| 프로세스 모델 | 쿼리마다 스폰 | 데몬이 모델 상주 유지 |
| LLM 캐시 | 재순위 점수 | 재순위 점수 + 확장기 출력 |
| 콜드 / 웜 쿼리 (M4 Max) | 9.5s / 840ms | **3.0s / 30ms** |

</details>

<details>
<summary><strong>개발 및 스키마</strong></summary>

```bash
cargo build [--release]
cargo test                   # 모델 불필요
cargo test -- --ignored      # 모델 의존 테스트
```

컬렉션별 스키마: `content` (해시 → 텍스트), `documents`, `documents_fts` (FTS5), `vectors_vec` (sqlite-vec, 코사인), `content_vectors` (청크 메타데이터), `llm_cache` (재순위 점수), `document_metadata` (프론트매터), `meta`, `doc_graph`, `ann_keys`, 그리고 코드 도메인의 `artifact_sources`, `artifacts`, `artifact_fts`, `artifact_anchors`, `artifact_edges` (코드 도메인이 없으면 비어 있음). 전역 `expander_cache.sqlite`는 확장 출력을 캐시한다. 스테이지드 비동기 데몬 설계는 [research/pipeline.md](research/pipeline.md) 참고.

</details>

## 고급 설정

검색 파이프라인 동작은 `config.yml`의 `retrieval:` 블록에서 컬렉션별로(그리고 전역으로) 설정한다. 모든 필드는 선택 사항이며 — 생략하면 기본값이 적용된다. 해석 우선순위는 **`config > env > default`**: 여기에 설정한 값이 최종 권한을 가지며, 떠도는 환경변수에 조용히 덮어써지지 않는다.

```yaml
# ~/.config/ir/config.yml

# 전역 (데몬): 인프로세스 LLM 쿼리 확장기 로드 여부
retrieval:
  expander: false            # 기본값 — 확장은 호출 에이전트의 몫

collections:
  - name: notes
    path: ~/notes
    # 컬렉션별 파이프라인 오버라이드
    retrieval:
      ann: true              # 벡터 kNN용 HNSW ANN (미빌드 시 정확 검색으로 폴백)
      t0_graph_expand: true  # 티어-0 doc-graph 시드 확장
      rerank_window: 100     # 재순위기로 보내는 후보 수
      rerank_keep_window: true
```

| 키 | 범위 | 기본값 | 의미 |
|---|---|---|---|
| `ann` | 컬렉션 | `true` | 벡터 검색용 HNSW ANN 인덱스; 없거나 오래되면 정확 브루트포스로 폴백. |
| `t0_graph_expand` | 컬렉션 | `true` | BM25 시드가 `doc_graph` 이웃을 티어-0 후보 목록으로 끌어온다. |
| `rerank_window` | 컬렉션 | `100` | 티어-2 재순위기로 보내는 후보 수. |
| `rerank_keep_window` | 컬렉션 | `true` | 판정된 문서를 미판정 꼬리보다 위에 유지. |
| `expander` | 전역 | `false` | 인프로세스 LLM 쿼리 확장기 로드; 아니면 확장은 호출자에게 위임. |

함께 검색되는 컬렉션들은 컬렉션별 값이 **일치**해야 적용된다; 충돌하면 기본값으로 폴백한다 (`routing:`과 동일한 규칙). `ir sync` / `ir embed`는 해당 노브가 켜져 있을 때 ANN 인덱스와 doc graph를 빌드한다 (빈 컬렉션은 건너뜀). 컬렉션을 0.18 이전 파이프라인으로 되돌리려면 `ann: false`, `t0_graph_expand: false`, `rerank_window: 20`, `rerank_keep_window: false`, 그리고 전역 `expander: true`로 설정한다.

**대체 설정 파일** — 데이터 디렉터리를 옮기지 않고(컬렉션과 캐시는 그대로) 다른 `config.yml`을 `ir`에 지정하여, 하나의 임베딩된 코퍼스 위에서 여러 파이프라인 설정을 비교할 수 있다:

```bash
ir --config-path ./variant.yml search "query" -c notes
```

우선순위: `--config-path` > `IR_CONFIG_FILE` > `<config-dir>/config.yml`.

## 라이선스

[MIT](LICENSE)
