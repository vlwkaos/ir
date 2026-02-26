---
name: codex
description: Use when the user asks to run Codex CLI (codex exec, codex resume) or references OpenAI Codex for code analysis, refactoring, or automated editing
context: fork
agent: general-purpose
---

# Codex Skill Guide

## Running a Task
1. Ask the user (via `AskUserQuestion`) for reasoning effort (`xhigh`, `high`, `medium`, or `low`).
2. Select sandbox mode for the task; default to `--sandbox read-only` unless edits or network access needed.
3. Assemble the command:
   - `-m, --model <MODEL>`
   - `--config model_reasoning_effort="<xhigh|high|medium|low>"`
   - `--sandbox <read-only|workspace-write|danger-full-access>`
   - `--full-auto`
   - `-C, --cd <DIR>`
   - `--skip-git-repo-check`
4. Always use `--skip-git-repo-check`.
5. When continuing a previous session: `echo "your prompt" | codex exec --skip-git-repo-check resume --last 2>/dev/null`
6. **Always** append `2>/dev/null` to suppress thinking tokens unless user asks for them.
7. Run command, capture output.

### Quick Reference
| Use case | Sandbox | Key flags |
|---|---|---|
| Read-only review | `read-only` | `--sandbox read-only 2>/dev/null` |
| Apply local edits | `workspace-write` | `--sandbox workspace-write --full-auto 2>/dev/null` |
| Network/broad access | `danger-full-access` | `--sandbox danger-full-access --full-auto 2>/dev/null` |
| Resume session | Inherited | `echo "prompt" \| codex exec --skip-git-repo-check resume --last 2>/dev/null` |
| Run from other dir | Match task | `-C <DIR>` plus other flags `2>/dev/null` |

## Return Output to Main Agent

After codex completes, collect and return a structured summary:

```bash
git diff --stat        # changed files
git diff --name-only   # file list
```

Return format:
```
## Codex Result

**Status**: success | partial | failed
**Changed files**:
- path/to/file (added|modified|deleted)

**Summary**: [1-3 sentences of what codex did]

**Output** (truncated if long):
[relevant stdout]
```

## Following Up
- After every `codex` command, use `AskUserQuestion` to confirm next steps or decide whether to resume.
- When resuming: `echo "new prompt" | codex exec resume --last 2>/dev/null`

## Critical Evaluation of Codex Output

Codex is powered by OpenAI models with their own knowledge cutoffs. Treat as a **colleague, not an authority**.

- **Trust your own knowledge** when confident — push back if Codex is wrong.
- **Research disagreements** via WebSearch before accepting Codex's claims.
- **Knowledge cutoffs** — Codex may not know recent releases or API changes.
- When disagreeing, identify yourself as Claude and your current model name:
  ```bash
  echo "This is Claude (<model>) following up. I disagree with [X] because [evidence]." | codex exec --skip-git-repo-check resume --last 2>/dev/null
  ```

## Error Handling
- Stop and report on non-zero exit from `codex --version` or `codex exec`.
- Ask permission before high-impact flags (`--full-auto`, `--sandbox danger-full-access`) unless already given.
- Summarize warnings/partial results and ask how to adjust.

## Model Override

Always use `codex-mini-latest`. Do not ask the user which model to use.
