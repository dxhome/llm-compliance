---
name: "git-commit"
description: "Commits and pushes changes to git with an auto-generated commit summary. Invoke when user asks to commit, push, or submit code changes."
---

# Git Commit & Push

This skill commits staged changes and pushes to remote with an auto-generated commit summary.

## Usage

Invoke when user says:
- "commit" / "commit and push" / "提交"
- "push to git" / "push changes"
- "save my changes"

## Workflow

1. **Check git status** - Run `git status` to see staged and unstaged changes
2. **Verify staging** - If any modified or untracked files are **not staged**, stop and list them. Ask the user to review and stage manually before proceeding. Do NOT run `git add`.
3. **Generate commit summary** - Analyze staged changes and create 2-3 sentence summary
4. **Commit** - `git commit -m "<summary>"`
5. **Push** - `git push`

**Important**: 
- Do NOT run `git add` — see global Git Staging Policy in project-context
- Do NOT create any files during skill execution

## Commit Summary Format

The summary must follow this structure:
- **MUST be in 中文（Chinese）**（本项目所有 commit message 一律用中文撰写，便于作者回顾与团队协作）
- First line: one sentence summarizing the overall commit
- Blank line
- One bullet per independent change, each a single sentence

Example:
```
本次提交新增特性 X 并修复 bug Y。

- 新增特性 X 以支持场景 Z
- 修复导致 W 行为异常的 bug Y
- 更新文档以反映新变更
```

## Rules

- Only commit if there are staged changes
- Never commit secrets, keys, or credentials
- Do NOT run `git add` — if unstaged files exist, stop and ask the user to handle them
- Do NOT create any files during skill execution
- If push fails, inform user and suggest checking remote configuration
