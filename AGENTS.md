# Agent Instructions (Codex)

Always read `~/.config/agent/POLICY.md` before doing any work; it defines the shell policy
and required wrappers for git and Beads commands.

## Issue Tracking
This project uses Beads. Use `bd_safe` only (no raw `bd` commands).

```bash
bd_safe ready              # Find available work
bd_safe show <id>          # View issue details
bd_safe update <id> --status in_progress  # Claim work
bd_safe close <id> --reason "<text>"      # Complete work
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git_net pull --rebase
   git_net push
   git_local status -sb  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes; prune remote branches only if explicitly requested
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Do not run raw `git` or `bd` commands; use `git_local`, `git_net`, and `bd_safe` only.
- Work is NOT complete until `git push` succeeds.
- NEVER stop before pushing - that leaves work stranded locally.
- NEVER say "ready to push when you are" - YOU must push.
- If push fails, resolve and retry until it succeeds.
