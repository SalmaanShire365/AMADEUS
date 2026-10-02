---
description: Check, commit (conventional commit) and push the current branch
allowed-tools: Bash(make check), Bash(make lint), Bash(make test), Bash(make eval), Bash(git status *), Bash(git diff *), Bash(git log *), Bash(git add *), Bash(git commit *), Bash(git push), Bash(git push -u origin *)
argument-hint: [commit message]
---

Ship the current work. Message (optional): $ARGUMENTS

1. `git status` and `git diff`. Stop if nothing to ship, or if the diff contains swap files,
   venvs, indexes, `.env` files or other secrets.
2. Run `make lint` and `make test`. If `amadeus/rag/` changed, also `make eval` and compare
   with the CLAUDE.md baseline. Stop and report on any failure.
3. Write a conventional commit message (`feat:`, `fix:`, `docs:`, `chore:`, ...). If retrieval
   changed, include the before/after eval numbers in the body. Use $ARGUMENTS if given.
4. Show me the message and the file list, and wait for my OK.
5. Commit, then `git push` (or `git push -u origin <branch>` for a new branch). Never force-push.
