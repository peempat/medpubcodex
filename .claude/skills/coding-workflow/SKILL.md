# Claude Code Techniques & Best Practices

This document summarizes practical techniques for using **Claude Code** effectively, with a focus on an agentic workflow:
**Explore → Explain → Plan → Implement → Test → Verify → Iterate**

---

## 1. Do Not Start by Asking Claude to Write Code Immediately

Before asking Claude to modify or add code, let it inspect the codebase and understand the system first.

### Example Prompt

```text
Explore this codebase and explain how the system works.
Do not modify anything yet.
```

or

```text
Explore this project first.

Explain:
- project structure
- main components
- data flow
- important dependencies
- entry points

Do not modify any files.
```

### Why this matters

If Claude does not understand the architecture and starts editing immediately, several problems can occur:

- Editing the wrong file
- Duplicating existing logic
- Breaking dependencies
- Changing existing behavior unintentionally
- Fixing symptoms instead of the root cause

---

# 2. Use Claude Code to Understand the Codebase

Claude Code is useful not only for writing code, but also for understanding an existing system.

### Example

```text
Explain how this feature works from start to finish.
```

```text
Trace the execution flow when this API endpoint is called.
```

```text
Where is this function used?
```

```text
Why does this function have so many parameters?
```

```text
Explain the relationship between these modules.
```

Claude can help with:

- Reading multiple files
- trace function calls
- Inspecting dependencies
- Reviewing Git history
- Analyzing the architecture
- Explaining code flow

---

# 3. Plan Before You Implement

For large tasks, do not let Claude start coding immediately.

Ask it to create a plan first.

### Prompt

```text
Analyze the task and create an implementation plan.

Do not modify any files yet.
```

or

```text
Explore the codebase first.

Then create a detailed implementation plan.

Wait before implementing.
```

Recommended workflow:

```text
Explore
   ↓
Understand
   ↓
Plan
   ↓
Review
   ↓
Implement
```

---

# 4. Use Explore → Plan → Implement

One of the most effective workflows is:

```text
Explore
   ↓
Explain
   ↓
Plan
   ↓
Implement
   ↓
Test
   ↓
Verify
```

Example Prompt:

```text
First explore the codebase.

Then explain how the relevant components work.

After that create an implementation plan.

Only then implement the change.
```

---

# 5. Have Claude Find the Root Cause Before Fixing a Bug

Do not ask Claude to fix a bug immediately.

Ask Claude to identify the true root cause first.

### Prompt

```text
Investigate the root cause first.

Do not modify any code yet.

Explain exactly why this bug happens.
```

or

```text
Find the root cause of this issue.

Provide evidence from the code before proposing a fix.
```

This reduces trial-and-error debugging.

---

# 6. Ask Claude to Show Evidence

Before changing the code, ask Claude to show where the issue comes from.

### Prompt

```text
Before changing anything, identify the exact files,
functions, and logic responsible for this behavior.
```

or

```text
Show evidence from the code that supports your diagnosis.
```

Evidence can include:

- file
- function
- code path
- config
- dataset
- git commit
- test result
- runtime output

---

# 7. Use a Feedback Loop

Claude performs much better when it can evaluate its own work.

Workflow:

```text
Generate
   ↓
Run
   ↓
Observe
   ↓
Evaluate
   ↓
Fix
   ↓
Repeat
```

Example:

```text
Implement the fix.

Then run the relevant tests.

Inspect failures.

Fix the remaining issues.

Repeat until the tests pass.
```

---

# 8. Generate + Verify

Do not let the AI only generate.

Make it verify its own output as well.

Bad workflow:

```text
Prompt
  ↓
Generate Code
  ↓
Done
```

Better workflow:

```text
Prompt
  ↓
Generate
  ↓
Run
  ↓
Verify
  ↓
Fix
  ↓
Run Again
```

---

# 9. Use Tests as Feedback

After modifying code, always have Claude run the relevant tests.

### Prompt

```text
After implementing the change:

1. Run the relevant tests.
2. Inspect any failures.
3. Identify the cause.
4. Fix the problem.
5. Run the tests again.
```

or

```text
Do not consider the task complete until the tests pass.
```

---

# 10. Have Claude Inspect the Actual Output

Do not look only at test pass/fail status.

Also have Claude inspect the actual runtime output.

### Prompt

```text
Run the program and inspect the actual output.

Check whether the behavior matches the expected result.
```

Useful for:

- benchmark
- ML
- API
- data pipeline
- CLI
- web app

---

# 11. UI Feedback Loop

For web/UI work, Claude can also inspect visual output.

Workflow:

```text
Implement UI
   ↓
Run app
   ↓
Capture screenshot
   ↓
Compare with design
   ↓
Fix
   ↓
Repeat
```

Example Prompt:

```text
Implement the UI.

Run the app.

Capture a screenshot.

Compare it with the target design.

Fix the visual differences.

Repeat until they are minimal.
```

---

# 12. Break Large Tasks into Agent-sized Tasks

Do not put an overly large task into a single prompt.

Bad:

```text
Fix my entire project.
```

Better:

```text
1. Inspect data loading.
2. Inspect preprocessing.
3. Inspect model inference.
4. Inspect output parsing.
5. Inspect evaluation.
```

Claude reasons more accurately when the task is clear and the scope is smaller.

---

# 13. One Agent = One Responsibility

You can split responsibilities across multiple Claude sessions.

Example:

```text
Claude Session 1
→ Backend

Claude Session 2
→ Frontend

Claude Session 3
→ Tests

Claude Session 4
→ Bug Investigation
```

or

```text
Agent 1 → Implement
Agent 2 → Test
Agent 3 → Review
```

---

# 14. Use a Reviewer Agent

After one Claude session modifies the code, use another session to review it.

### Prompt

```text
Review these changes as a senior engineer.

Look for:
- bugs
- edge cases
- regressions
- security risks
- unnecessary complexity
```

The advantage is that the second agent is not anchored to the first agent's reasoning.

---

# 15. Let Claude Use Git

Claude Code can work directly with a Git workflow.

Example:

```text
Review the current git diff.
```

```text
Explain what changed in this branch.
```

```text
Create a commit for these changes.
```

```text
Commit and push this.
```

---

# 16. Use Git History to Investigate Problems

Claude can use:

```bash
git log
git diff
git blame
```

to identify when a behavior changed.

### Prompt

```text
Inspect git history and identify when this behavior changed.
```

```text
Find the commit that introduced this bug.
```

---

# 17. Use CLAUDE.md

The `CLAUDE.md` file stores project context and project-specific rules.

Example:

```md
# Project

Medical LLM Benchmark

## Architecture

- Python
- Hugging Face Transformers
- PyTorch

## Commands

Run benchmark:

python benchmark.py

Run tests:

pytest tests/

## Rules

- Never modify dataset labels
- Use deterministic inference
- Always save raw outputs
- Do not change evaluation metrics without approval
```

---

# 18. What to Put in CLAUDE.md

Recommended sections include:

## Project Overview

```md
## Project Overview

This project benchmarks quantized medical LLMs.
```

## Architecture

```md
## Architecture

Dataset
→ Prompt Builder
→ Model
→ Generation
→ Answer Extraction
→ Evaluation
```

## Commands

```md
## Commands

Run:

python benchmark.py

Test:

pytest tests/
```

## Coding Rules

```md
## Rules

- Use type hints
- Follow existing architecture
- Do not duplicate functions
```

## Important Files

```md
## Important Files

src/model.py
src/evaluation.py
src/datasets.py
```

---

# 19. Do Not Make CLAUDE.md Too Long

Too much context can cause:

- Higher token usage
- Important instructions can get buried
- The context window is consumed unnecessarily

Keep only information that Claude needs frequently.

---

# 20. Context Engineering

Claude performs better when it has the right context.

Useful context includes:

- architecture
- coding conventions
- tools
- dependencies
- commands
- constraints
- test procedures
- expected behavior

Principle:

```text
Better Context
      ↓
Better Reasoning
      ↓
Better Code
```

---

# 21. Use CLI Tools

Claude Code can invoke command-line tools such as:

```text
git
docker
pytest
npm
kubectl
python
```

Example Prompt:

```text
Run the existing test suite and inspect the failures.
```

or

```text
Use the project's CLI tools to verify the implementation.
```

---

# 22. Let Claude Learn a Tool from --help

If Claude does not know a command,

you can have it inspect the CLI documentation.

```text
Use --help to understand this CLI before using it.
```

For example:

```bash
tool --help
```

---

# 23. Use MCP Tools

Claude Code can connect to external systems through MCP.

For example::

- database
- browser
- issue tracker
- internal tools
- documentation
- APIs

This lets the agent access information outside the codebase.

---

# 24. Restrict Permissions for Dangerous Commands

Claude Code can execute Bash commands.

So permissions should be configured carefully.

Be especially careful with commands such as:

```bash
rm -rf
git reset --hard
DROP DATABASE
docker system prune
```

Example rule:

```text
Ask before:
- deleting files
- resetting git history
- modifying production systems
- deleting databases
```

---

# 25. Auto-approve Only Safe Tasks

Tasks that are usually safe to let Claude do automatically include:

- read files
- search code
- run tests
- inspect git
- run formatter

Tasks that should require approval include:

- delete
- deploy production
- modify database
- overwrite important files

---

# 26. Use Escape to Stop the Agent

If Claude is going in the wrong direction, you can stop it.

Concept:

```text
Agent working
     ↓
Wrong direction
     ↓
Stop
     ↓
Give correction
     ↓
Continue
```

You do not need to restart the entire session.

---

# 27. Resume Session

For long tasks, you can resume the same session later.

Concept:

```text
Start Work
   ↓
Pause
   ↓
Resume
   ↓
Continue with existing context
```

This reduces the need to explain the project again.

---

# 28. Use Claude Code Like a Unix Tool

Claude Code can be used as part of a CLI pipeline.

Example:

```bash
git status | claude -p "Summarize these changes"
```

or

```bash
cat error.log | claude -p "Explain the root cause"
```

Concept:

```text
Unix Tool
   +
LLM Intelligence
```

---

# 29. Use Structured Output

For automation, it is better to use structured output.

Example:

```text
Return JSON with:

{
  "status": "",
  "root_cause": "",
  "files": [],
  "recommended_fix": ""
}
```

This makes the result easier to consume in scripts.

---

# 30. Parallel Claude Sessions

Power users can run multiple Claude Code sessions in parallel.

Example:

```text
Terminal 1 → backend
Terminal 2 → frontend
Terminal 3 → tests
Terminal 4 → documentation
```

This reduces waiting caused by sequential work.

---

# 31. Use Git Worktrees

If multiple Claude agents are working on the same repository,

you can use Git worktrees to separate their workspaces.

Concept:

```text
main repo

├── worktree-backend
├── worktree-frontend
├── worktree-tests
└── worktree-refactor
```

Each Claude session can work independently with fewer conflicts.

---

# 32. Compare Before / After

Every optimization should have a baseline.

Example:

```text
Before:
accuracy = 65%
speed = 15 tokens/sec

After:
accuracy = 66%
speed = 28 tokens/sec
```

### Prompt

```text
Compare the system before and after the change.

Report measurable differences.

Do not claim improvement unless the data supports it.
```

---

# 33. Do Not Let Claude Make Claims Without Evidence

A very useful prompt:

```text
Do not claim that the issue is fixed unless you verify it with tests or output.
```

or

```text
Do not assume success.

Verify the result.
```

---

# 34. Have Claude Check Edge Cases

Prompt:

```text
Identify edge cases that could break this implementation.
```

or

```text
Test normal cases, boundary cases, and failure cases.
```

---

# 35. Have Claude Check for Regressions

```text
Check whether this fix could break existing behavior.
```

or

```text
Run regression tests after the change.
```

---

# 36. Make the Smallest Necessary Change

When fixing a bug, avoid large unrelated refactors.

### Prompt

```text
Make the smallest safe change that fixes the issue.

Avoid unrelated refactoring.
```

Benefits:

- Easier to review
- Fewer regressions
- Easier to debug
- Smaller Git diff

---

# 37. Have Claude Follow the Existing Style

```text
Follow the existing project architecture and coding style.

Do not introduce new patterns unless necessary.
```

This helps keep the code consistent with the existing project.

---

# 38. Use Claude as a Debugging Partner

Instead of asking:

```text
Fix this.
```

Use:

```text
Investigate this issue like a debugging engineer.

Form hypotheses.

Test each hypothesis.

Use evidence to eliminate incorrect explanations.
```

This encourages more systematic reasoning.

---

# 39. Hypothesis-driven Debugging

Workflow:

```text
Observe problem
     ↓
Create hypotheses
     ↓
Test hypothesis
     ↓
Collect evidence
     ↓
Eliminate hypothesis
     ↓
Find root cause
```

Example:

```text
Generate the three most likely causes.

Test each one systematically.

Do not modify code until the cause is confirmed.
```

---

# 40. Have Claude Analyze Failures in Groups

If there are many errors, do not fix them one by one blindly.

### Prompt

```text
Group the failures by root cause.

Identify the highest-impact problem.

Fix that first.
```

This is better for identifying patterns than fixing failures randomly.

---

# 41. Run a Smoke Test Before a Full Test

For large tasks, such as an ML benchmark,

do not run the full dataset immediately.

Use:

```text
Small Sample
   ↓
Smoke Test
   ↓
Verify Pipeline
   ↓
Full Benchmark
```

Example:

```text
Run a 10-sample smoke test first.

Verify that the pipeline works correctly.

Only then run the full benchmark.
```

---

# 42. Inspect the Pipeline Stage by Stage

Useful for data, ML, and benchmark pipelines.

```text
Dataset
   ↓
Prompt
   ↓
Tokenization
   ↓
Model
   ↓
Generation
   ↓
Parsing
   ↓
Evaluation
```

Have Claude inspect every stage.

---

# 43. Example: Debug Benchmark Accuracy = 0%

Prompt:

```text
Explore the entire benchmark pipeline.

The model accuracy is unexpectedly 0%.

Do not modify code yet.

Trace:

1. Dataset loading
2. Label mapping
3. Prompt construction
4. Tokenization
5. Model generation
6. Output truncation
7. Answer extraction
8. Prediction normalization
9. Accuracy calculation

Inspect several raw examples.

Identify the root cause with evidence.

Then propose the smallest safe fix.
```

---

# 44. Inspect Raw Output

For LLM benchmarks, raw generations must be inspected.

Example:

```text
Show 20 raw model outputs.

Compare:

expected answer
predicted answer
parsed answer

Identify parsing failures.
```

This can reveal:

- truncation
- invalid formatting
- hallucinated text
- wrong label extraction

---

# 45. Check for Truncation

An LLM may answer correctly but get truncated before the parser can read the answer.

Prompt:

```text
Inspect whether outputs are being truncated.

Check:
- max_new_tokens
- stop tokens
- EOS handling
- prompt length
- context length
```

---

# 46. Verify Label Mapping

MCQ benchmarks often suffer from mapping mismatches such as:

```text
A/B/C/D
```

versus

```text
0/1/2/3
```

Prompt:

```text
Verify the mapping between dataset labels,
prompt choices, model answers, and evaluation labels.
```

---

# 47. Inspect the Parser

Example model output:

```text
The correct answer is B.
```

The parser should extract:

```text
B
```

rather than comparing the entire sentence.

Prompt:

```text
Inspect the answer extraction logic.

Test it against different realistic model outputs.
```

---

# 48. Deterministic Testing

For benchmark testing, reduce randomness as much as possible.

Example:

```python
do_sample = False
```

and keep the configuration consistent.

Claude can help verify whether the experiment is reproducible.

---

# 49. Performance Verification

Optimization should be evaluated using multiple metrics.

For example::

```text
Accuracy
Latency
Tokens/sec
VRAM
RAM
Model size
```

Prompt:

```text
Measure performance before and after optimization.

Report:

- accuracy
- tokens/sec
- latency
- GPU memory
- model size
```

---

# 50. Prompt Patterns Worth Remembering

Core formula:

```text
Explore
→ Explain
→ Plan
→ Implement
→ Test
→ Verify
→ Iterate
```

Or, in short:

```text
Understand before changing.
Verify before claiming success.
```

---

# Master Prompt

This prompt can be used as a template for large coding tasks.

```text
Explore the relevant parts of this codebase first.

Do not modify anything yet.

Understand:
- architecture
- execution flow
- relevant files
- dependencies
- existing tests

Then analyze the requested task.

Identify possible risks and edge cases.

Create a minimal implementation plan.

After that, implement the smallest safe change.

Then:

1. Run the relevant tests.
2. Inspect failures.
3. Check actual runtime output.
4. Verify expected behavior.
5. Check for regressions.
6. Fix remaining issues.
7. Repeat until the result is verified.

Do not claim the task is complete unless the evidence supports it.

Avoid unrelated refactoring.
Follow the existing project style and architecture.
```

---

# Master Debugging Prompt

```text
Investigate this issue systematically.

Do not modify code yet.

First:

1. Reproduce the problem.
2. Trace the execution path.
3. Identify the relevant files and functions.
4. Generate possible root-cause hypotheses.
5. Test each hypothesis using code, logs, tests, or runtime output.
6. Eliminate unsupported hypotheses.
7. Identify the root cause with evidence.

Then propose the smallest safe fix.

After implementing:

1. Run the original failing case.
2. Run relevant tests.
3. Test edge cases.
4. Check for regressions.
5. Inspect actual output.

Do not claim the issue is fixed unless it is verified.
```

---

# Master Benchmark Debugging Prompt

```text
Explore the entire benchmark pipeline.

Do not modify the code yet.

Trace the full pipeline:

Dataset
→ Label Mapping
→ Prompt Construction
→ Tokenization
→ Model Inference
→ Generation
→ Output Truncation
→ Answer Extraction
→ Prediction Normalization
→ Evaluation

Inspect raw examples from each stage.

Check for:

- incorrect labels
- prompt formatting issues
- context truncation
- generation truncation
- invalid answers
- parser failures
- answer mapping errors
- metric calculation bugs

Group failures by root cause.

Identify the highest-impact issue using evidence.

Create a minimal fix plan.

After implementing the fix:

1. Run a small smoke test.
2. Inspect raw predictions.
3. Compare expected vs generated vs parsed answers.
4. Calculate accuracy.
5. Run the previous failing cases.
6. Compare before/after metrics.

Do not run the full benchmark until the smoke test is verified.

Do not claim improvement unless the measurements support it.
```

---

# Recommended Claude Code Workflow

```text
                 ┌───────────────┐
                 │    Request    │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │    Explore    │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │    Explain    │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │     Plan      │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │   Implement   │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │     Test      │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │    Verify     │
                 └───────┬───────┘
                         ↓
                   Success?
                    /     \
                  No       Yes
                  ↓         ↓
                Fix       Done
                  ↓
                 Test
```

---

# Key Principles

## Principle 1

**Understand before changing.**

Do not modify a system you do not yet understand.

## Principle 2

**Evidence before conclusions.**

Do not conclude a root cause without evidence.

## Principle 3

**Plan before large changes.**

Large tasks should have a plan before implementation.

## Principle 4

**Generate + Verify.**

The AI should verify what it creates.

## Principle 5

**Use feedback loops.**

Test → Observe → Fix → Repeat

## Principle 6

**Small safe changes.**

Change only what is necessary.

## Principle 7

**Measure before claiming improvement.**

Optimization must include before/after metrics.

## Principle 8

**Context matters.**

Claude performs better when it understands the project, rather than seeing only a single isolated prompt.

---

# Short Cheat Sheet

```text
Before Coding
-------------
Explore
Understand
Trace
Plan

During Coding
-------------
Small changes
Follow existing style
Avoid unnecessary refactor

After Coding
-------------
Run
Test
Inspect
Verify
Regression Test

Debugging
-------------
Reproduce
Hypothesize
Test
Evidence
Root Cause
Fix
Verify

Optimization
-------------
Baseline
Change
Measure
Compare

Claude Code
-------------
CLAUDE.md
Git
CLI
MCP
Multiple Agents
Worktrees
Feedback Loops
```

---

# One-line Summary

> Claude Code is most effective when used not merely as a code generator, but as a software engineering agent that can inspect a system, plan, implement, test, verify, and iteratively improve its own work.
