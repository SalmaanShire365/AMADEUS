# AMADEUS demo

About five minutes. Everything runs on your machine, on made-up files for a fictional tea shop, Quill & Kettle.

```bash
make models   # once, about 8 GB
make demo
```

`make demo` builds the TUI, runs `amadeus doctor`, and opens the TUI on a fresh copy of `demo/workspace` in a temp directory. Each run starts clean, and the repo stays untouched. If the doctor reports a problem, it prints the command that fixes it.

In the TUI, `/use <template>` picks a template and fills in its example task, which is the one shown below. Press enter to run it. To run something else, edit the text first.

## 1. Summarize a folder

```
/use folder-summary
Summarize the meeting notes in meeting-notes/ and save the briefing as meeting-notes/SUMMARY.md
```

The model lists the folder and reads each note without asking. Then it stops at a card showing the file it wants to create. Press `y`.

## 2. Turn notes into a spreadsheet, and say no

```
/use notes-to-spreadsheet
Turn expenses.txt into a spreadsheet with columns date, vendor, category, amount, paid_by. Save it as expenses.csv
```

Read the card closely, because this is the point of the demo. The last line of `expenses.txt` says "Theo has the receipt", which doesn't say who paid. If the model fills in `Theo`, press `n`, type the reason, and press enter:

```
leave paid_by empty when a line does not name who paid
```

The model gets your reason and proposes the file again. Approve the corrected version with `y`.

## 3. Draft replies

```
/use draft-replies
Draft a reply to each email in inbox/ using our policy in support-policy.md. Save each draft in drafts/ with the same file name
```

Each draft gets its own card. Press `a` to approve the rest of the task. Nothing is sent; the drafts are just files.

Now check them against `support-policy.md`. Look for a refund refused on a wrong item, and for a promised delivery date.

## 4. See what happened

```
/log
```

This prints the path of the last task's audit log. Quit with `/quit`, then read it:

```bash
amadeus replay
```

Run this from `amadeus/amadeus.sh` if you haven't linked `amadeus` onto your `PATH`. Run it in the same directory, since logs are stored per workspace.

## What to expect

Measured once, on 2026-11-04, on an RTX 5070 Laptop (8 GB) with `qwen3:8b` already loaded: each task took 8 to 18 seconds. Your times will vary, and a cold model load adds to the first one.

Model output changes from run to run, and **it contains mistakes**. In that run, the expense sheet invented vendors, and two of the three replies broke the support policy. The README lists them all. Don't present this as accurate output: what it shows is a local model doing real work where you can see and stop every change. Rehearse before showing anyone.
