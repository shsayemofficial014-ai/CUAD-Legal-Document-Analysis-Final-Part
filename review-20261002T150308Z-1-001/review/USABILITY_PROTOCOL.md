# Minimal usability check — 15 minutes per participant, 2 participants

This is deliberately the smallest thing that would let the thesis say something concrete
instead of "no user has ever been observed using the application". It is **not** a study
and must not be written up as one. Two participants, fifteen minutes each.

## Who
Two people who are **not** law students and **not** on this project. A friend from
another department is ideal — the target user is a non-lawyer signing a contract.

## Setup (2 min)
1. Run the app: `streamlit run app.py`
2. Load `test_contract.txt` (the synthetic Master Software License agreement).
3. Say only this: *"This tool reads a contract and tells you what to watch out for.
   Have a look and tell me what you'd do next."* Then stop talking.

## Observe (8 min) — write down, do not prompt
- Where do they look first? Do they read the High-risk group before the others?
- Do they click into any clause card, or only read the badges?
- Do they touch the confidence slider? Do they say anything about what the number means?
- Do they at any point treat the quoted clause text as authoritative?
- Anything they say out loud that suggests they trust it more than they should.

## Ask (5 min) — four questions, record the answers verbatim
1. In your own words, what did this tell you about the contract?
2. If this were your contract, what would you do next?
3. Was there anything you didn't believe, or weren't sure about?
4. Would you sign a contract on the strength of this? Why or why not?

## What to report in the thesis (Section 7.6 and 8.2)
Write **three to five sentences**, no more, and be explicit about the limits:

> We ran an informal walkthrough with two non-lawyer participants (15 minutes each),
> which is far too small to be a usability study and is reported only to replace an
> absence of evidence with a little. [What they did.] [What they said.] [Anything that
> suggested over-trust.] No task-completion or comprehension measurement was attempted,
> and with n=2 no quantitative claim is possible.

**The single most valuable thing to look for is over-trust** — a participant treating the
quoted clause text as reliable, or reading the confidence bar as a probability. Chapter 6
shows both are unwarranted (ECE 0.201; the quoted text matches the real clause in about
one case in five overall). If a participant does that, it is a finding worth reporting,
and it belongs in the ethics discussion as well.
