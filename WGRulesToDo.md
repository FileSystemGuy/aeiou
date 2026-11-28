# WGRulesToDo.md

What the MLPerf Storage WG's `Rules.md` (in the WG's repository, not this one) will need to say
once the KV-cache workload's comparability changes. A list to write from, not rules: nothing here
is in force. This file is WG process (`PROJECT_BRIEF.md` §8). Started 2026-11-27 at the user's
request, from the debate recorded in `DESIGN_REVIEW.md` §3.100, the note "Task scheduling that
follows KV residency" and what follows it; each entry says whether it was agreed in that debate,
is proposed, or is open. Nothing below is built.

## 1. Comparability is declared per workload

1. Every workload states which of two kinds of comparability it has, and every published result
   states it too. *(Agreed 2026-11-27.)*
   - **Operation-level** (training, checkpointing): every system under test receives the same
     POSIX operations; the run's fingerprint is known before the run and equal across
     submissions.
   - **Demand-level** (the KV-cache workload on vLLM with LMCache): every system under test
     receives the same sessions; the operations follow from the application's own reaction to
     the order in which the storage completes them.
2. For a demand-level workload, CLOSED means the session trees are fixed for every submitter:
   the sessions, their arrival times, every turn's prompt and its keys, the think and tool
   times. *(Agreed.)*
3. The line between application and solution does not move. What an unmodified application does
   in reaction to timing is the application's; the interposition test stands as written.
   *(Agreed.)*
4. What may follow the run, and nothing else: which tier holds a chunk when it is asked for, and
   so the loads, the repeated stores, and the unlinks. *(Proposed.)*
5. What never follows the run: the demand. No timeout, retry, dropped store, or routing on
   observed queue depth is modelled; each is an invalid run or a reported number. *(Agreed.)*
6. Results of an operation-level workload and of a demand-level workload are never compared with
   one another, nor are byte or operation counts between two submissions of a demand-level
   workload. *(Proposed.)*
7. A run under a backend other than the one the abstract declares remains another workload.
   *(Unchanged; restated.)*

## 2. The score of a demand-level workload

8. The score is the load: the number N of session roots (main-line sessions, each with its
   sub-agents) the system carries at the required SLA. *(Agreed.)*
9. N counts session roots, not users. How many session roots a user has is left to the reader of
   the results; the rules give no conversion. *(Agreed.)*
10. Stated limit: the session mix assumes independent arrivals. Session roots of one user may be
    correlated (bursts, a shared project prompt); the corpus the workload is fitted to cannot
    show it. *(Agreed as a limit to state.)*
11. Session *i* is the same session at every N. *(Proposed.)*
12. The SLA is on the user-visible latency of a turn (from the user's input to the answer, the
    stalls of its sub-agents inside it), at a percentile and a threshold the WG fixes; never on
    each load. **The percentile and threshold are open.** *(The form agreed; the values open.)*
13. Several runs at a load point; the point counts only if every run passes. **The number of
    runs is open.** *(Agreed in form.)*
14. A difference in N smaller than the spread between runs is not a difference. **The step
    between load points and how the spread is stated are open.**
15. Open: how long a load point runs and what part of it is measured (the plan of the present
    case is 51.9 hours); whether the application side is fixed or engines scale with N.

## 3. What a submission of a demand-level workload contains

16. The demand's fingerprint, known before the run, equal for every submission at that N.
    *(Agreed.)*
17. The run's log of the order of the events that touch shared cache state. *(Agreed.)*
18. The result of replaying that log in a dry run: the same operations as the run issued. The
    replay is part of every submission run, not optional and not once for a group of runs; an
    exploration run may leave it out. The reviewer can repeat it from the log. *(Agreed.)*
19. The zero-latency plan of the same case and seed, the ideal-storage reference the run's
    latency is measured against. Never a comparison of operation order. *(Agreed.)*

## 4. When a run is invalid

20. The storage does not return a chunk the benchmark's books hold. *(As now.)*
21. A request finds missing a chunk the run stored earlier: the case's lowest tier is sized to
    hold the run, so this is a mis-sized case. *(Agreed 2026-11-27.)*
22. A would-be drop (staging past its size). *(Decided 2026-11-14.)*
23. The benchmark host's own lateness past its limit, and, once caches are shared across hosts,
    the coordination's latency past its limit. **The limits are to be set again**, the host's
    timing now moving which operations issue and not only when. *(Open.)*
24. Session arrivals are on absolute time; a run whose lag passes the ceiling did not carry the
    load. *(As now; restated.)*

## 5. What a run reports beside the score

25. The times the rule "G2 has no room, so nothing is loaded" fired, and the tokens each would
    recompute. *(Agreed.)*
26. The time requests waited at admission for HBM blocks held by loads in flight. *(Agreed, a
    diagnostic.)*
27. The run's loads and stores per tier beside the zero-latency plan's. *(Proposed.)*

## 6. What is legitimate for a solution

28. A solution's timing shapes the workload it receives (which loads finish first decides what
    is asked next), as in production. This is permitted and is to be said outright. *(Agreed.)*
29. Open: whether any ordering of completions is out of bounds. None is named so far.

## 7. The benchmark host

30. A demand-level workload depends more on the machine running the benchmark than an
    operation-level one does (a cost accepted with regret, 2026-11-27). The rules need minimum
    host requirements and the gates of item 23. *(Open.)*

## 8. What the application side is

31. The workload is vLLM with LMCache at stated versions, its tier sizes, slots, and model fixed
    by the WG; a submitter chooses none of them. *(As decided before; restated.)*
32. Another inference framework is another abstract and another workload, not compared with
    this one. *(Agreed.)*

## 9. Changing these rules

33. "Never learn from the run" is weakened by one input, the order of the events that touch
    shared cache state. It is weakened further only on clear and present evidence that the gain
    in fidelity justifies it. *(Agreed.)*

## 10. Undecided in the design, to be added when decided

34. For a cache shared by engines on several hosts: whether the benchmark's coordinator orders
    its events or the storage's own answer to `stat` decides a hit.
35. Whether a prompt is built from its own agent's earlier prompts alone (the corpus supports
    it: under 0.4 % of blocks come from another branch).
