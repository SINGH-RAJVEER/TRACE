# Fixed-checkpoint repeated inference report

Verification ok: True  (21 complete passes: 21, bad records: 0, input hash mismatches: 0)

| model | correct per pass | mean acc | repeat SD | range | correctness agree | exact-output agree | family 95% CI |
|---|---|---|---|---|---|---|---|
| base | 386, 386, 386 | 0.5174 | 0.0000 | 0.5174-0.5174 | 1.0000 | 1.0000 | 0.4812-0.5523 |
| A2/full | 527, 527, 527 | 0.7064 | 0.0000 | 0.7064-0.7064 | 1.0000 | 1.0000 | 0.6743-0.7386 |
| A2/instantaneous | 528, 528, 528 | 0.7078 | 0.0000 | 0.7078-0.7078 | 1.0000 | 1.0000 | 0.6756-0.7399 |
| A5/full | 523, 523, 523 | 0.7011 | 0.0000 | 0.7011-0.7011 | 1.0000 | 1.0000 | 0.6702-0.7332 |
| A5/instantaneous | 548, 548, 548 | 0.7346 | 0.0000 | 0.7346-0.7346 | 1.0000 | 1.0000 | 0.7038-0.7654 |
| unprotected | 548, 548, 548 | 0.7346 | 0.0000 | 0.7346-0.7346 | 1.0000 | 1.0000 | 0.7011-0.7668 |
| teacher | 626, 620, 619 | 0.8333 | 0.0051 | 0.8298-0.8391 | 0.9692 | 0.6005 | 0.8074-0.8597 |

## Paired differences (item-mean, family bootstrap 95%)

| a - b | diff | 95% interval | excludes 0 |
|---|---|---|---|
| A5/instantaneous - A2/full | +0.0282 | +0.0013 to +0.0563 | True |
| A5/instantaneous - A5/full | +0.0335 | +0.0067 to +0.0590 | True |
| unprotected - A5/instantaneous | +0.0000 | -0.0295 to +0.0295 | False |
| unprotected - A2/full | +0.0282 | -0.0027 to +0.0576 | False |
| A2/full - A5/full | +0.0054 | -0.0214 to +0.0308 | False |
| A2/instantaneous - A2/full | +0.0013 | -0.0268 to +0.0295 | False |
| A2/full - base | +0.1890 | +0.1488 to +0.2306 | True |
| teacher - unprotected | +0.0987 | +0.0693 to +0.1287 | True |

## Limitations

- n_items is 746; 3 passes x 746 is not 2238 independent problems.
- Student passes are greedy with fixed weights and are expected to be identical; they measure inference stability only, not training-seed or campaign variability.
- Teacher uses a GGUF llama.cpp server; its differences between passes reflect serving nondeterminism.
- Question-family intervals resample normalized question families and condition on the observed runs; families are not certified problem lineages.
- One training seed (42) per student; paired intervals are uncorrected for multiple comparisons.
