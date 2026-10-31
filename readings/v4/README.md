# v4: frozen gateway MMD audit

**Date:** 2026-10-30; independently replayed on 2026-10-31.

This offline audit examines the [v3 gateway observations](../v3/) and explains why the frozen MMD policy did not restrict the A1 extraction campaign. Replaying the recorded calibration reproduced all 60 completed-window p-values for each A1 MMD arm and all 15 in the interrupted A2 arm. The full MMD policy released all 6,000 A1 responses and stayed at L0. These are historical results for one study, not measurements of the current deployment.

The audit found that broad extraction traffic resembled the pooled reference more closely than many topic-focused benign accounts, leaving the upper-tail detector without evidence to escalate. The synthetic benign generator also overrepresented early categories in some calibration pools. Exploratory alternative tail tests did not reliably prevent reordered extraction and require independent validation.

The detailed [audit and source checks](analysis/) explain the findings and their limits. [Results](results/) contain the original diagnostics; [verification](verified-20261031/) records the independent replay. The [v2 A1 and A7 evidence](evidence/v2/) supplies the historical token counts used in the audit. [Method checks](method-checks/) contain the scripts and reproduction instructions. The original report and verification manifest retain their historical paths and hashes; the scripts were adjusted for the current directory layout.
