# Unprotected

The unprotected run, taken from the A6 schedule (formerly `campaigns/A6/unprotected/`, moved without changing any file): all 6,000 requests were served with no intervention, releasing 6,000 replies and 806,570 output tokens. They include 1,500 GSM8K replies, of which 1,473 finished and 1,402 are correct against the gold answers.

It was collected through the quota arm, whose quota never fired on A6 (every decision is level 0, `serve`). Its files are therefore identical to `../quota/`, and `harness.env` names the quota policy.

Its GSM8K prompts include every GSM8K prompt the A5 arms received, so it is the unprotected upper bound for the A5 student comparison.
