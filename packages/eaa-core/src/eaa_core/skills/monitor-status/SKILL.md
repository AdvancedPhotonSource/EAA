---
name: monitor-status
description: Monitor a certain object or state according to the instructions of the user using sleep functions. 
---

## Overview

The user may need you to regularly check the status of a certain object or state and
keep monitoring it. To do this, you should repeat a monitoring action repeatedly.
When observables are obtained from the monitoring action, decide based on the
observables whether the state is normal. If an anomaly is detected, perform response
actions, which may include running a corrective procedure to fix the issue and
notifying the user.

Instead of running the monitoring action continuously without pause, you should run
a sleep function in Python or bash to "sleep" for a set interval and let the tool
response "wake you up".

## What you need

- (Mandatory) The monitoring action: what should be done to obtain the observables of the
  state, process, or object being monitored.
- (Mandatory) The anomaly criteria: if the observables meet these conditions, the state is
  considered normal.
- (Mandatory) The response action: what should be done if the state is abnormal.
- (Mandatory) The interval between monitoring actions.
- The number of times to run the monitoring actions. If it is not specified by the
  user, assume it is indefinite.

If any of the mandatory information is missing, keep asking the user until you obtain
all necessary information.

## Procedure

1. Receive the instructions from the user. If anything is unclear, ask the user. Before
   you start, remind the user to check the approval settings of Python and Bash coding
   tools, as requiring user approval would jam the process. The user should turn
   off approval for these tools to make the process truly automated. For safety, consider
   using sandbox.
2. Perform the monitoring action once. 
3. Decide whether the state is normal based on the observables and the anomaly criteria.
4. If the state is abnormal, perform the response action.
5. If the state is normal, or if the state is abnormal but you have finished the response
   action, enter the sleep phase. Either use the Bash tool to run `sleep` for the user
   specified interval, or call the `time.sleep` function in the Python tool.
6. When you see the tool response after the sleep time is over, run the monitoring action
   again. 
7. Repeat the process until the user-specified run count is reached.

## Hints
- If the monitoring action is very complicated, spawn a sub-agent to perform it.
- If the user's instructions contradict with this document, follow the user instructions.
  For example, the user might ask you to stop and wait for further instructions if an
  anomaly is detected; in that case, stop after step 4 and wait for user input.
