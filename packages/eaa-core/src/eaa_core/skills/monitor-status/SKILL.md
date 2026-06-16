---
name: monitor-status
description: Monitor a certain object or state according to the instructions of the user using sleep functions. 
---

## Overview

The user may need you to regularly check the status of a certain object or state and
keep monitoring it. To do this, you should repeat a monitoring action repeatedly.
However, repeating the action continuously is undesirable. Instead, you should run
a sleep function in Python or bash to "sleep" for a set interval and let the tool
response "wake you up".

## What you need

- You need to know what the user asks you to monitor. It might be an API endpoint, a
  file, or you might need to control a camera or experiment instrument to collect and
  check data.
- You need either the Python or Bash tool.
- The interval between monitoring actions. This must be specified by the user.
- The number of times to run the monitoring actions. If it is not specified by the
  user, assume it is indefinite.

## Procedure

1. Receive the instructions from the user. If anything is unclear, ask the user. Before
   you start, remind the user to check the approval settings of Python and Bash coding
   tools, as requiring user approval would jam the process. The user should turn
   off approval for these tools to make the process truly automated. For safety, consider
   using sandbox.
2. Perform the monitoring action once. 
3. Either use the Bash tool to run `sleep` for the user specified interval, or call
   the `time.sleep` function in the Python tool.
4. When you see the tool response after the sleep time is over, run the monitor action
   again. 
5. Repeat the process until the user-specified run count is reached.

## Hints
- If the monitoing action is very complicated, spawn a sub-agent to perform it.
