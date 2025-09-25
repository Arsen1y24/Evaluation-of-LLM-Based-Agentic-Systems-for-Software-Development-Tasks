# ReAct Python Bug-Fixing Agent

This project implements a **ReAct-style agent** that automatically fixes buggy Python functions and verifies them against tests.
The agent uses a sandboxed environment for safe execution and can run **fully offline** after setup.

---

## Features

- Automatic bug-fixing for Python functions
- Sandbox execution with `RestrictedPython`
- Iterative testing and reasoning with a ReAct agent
- Works offline after initial setup
- Logs all runs to `???`

---

##  Quick Start

### Setup Conda Environment

<button style="background:#4CAF50;color:white;padding:8px 12px;border:none;border-radius:4px;cursor:pointer;">
💻 Linux/MacOS
</button>

```bash

chmod +x setup.sh
./setup.sh
