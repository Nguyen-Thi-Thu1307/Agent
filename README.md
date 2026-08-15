# Vacuum world agents

| Agent | What it keeps inside |
|---|---|
| Table-driven | Lookup table from percept sequence to action |
| Simple reflex | Nothing |
| Model-based reflex | Which cells it knows to be clean |
| Goal-based | An explicit goal, plus a plan found by search |
| Utility-based | Expected gain weighed against move cost |
| Learning (model-based) | Estimated regrowth rate per cell |
| Reinforcement learning | Q values for each (percept, action) pair |

The agent perceives only the cell it stands on, so the environment is
partially observable. Score is +1 per clean cell per step, minus the move
cost for each move.

## Files

```
agents.py             environment and the seven agents, no dependencies
app.py                Streamlit interface
machine.svg dirt.svg  drawings shared by both interfaces
```

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

`python agents.py` prints the comparison table without Streamlit.

## Deploy

Push to GitHub, then create an app at share.streamlit.io with `app.py` as the
main file.
