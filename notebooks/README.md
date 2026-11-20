# notebooks/

For your own exploration. Everything in `src/etod` is importable once the project is installed (`pip install -e .`):

```python
import pandas as pd
from etod.utils.paths import latest_run
run = latest_run()
summary = pd.read_csv(run / "metrics" / "summary.csv")
frames = pd.read_csv(run / "per_frame" / "baseline.csv")   # D_t of every frame
```
