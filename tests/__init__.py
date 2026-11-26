"""Unit tests. Standard library only, no GPU, no models, no network:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

They cover the parts of the pipeline whose correctness is logic rather than
model quality - agreement, segmentation, gating, the engine loop's ordering
and recovery - using scripted stand-ins for the recogniser and translators.
Model quality is measured, not asserted; that is what bench/ is for.
"""

import logging

# Several tests make the pipeline fail on purpose. The tracebacks it logs
# while recovering are the behaviour under test, not something to read.
logging.disable(logging.CRITICAL)
