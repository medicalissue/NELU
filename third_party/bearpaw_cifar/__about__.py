"""Provenance metadata for files vendored from bearpaw/pytorch-classification.

Only files we actually use are copied in; the rest of the upstream repo is
omitted. License is kept alongside as ``LICENSE`` (MIT, Copyright © 2017
Wei Yang).

Upstream:    https://github.com/bearpaw/pytorch-classification
License:     MIT (see LICENSE)
Commit SHA:  24f1c456f48c78133088c4eefd182ca9e6199b03
Vendored:    2026-04-24

Files used:
    models/cifar/resnet.py    →  resnet.py       (verbatim)
    models/cifar/densenet.py  →  densenet.py     (local modification, see
                                                  the header note in that
                                                  file — Bottleneck's
                                                  single ``self.relu`` is
                                                  split into two distinct
                                                  modules ``relu1`` /
                                                  ``relu2`` so each call
                                                  site gets its own
                                                  Gate-Normalization axes
                                                  after activation swap).
"""

COMMIT_SHA = "24f1c456f48c78133088c4eefd182ca9e6199b03"
UPSTREAM = "https://github.com/bearpaw/pytorch-classification"
