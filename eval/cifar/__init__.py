"""CIFAR-100 representation-quality evaluation suite.

Each submodule exports a stand-alone CLI that takes one (model, activation,
checkpoint) tuple and writes a JSON result. The shared loader and
feature-extraction utilities live in :mod:`eval.cifar._common`.
"""
