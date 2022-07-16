.. Smarty documentation master file, created by
   sphinx-quickstart on Thu Jul  7 11:36:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Smarty's!
==================================

========
Overview
========
**Smarty** is an open source end-to-end maschine learning python library. Providing both high-level and low-level api, it allows easy development of basic models as well as preprocessing and visualizing the data. Based on *numpy*, it does not support gpu operations, however remains well-scalable, with full control over dtypes and their structure.

.. note::
   This project is under active development, currently each model works under assumption that there is only one target class

-------------
Documentation
-------------
   .. toctree::
      :maxdepth: 3
      
      smarty.datasets
      smarty.models
      smarty.metrics
      smarty.callbacks
      smarty.preprocessing
      smarty.config
      smarty.errors

------------------
Indices and tables
------------------
   1. :ref:`genindex`
   2. :ref:`modindex`
   3. :ref:`search`