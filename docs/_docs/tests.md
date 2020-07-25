# Testing

Tests can be run using the test running tool pytest:

~~~ bash
cd DEERpredict
python -m pytest
~~~

The tests reproduce reference data for four protein systems:
- HIV-1 Protease
- T4 Lysozyme
- Acyl-CoA-Binding Protein
- A discoidal complex of Apolipoprotein A-I

Test systems can be further explored through the Jupyter Notebooks in the `tests/data` folder:
- `HIV-1PR/HIV-1PR.ipynb`
- `nanodisc/nanodisc.ipynb`
- `ACBP/ACBP.ipynb`
- `article.ipynb`
