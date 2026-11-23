# Data

This project uses the **Employment Scam Aegean Dataset (EMSCAD)**, released
publicly for research purposes by the Laboratory of Information & Communication
Systems Security, University of the Aegean:

- 17,880 job ads collected from employment websites (2012-2014)
- hand-annotated: 17,014 real, 866 fraudulent
- reference: van de Ven, Shiaeles et al., University of the Aegean

The raw CSV (~50 MB) is not committed to this repository. To reproduce the
training run, obtain `fake_job_postings.csv` (a public mirror is the GitHub
repository `abbylmm/fake_job_posting`, which redistributes the raw file) and
place it at `data/fake_job_postings.csv`.

If you use this dataset, credit the original authors, not this repository.
