# 125.785 Research and Analytics in Economics and Finance – Python Labs

UEH–Massey Master of Finance · Lecturer: Nguyễn Phương Quỳnh

Practical work for this course uses **Python on Google Colab**. You do not need to install anything.

## Start a session
1. Open the notebook for the session in `notebooks/` and click **Open in Colab**
   (or go to colab.research.google.com → File → Open notebook → GitHub → paste this repository's address).
2. **File → Save a copy in Drive** before you start.
3. Run the setup cell first, then work from top to bottom.

| Session | Topic | Notebook | Data |
|---|---|---|---|
| 1 | Getting started: Python, Colab and financial data | `S01_student.ipynb` | vn_corp_panel |
| 2 | Simple regression: the CAPM beta | `S02_student.ipynb` | vn_stock_returns_monthly |
| 3 | Multiple regression: estimation and inference | `S03_student.ipynb` | vn_corp_panel |
| 4 | Multiple regression: diagnostics | `S04_student.ipynb` | vn_corp_panel |
| 5 | Panel data I: pooled, FE, RE | `S05_student.ipynb` | vn_bank_panel |
| 6 | Panel data II: Fama–MacBeth, two-way FE | `S06_student.ipynb` | vn_stock_returns_monthly, vn_corp_panel |
| 7 | Binary outcome models: logit and probit | `S07_student.ipynb` | vn_corp_panel |
| 8 | Univariate time series: ARIMA | `S08_student.ipynb` | vn_macro_quarterly |
| 9 | Multivariate time series: VAR | `S09_student.ipynb` | vn_macro_quarterly |
| 10 | Event studies: earnings announcements | `S10_student.ipynb` | vn_earnings_events |
| 11 | Endogeneity: FE and instrumental variables | `S11_student.ipynb` | vn_corp_panel |
| 12 | Forecasting and forecast evaluation | `S12_student.ipynb` | vn_macro_quarterly |

The full guide is `docs/125785_Python_Lab_Book_Student.docx`.
Companion text: Wichmann & Brooks, *Python Guide for Introductory Econometrics for Finance* (Cambridge University Press).

## Data
The five CSV files in `data/` are **simulated for teaching** (seed 125785, see `data/make_data.py`).
They mimic Vietnamese listed firms, banks and markets but are not real company data. Do not cite them as evidence.

## Your notebook is your log
For A1/A2, submit your notebook (.ipynb) and a PDF of it, run from top to bottom, so that every number in your tables can be re-produced.
