# Data

`raw/MI.data` is the **Myocardial infarction complications** dataset from the UCI Machine
Learning Repository (dataset id 579), licensed CC BY 4.0.

* Source: https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications
* Authors: S.E. Golovenkin, A. Gorban, E. Mirkes, V.A. Shulman, D.A. Rossiev,
  P.A. Shesternya, S.Yu. Nikulina, Yu.V. Orlova, M.G. Dorrer
  (collected in Krasnoyarsk, Russia, 1992-1995)
* 1,700 patients, 124 columns: ID, 111 clinical input features, 12 outcomes
* Comma-separated, **no header row**, missing values are written as `?`
  (column names are defined in `src/config.py`)

If the file is missing, download the zip from the link above and place `MI.data` here.
