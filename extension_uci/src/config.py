"""Project-wide configuration: paths, column names, feature groups and targets.

Dataset: "Myocardial infarction complications" (UCI ML Repository, id 579).
Golovenkin S.E. et al., Krasnoyarsk Interdistrict Clinical Hospital, 1992-1995.
1,700 patients, 111 clinical input features, 12 outcome columns. Missing values are '?'.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "MI.data"
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# The raw file has no header row; these are the official UCI column names (1..124).
COLUMNS = [
    "ID", "AGE", "SEX", "INF_ANAM", "STENOK_AN", "FK_STENOK", "IBS_POST", "IBS_NASL", "GB",
    "SIM_GIPERT", "DLIT_AG", "ZSN_A", "nr_11", "nr_01", "nr_02", "nr_03", "nr_04", "nr_07",
    "nr_08", "np_01", "np_04", "np_05", "np_07", "np_08", "np_09", "np_10", "endocr_01",
    "endocr_02", "endocr_03", "zab_leg_01", "zab_leg_02", "zab_leg_03", "zab_leg_04",
    "zab_leg_06", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT", "O_L_POST",
    "K_SH_POST", "MP_TP_POST", "SVT_POST", "GT_POST", "FIB_G_POST", "ant_im", "lat_im",
    "inf_im", "post_im", "IM_PG_P", "ritm_ecg_p_01", "ritm_ecg_p_02", "ritm_ecg_p_04",
    "ritm_ecg_p_06", "ritm_ecg_p_07", "ritm_ecg_p_08", "n_r_ecg_p_01", "n_r_ecg_p_02",
    "n_r_ecg_p_03", "n_r_ecg_p_04", "n_r_ecg_p_05", "n_r_ecg_p_06", "n_r_ecg_p_08",
    "n_r_ecg_p_09", "n_r_ecg_p_10", "n_p_ecg_p_01", "n_p_ecg_p_03", "n_p_ecg_p_04",
    "n_p_ecg_p_05", "n_p_ecg_p_06", "n_p_ecg_p_07", "n_p_ecg_p_08", "n_p_ecg_p_09",
    "n_p_ecg_p_10", "n_p_ecg_p_11", "n_p_ecg_p_12", "fibr_ter_01", "fibr_ter_02",
    "fibr_ter_03", "fibr_ter_05", "fibr_ter_06", "fibr_ter_07", "fibr_ter_08", "GIPO_K",
    "K_BLOOD", "GIPER_NA", "NA_BLOOD", "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD",
    "ROE", "TIME_B_S", "R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_KB", "NOT_NA_KB", "LID_KB",
    "NITR_S", "NA_R_1_n", "NA_R_2_n", "NA_R_3_n", "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
    "LID_S_n", "B_BLOK_S_n", "ANT_CA_S_n", "GEPAR_S_n", "ASP_S_n", "TIKL_S_n", "TRENT_S_n",
    # ---- outcomes (columns 113-124) ----
    "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK", "OTEK_LANC", "RAZRIV",
    "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN", "LET_IS",
]
assert len(COLUMNS) == 124

OUTCOMES = COLUMNS[112:]
INPUTS = COLUMNS[1:112]

OUTCOME_NAMES = {
    "FIBR_PREDS": "Atrial fibrillation",
    "PREDS_TAH": "Supraventricular tachycardia",
    "JELUD_TAH": "Ventricular tachycardia",
    "FIBR_JELUD": "Ventricular fibrillation",
    "A_V_BLOK": "Third-degree AV block",
    "OTEK_LANC": "Pulmonary edema",
    "RAZRIV": "Myocardial rupture",
    "DRESSLER": "Dressler syndrome",
    "ZSN": "Chronic heart failure",
    "REC_IM": "Relapse of MI",
    "P_IM_STEN": "Post-infarction angina",
    "LET_IS": "Death (any cause)",
}

# ---- Leakage control -------------------------------------------------------
# We predict risk AT THE TIME OF HOSPITAL ADMISSION. Following the dataset authors'
# guidance, columns recorded later (pain relapses and drugs given in the first 3 days)
# are excluded; otherwise the model would "see the future".
LATER_THAN_ADMISSION = [
    "R_AB_1_n", "R_AB_2_n", "R_AB_3_n",
    "NA_R_1_n", "NA_R_2_n", "NA_R_3_n",
    "NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n",
]
# Columns that are almost entirely missing carry no usable signal.
MOSTLY_MISSING_THRESHOLD = 0.90

# Real-valued clinical measurements; everything else is binary / ordinal.
CONTINUOUS = [
    "AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT", "K_BLOOD", "NA_BLOOD",
    "ALT_BLOOD", "AST_BLOOD", "KFK_BLOOD", "L_BLOOD", "ROE",
]

# Readable names for commonly important features (plots, demo).
FEATURE_NAMES = {
    "AGE": "Age",
    "SEX": "Sex (1 = male)",
    "INF_ANAM": "No. of previous MIs",
    "STENOK_AN": "Angina history",
    "FK_STENOK": "Angina functional class",
    "IBS_POST": "Recent coronary heart disease",
    "GB": "Hypertension stage",
    "SIM_GIPERT": "Symptomatic hypertension",
    "DLIT_AG": "Duration of hypertension",
    "ZSN_A": "Prior chronic heart failure",
    "S_AD_KBRIG": "Systolic BP (ambulance)",
    "D_AD_KBRIG": "Diastolic BP (ambulance)",
    "S_AD_ORIT": "Systolic BP (ICU)",
    "D_AD_ORIT": "Diastolic BP (ICU)",
    "O_L_POST": "Pulmonary edema at ICU admission",
    "K_SH_POST": "Cardiogenic shock at ICU admission",
    "MP_TP_POST": "Paroxysmal AF at ICU admission",
    "SVT_POST": "SVT at ICU admission",
    "GT_POST": "Ventricular tachycardia at ICU admission",
    "FIB_G_POST": "Ventricular fibrillation at ICU admission",
    "ant_im": "Anterior MI (ECG)",
    "lat_im": "Lateral MI (ECG)",
    "inf_im": "Inferior MI (ECG)",
    "post_im": "Posterior MI (ECG)",
    "IM_PG_P": "Right ventricular MI",
    "K_BLOOD": "Serum potassium",
    "NA_BLOOD": "Serum sodium",
    "ALT_BLOOD": "Serum ALT",
    "AST_BLOOD": "Serum AST",
    "L_BLOOD": "White blood cell count",
    "ROE": "ESR",
    "TIME_B_S": "Time from pain onset to hospital",
    "NA_KB": "Opioids given by ambulance",
    "NOT_NA_KB": "NSAIDs given by ambulance",
    "LID_KB": "Lidocaine given by ambulance",
    "NITR_S": "Liquid nitrates in ICU",
    "endocr_01": "Diabetes mellitus",
    "zab_leg_01": "Chronic bronchitis",
    "zab_leg_02": "Obstructive chronic bronchitis",
    "nr_11": "Arrhythmia history",
    "n_p_ecg_p_12": "Complete RBBB on ECG",
    "ritm_ecg_p_01": "Sinus rhythm on ECG",
    "ritm_ecg_p_02": "Atrial fibrillation on ECG",
    "ritm_ecg_p_07": "Sinus tachycardia on ECG",
    "GEPAR_S_n": "Heparin in ICU",
    "TRENT_S_n": "Trental in ICU",
    "ASP_S_n": "Aspirin in ICU",
    "B_BLOK_S_n": "Beta-blockers in ICU",
    "LID_S_n": "Lidocaine in ICU",
    "ANT_CA_S_n": "Calcium blockers in ICU",
    "TIKL_S_n": "Ticlid in ICU",
}

PRIMARY_TARGET = "LET_IS"   # binarised: 0 = survived, 1 = died (any cause)
SECONDARY_TARGET = "ZSN"    # chronic heart failure


def pretty(col: str) -> str:
    """Human-readable name for a feature column (falls back to the raw code)."""
    if col.startswith("missing_"):
        return f"{pretty(col[8:])} [missing]"
    if col.startswith("cluster_"):
        return f"Patient cluster {col.split('_')[1]}"
    return FEATURE_NAMES.get(col, col)
