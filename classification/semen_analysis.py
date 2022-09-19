import pandas as pd
import os

PATH = "./data/archive/semen_analysis_data_Train.csv"


def _header_conversion(df_header_list) -> dict:
    """Header conversion for semen analysis only

    Args:
        df_header_list (list): DataFrame header list

    Returns:
        dict: Mapper dict, this was meant for df.rename function
    """
    mapping = {}
    rules = {
        "ID": "id",
        "Total sperm count": "total_sperm_count",
        "Sperm concentration": "sperm_concentration",
        "Ejaculate volume": "ejaculate_volume",
        "Sperm vitality ": "percent_sperm_vitality",
        "Normal spermatozoa": "percent_normal_spermatozoa",
        "Head defects": "percent_head_defect",
        "Midpiece and neck defects": "percent_mid_neck_defect",
        "Tail defects": "percent_tail_defect",
        "Cytoplasmic droplet": "percent_cytoplasmic_droplet",
        "Teratozoospermia index": "tera_index",
        "Progressive motility": "percent_progressive",
        "Non progressive sperm motility": "percent_non_progressive",
        'Immotile sperm': "percent_immotile",
        "HDS": "percent_hds",
        "DFI": "percent_dfi"
    }
    for col_name in df_header_list:
        for rule_key in rules.keys():
            if rule_key not in col_name: continue
            mapping[col_name] = rules[rule_key]

    return mapping


def prepare_data(path=PATH, includes=None):
    df = pd.read_csv(PATH)
    mapping = _header_conversion(list(df.columns))
    includes = includes or [
        "id", "percent_progressive", "percent_non_progressive",
        "percent_immotile"
    ]
    return df.rename(columns=mapping)[includes]


if __name__ == "__main__":
    print(prepare_data())
    pass
