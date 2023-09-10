from dataset.dataset import Dataset
import pandas as pd


table_name_join_key_mapping = {
    'stays': ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
    'patients': ['SUBJECT_ID'],
    'admissions': ['HADM_ID'],
    'diagnoses': ['HADM_ID'],
    'events': ['ICUSTAY_ID']
}
label_info = { 'events': 'LABEL' }


class MIMIC(Dataset):
    name = 'MIMIC-III'
    num_class = 2
    task = 'classification'


    def __init__(self):
        super().__init__(table_name_join_key_mapping, label_info)
    

    def extract_data(self, table_name, df: pd.DataFrame):
        join_key = self.table_name_join_key_mapping[table_name]
        df = df.drop(columns=[*join_key])
        if table_name in self.label_info:
            df = df.drop(columns=[self.label_info[table_name]])
        if table_name == 'stays':
            df.drop(columns=['MORTALITY'], inplace=True)
        return df.values
    
    def build_mapping(self, table_meta_dict):
        table_index_mapping = pd.merge(table_meta_dict["stays"], table_meta_dict["patients"], on="SUBJECT_ID") \
            .merge(table_meta_dict["admissions"], on="HADM_ID") \
            .merge(table_meta_dict["diagnoses"], on="HADM_ID") \
            .merge(table_meta_dict["events"], on="ICUSTAY_ID") \
            .reset_index()
        f = table_index_mapping.drop(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], axis=1).values # Drop join keys
        G = self.get_G(table_index_mapping, table_meta_dict)
        return f, G, table_index_mapping