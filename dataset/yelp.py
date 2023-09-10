from dataset.dataset import Dataset
import pandas as pd


table_name_join_key_mapping = {
    'review': ['review_id', 'user_id', 'business_id'],
    'restaurant': ['business_id'],
    'user': ['user_id']
}
label_info = { 'review': 'label' }


class Yelp(Dataset):
    name = 'Yelp'
    num_class = 5
    task = 'classification'


    def __init__(self):
        super().__init__(table_name_join_key_mapping, label_info)

    # Override this method because the label need to be subtracted one
    # def extract_meta(self, table_name, df):
    #     join_key = self.table_name_join_key_mapping[table_name]
    #     label = None
    #     if table_name in self.label_info:
    #         label = df[self.label_info[table_name]].values
    #         label = label - 1
    #     return df.reset_index().rename(columns={'index': table_name})[[*join_key, table_name]], label
    
    def build_mapping(self, table_meta_dict):
        table_index_mapping = pd.merge(table_meta_dict['review'], table_meta_dict['restaurant'], on='business_id') \
            .merge(table_meta_dict['user'], on='user_id') \
            .reset_index()
        f = table_index_mapping.drop(['review_id', 'business_id', 'user_id'], axis=1).values
        G = self.get_G(table_index_mapping, table_meta_dict)
        return f, G, table_index_mapping