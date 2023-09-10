from dataset.dataset import Dataset
import pandas as pd


"""
The key order of table_name_join_key_mapping should be consistent with f !!! 
"""
table_name_join_key_mapping = {
    'ratings': ['userid', 'movieid'],
    # 'movies': ['movieid'],
    'movies_genres_split': ['movieid'],
    'users': ['userid'],
}
label_info = { 'ratings': 'rating' }


class MovieLens_1M(Dataset):
    name = 'MovieLens-1M'
    num_class = 1
    task = 'regression'


    def __init__(self):
        super().__init__(table_name_join_key_mapping, label_info)
    
    def build_mapping(self, table_meta_dict):
        table_index_mapping = pd.merge(table_meta_dict['ratings'], table_meta_dict['movies_genres_split'], on='movieid') \
            .merge(table_meta_dict['users'], on='userid') \
            .reset_index()
        f = table_index_mapping.drop(['movieid', 'userid'], axis=1).values
        G = self.get_G(table_index_mapping, table_meta_dict)
        return f, G, table_index_mapping