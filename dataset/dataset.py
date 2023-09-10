import torch
import abc


class Dataset:
    def __init__(self, table_name_join_key_mapping, label_info):
        self.table_name_join_key_mapping = table_name_join_key_mapping
        self.label_info = label_info
    
    def extract_meta(self, table_name, df):
        join_key = self.table_name_join_key_mapping[table_name]
        label = None
        if table_name in self.label_info:
            label = df[self.label_info[table_name]].values
        return df.reset_index().rename(columns={'index': table_name})[[*join_key, table_name]], label

    def extract_data(self, table_name, df):
        join_key = self.table_name_join_key_mapping[table_name]
        df = df.drop(columns=[*join_key])
        if table_name in self.label_info:
            df = df.drop(columns=[self.label_info[table_name]])
        return df.values
    
    def get_G(self, table_index_mapping, table_meta_dict):
        G = []
        for table_name in self.table_name_join_key_mapping.keys():
            matches = [[] for _ in range(table_meta_dict[table_name].shape[0])]
            groups = table_index_mapping.groupby(table_name).groups
            for index, values in groups.items():
                matches[index] = list(values)
            G.append(matches)
        return G
    
    @abc.abstractmethod
    def build_mapping(self):
        pass


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]