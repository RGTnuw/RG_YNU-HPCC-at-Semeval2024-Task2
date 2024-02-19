import json
import torch
from torch.utils.data import Dataset

Label2num = {'Entailment': 1, "Contradiction": 0}

class Recorddataset(Dataset):
    def __init__(self, args, path, split='train'):
        super(Recorddataset, self).__init__()
        self.args = args
        self.split = split
        self.statement = []
        self.trail1 = []
        self.trail2 = []
        self.label = []
        self.section = []
        ctpath = path + "/CT json/"
        jspath = path + f"/{split}" + ".json" if split != 'trn&dev' else path + '/train.json'
        with open(jspath) as file:
            self.data = json.load(file)
            self.uuid_list = list(self.data.keys())
        if split == 'trn&dev':  # 
            with open(path + '/dev.json') as file:
                self.data = {**self.data, **json.load(file)} 
                self.uuid_list = list(self.data.keys())
        for id in self.uuid_list:
            self.statement.append(self.data[id]['Statement'])
            if split != 'test':
                self.label.append(Label2num[self.data[id]['Label']])
            section = self.data[id]['Section_id']
            self.section.append(section)
            with open(
                    ctpath + f"{self.data[id]['Primary_id']}" + ".json") as file:  
                ct = json.load(file)
                trail1 = ct[section]
                self.trail1.append(self.format_change(trail1))
            if self.data[id]['Type'] == "Comparison": 
                with open(ctpath + f"{self.data[id]['Secondary_id']}" + ".json") as file:
                    ct = json.load(file)
                    trail2 = ct[section]
                    self.trail2.append(self.format_change(trail2))
            else:
                self.trail2.append("_")  # 代表空数据

    def __getitem__(self, index):
        if self.args.prompt == 0:
            if self.trail2[index] == '_':
                sent = "{} [SEP] {} [SEP] {}".format(self.statement[index], self.section[index],
                                                     self.trail1[index])
            else:
                sent = "{} [SEP] {} [SEP] {} [SEP] {}".format(self.statement[index], self.section[index],
                                                              self.trail1[index], self.trail2[index])

        elif self.args.prompt == 1:
            sent = "{}".format(self.statement[index])

        else:
            raise NotImplementedError("Prompt not implemented")
        if self.split != 'test':
            return sent, torch.tensor(self.label[index])
        else:
            return sent, self.uuid_list[index]

    def __len__(self):
        return len(self.uuid_list)

    def format_change(self, sentence):
        s = ""
        for sent in sentence:
            s += sent.strip() + ","
        return s

    def get_max_length(self):
        print([len(self.__getitem__(i)[0].split(' ')) for i in range(self.__len__())])
        return max([len(self.__getitem__(i)[0].split(' ')) for i in range(self.__len__())])