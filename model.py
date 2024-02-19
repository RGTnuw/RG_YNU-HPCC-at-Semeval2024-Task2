import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

class Model(torch.nn.Module):
    def __init__(self,args,model_name,from_check_point = False,tokenizer_dir = None, model_dir = None):
        super(Model,self).__init__()
        assert(type(from_check_point) == bool)  

        self.args = args
                # 加载预训练模型，忽略分类器层的不匹配尺寸
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

        # 替换分类器层以适应二分类任务
        classifier_size = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(classifier_size, 2)
        if from_check_point:
                config = torch.load(model_dir)
                self.model.load_state_dict(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case = True) if from_check_point == False else AutoTokenizer.from_pretrained(tokenizer_dir,do_lower_case = True)

    def forward(self, sent, label, device):
        # 分词并准备输入数据
        token = self.tokenizer(sent, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
        output = self.model(**token, labels=label)
        return output

    def save_model(self, dir):
        # 保存模型和分词器
        self.tokenizer.save_pretrained(dir)
        torch.save(self.model.state_dict(), dir + f"/dev_best_seed{self.args.seed}.pth")
