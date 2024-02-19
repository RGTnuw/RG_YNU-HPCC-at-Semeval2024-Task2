import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from model import Model
from dataloader import Recorddataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--ptlm", default='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', type=str)
    parser.add_argument("--prompt", default=0, choices=[0, 1], type=int)
    parser.add_argument('--data', default='./training_data/',type=str, help='data dir')
    parser.add_argument("--tokenizer_dir", default='./result/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli_prompt0_modetrn_epoch10_eval10/', type=str, help='the tokenizer check point dir')
    parser.add_argument("--model_dir", default='./result/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli_prompt0_modetrn_epoch10_eval10/dev_best_seed621.pth', type=str, help='the model check point dir')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 加载模型
    model = Model(args, args.ptlm, True, args.tokenizer_dir, args.model_dir)
    model.to(device)

    # 加载测试数据
    tst_dataset = Recorddataset(args, args.data, "test")
    tst_loader = DataLoader(tst_dataset, batch_size=16, shuffle=False)

    # 推理并保存结果
    model.eval()
    Test_Results = {}
    with torch.no_grad():
        for sent, uuid in tqdm(tst_loader):
            output = model(sent, None, device)
            logits = output.logits  # 使用logits属性
            for i in range(logits.size(0)):
                if torch.argmax(logits[i]) == 0:
                    Test_Results[str(uuid[i])] = {"Prediction": 'Contradiction'}
                else:
                    Test_Results[str(uuid[i])] = {"Prediction": "Entailment"}

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", 'w') as jsonFile:
        jsonFile.write(json.dumps(Test_Results, indent=4))
    print("Results saved to", output_dir + "/results.json")

if __name__ == '__main__':
    main()