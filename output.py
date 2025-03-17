import argparse
import pandas as pd
from tqdm import tqdm
import json

from datasets import load_dataset

def main(args):
    dataset_df = load_dataset("ScottHan/CorrelationQA")['train']['true_answer']

    with open(args.input_path) as f:
        mllm_answer = f.readlines()
 
    true = 0
    wrong = 0
    print(len(mllm_answer))
    assert len(dataset_df) == len(mllm_answer) 
    for i in range(len(mllm_answer)):
        if dataset_df[i].lower() in mllm_answer[i].lower():
            true+=1
        else:
            wrong+=1
        
    print(f'Average Accuracy: { true / (true+wrong) }')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default= "output.txt")
    args = parser.parse_args()
    main(args)