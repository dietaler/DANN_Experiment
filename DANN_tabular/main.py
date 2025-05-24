import torch
import argparse
import pandas as pd
from dataset import load_data, change_label, down_sampling
import train
import model

def read_large_csv(file_path, chunksize=100000):
    """ 分批讀取大型 CSV 文件 """
    df_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        df_chunks.append(chunk)
    df = pd.concat(df_chunks, ignore_index=True)
    return df

def main():
    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str,default='data/tabular_6class/cicids2017/all_data_6class.csv', help="Path to source dataset")
    parser.add_argument('--target_data', type=str, default='data/tabular_6class/cicids2018/all_data_6class.csv', help="Path to target dataset")
    # parser.add_argument('--method', type=str, choices=['standard', 'dann'], default='standard', help="Choose training method")
    # parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    # parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    # parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # 設定 GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載source domain
    if args.source_data:
        print(f'source domain: {args.source_data}')
        # df_source = pd.read_csv(args.source_data)
        df_source = read_large_csv(args.source_data)  # **限制讀取數量**
        df_source = change_label(df_source)  # 標籤映射
        df_source = down_sampling(df_source)  # 下採樣
        source_train_loader, source_test_loader = load_data(df_source) 
    else:
        raise ValueError("必須提供 --source_data 參數！")

    # 加載target domain
    if args.target_data:
        print(f'target domain: {args.target_data}')
        # df_target = pd.read_csv(args.target_data)
        df_target = read_large_csv(args.target_data)  # **限制讀取數量**
        df_target = change_label(df_target)  # 標籤映射
        df_target = down_sampling(df_target)  # **可選**
        target_train_loader, target_test_loader = load_data(df_target)  # 只取 test
  
    if torch.cuda.is_available():
        encoder_source_only = model.Extractor().cuda()
        classifier_source_only = model.Classifier(input_dim=encoder_source_only.flatten_dim).cuda(device)

        encoder_dann = model.Extractor().cuda()
        classifier_dann = model.Classifier(input_dim=encoder_dann.flatten_dim).cuda(device)
        discriminator_dann = model.Discriminator(input_dim=encoder_dann.flatten_dim).cuda(device)
        train.source_only(encoder_source_only, classifier_source_only, source_train_loader, target_train_loader, source_test_loader, target_test_loader)
        train.dann(encoder_dann, classifier_dann, discriminator_dann, source_train_loader, target_train_loader, source_test_loader, target_test_loader)

    else:
        print("No GPUs available.")


if __name__ == "__main__":
    main()
