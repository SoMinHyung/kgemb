import os, sys
import csv
import boto3
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def download_kg_triple():
    def check_and_download(s3, bucket_name, local_download_path, s3_file_path):
        dir = local_download_path.rsplit("/", 1)[0]
        if '/' in dir and not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(local_download_path):
            s3.download_file(bucket_name, s3_file_path, local_download_path)
    s3 = boto3.client('s3')
    bucket_name = "nlp-entity-linking-train-set"
    s3_file_path = 'rsc/full_output_trim.tsv'
    local_download_path = 'kg_triple/full_output_trim.tsv'
    check_and_download(s3, bucket_name, local_download_path, s3_file_path)

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        if not os.path.exists(target):
            bucket.download_file(obj.key, target)

def make_total_tsv():
    out = []
    files = os.listdir("kg_triple")
    for file in files:
        with open(os.path.join('kg_triple',file)) as f:
            lines = list(csv.reader(f, delimiter="\t"))
            for line in lines:
                if len(line) is not 3:
                    continue
                out.append(line)
    random.shuffle(out)
    with open("s_total_data.tsv", 'w') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for row in out:
            tsv_output.writerow(row)

def split_total_tsv():
    df = pd.read_csv("s_total_data.tsv", delimiter='\t', names=["head", "relation", "tail"])
    df = df.dropna()
    print(df.count())

    trainDF, testDF = train_test_split(df, test_size=0.05)
    trainDF, validDF = train_test_split(trainDF, test_size=0.1)
    print(trainDF.shape, validDF.shape, testDF.shape)
    trainDF.to_csv("train.tsv", header=None, index=None, sep="\t")
    validDF.to_csv("valid.tsv", header=None, index=None, sep="\t")
    testDF.to_csv("test.tsv", header=None, index=None, sep="\t")

def upload_trained_data(s3_dirname='KGE'):
    s3 = boto3.client('s3')
    bucket_name = "nlp-entity-linking-train-set"
    s3_file_path_npy = s3_dirname + '/t1.npy'
    s3_file_path_entities = s3_dirname + '/e1.tsv'
    local_npy = 'model/TransE_l2_data_0/data_TransE_l2_entity.npy'
    local_entities = 'entities.tsv'
    s3.upload_file(local_npy,bucket_name,s3_file_path_npy)
    s3.upload_file(local_entities,bucket_name,s3_file_path_entities)

def run(algorithm='TransE'):
    if algorithm=='TransE':
        os.system("DGLBACKEND=pytorch dglke_train \
                --dataset data \
                --model_name TransE_l2 \
                --batch_size 1000 \
                --neg_sample_size 200 \
                --hidden_dim 1000 \
                --gamma 10 \
                --lr 0.1 \
                --max_step 3000 \
                --log_interval 10000 \
                --gpu 0 \
                --batch_size_eval 1000 \
                --regularization_coef 1.00E-09 \
                --eval_interval 10000 \
                --test --valid -adv \
                --save_path ./model/ \
                --data_path ./ \
                --format raw_udd_hrt \
                --data_files train.tsv valid.tsv test.tsv \
                --neg_sample_size_eval 1000")
    elif algorithm=='DistMult':
        os.system("DGLBACKEND=pytorch dglke_train \
                --dataset data \
                --model_name DistMult \
                --batch_size 1024 \
                --neg_sample_size 256 \
                --hidden_dim 400 \
                --gamma 143.0 \
                --lr 0.08 \
                --max_step 50000 \
                --log_interval 1000 \
                --gpu 0 \
                --batch_size_eval 1000 \
                --eval_interval 10000 \
                --test --valid -adv \
                --save_path ./model/ \
                --data_path ./ \
                --format raw_udd_hrt \
                --data_files train.tsv valid.tsv test.tsv \
                --neg_sample_size_eval 1000")
    elif algorithm=='ComplEx':
        os.system("DGLBACKEND=pytorch dglke_train \
                --dataset data \
                --model_name ComplEx \
                --batch_size 1024 \
                --neg_sample_size 256 \
                --hidden_dim 400 \
                --gamma 143.0 \
                --lr 0.1 \
                --max_step 50000 \
                --log_interval 1000 \
                --gpu 0 \
                --batch_size_eval 1000 \
                --eval_interval 10000 \
                --test --valid -adv \
                --save_path ./model/ \
                --data_path ./ \
                --format raw_udd_hrt \
                --data_files train.tsv valid.tsv test.tsv \
                --neg_sample_size_eval 1000")
    elif algorithm=='TransR':
        os.system("DGLBACKEND=pytorch dglke_train \
               --dataset data \
               --model_name TransR \
               --batch_size 1024 \
               --neg_sample_size 256 \
               --hidden_dim 200 \
               --gamma 8.0 \
               --lr 0.01 \
               --max_step 50000 \
               --log_interval 1000 \
               --gpu 0 \
               --batch_size_eval 1000 \
               --eval_interval 10000 \
               --test --valid -adv \
               --save_path ./model/ \
               --data_path ./ \
               --format raw_udd_hrt \
               --data_files train.tsv valid.tsv test.tsv \
               --neg_sample_size_eval 1000")
    elif algorithm=='RotatE':
        os.system("DGLBACKEND=pytorch dglke_train \
                --dataset data \
                --model_name RotatE \
                --batch_size 1024 \
                --neg_sample_size 256 \
                --hidden_dim 200 \
                --gamma 12.0 \
                -de \
                --lr 0.01 \
                --regularization_coef 1e-7 \
                --max_step 50000 \
                --log_interval 1000 \
                --gpu 0 \
                --batch_size_eval 1000 \
                --eval_interval 10000 \
                --test --valid -adv \
                --save_path ./model/ \
                --data_path ./ \
                --format raw_udd_hrt \
                --data_files train.tsv valid.tsv test.tsv \
                --neg_sample_size_eval 1000")

def main(action):
    # Download data for KGE
    download_s3_folder('nlp-entity-linking-train-set','kg_triple')
    download_kg_triple()
    # Trim triple & Make total_data.tsv
    make_total_tsv()
    # Make train, valid, test dataset
    split_total_tsv()
    # Train & Upload Trained Model
    # run(action.strip('--'))
    # upload_trained_data('KGE')
    # print("----------------------------------------")
    # print("Finished")

if __name__ == '__main__':
    action = sys.argv[1]
    assert action in ['--TransE', '--DistMult', '--ComplEx','--TransR','--RotatE'], 'Only train, deploy, delete, docker is supported'
    main(action)