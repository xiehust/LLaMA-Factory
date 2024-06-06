import os
import json
import socket
import yaml

if __name__ == "__main__":
   
    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))
    
    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)
    
    os.environ['DS_BUILD_FUSED_ADAM'] = '1'
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    
    # backend env config
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
   # os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'
    
    num_machines = int(os.environ["NODE_NUMBER"])
    num_processes = int(os.environ["SM_NUM_GPUS"]) * num_machines
    # os.system("wandb disabled")
    sg_config = os.environ["sg_config"]
    sg_lora_merge_config = os.environ["sg_lora_merge_config"]
    s3_data_paths = os.environ.get('s3_data_paths')
    GPUS_PER_NODE = int(os.environ["SM_NUM_GPUS"])
    DEVICES = ','.join([str(i) for i in range(GPUS_PER_NODE)])
    
    # file_name = './sg_config.yaml'
    # with open(file_name) as f:
    #     doc = yaml.safe_load(f)
    # doc['machine_rank'] = host_rank
    # doc['main_process_ip'] = str(master_addr)
    # doc['num_machines'] = num_machines  # how many intances in this training job
    # doc['num_processes'] = num_processes  # how many GPU cards in total
    # with open('./sg_config.yaml', 'w') as f:
    #     yaml.safe_dump(doc, f)

    #Install LLama Factory 
    os.system("pip install --no-deps -e .")
    os.system("pip install -r requirements.txt")
    
    #invoke the torch launcher shell script.
    #Note: we will use the s5cmd to speed up the uploading model assets to S3.
    # os.system("chmod +x ./train_script_sagemaker.sh")
    os.system("chmod +x ./s5cmd")

    os.system("ls /opt/ml/code")

    if s3_data_paths:
        paths = s3_data_paths.split(',')
        for s3_path in paths:
            os.system("./s5cmd sync {0} {1}".format(s3_path+'/*', '/opt/ml/code/data/'))

    # print("*****************start cp pretrain model*****************************")
    # os.system("./s5cmd sync {0} {1}".format(os.environ['MODEL_S3_PATH'], os.environ["MODEL_LOCAL_PATH"]))
    # print(f'-----finished cp-------')
    print(f'------envs------\nnum_machines:{num_machines}\nnum_processes:{num_processes}\nhost_rank:{host_rank}\n')
    if host_rank == 0:
        os.system(f"CUDA_VISIBLE_DEVICES={DEVICES} NNODES=2 RANK=0 MASTER_ADDR={master_addr} MASTER_PORT=29500 llamafactory-cli train {sg_config}")
    else:
        os.system(f"CUDA_VISIBLE_DEVICES={DEVICES} NNODES=2 RANK=1 MASTER_ADDR={master_addr} MASTER_PORT=29500 llamafactory-cli train {sg_config}")

    if os.environ.get("merge_lora") == '1' and host_rank == 0:
        print(f'-----start merge lora-------')
        os.system(f'CUDA_VISIBLE_DEVICES=0 llamafactory-cli export {sg_lora_merge_config}')
        print(f'-----end merge lora-------')
        os.system("./s5cmd sync {0} {1}".format("/tmp/finetuned_model_merged", os.environ['OUTPUT_MODEL_S3_PATH']))

          
    if host_rank == 0:
        print("*****************finished training, start cp finetuned model*****************************")
        os.system("./s5cmd sync {0} {1}".format("/tmp/finetuned_model", os.environ['OUTPUT_MODEL_S3_PATH']))
        print(f'-----finished cp-------')
  
