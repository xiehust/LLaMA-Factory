import os
import json
import socket
import yaml
import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, check=True, shell=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

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
    os.environ['NCCL_DEBUG'] = 'ERROR'
    os.environ['HCCL_OVER_OFI'] = '1'
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    
    num_machines = int(os.environ["NODE_NUMBER"])
    num_processes = int(os.environ["SM_NUM_GPUS"]) * num_machines
    # os.system("wandb disabled")
    sg_config = os.environ["sg_config"]
    sg_lora_merge_config = os.environ["sg_lora_merge_config"]
    s3_data_paths = os.environ.get('s3_data_paths')
    GPUS_PER_NODE = int(os.environ["SM_NUM_GPUS"])
    DEVICES = ','.join([str(i) for i in range(GPUS_PER_NODE)])
    

    #Install LLama Factory 
    os.system("pip install --no-deps -e .")
    os.system("pip install -r requirements.txt")
    
    os.system("chmod +x ./s5cmd")

    os.system("ls /opt/ml/code")

    if s3_data_paths:
        paths = s3_data_paths.split(',')
        for s3_path in paths:
            # 同步S3数据到本地
            s3_sync_command = f"./s5cmd sync {s3_path}/* /opt/ml/code/data/"
            run_command(s3_sync_command)

    print(f'------envs------\nnum_machines:{num_machines}\nnum_processes:{num_processes}\nhost_rank:{host_rank}\n')
    train_command = f"CUDA_VISIBLE_DEVICES={DEVICES} NNODES={num_machines} RANK={host_rank} MASTER_ADDR={master_addr} MASTER_PORT=29500 llamafactory-cli train {sg_config}"
    # run_command(train_command)
    exit_code = os.system(train_command)
    if exit_code != 0:
        print(f"Train failed with exit code: {exit_code}")
        sys.exit(1)
        
    if os.environ.get("merge_lora") == '1' and host_rank == 0:
        print(f'-----start merge lora-------')
        merge_command = f'CUDA_VISIBLE_DEVICES=0 llamafactory-cli export {sg_lora_merge_config}'
        run_command(merge_command)

        print(f'-----end merge lora-------')
        sync_merged_command = f"./s5cmd sync /tmp/finetuned_model_merged {os.environ['OUTPUT_MODEL_S3_PATH']}"
        run_command(sync_merged_command)

          
    if host_rank == 0:
        print("*****************finished training, start cp finetuned model*****************************")
        sync_final_command = f"./s5cmd sync /tmp/finetuned_model {os.environ['OUTPUT_MODEL_S3_PATH']}"
        run_command(sync_final_command)
        print(f'-----finished cp-------')
  
