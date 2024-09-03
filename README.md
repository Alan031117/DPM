# DeepPointMap

## 使用方法

### Tensorboard

1. tensorboard --logdir=log_tb --port 6006
2. 控制台执行`ssh -p 20004 -L 16006:127.0.0.1:6006 root@10.176.55.26`
3. 浏览器访问`http://localhost:16006`

### 训练  
- 需指定相应的yaml配置文件路径
```bash
python pipeline/train.py --yaml_file my_yaml.yaml
```

- 训练中断后的恢复
```bash
python pipeline/train.py --yaml_file my_yaml.yaml --checkpoint my_ckpt.ckpt
```

- 使用GPU加速训练 
```bash
python pipeline/train.py --yaml_file my_yaml.yaml --gpu_index 0
```

- 使用PyTorch的分布式数据并行(DDP)进行多GPU训练  
tips: 多卡训练依赖通讯端口进行通讯，默认端口号29500，当执行多个DDP任务时，需指定不同端口以避免端口出现被占用的情况  
方法：增加参数 ```--master_port 61234```，其中端口号可自行替换为其他空端口
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LOCAL_RANK=0 torchrun --nproc_per_node=6 pipeline/train.py --use_ddp --word_size 6 --yaml_file my_yaml.yaml
```
torchrun启动方式仅适用于torch版本>=1.9.0，早期版本请替换为
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 pipeline/train.py --local_rank 0 --use_ddp --word_size 6 --yaml_file my_yaml.yaml
```

### 推理
指定yaml配置文件路径和模型权重路径
```bash
python pipeline/infer.py --yaml_file my_yaml.yaml --gpu_index 0 --weight my_weight.pth
```

### 输出点云配准指标
```bash
python ./SlamSystem/Metrics/eval_registration.py -pred {evo_pred_key_path} -gt {evo_gt_key_path}
```
