# distributeTensorflowExample
distribute tensorflow  example

this is a distribute tensorflow example to compute y = w*x + b


# run example

ps server:
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=192.168.100.42:2222,192.168.100.22:2223 --worker_hosts=192.168.100.30:2224,192.168.100.253:2225 --job_name=ps --task_index=0
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=192.168.100.42:2222,192.168.100.22:2223 --worker_hosts=192.168.100.30:2224,192.168.100.253:2225 --job_name=ps --task_index=1


worker server:
CUDA_VISIBLE_DEVICES=0 python distribute.py --ps_hosts=192.168.100.42:2222,192.168.100.22:2223 --worker_hosts=192.168.100.30:2224,192.168.100.253:2225 --job_name=worker --task_index=0
CUDA_VISIBLE_DEVICES=0 python distribute.py --ps_hosts=192.168.100.42:2222,192.168.100.22:2223 --worker_hosts=192.168.100.30:2224,192.168.100.253:2225 --job_name=worker --task_index=1

