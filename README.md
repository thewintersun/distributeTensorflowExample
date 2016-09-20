# distributeTensorflowExample
distribute tensorflow  example

this is a distribute tensorflow example to compute y = weight * x + biasis


## Introduce

```
This is a most simple example for distributed tensorflow.

The task is to estimate the paramters of the formula : Y = 2 * X + 10

the paramter weight is the number 2, 

the paramter biasis is the number 10.



```



## run example


```
ps server:

CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=192.168.100.42:2222 --worker_hosts=192.168.100.42:2224,192.168.100.253:2225 --job_name=ps --task_index=0



worker server:

CUDA_VISIBLE_DEVICES=0 python distribute.py --ps_hosts=192.168.100.42:2222 --worker_hosts=192.168.100.42:2224,192.168.100.253:2225 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES=0 python distribute.py --ps_hosts=192.168.100.42:2222 --worker_hosts=192.168.100.42:2224,192.168.100.253:2225 --job_name=worker --task_index=1

```



## 说明

```
这是一个最简单的分布式tensorflow的例子。

实现的功能是估计这个公式的2个参数：  Y = 2 * X + 10

要估计的参数是weight是2， biasis 是10.


程序执行的ps节点1个， worker节点2个。 执行命令示例在下面。

详细关于tensorflow的分布式示例介绍：

```

## 执行命令示例


```
ps 节点执行： 

CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=192.168.100.42:2222 --worker_hosts=192.168.100.42:2224,192.168.100.253:2225 --job_name=ps --task_index=0



worker 节点执行:

CUDA_VISIBLE_DEVICES=0 python distribute.py --ps_hosts=192.168.100.42:2222 --worker_hosts=192.168.100.42:2224,192.168.100.253:2225 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES=0 python distribute.py --ps_hosts=192.168.100.42:2222 --worker_hosts=192.168.100.42:2224,192.168.100.253:2225 --job_name=worker --task_index=1

```

## 分布式介绍中文文档
```
http://blog.csdn.net/luodongri/article/details/52596780
```
