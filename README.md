<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu"></a>
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)


# distributeTensorflowExample

## 分布式介绍中文文档
```
http://blog.csdn.net/luodongri/article/details/52596780
```

## 更多tensorflow和深度学习的内容，请参考我的书《tensorflow入门与实战》 
```
这本书深度学习入门的内容占了一半，都是很基础和入门的.

如果刚入门的可以看看，自认为比网上看吴恩达的教程更容易看懂。

如果是已经比较熟悉Tensorflow和深度学习了，可以不用看了。

```
链接：
[https://item.jd.com/12307221.html](https://item.jd.com/12307221.html)





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


