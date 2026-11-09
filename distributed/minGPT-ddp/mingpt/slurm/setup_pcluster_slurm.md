# Setup AWS cluster with pcluster

使用 pcluster 配置 AWS 集群

Refer https://www.hpcworkshops.com/04-pcluster-cli.html

参考教程：https://www.hpcworkshops.com/04-pcluster-cli.html。

## 1. Sign in to an AWS instance

1．登录 AWS 实例

## 2. Install pcluster

2．安装 pcluster

```
pip3 install awscli -U --user
pip3 install "aws-parallelcluster" --upgrade --user
```

## 3. Create a cluster config file

3．创建集群配置文件

```
pcluster configure --config config.yaml
```

See config.yaml.template for an example. Ensure you have a valid EC2 key-pair file

示例见 config.yaml.template；请确保拥有有效的 EC2 密钥对文件。

## 4. Create the cluster

4．创建集群

```
pcluster create-cluster --cluster-name dist-ml --cluster-configuration config.yaml
```

### 4a. Track progress

4a．跟踪进度

```
pcluster list-clusters
```

## 5. Login to cluster headnode

5．登录集群主节点

```
pcluster ssh --cluster-name dist-ml -i your-keypair-file
```

## 6. Install dependencies

6．安装依赖

```
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv /shared/venv/
source /shared/venv/bin/activate
pip install wheel
echo 'source /shared/venv/bin/activate' >> ~/.bashrc
```

## 7. Download training code and install requirements

7．下载训练代码并安装依赖

```
cd /shared
git clone --depth 1 https://github.com/pytorch/examples;
cd /shared/examples
git filter-branch --prune-empty --subdirectory-filter distributed/minGPT-ddp
python3 -m pip install setuptools==59.5.0
pip install -r requirements.txt
```
