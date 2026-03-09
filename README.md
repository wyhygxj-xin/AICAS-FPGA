# AICAS-FPGA

## 文档内容

./
├── doc                       // 相关论文
├── README.md       // README
├── models                // 模型权重 
├── smolvlm               // 模型加载及推理代码 
└── docker-image	  // 可直接运行的docker环境

## 其他数据

因为docker_images和model权重太大了，github上无法上传，所以得自行下载

## Docker使用

1. 下载docker

2. docker load -i docker_image.tar

3. 使用以下命令创建镜像:

   ```bash
   docker run -it --gpus all --name test -e http_proxy="http://172.17.0.1:7897"  -e https_proxy="http://172.17.0.1:7897"  -v 挂载卷:/workspace/AICAS  镜像名 /bin/bash
   ```

4. 常用docker命令

   ```bash
   docker images					// 查看docker images
   docker ps -a					// 查看docker容器
   docker start container-name     // 开始docker容器
   docker exec -it container-name /bin/bash		// 推出容器后重新进入容器
   ```


## Quick_start

1. 进入容器后执行

   ```bash
   source ~/.venv/bin/activate
   ```

2. 进入`smolvlm`中执行`python3 model.py`即可