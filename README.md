### Dockerのインストール
[イメージ](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
```
docker pull nvcr.io/nvidia/tritonserver:24.03-py3
```

### server 
https://github.com/triton-inference-server/server

### tutorials
https://github.com/triton-inference-server/tutorials

### python_backend
https://github.com/triton-inference-server/python_backend

### quickstart
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html

```
curl https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/fetch_models.sh -O ./
chmod 755 fetch_models.sh
sh fetch_models.sh
```

```
$ tree model_repository
model_repository
├── densenet_onnx
│   └── 1
│       └── model.onnx
└── inception_graphdef
    └── 1
        └── model.graphdef

5 directories, 2 files
```

```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository2:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models
```

API Reference
```
https://github.com/triton-inference-server/server/blob/main/docs/protocol/README.md
```

```
$ curl -s localhost:8000/v2/models/densenet_onnx
{"name":"densenet_onnx","versions":["1"],"platform":"onnxruntime_onnx","inputs":[{"name":"data_0","datatype":"FP32","shape":[1,3,224,224]}],"outputs":[{"name":"fc6_1","datatype":"FP32","shape":[1,1000,1,1]}]}
```

```
$ curl -s localhost:8000/v2/models/densenet_onnx/versions/1/config
{"name":"densenet_onnx","platform":"onnxruntime_onnx","backend":"onnxruntime","runtime":"","version_policy":{"latest":{"num_versions":1}},"max_batch_size":0,"input":[{"name":"data_0","data_type":"TYPE_FP32","format":"FORMAT_NONE","dims":[1,3,224,224],"is_shape_tensor":false,"allow_ragged_batch":false,"optional":false}],"output":[{"name":"fc6_1","data_type":"TYPE_FP32","dims":[1,1000,1,1],"label_filename":"","is_shape_tensor":false}],"batch_input":[],"batch_output":[],"optimization":{"priority":"PRIORITY_DEFAULT","input_pinned_memory":{"enable":true},"output_pinned_memory":{"enable":true},"gather_kernel_buffer_threshold":0,"eager_batching":false},"instance_group":[{"name":"densenet_onnx","kind":"KIND_CPU","count":2,"gpus":[],"secondary_devices":[],"profile":[],"passive":false,"host_policy":""}],"default_model_filename":"model.onnx","cc_model_filenames":{},"metric_tags":{},"parameters":{},"model_warmup":[]}
```

https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/ONNX

```
curl https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Quick_Deploy/ONNX/client.py -O
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

```
 docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:24.03-py3-sdk bash
pip install torchvision
```

```
root@docker-desktop:/workspace# python client.py
['11.548583:92' '11.231403:14' '7.527273:95' '6.922707:17' '6.576274:88']
```
