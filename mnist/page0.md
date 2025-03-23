## 入力からクラス分類まで
モデルとテストデータは以下にある。
[onnx/models -> mnist](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)

```
curl -OL https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.tar.gz
tar -zxvf mnist-12.tar.gz
rm mnist-12.tar.gz
```

```
$ tree mnist-12
mnist-12
├── mnist-12.onnx
└── test_data_set_0
    ├── input_0.pb
    └── output_0.pb

2 directories, 3 files
```

画像はこことかなどからダウンロードできる
https://www.kaggle.com/datasets/alexanderyyy/mnist-png

### モデルディレクトリ作成
名前はmodel.onnxでなければならない（Expected model name of the form 'model.<backend_name>'.）
```
$ tree model_repository 
model_repository
└── mnist_onnx
    └── 1
        └── model.onnx

3 directories, 1 file
```

### Triton Server起動
```
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --env PYTHONIOENCODING=utf8 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models
```

[API Reference](https://github.com/triton-inference-server/server/blob/main/docs/protocol/README.md)

モデルの確認
```
$ curl -s localhost:8000/v2/models/mnist_onnx | jq
{
  "name": "mnist_onnx",
  "versions": [
    "1"
  ],
  "platform": "onnxruntime_onnx",
  "inputs": [
    {
      "name": "Input3",
      "datatype": "FP32",
      "shape": [
        1,
        1,
        28,
        28
      ]
    }
  ],
  "outputs": [
    {
      "name": "Plus214_Output_0",
      "datatype": "FP32",
      "shape": [
        1,
        10
      ]
    }
  ]
}
```

configの確認
```
$ curl -s localhost:8000/v2/models/mnist_onnx/config | jq
{
  "name": "mnist_onnx",
  "platform": "onnxruntime_onnx",
  "backend": "onnxruntime",
  "runtime": "",
  "version_policy": {
    "latest": {
      "num_versions": 1
    }
  },
  "max_batch_size": 0,
  "input": [
    {
      "name": "Input3",
      "data_type": "TYPE_FP32",
      "format": "FORMAT_NONE",
      "dims": [
        1,
        1,
        28,
        28
      ],
      "is_shape_tensor": false,
      "allow_ragged_batch": false,
      "optional": false
    }
  ],
  "output": [
    {
      "name": "Plus214_Output_0",
      "data_type": "TYPE_FP32",
      "dims": [
        1,
        10
      ],
      "label_filename": "",
      "is_shape_tensor": false
    }
  ],
  "batch_input": [],
  "batch_output": [],
  "optimization": {
    "priority": "PRIORITY_DEFAULT",
    "input_pinned_memory": {
      "enable": true
    },
    "output_pinned_memory": {
      "enable": true
    },
    "gather_kernel_buffer_threshold": 0,
    "eager_batching": false
  },
  "instance_group": [
    {
      "name": "mnist_onnx",
      "kind": "KIND_CPU",
      "count": 2,
      "gpus": [],
      "secondary_devices": [],
      "profile": [],
      "passive": false,
      "host_policy": ""
    }
  ],
  "default_model_filename": "model.onnx",
  "cc_model_filenames": {},
  "metric_tags": {},
  "parameters": {},
  "model_warmup": []
}
```

### 推論
[API Reference](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)

```
$ curl -s localhost:8000/v2/models/mnist_onn
```
の結果を元にjsonを作成（requests/data0_classification.json）。

```
$ python img2data.py mnist_png/test/0/10.png 
```
の結果をinputsのdataに書き込み。

outputsのparametersの`{ "classification" : 10 }`は上位10件を出力することを表す。

```
$ curl -s -X POST --data @./requests/data0_classification.json -H "Content-Type: application/json" localhost:8000/v2/models/mnist_onnx/infer | jq
{
  "model_name": "mnist_onnx",
  "model_version": "1",
  "outputs": [
    {
      "name": "Plus214_Output_0",
      "datatype": "BYTES",
      "shape": [
        10
      ],
      "data": [
        "21.535694:0",
        "1.952763:5",
        "1.597451:6",
        "1.475181:9",
        "0.888193:2",
        "-2.370755:8",
        "-3.199228:3",
        "-3.245501:7",
        "-9.395827:4",
        "-16.862560:1"
      ]
    }
  ]
}
```
0の確率が一番高い。

### ラベルを表示してみる
+ [Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
+ [Labelファイルの例](https://github.com/triton-inference-server/server/blob/main/docs/examples/model_repository/densenet_onnx/densenet_labels.txt)

```
$ curl -s -X POST --data @./requests/data0_classification.json -H "Content-Type: application/json" localhost:8000/v2/models/mnist_onnx/infer | jq
{
  "model_name": "mnist_onnx",
  "model_version": "1",
  "outputs": [
    {
      "name": "Plus214_Output_0",
      "datatype": "BYTES",
      "shape": [
        10
      ],
      "data": [
        "21.535694:0:零",
        "1.952763:5:伍",
        "1.597451:6:陸",
        "1.475181:9:玖",
        "0.888193:2:弐",
        "-2.370755:8:捌",
        "-3.199228:3:参",
        "-3.245501:7:漆",
        "-9.395827:4:肆",
        "-16.862560:1:壱"
      ]
    }
  ]
}
```