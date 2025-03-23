## 後処理

+ [python_backend](https://github.com/triton-inference-server/python_backend)
    + [examples](https://github.com/triton-inference-server/python_backend/tree/main/examples)  
+ [Model Configuration - datatypes](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes)
+ [triton_python_backend_utils](https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py)
    
### model.pyの作成
[ファイル構成](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#model-config-file)
```
$ mkdir -p model_repository/postprocess/1
$ touch model_repository/postprocess/1/model.py
```
    
[Usage](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#usage)

#### auto_complete_config
Optional。

Tritonでconfigファイルが無いときに最低限必要なmax_batch_size、dynamic_batching、input、outputを設定する。

#### initialize
Optional。

args（Dict）によって以下の値を扱える。
| key | description | 
| --- | :--- |
| model_config | A JSON string containing the model configuration |
| model_instance_kind | A string containing model instance kind |
| model_instance_device_id | A string containing model instance device ID |
| model_repository | Model repository path |
| model_version | Model version |
| model_name | Model name |

例
```
Initialized...
model_config {"name":"postprocess","platform":"","backend":"python","runtime":"","version_policy":{"latest":{"num_versions":1}},"max_batch_size":0,"input":[{"name":"INPUT0","data_type":"TYPE_STRING","format":"FORMAT_NONE","dims":[10],"is_shape_tensor":false,"allow_ragged_batch":false,"optional":false}],"output":[{"name":"OUTPUT0","data_type":"TYPE_STRING","dims":[1],"label_filename":"","is_shape_tensor":false}],"batch_input":[],"batch_output":[],"optimization":{"priority":"PRIORITY_DEFAULT","input_pinned_memory":{"enable":true},"output_pinned_memory":{"enable":true},"gather_kernel_buffer_threshold":0,"eager_batching":false},"instance_group":[{"name":"postprocess_0","kind":"KIND_CPU","count":1,"gpus":[],"secondary_devices":[],"profile":[],"passive":false,"host_policy":""}],"default_model_filename":"model.py","cc_model_filenames":{},"metric_tags":{},"parameters":{},"model_warmup":[]}
model_instance_kind CPU
model_instance_name postprocess_0_0
model_instance_device_id 0
model_repository /models/postprocess
model_version 1
model_name postprocess
```

#### execute
Mandatory。

```
Parameters
----------
requests : list
  A list of pb_utils.InferenceRequest

Returns
-------
list
  A list of pb_utils.InferenceResponse. The length of this list must
  be the same as `requests`
```

#### finalize
モデルアンロード時の終了処理。

### config.pbtxtの作成

```
$ touch model_repository/postprocess/config.pbtxt
```

[Model Configuration - datatypes](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes)などを参考に作成。


```
$  cat model_repository/postprocess/config.pbtxt 
name: "postprocess"
backend: "python"

input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [
        1,
        10
    ]
    reshape: { shape: [ 10 ] }
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [{ kind: KIND_CPU }]
```

### ensemble
+ [Tutorial](https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_5-Model_Ensembles/README.md)

```
$ mkdir -p model_repository/ensemble_model/1
$ touch model_repository/ensemble_model/config.pbtxt
```

### 推論
元のrequestsからnameを修正
```
$ cat requests/data0_ensemble.json 
{
    "inputs": [
        {
            "name": "ensemble_input0",
            "datatype": "FP32",
            "shape": [
                1,
                1,
                28,
                28
              ],
            "data": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.239, 0.012, 0.165, 0.463, 0.757, 0.463, 0.463, 0.239, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.055, 0.702, 0.961, 0.925, 0.949, 0.996, 0.996, 0.996, 0.996, 0.961, 0.922, 0.329, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.592, 0.996, 0.996, 0.996, 0.835, 0.753, 0.698, 0.698, 0.706, 0.996, 0.996, 0.945, 0.180, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.169, 0.922, 0.996, 0.886, 0.251, 0.110, 0.047, 0.000, 0.000, 0.008, 0.502, 0.988, 1.000, 0.678, 0.067, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.220, 0.996, 0.992, 0.420, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.525, 0.980, 0.996, 0.294, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.247, 0.996, 0.620, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.867, 0.996, 0.616, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.761, 0.996, 0.404, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.588, 0.996, 0.835, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.133, 0.863, 0.937, 0.227, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.329, 0.996, 0.835, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.494, 0.996, 0.671, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.329, 0.996, 0.835, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.839, 0.937, 0.235, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.329, 0.996, 0.835, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.839, 0.780, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.329, 0.996, 0.835, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.043, 0.859, 0.780, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.329, 0.996, 0.835, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.384, 0.996, 0.780, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.635, 0.996, 0.820, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.384, 0.996, 0.780, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.200, 0.933, 0.996, 0.294, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.384, 0.996, 0.780, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.200, 0.647, 0.996, 0.765, 0.016, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.259, 0.945, 0.780, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.012, 0.655, 0.996, 0.890, 0.216, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.839, 0.835, 0.078, 0.000, 0.000, 0.000, 0.000, 0.000, 0.180, 0.596, 0.792, 0.996, 0.996, 0.247, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.839, 0.996, 0.800, 0.706, 0.706, 0.706, 0.706, 0.706, 0.922, 0.996, 0.996, 0.918, 0.612, 0.039, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.318, 0.804, 0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.996, 0.988, 0.918, 0.471, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.102, 0.824, 0.996, 0.996, 0.996, 0.996, 0.996, 0.600, 0.408, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
        }
    ],
    "outputs": [
        {
          "name": "ensemble_output0"
        }
    ]
}
```

```
$ curl -s -X POST --data @./requests/data0_ensemble.json -H "Content-Type: application/json" localhost:8000/v2/models/ensemble_model/infer | jq
{
  "model_name": "ensemble_model",
  "model_version": "1",
  "parameters": {
    "sequence_id": 0,
    "sequence_start": false,
    "sequence_end": false
  },
  "outputs": [
    {
      "name": "ensemble_output0",
      "datatype": "BYTES",
      "shape": [
        1
      ],
      "data": [
        "零"
      ]
    }
  ]
}
```


