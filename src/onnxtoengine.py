import tensorrt as trt
import onnx

onnx_model_path = "/home/ljy/project/cpr-fusion-ui-main/models/1best_model_fold_2.onnx"
engine_file_path = "/home/ljy/project/cpr-fusion-ui-main/models/1best_model_fold_2.engine"

# 创建 TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 加载 ONNX 模型
onnx_model = onnx.load(onnx_model_path)

# 获取输入张量的尺寸
input_name = onnx_model.graph.input[0].name  # 获取模型第一个输入的名称
input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
# 假设输入的形状为 (batch_size, channels, height, width)
# 获取输入的维度 (batch_size, channels, height, width)
min_shape = (1, 3, 224, 224)  # 默认最小尺寸，修改为你的实际尺寸
opt_shape = (16, 3, 224, 224)  # 最优尺寸
max_shape = (32, 3, 224, 224)  # 最大尺寸

# 创建 TensorRT 构建器并指定显式批量维度
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # 使用显式批量维度
)

# 创建 IBuilderConfig 对象
config = builder.create_builder_config()

# 设定优化配置文件（Profile）
profile = builder.create_optimization_profile()

# 使用从 ONNX 模型中获取的输入尺寸设置优化配置文件
profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)

# 将优化配置文件应用到构建器配置
config.add_optimization_profile(profile)

# 创建解析器并解析 ONNX 模型
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_model_path, 'rb') as f:
    if not parser.parse(f.read()):
        print('Error parsing ONNX file')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
    else:
        # 使用优化配置文件构建引擎
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"Engine successfully saved to {engine_file_path}")
        else:
            print("Failed to build the engine.")
