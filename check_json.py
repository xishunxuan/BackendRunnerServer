import sys
import pathlib
from SerializeTool import deserialize_from_file_with_types, serialize_to_file_with_types
from ConfClass import OPTIMIZER_CONF, GLOBAL_CONF, TRAINER_CONF, TEST_SURROGATE_MODULE_CONF

class ConfigManager:
    def __init__(self):
        self.args = None
        self.optim_conf = None
        self.global_conf = None
        self.trainer_conf = None
        self.test_surrogate_module_conf = None

    def init_conf(self, args=None):
        """使用反序列化的args初始化所有配置"""
        self.args = args
        self.optim_conf = OPTIMIZER_CONF(args)
        self.global_conf = GLOBAL_CONF(args)
        self.trainer_conf = TRAINER_CONF(args)
        self.test_surrogate_module_conf = TEST_SURROGATE_MODULE_CONF(args)
        
        # 序列化输出
        output_path = pathlib.Path(args.output_path) / "OptimizerMainConfig.conf"
        serialize_to_file_with_types(output_path, args, 
                                   ignore_unsupported=True, 
                                   pretty_json=True)
        print(f"配置已序列化到: {output_path}")

def main():
    # 写死的配置文件路径
    CONFIG_FILE_PATH = "F:\前后端开发\electromagnet_back-master\config_output.json"  # 替换为你的输入配置文件路径
    
    try:
        # 1. 反序列化输入配置
        print(f"正在反序列化配置文件: {CONFIG_FILE_PATH}")
        args = deserialize_from_file_with_types(CONFIG_FILE_PATH)
        print("\n✅ 反序列化成功！")
        
        # 2. 检查反序列化后的对象
        print(f"对象类型: {type(args)}")
        attributes = [attr for attr in dir(args) if not attr.startswith('__')]
        print(f"\n包含 {len(attributes)} 个属性:")
        for attr in attributes[:5]:  # 只显示前5个属性避免输出过长
            try:
                value = getattr(args, attr)
                print(f"  - {attr}: {type(value)}")
            except Exception as e:
                print(f"  - {attr}: [无法获取值: {str(e)}]")
        if len(attributes) > 5:
            print(f"  ...(共 {len(attributes)} 个属性，只显示前5个)")
        
        # 3. 初始化所有配置
        print("\n正在初始化配置...")
        config_manager = ConfigManager()
        config_manager.init_conf(args)
        print("✅ 所有配置初始化完成！")
        
        return 0
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 文件不存在 - {CONFIG_FILE_PATH}")
        return 1
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())