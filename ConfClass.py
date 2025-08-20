import copy
from typing import Union, List, Dict
from types import SimpleNamespace
import json
import pathlib
import numpy as np
import re
from munch import DefaultMunch


def get_float_list(edit_text: str, mode: int = 0) -> Union[List[float], List[int], str]:
    # 将字符串按逗号分割成列表
    if mode == 0:
        str_list: List[str] = re.split("[,，]", edit_text)
        float_list: List[float] = [float(x) for x in str_list]
        return float_list
    elif mode == 1:
        str_list: List[str] = re.split("[,，]", edit_text)
        int_list: List[int] = [int(x) for x in str_list]
        return int_list


class CONF:
    """
    抽象CONF类
    todo:可迭代对象中不可含有不可序列化的对象
    """

    def __init__(self):
        self.parsed_dict = {}
        self.parsed_type = {}

    # def save_json(self):
    #     for keys, values in self.__dict__.items():
    #         if isinstance(values, pathlib.Path):
    #             self.parsed_dict[keys] = str(values)
    #             self.parsed_type[keys] = "pathlib.Path"
    #         elif isinstance(values, np.ndarray):
    #             try:
    #                 self.parsed_dict[keys] = values.tolist()
    #                 self.parsed_type[keys] = "np.ndarray"
    #             except:
    #                 raise TypeError
    #         elif keys == 'parsed_type' or keys == 'parsed_dict':
    #             continue
    #         else:
    #             self.parsed_dict[keys] = copy.deepcopy(values)
    #             self.parsed_type[keys] = "normal"
    #     return {"dict": self.parsed_dict, "type": self.parsed_type}
    #
    # def load_json(self, path):
    #     with open(pathlib.Path(path), mode='r') as fp:
    #         conf_data = json.load(fp)
    #         _parsed_dict: Dict = conf_data["dict"]
    #         _parsed_type: Dict = conf_data["type"]
    #     for keys, values in _parsed_type.items():
    #         if values == 'pathlib.Path':
    #             _parsed_dict[keys] = pathlib.Path(_parsed_dict[keys])
    #         elif values == 'np.ndarray':
    #             try:
    #                 _parsed_dict[keys] = np.array(_parsed_dict[keys])
    #             except:
    #                 raise TypeError
    #         elif keys == '_parsed_type' or keys == '_parsed_dict':
    #             continue
    #         else:
    #             continue
    #     return _parsed_dict, _parsed_type


class TRAINER_CONF(CONF):
    def __init__(self, args=None, path=None):

        """
        从args及self中读取训练器所需要的一些参数
        有一些参数不能静态确定，具体如下：
        Trainer还需要读入data_index_to_frequency，y_dim：
          该变量来源于dataset。从self.data_index_to_frequency,self.y_dim读入。需要在初始化数据集后，再读入。
        Trainer还需要读入model_pretrained：
           需要根据iter_n 与train_from_scratch_per_rounds的关系，手动判断model_pretrained取值。
        """

        """
        ******训练超参数******
        training_ratio:训练集比例;float;1>t>0;0.8
        validation_ratio：验证集比例;float;1-training_ratio>t>0

        normal_lr:训练网络的学习率
        batch_size:训练网络的批量大小;int;>0;t0=128
        normal_epoch:训练网络的轮数;int;>0;t0=300
        train_from_scratch_per_rounds：正整数，每隔N个优化轮次后，重新从零训练;int;>=0;t0=10
        train_from_scratch_lr:重新从零训练网络的学习率;float;>0;t0=0.001
        train_from_scratch_epoch:重新从零训练网络的轮数;int;>0;t0=500
        weight_decay：正则项loss系数（Adam）;float;>0;t0=0
        criterion：LOSS类型 str:"l2","l1";to="l2"
        NN_optimizer: 训练神经网络的优化器 str: "Adam","SGD",为"SGD"时 weight_decay失效;t0="Adam"
        model_name:神经网络模型结构 str: "resnet20", "resnet32", "resnet44", "resnet56", "resnet101";t0="resnet32"
        pareto_loss_coeff：帕累托学习LOSS的系数;float;>0;t0="0.9"
        """
        super(TRAINER_CONF, self).__init__()

        if args is None:
            assert path is not None
            self.load_json(path)
            return
        if hasattr(args, "training_param"):
            args.training_param = DefaultMunch.fromDict(args.training_param)
        self.training_policy = 'StandardNet'
        # todo:完善typing
        self.lr: Union[float, None] = None
        self.epoch: Union[int, None] = None
        self.fragment_shape: Union[List, None] = None
        self.fragment_setting_list: Union[List, None] = None
        self.data_index_to_frequency: Union[List, None] = None
        self.y_dim: Union[int, None] = None
        self.model_pretrained: [str, None, bool] = None
        # todo:界面接入
        self.device: str = "cuda:0"

        if args.predict_method == "MultiResVAE" or \
                args.predict_method == 'KEP' or \
                args.predict_method == 'VitKEP':
            self.multi_resnet_vae_train_vae = args.training_param.multi_resnet_vae_train_vae
            self.multi_resnet_vae_epochs = args.training_param.multi_resnet_vae_epochs
            self.multi_resnet_vae_vae_batch_size = args.training_param.multi_resnet_vae_vae_batch_size
            self.multi_resnet_vae_vae_lr = args.training_param.multi_resnet_vae_vae_lr
            self.multi_resnet_vae_use_discriminator = args.training_param.multi_resnet_vae_use_discriminator
            self.multi_resnet_vae_discriminator_lr = args.training_param.multi_resnet_vae_discriminator_lr
            self.multi_resnet_vae_finetune_decoder = args.training_param.multi_resnet_vae_finetune_decoder
            self.multi_resnet_vae_decoder_lr = args.training_param.multi_resnet_vae_decoder_lr
            self.multi_resnet_vae_resnet_epoch = args.training_param.multi_resnet_vae_resnet_epoch
            self.multi_resnet_vae_res_batch_size = args.training_param.multi_resnet_vae_res_batch_size
            self.multi_resnet_vae_resnet_lr = args.training_param.multi_resnet_vae_resnet_lr
            self.multi_resnet_vae_resnet_fc_lr = args.training_param.multi_resnet_vae_resnet_fc_lr
            self.multi_resnet_vae_resnet_update_epoch = args.training_param.multi_resnet_vae_resnet_update_epoch
            self.multi_resnet_vae_res_update_batch_size = args.training_param.multi_resnet_vae_res_update_batch_size
            self.multi_resnet_vae_vae_input_dim = args.training_param.multi_resnet_vae_vae_input_dim
            self.multi_resnet_vae_hidden_dim = args.training_param.multi_resnet_vae_hidden_dim
            self.multi_resnet_vae_latent_dim = args.training_param.multi_resnet_vae_latent_dim
            self.multi_resnet_vae_alpha = args.training_param.multi_resnet_vae_alpha
            self.multi_resnet_vae_train_resnet_from_scratch_or_not = args.training_param.multi_resnet_vae_train_resnet_from_scratch_or_not
            self.multi_resnet_vae_transfer_learning = args.training_param.multi_resnet_vae_transfer_learning
            if args.training_param.multi_resnet_vae_update_method == "每次更新重新训练":
                self.multi_resnet_vae_update_method = "continue_train_from_scratch"
            elif args.training_param.multi_resnet_vae_update_method == "持续微调":
                self.multi_resnet_vae_update_method = "continue_finetune"
            else:
                self.multi_resnet_vae_update_method = 'continue_with_scratch_and_finetune'
            self.multi_resnet_vae_train_from_scratch_n = args.training_param.multi_resnet_vae_train_from_scratch_n

        if args.predict_method == 'KEFormer':
            self.vqvae_epochs = args.training_param.vqvae_epochs

        if args.predict_method == 'LightningKEP':
            # todo 临时处理
            # self.selected_model = args.selected_model
            self.train_model_from_scratch_or_not = args.training_param.train_model_from_scratch_or_not
            self.model_epoch = args.training_param.model_epoch
            self.batch_size = args.training_param.batch_size
            self.model_update_epoch = args.training_param.model_update_epoch
            self.update_batch_size = args.training_param.update_batch_size
            if args.training_param.update_method == "每次更新重新训练":
                self.update_method = "continue_train_from_scratch"
            elif args.training_param.update_method == "持续微调":
                self.update_method = "continue_finetune"
            else:
                self.update_method = 'continue_with_scratch_and_finetune'
            self.train_from_scratch_n = args.training_param.train_from_scratch_n
            self.check_val_every_n_epoch = args.training_param.check_val_every_n_epoch
            self.transfer_learning_or_not = args.training_param.transfer_learning_or_not
            self.pretrained_model_path = args.training_param.pretrained_model_path

            if hasattr(args.training_param,"seed") and args.training_param.seed is not None:
                self.seed = args.training_param.seed
            else:
                self.seed = 0

            if args.selected_model == "LightningResNet":
                # todo 临时处理
                self.selected_model = "ResNet"
                self.resnet_lr = args.training_param.selected_model_param.resnet_lr
                self.fc_lr = args.training_param.selected_model_param.fc_lr
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.pretrained = args.training_param.selected_model_param.pretrained
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
                self.model_depth = int(args.training_param.selected_model_param.model_depth)

            if args.selected_model == "LightningGoogleNet":
                # todo 临时处理
                self.selected_model = "GoogleNet"
                self.googlenet_lr = args.training_param.selected_model_param.googlenet_lr
                self.weight_decay = args.training_param.selected_model_param.weight_decay
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.pretrained = args.training_param.selected_model_param.pretrained
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape

            if args.selected_model == "MultiResVAE":
                self.pretrained_vae = args.training_param.selected_model_param.pretrained_vae
                self.vae_batch_size = args.training_param.selected_model_param.vae_batch_size
                self.vae_batch_epoch = args.training_param.selected_model_param.vae_batch_epoch
                self.vae_input_dim = args.training_param.selected_model_param.vae_input_dim
                self.vae_hidden_dim = args.training_param.selected_model_param.vae_hidden_dim
                self.vae_latent_dim = args.training_param.selected_model_param.vae_latent_dim
                self.vae_lr = args.training_param.selected_model_param.vae_lr
                self.decoder_lr = args.training_param.selected_model_param.decoder_lr

                self.resnet_lr = args.training_param.selected_model_param.resnet_lr
                self.fc_lr = args.training_param.selected_model_param.fc_lr
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.pretrained = args.training_param.selected_model_param.pretrained
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape

            if args.selected_model == "NormResNet":
                self.resnet_lr = args.training_param.selected_model_param.resnet_lr
                self.fc_lr = args.training_param.selected_model_param.fc_lr
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.pretrained = args.training_param.selected_model_param.pretrained
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape

            if args.selected_model == "LightningResNet18":
                # todo 临时处理
                self.encoder = DefaultMunch()
                self.decoder = DefaultMunch()
                self.selected_model = "ResNet18"
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
                self.encoder.lr = args.training_param.selected_model_param.encoder_lr
                self.decoder.lr = args.training_param.selected_model_param.decoder_lr
                self.decoder.name = args.training_param.selected_model_param.decoder_name

            if args.selected_model == "LightningDeepONet":
                # todo 临时处理
                self.selected_model = "DeepONet"
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
                self.lr = args.training_param.selected_model_param.lr

            if args.selected_model == "LightningVIT":
                # todo 临时处理
                self.encoder = DefaultMunch()
                self.decoder = DefaultMunch()
                self.selected_model = "VIT"
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
                self.encoder.lr = args.training_param.selected_model_param.encoder_lr
                self.decoder.lr = args.training_param.selected_model_param.decoder_lr

            if args.selected_model == "LightningTVM":
                self.selected_model = "TVM"
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
                self.num_source_scenarios = args.training_param.selected_model_param.num_source_scenarios
                self.top_k = args.training_param.selected_model_param.top_k
                self.is_geting_source_model = args.training_param.selected_model_param.is_geting_source_model
                self.source_models_path = args.training_param.selected_model_param.source_models_path
                # todo：添加其他参数

            if args.selected_model == "LightningSwinTransformer":
                # todo 临时处理
                self.encoder = DefaultMunch()
                self.decoder = DefaultMunch()
                self.selected_model = "SwinTransformer"
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
                self.encoder.lr = args.training_param.selected_model_param.encoder_lr
                self.decoder.lr = args.training_param.selected_model_param.decoder_lr
                self.use_pretrained = args.training_param.selected_model_param.use_pretrained
                self.unfreeze_strategy = args.training_param.selected_model_param.unfreeze_strategy

        # if args.predict_method == 'LightningKEPDemo':
        #     self.selected_model = args.selected_model
        #     self.train_model_from_scratch_or_not = args.training_param.train_model_from_scratch_or_not
        #     self.model_epoch = args.training_param.model_epoch
        #     self.batch_size = args.training_param.batch_size
        #     self.model_update_epoch = args.training_param.model_update_epoch
        #     self.update_batch_size = args.training_param.update_batch_size
        #     if args.training_param.update_method == "每次更新重新训练":
        #         self.update_method = "continue_train_from_scratch"
        #     elif args.training_param.update_method == "持续微调":
        #         self.update_method = "continue_finetune"
        #     else:
        #         self.update_method = 'continue_with_scratch_and_finetune'
        #     self.train_from_scratch_n = args.training_param.train_from_scratch_n
        #     self.check_val_every_n_epoch = args.training_param.check_val_every_n_epoch
        #     self.transfer_learning_or_not = args.training_param.transfer_learning_or_not
        #     self.pretrained_model_path = args.training_param.pretrained_model_path
        #
        #     if args.selected_model == "ResNet":
        #         self.resnet_lr = args.training_param.selected_model_param.resnet_lr
        #         self.fc_lr = args.training_param.selected_model_param.fc_lr
        #         self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
        #         self.pretrained = args.training_param.selected_model_param.pretrained
        #         self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
        #
        #     if args.selected_model == "GoogleNet":
        #         self.googlenet_lr = args.training_param.selected_model_param.googlenet_lr
        #         self.weight_decay = args.training_param.selected_model_param.weight_decay
        #         self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
        #         self.pretrained = args.training_param.selected_model_param.pretrained
        #         self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
        #
        #     if args.selected_model == "MultiResVAE":
        #         self.pretrained_vae = args.training_param.selected_model_param.pretrained_vae
        #         self.vae_batch_size = args.training_param.selected_model_param.vae_batch_size
        #         self.vae_batch_epoch = args.training_param.selected_model_param.vae_batch_epoch
        #         self.vae_input_dim = args.training_param.selected_model_param.vae_input_dim
        #         self.vae_hidden_dim = args.training_param.selected_model_param.vae_hidden_dim
        #         self.vae_latent_dim = args.training_param.selected_model_param.vae_latent_dim
        #         self.vae_lr = args.training_param.selected_model_param.vae_lr
        #         self.decoder_lr = args.training_param.selected_model_param.decoder_lr
        #
        #         self.resnet_lr = args.training_param.selected_model_param.resnet_lr
        #         self.fc_lr = args.training_param.selected_model_param.fc_lr
        #         self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
        #         self.pretrained = args.training_param.selected_model_param.pretrained
        #         self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape
        #
        #     if args.selected_model == "NormResNet":
        #         self.resnet_lr = args.training_param.selected_model_param.resnet_lr
        #         self.fc_lr = args.training_param.selected_model_param.fc_lr
        #         self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
        #         self.pretrained = args.training_param.selected_model_param.pretrained
        #         self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape

        """
        ******控制参数******
        save_model:是否在每轮优化后，保存模型文件;bool;False时train_from_scratch_xxx参数失效，每一轮网络训练按照一般参数(lr,epoch)进行;t0=True
        output_path:每轮优化后，训练完成的模型文件将保存在(output_path / const.MODEL_PATH / args.model_name)，str;t0="./output"
        print_freq：正整数，训练时每隔N轮显示训练集、验证集LOSS情况;int;>0;t0=20
        """
        self.output_path = args.output_path
        self.save_model = args.save_model
        # todo:界面接入
        self.print_freq = args.print_freq

        # todo: 这些参数对应的polydataset（曲线傅里叶变换模块）暂未实现，读取后再次调整为False
        # self.using_polydataset = args.using_polydataset
        # self.using_polydataset = False
        # self.polyfit_dim = None

    # def save_json(self, path=None):
    #     this_dict = super(TRAINER_CONF, self).save_json()
    #     if path is None:
    #         with open(pathlib.Path(self.output_path) / "./trainer.conf", mode='w+') as fp:
    #             json.dump(this_dict, fp)
    #     else:
    #         with open(pathlib.Path(path), mode='w+') as fp:
    #             json.dump(this_dict, fp)
    #
    # def load_json(self, path=None):
    #     if path is None:
    #         my_path = pathlib.Path(self.output_path) / "./trainer.conf"
    #     else:
    #         my_path = path
    #     parsed_dict, parsed_type_dict = super(TRAINER_CONF, self).load_json(my_path)
    #     self.__dict__ = parsed_dict
    #     self.parsed_type = copy.deepcopy(parsed_type_dict)
    #     self.parsed_dict = copy.deepcopy(parsed_dict)


class OPTIMIZER_CONF(CONF):
    def __init__(self, args=None, path=None):

        """
        ******下列参数需要在CONF完成初始化后额外赋值，因为这些变量动态依赖于其他变量，所以这些变量不能从args中读取
        变量名                    中文名称             类型&默认值          说明
        fragment_shape         Fragment信息表        List[]        描述各个Fragment形状的List。依赖于Optimize.fragment_conf
        data_index_to_frequenct 曲线横坐标索引         List[]        描述原始曲线横坐标值与索引下标关系的List。依赖于Optimize.data_index_to_frequenct或Optimize.dataset.data_index_to_frequenct
        LDI_0                   相似样本判定距离        List[]        局部优化阶段，两样本像素距离小于该LDI_0，则被判定为在同一个区域内。依赖于Optimize.pix_n*OPTIMIZER_CONF.LDI_0_coeff
        filter_eps              局部搜索过滤样本距离    int           局部优化阶段创建区域时，距离中心样本小于filter_eps的样本会被过滤。依赖于OPTIMIZER_CONF.LDI_0
        ******下列参数需要从args中读取，但是不应该放在优化设置界面中
        output_path            输出文件path          str            场景的project文件夹存放目录

        ******下列参数需要从args中读取

        ******全局参数-主要部分
        @optimization_iter_n   优化迭代次数  int 100
        @samples_per_generation 每轮优化样本量     int 10
        @local_search_per_Nround  全局搜索最大周期           int　>0 15        每隔N轮全局搜索，强制结束，进入局部搜索
        @max_global_search_iter 全局搜索-筛选次数        int 　>0 100      每轮全局搜索筛选总样本量等于筛选次数乘以单次样本量
        @max_local_search_iter  局部搜索-筛选次数        int  >0　50      每轮局部搜索筛选总样本量等于筛选次数乘以单次样本量
        @resample_num           单次筛选样本量           int  >0　200
        @max_search_time        模拟退火迭代次数      int  >0　9999    全局搜索环节，模拟退火的迭代次数
        @using_filter            是否启用滤波器         bool False   是否对样本执行滤波操作；优化走线型场景时，请勿开启
        @random_weight_coeff    权重均匀度            float ＜10 4     均匀权重生成中，最大权重/最小权重不会超过(10+random_weight_coeff)/random_weight_coeff
        ******全局参数-次要部分
        @filter_type            滤波器类型              int 2        1:中值滤波 2：高斯滤波
        @filter_std             高斯滤波标准差           float 0.7    仅在高斯滤波时可用
        @update_sampler_per_rounds 采样器优化周期       int 5           每隔一段时间优化采样器的参数
        @force_update_sampler_rounds 强制优化采样器轮次   List[int]  [1,2,3]   当当前优化轮次在List当中，则强制优化采样器
        @n_mat_pool               筛选缓冲区大小        int 20000       缓存已筛选样本的矩阵池，应大于resample_number
        @using_known_initial_point  从较优点开始采样    bool False  是否将已有较好样本作为下次搜索的初始点
        ******局部优化-主要参数
        @neighbour_num           每次探索区域数量        int 3
        @neighbour_size          区域样本容量           int 10
        @neighbour_lifelongneighbour_lifelong      区域可容忍无改进次数     int 20
        @LDI_0_coeff             同区域样本相似度           float 0.1

        ******自适应权重参数
        @adapt_weight_smooth_coeff   自适应权重光滑系数   float 0.5
        @using_adapt_weight          启用自适应权重功能   bool True
        @adapt_weight_min_weight_prop  自适应权重权重均匀度   float ＜10 4


        ******全局优化参数




        ******采样器参数
        @max_mutate_length_ratio_shape_modify    形状修改-顶点改变度 float 0<t<1 0.1
        @add_mutate_layer_shape_modify           形状修改-增加随机线段单元  bool True
        @mutate_layer_max_changed_pixel_ratio_shape_modify  形状修改-线段单元最大像素比例 float 1>t>0 0.05
        @v_prob_shape_modify                     形状修改-顶点修改概率         float 1>t>0 0.05

        ******优化目标函数设置
        @using_soft_distance_target          差距线性加权    bool False    当using_distance_target为真时生效
        @using_distance_target          取最大差距      bool   True  ：using_soft_distance_target



        ******StableBacchus特有参数
        adaptive_mesh_size （arg.adaptive_mesh_size） 是否启用自适应网格数量 bool True
        min_mesh_size （arg.min_mesh_size）   自适应网格数量初始值 int 4
        ******GA算法参数
        ga_population_size (arg.ga_population_size) 种群大小
        ga_cross_area_ratio    (arg.ga_cross_area_ratio)    交叉算子-交叉区域比例
        ga_samples_per_generation  (arg.samples_per_generation)  每代产生样本，与samples_per_generation一致
        ga_crossover_probability   (arg.ga_crossover_probability)   交叉算子-概率
        ga_mutate_probability   (arg.ga_mutation_probability)   变异算子-概率
        ga_tournament_n   (ga_tournament_coefficient) 竞标赛选择算子-小组赛成员数


        ******专家训练特有参数
        expert_lr (arg.expert_lr) 专家模型学习率

        """
        super(OPTIMIZER_CONF, self).__init__()
        if args is None:
            assert path is not None
            self.load_json(path)
            return
        if hasattr(args, "optimize_param"):
            args.optimize_param = DefaultMunch.fromDict(args.optimize_param)
        # 这里是需要开始优化后赋值的变量
        self.fragment_shape: Union[List[int, int], None] = None
        self.data_index_to_frequency: Union[List[np.ndarray], None] = None
        # todo 计算距离
        self.LDI_0: Union[float, None] = None
        self.filter_eps: Union[int, None] = None

        self.output_path: str = args.output_path

        # todo 单轮优化最大筛选时间还未引入
        self.using_known_initial_point: bool = args.using_known_initial_point

        self.constrain_list = args.constraint_list
        self.variable_dict = args.variable_dict



        if args.optimize_method == 'Bacchus-GA' or \
                args.optimize_method == 'StableBacchus' or \
                args.optimize_method == 'StableBacchus_ada':
            self.samples_per_generation: int = args.optimize_param.sampleSizePerGeEdt
            self.local_search_per_Nround: int = args.optimize_param.localNroundEdt
            self.max_global_search_iter: int = args.optimize_param.maxGlobalIterEdt
            self.max_local_search_iter: int = args.optimize_param.maxLocalIterEdt
            self.resample_num: int = args.optimize_param.resampleNumEdt
            self.max_search_time: int = args.optimize_param.maxSearchTimeEdt
            self.using_filter: bool = args.optimize_param.filterOrNotCbx
            self.random_weight_coeff: float = args.optimize_param.randomWeigCoeEdt
            self.using_adapt_weight: bool = args.optimize_param.adapWeigOrNotCbx

            self.max_mutate_length_ratio_shape_modify: float = args.optimize_param.max_mutate_length_ratio_shape_modify
            self.add_mutate_layer_shape_modify: bool = args.optimize_param.add_mutate_layer_shape_modify
            self.mutate_layer_max_changed_pixel_ratio_shape_modify: float = args.optimize_param.mutate_layer_max_changed_pixel_ratio_shape_modify
            self.v_prob_shape_modify: float = args.optimize_param.v_prob_shape_modify

            self.filter_type: int = args.optimize_param.filterTypeEdt
            self.filter_std: float = args.optimize_param.filterStdEdt
            self.update_sampler_per_rounds: int = args.optimize_param.updateSamplerEdt
            self.force_update_sampler_rounds = get_float_list(args.optimize_param.forceUpdateEdt, 1)
            self.n_mat_pool: int = args.optimize_param.matPoolEdt
            self.adapt_weight_smooth_coeff: float = args.optimize_param.adapWeigEMCEdt
            self.adapt_weight_min_weight_prop: float = args.optimize_param.adapWeigMinPropEdt

            self.neighbour_num: int = args.optimize_param.neighborNumEdt
            self.neighbour_size: int = args.optimize_param.neighbourSizeEdt
            self.neighbour_lifelong: int = args.optimize_param.maxNoImproveEdt
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.similar_sample_tolerance = args.optimize_param.similarToleranceEdt

            # GA算法参数
            self.ga_population_size = args.optimize_param.populationSizeEdt
            self.ga_cross_area_ratio = args.optimize_param.crossAreaRatioEdt
            self.ga_samples_per_generation = args.optimize_param.sampleSizePerGeEdt
            self.ga_crossover_probability = args.optimize_param.crossoverEdt
            self.ga_mutate_probability = args.optimize_param.mutationEdt
            self.ga_tournament_n = args.optimize_param.tournamentEdt

        if args.optimize_method == 'StableBacchus_ada':
            self.start_mesh = get_float_list(args.optimize_param.start_mesh, 1)
            self.progress_var_n = args.optimize_param.progress_var_n
            self.max_mutate_length_ratio_shape_modify: float = args.optimize_param.max_mutate_length_ratio_shape_modify
            self.add_mutate_layer_shape_modify: bool = args.optimize_param.add_mutate_layer_shape_modify
            self.mutate_layer_max_changed_pixel_ratio_shape_modify: float = args.optimize_param.mutate_layer_max_changed_pixel_ratio_shape_modify
            self.v_prob_shape_modify: float = args.optimize_param.v_prob_shape_modify

        # GA算法参数
        if args.optimize_method == 'GA_elite':
            self.ga_population_size = args.optimize_param.populationSizeEdt
            self.ga_cross_area_ratio = args.optimize_param.crossAreaRatioEdt
            self.ga_samples_per_generation = args.optimize_param.sampleSizePerGeEdt
            self.ga_crossover_probability = args.optimize_param.crossoverEdt
            self.ga_mutate_probability = args.optimize_param.mutationEdt
            self.ga_tournament_n = args.optimize_param.tournamentEdt

        # 专家训练参数
        if args.optimize_method == 'ExpertTraining':
            self.expertTrainingEpoch = args.optimize_param.expertTrainingEpoch
            self.expertTrainingLr = args.optimize_param.expertTrainingLr
            self.expertTrainingBatchSize = args.optimize_param.expertTrainingBatchSize
            self.expertTrainingNetType = args.optimize_param.expertTrainingNetType
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt


        if args.optimize_method == 'BoostingTransfer':
            self.boostingEpoch = args.optimize_param.boostingEpoch
            self.boostingLr = args.optimize_param.boostingLr
            self.boostingBatchSize = args.optimize_param.boostingBatchSize
            self.boostingNetType = args.optimize_param.boostingNetType
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt

        if args.optimize_method == 'EmoPredictor' or \
                args.optimize_method == 'ResVAE':
            self.expertTrainingEpoch = args.optimize_param.emoEpoch
            self.expertTrainingLr = args.optimize_param.emoLr
            self.expertTrainingBatchSize = args.optimize_param.emoBatchSize
            self.compressed_dimension = args.optimize_param.compressedDimension
            self.target_weight = args.optimize_param.targetWeight
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == 'SurrogateEmoGA':
            self.emo_ga_proC = args.optimize_param.emo_ga_proC
            self.emo_ga_disC = args.optimize_param.emo_ga_disC
            self.emo_ga_proM = args.optimize_param.emo_ga_proM
            self.emo_ga_disM = args.optimize_param.emo_ga_disM
            self.emo_ga_tournament_K = args.optimize_param.emo_ga_tournament_K
            self.emo_ga_population_size = args.optimize_param.emo_ga_population_size
            self.emo_ga_using_elite = args.optimize_param.emo_ga_using_elite
            # self.emo_ga_method = args.optimize_param.emo_ga_method
            # self.emo_ga_surrogate_method_n = args.optimize_param.emo_ga_surrogate_method_n
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)
        if args.optimize_method == 'SimulatedAnnealing':
            self.T0 = args.optimize_param.T0
            self.T_min = args.optimize_param.T_min
            self.delta = args.optimize_param.delta
            self.alpha = args.optimize_param.alpha
            self.using_re_annealing = args.optimize_param.using_re_annealing
            self.re_annealing_iter = args.optimize_param.re_annealing_iter
            self.fix_range_num = args.optimize_param.fix_range_num
            self.trial_N_neighbour = args.optimize_param.trial_N_neighbour
            self.tral_N_line_search = args.optimize_param.tral_N_line_search
            self.neighbour_size = args.optimize_param.neighbour_size
            self.population_n = args.optimize_param.population_n
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration
            # todo debug
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)
        if args.optimize_method == 'SimulatedAnnealing_Discrete':
            self.population_n = args.optimize_param.population_n
            self.progress_var_n = args.optimize_param.progress_var_n
            self.start_var_n = args.optimize_param.start_var_n
            self.start_depth = args.optimize_param.start_depth
            self.T0 = args.optimize_param.T0
            self.delta = args.optimize_param.delta
            self.alpha = args.optimize_param.alpha
            self.using_re_annealing = args.optimize_param.using_re_annealing
            self.T_min = args.optimize_param.T_min
            self.re_annealing_iter = args.optimize_param.re_annealing_iter
            self.global_sampling_for_re_annealing = args.optimize_param.global_sampling_for_re_annealing
            # 加载仿真策略参数
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)

        if args.optimize_method == 'TSDDEOOptimizer':
            self.delta = args.optimize_param.delta
            self.population_n = args.optimize_param.population_n
            self.local_population_n = args.optimize_param.local_population_n
            self.stage1_eval_threshold = args.optimize_param.stage1_eval_threshold
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)

        if args.optimize_method == 'GreyWolfOptimizer':
            self.a = args.optimize_param.a
            self.delta = args.optimize_param.delta
            self.population_n = args.optimize_param.population_n
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)

        if args.optimize_method == 'TestOptimizationStrategy':
            self.emo_ga_method = args.optimize_param.emo_ga_method
            self.emo_ga_surrogate_method_n = args.optimize_param.emo_ga_surrogate_method_n
            self.emo_ga_surrogate_method_k = args.optimize_param.emo_ga_surrogate_method_k
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration
        if args.optimize_method == "TestLightningKEP":
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == 'Surrogate-RandomSearch':
            self.srs_simulation_per_generation = args.optimize_param.samplesPerGeneration
            self.srs_population_size = args.optimize_param.srs_population_size

            self.emo_ga_surrogate_method_n = args.optimize_param.emo_ga_surrogate_method_n
            self.emo_ga_train_vae = args.optimize_param.emo_ga_train_vae
            self.emo_ga_method = args.optimize_param.emo_ga_method
            self.emo_ga_epochs = args.optimize_param.emo_ga_epochs
            self.emo_ga_vae_batch_size = args.optimize_param.emo_ga_vae_batch_size
            self.emo_ga_vae_lr = args.optimize_param.emo_ga_vae_lr
            self.emo_ga_finetune_decoder = args.optimize_param.emo_ga_finetune_decoder
            self.emo_ga_decoder_lr = args.optimize_param.emo_ga_decoder_lr
            self.emo_ga_resnet_epoch = args.optimize_param.emo_ga_resnet_epoch
            self.emo_ga_res_batch_size = args.optimize_param.emo_ga_res_batch_size
            self.emo_ga_resnet_lr = args.optimize_param.emo_ga_resnet_lr
            self.emo_ga_resnet_update_epoch = args.optimize_param.emo_ga_resnet_update_epoch
            self.emo_ga_res_update_batch_size = args.optimize_param.emo_ga_res_update_batch_size
            self.emo_ga_input_dim = args.optimize_param.emo_ga_input_dim
            self.emo_ga_hidden_dim = args.optimize_param.emo_ga_hidden_dim
            self.emo_ga_latent_dim = args.optimize_param.emo_ga_latent_dim
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)
            # 通用参数
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == "StableBacchusAda_new":
            self.samples_per_generation: int = args.optimize_param.sampleSizePerGeEdt
            self.local_search_per_Nround: int = args.optimize_param.localNroundEdt
            self.max_global_search_iter: int = args.optimize_param.maxGlobalIterEdt
            self.max_local_search_iter: int = args.optimize_param.maxLocalIterEdt
            self.resample_num: int = args.optimize_param.resampleNumEdt
            self.max_search_time: int = args.optimize_param.maxSearchTimeEdt
            self.using_filter: bool = args.optimize_param.filterOrNotCbx
            self.random_weight_coeff: float = args.optimize_param.randomWeigCoeEdt
            self.using_adapt_weight: bool = args.optimize_param.adapWeigOrNotCbx

            self.filter_type: int = args.optimize_param.filterTypeEdt
            self.filter_std: float = args.optimize_param.filterStdEdt
            self.update_sampler_per_rounds: int = args.optimize_param.updateSamplerEdt
            self.force_update_sampler_rounds = get_float_list(args.optimize_param.forceUpdateEdt, 1)
            self.n_mat_pool: int = args.optimize_param.matPoolEdt
            self.adapt_weight_smooth_coeff: float = args.optimize_param.adapWeigEMCEdt
            self.adapt_weight_min_weight_prop: float = args.optimize_param.adapWeigMinPropEdt

            self.neighbour_num: int = args.optimize_param.neighborNumEdt
            self.neighbour_size: int = args.optimize_param.neighbourSizeEdt
            self.neighbour_lifelong: int = args.optimize_param.maxNoImproveEdt
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.similar_sample_tolerance = args.optimize_param.similarToleranceEdt

            # GA算法参数
            self.ga_population_size = args.optimize_param.populationSizeEdt
            self.ga_cross_area_ratio = args.optimize_param.crossAreaRatioEdt
            self.ga_samples_per_generation = args.optimize_param.sampleSizePerGeEdt
            self.ga_crossover_probability = args.optimize_param.crossoverEdt
            self.ga_mutate_probability = args.optimize_param.mutationEdt
            self.ga_tournament_n = args.optimize_param.tournamentEdt

            self.start_mesh = get_float_list(args.optimize_param.start_mesh, 1)
            self.progress_var_n = args.optimize_param.progress_var_n

            self.max_mutate_length_ratio_shape_modify: float = args.optimize_param.max_mutate_length_ratio_shape_modify
            self.add_mutate_layer_shape_modify: bool = args.optimize_param.add_mutate_layer_shape_modify
            self.mutate_layer_max_changed_pixel_ratio_shape_modify: float = args.optimize_param.mutate_layer_max_changed_pixel_ratio_shape_modify
            self.v_prob_shape_modify: float = args.optimize_param.v_prob_shape_modify

            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method,surrogate_param=args.optimize_param.surrogate_method_param)
            # todo debug

        if args.optimize_method == "KnowledgeInjection":
            self.LDI_0_coeff: float = 0.1
            self.samples_per_generation: int = 1
        if args.optimize_method == "KEPHPSearch":
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == "CGAN_Using_Was_Loss":
            self.cgan_epoch: int = args.optimize_param.cgan_epoch
            self.vae_epoch: int = args.optimize_param.vae_epoch
            self.vae_batch_size: int = args.optimize_param.vae_batch_size
            self.cgan_lr: int = args.optimize_param.cgan_lr
            self.vae_lr: int = args.optimize_param.vae_lr
            self.cgan_batch_size: int = args.optimize_param.cgan_batch_size
            self.cgan_latent_dim: int = args.optimize_param.cgan_latent_dim
            self.embedding_dim: int = args.optimize_param.embedding_dim
            self.cgan_optimizer_b1: int = args.optimize_param.cgan_optimizer_b1
            self.cgan_optimizer_b2: int = args.optimize_param.cgan_optimizer_b2
            self.cgan_fake_data_per_epoch: int = args.optimize_param.cgan_fake_data_per_epoch
            self.cgan_condition_value: int = args.optimize_param.cgan_condition_value
            self.gen_condition_value_min: int = args.optimize_param.gen_condition_value_min
            self.gen_condition_value_max: int = args.optimize_param.gen_condition_value_max
            self.train_predictor_from_zero_or_not = args.optimize_param.train_predictor_from_zero_or_not
            self.cgan_update_dataset_enable: int = args.optimize_param.cgan_update_dataset_enable
            self.cgan_expert_solution_pretrain_enable: int = args.optimize_param.cgan_expert_solution_pretrain_enable
            self.sharp_value: int = args.optimize_param.sharp_value
            self.g_train_frequency: int = args.optimize_param.g_train_frequency
            self.lambda_gp: int = args.optimize_param.lambda_gp
            self.augment_factor = args.optimize_param.augment_factor
            self.cgan_simu_test_or_not = args.optimize_param.cgan_simu_test_or_not
            self.scheduler_step: int = args.optimize_param.scheduler_step
            self.scheduler_gamma: int = args.optimize_param.scheduler_gamma
            self.cgan_simu_num: int = args.optimize_param.cgan_simu_num
            self.cgan_design_num: int = args.optimize_param.cgan_design_num
            self.except_value: int = args.optimize_param.except_value
            self.cgan_sample_interval: int = args.optimize_param.cgan_sample_interval
            self.cgan_simulation_sample_n: int = args.optimize_param.cgan_simulation_sample_n
            self.cgan_n_cpu: int = args.optimize_param.cgan_n_cpu
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == "TYCVAEO":
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == "IDN":
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration

        if args.optimize_method == 'VQVAE_GenCO':
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.selected_model = args.selected_model
            self.train_model_from_scratch_or_not = args.training_param.train_model_from_scratch_or_not
            self.model_epoch = args.training_param.model_epoch
            self.batch_size = args.training_param.batch_size
            self.model_update_epoch = args.training_param.model_update_epoch
            self.update_batch_size = args.training_param.update_batch_size
            if args.training_param.update_method == "每次更新重新训练":
                self.update_method = "continue_train_from_scratch"
            elif args.training_param.update_method == "持续微调":
                self.update_method = "continue_finetune"
            else:
                self.update_method = 'continue_with_scratch_and_finetune'
            self.train_from_scratch_n = args.training_param.train_from_scratch_n
            self.check_val_every_n_epoch = args.training_param.check_val_every_n_epoch
            self.transfer_learning_or_not = args.training_param.transfer_learning_or_not
            self.pretrained_model_path = args.training_param.pretrained_model_path

            if args.selected_model == "ResNet":
                self.resnet_lr = args.training_param.selected_model_param.resnet_lr
                self.fc_lr = args.training_param.selected_model_param.fc_lr
                self.upsample_size = int(args.training_param.selected_model_param.upsample_size)
                self.pretrained = args.training_param.selected_model_param.pretrained
                self.split_mat_as_fragment_shape = args.training_param.selected_model_param.split_mat_as_fragment_shape

                # 导入仿真策略参数
                self.surrogate_method = DefaultMunch()
                self.surrogate_method.method = args.surrogate_method
                self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)

        if args.optimize_method == "RepTestPredictor":
            self.rep = args.optimize_param.rep



        if args.optimize_method == "GenerativeOptimizer":

            self.selected_model = args.selected_generative_model
            self.population_n = args.optimize_param.population_n
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)
            if self.selected_model == "DCGAN":
                """
                ---DCGAN参数---
                与cGAN相同
                """
                self.img_size = args.optimize_param.model_method_param.img_size
                # self.num_channels = args.optimize_param.model_method_param.num_channels
                self.latent_dim = args.optimize_param.model_method_param.latent_dim
                self.epoch = args.optimize_param.model_method_param.epoch
                self.g_lr = args.optimize_param.model_method_param.g_lr
                self.d_lr = args.optimize_param.model_method_param.d_lr
                self.initial_train_ratio = args.optimize_param.model_method_param.initial_train_ratio
                self.split_dataset_or_not = args.optimize_param.model_method_param.split_dataset_or_not
                self.seed = args.optimize_param.model_method_param.seed
                self.evaluate_sample_n = args.optimize_param.model_method_param.evaluate_sample_n
                self.check_val_every_n_epoch = args.optimize_param.model_method_param.check_val_every_n_epoch
                self.g_clip_grad_norm = args.optimize_param.model_method_param.g_clip_grad_norm
                self.d_clip_grad_norm = args.optimize_param.model_method_param.d_clip_grad_norm
            if self.selected_model == "cGAN":
                """
                ---cGAN参数---
                img_size: int = 128
                num_channels: int = 1
                 # 需要后处理读入, 不在这里读入
                fragment_shape: List[Tuple[int, int]] = field(default_factory=lambda: list())

                latent_dim: int = 128
                condition_dim: int = 1  # 暂时只支持从目标函数值生成,先不读入
                g_lr: float = 2e-4
                d_lr: float = 2e-4
                # 训练相关参数
                batch_size: int = 128
                initial_train_ratio: float = 0.9
                split_dataset_or_not: bool = True
                seed: int = 0
                """
                self.img_size = args.optimize_param.model_method_param.img_size
                # self.num_channels = args.optimize_param.model_method_param.num_channels
                self.latent_dim = args.optimize_param.model_method_param.latent_dim
                self.epoch = args.optimize_param.model_method_param.epoch
                self.g_lr = args.optimize_param.model_method_param.g_lr
                self.d_lr = args.optimize_param.model_method_param.d_lr
                self.initial_train_ratio = args.optimize_param.model_method_param.initial_train_ratio
                self.split_dataset_or_not = args.optimize_param.model_method_param.split_dataset_or_not
                self.seed = args.optimize_param.model_method_param.seed
                self.evaluate_sample_n = args.optimize_param.model_method_param.evaluate_sample_n
                self.check_val_every_n_epoch = args.optimize_param.model_method_param.check_val_every_n_epoch
                self.g_clip_grad_norm = args.optimize_param.model_method_param.g_clip_grad_norm
                self.d_clip_grad_norm = args.optimize_param.model_method_param.d_clip_grad_norm
            if self.selected_model == "cVQVAE":

                self.img_size = args.optimize_param.model_method_param.img_size
                self.embedding_dim = args.optimize_param.model_method_param.embedding_dim
                self.num_embeddings = args.optimize_param.model_method_param.num_embeddings
                self.commitment_cost = args.optimize_param.model_method_param.commitment_cost
                self.decay = args.optimize_param.model_method_param.decay
                self.learnable_codebook = args.optimize_param.model_method_param.learnable_codebook
                self.epoch = args.optimize_param.model_method_param.epoch
                self.ae_warmup_epoch = args.optimize_param.model_method_param.ae_warmup_epoch
                self.lr = args.optimize_param.model_method_param.lr
                self.batch_size = args.optimize_param.model_method_param.batch_size
                self.initial_train_ratio = args.optimize_param.model_method_param.initial_train_ratio
                self.split_dataset_or_not = args.optimize_param.model_method_param.split_dataset_or_not
                self.seed = args.optimize_param.model_method_param.seed
                self.evaluate_sample_n = args.optimize_param.model_method_param.evaluate_sample_n
                self.check_val_every_n_epoch = args.optimize_param.model_method_param.check_val_every_n_epoch
                self.kl_weight = args.optimize_param.model_method_param.kl_weight
                self.warmup_epochs = args.optimize_param.model_method_param.warmup_epochs

            if self.selected_model == "cVAE":

                self.img_size = args.optimize_param.model_method_param.img_size
                self.embedding_dim = args.optimize_param.model_method_param.embedding_dim
                self.num_embeddings = args.optimize_param.model_method_param.num_embeddings
                self.commitment_cost = args.optimize_param.model_method_param.commitment_cost
                self.decay = args.optimize_param.model_method_param.decay
                self.learnable_codebook = args.optimize_param.model_method_param.learnable_codebook
                self.epoch = args.optimize_param.model_method_param.epoch
                self.lr = args.optimize_param.model_method_param.lr
                self.batch_size = args.optimize_param.model_method_param.batch_size
                self.initial_train_ratio = args.optimize_param.model_method_param.initial_train_ratio
                self.split_dataset_or_not = args.optimize_param.model_method_param.split_dataset_or_not
                self.seed = args.optimize_param.model_method_param.seed
                self.evaluate_sample_n = args.optimize_param.model_method_param.evaluate_sample_n
                self.check_val_every_n_epoch = args.optimize_param.model_method_param.check_val_every_n_epoch
                self.kl_weight = args.optimize_param.model_method_param.kl_weight

            if self.selected_model == "InvGrad":

                self.img_size = args.optimize_param.model_method_param.img_size
                self.num_channels = args.optimize_param.model_method_param.num_channels
                self.lr = args.optimize_param.model_method_param.lr
                self.batch_size = args.optimize_param.model_method_param.batch_size
                self.epoch = args.optimize_param.model_method_param.epoch
                self.variation = args.optimize_param.model_method_param.variation
                self.initial_train_ratio = args.optimize_param.model_method_param.initial_train_ratio
                self.split_dataset_or_not = args.optimize_param.model_method_param.split_dataset_or_not
                self.seed = args.optimize_param.model_method_param.seed
                self.using_transform = args.optimize_param.model_method_param.using_transform
                self.evaluate_sample_n = args.optimize_param.model_method_param.evaluate_sample_n
                self.grad_epoch = args.optimize_param.model_method_param.grad_epoch
                self.grad_lr = args.optimize_param.model_method_param.grad_lr

            if self.selected_model == "GenCO":

                self.img_size = args.optimize_param.model_method_param.img_size
                self.num_channels = args.optimize_param.model_method_param.num_channels
                self.embedding_dim = args.optimize_param.model_method_param.embedding_dim
                self.lr = args.optimize_param.model_method_param.lr
                self.batch_size = args.optimize_param.model_method_param.batch_size
                self.epoch = args.optimize_param.model_method_param.epoch
                self.variation = args.optimize_param.model_method_param.variation
                self.grad_epoch = args.optimize_param.model_method_param.grad_epoch
                self.grad_lr = args.optimize_param.model_method_param.grad_lr

            if self.selected_model == "IDN":

                self.img_size = args.optimize_param.model_method_param.img_size
                self.num_channels = args.optimize_param.model_method_param.num_channels
                self.lr = args.optimize_param.model_method_param.lr
                self.batch_size = args.optimize_param.model_method_param.batch_size
                self.epoch = args.optimize_param.model_method_param.epoch
                self.initial_train_ratio = args.optimize_param.model_method_param.initial_train_ratio
                self.split_dataset_or_not = args.optimize_param.model_method_param.split_dataset_or_not
                self.seed = args.optimize_param.model_method_param.seed
                self.using_transform = args.optimize_param.model_method_param.using_transform
                self.evaluate_sample_n = args.optimize_param.model_method_param.evaluate_sample_n

                # 导入仿真策略参数
                self.surrogate_method = DefaultMunch()
                self.surrogate_method.method = args.surrogate_method
                self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)


        if args.optimize_method == "SurrogateGA":
            self.population_n = args.optimize_param.population_n
            self.mutation_rate = args.optimize_param.mutation_rate
            self.tournament_size = args.optimize_param.tournament_size

            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)

        if args.optimize_method == "SAHSO":
            self.population_n = args.optimize_param.population_n
            self.T0 = args.optimize_param.T0
            self.w = args.optimize_param.w
            self.c1 = args.optimize_param.c1
            self.c2 = args.optimize_param.c2
            self.total_sample_nums = args.optimize_param.total_sample_nums
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)

        if args.optimize_method == 'CMAES':
            self.population_n = args.optimize_param.population_n
            self.mu = args.optimize_param.mu
            self.sigma = args.optimize_param.sigma
            # 这两个参数仅仅是为了兼容算法框架
            self.LDI_0_coeff: float = args.optimize_param.similarDistanceEdt
            self.samples_per_generation: int = args.optimize_param.samplesPerGeneration
            # todo debug
            # 导入仿真策略参数
            self.surrogate_method = DefaultMunch()
            self.surrogate_method.method = args.surrogate_method
            self.surrogate_method = load_surrogate_method_param(self.surrogate_method, surrogate_param=args.optimize_param.surrogate_method_param)



    # def save_json(self, path=None):
    #     this_dict = super(OPTIMIZER_CONF, self).save_json()
    #     if path is None:
    #         with open(pathlib.Path(self.output_path) / "./optimizer.conf", mode='w+') as fp:
    #             json.dump(this_dict, fp)
    #     else:
    #         with open(pathlib.Path(path), mode='w+') as fp:
    #             json.dump(this_dict, fp)

    # def load_json(self, path=None):
    #     if path is None:
    #         my_path = pathlib.Path(self.output_path) / "./optimizer.conf"
    #     else:
    #         my_path = path
    #     parsed_dict, parsed_type_dict = super(OPTIMIZER_CONF, self).load_json(my_path)
    #     self.__dict__ = parsed_dict
    #     self.parsed_type = copy.deepcopy(parsed_type_dict)
    #     self.parsed_dict = copy.deepcopy(parsed_dict)
    #


class GLOBAL_CONF(CONF):
    def __init__(self, args=None, path=None):
        """
        ******下列参数需要在CONF完成初始化后额外赋值，因为这些变量动态依赖于其他变量，所以这些变量不能从args中读取

        ******下列参数需要从args中读取，但是不应该放在优化设置界面中

        ******下列参数需要从args中读取
        变量名                    中文名称             类型&默认值          说明
        initial_sample_number   初始样本量           int   100
        initialize_data         是否初始化数据        bool  True
        sampler_sequence_config   图层设置序列          List  None          必须正确赋值

        """
        super(GLOBAL_CONF, self).__init__()
        if args is None:
            assert path is not None
            self.load_json(path)
            return
        if not hasattr(args, "global_param"):
            return
        if hasattr(args, "global_param"):
            args.global_param = DefaultMunch.fromDict(args.global_param)

        self.sampler_sequence_config = args.sampler_sequence_config

        self.optimization_iter_n = args.global_param.optimization_iter_n
        self.initial_sample_number = args.global_param.initial_sample_number
        self.initialize_data = args.global_param.initialize_data

        self.using_soft_distance_target: bool = args.global_param.using_soft_distance_target
        self.using_distance_target: bool = args.global_param.using_distance_target
        self.surrogate_module_sim: bool = args.global_param.surrogate_module_sim

    # def save_json(self, path=None):
    #     this_dict = super(GLOBAL_CONF, self).save_json()
    #     if path is None:
    #         with open(pathlib.Path(self.output_path) / "./Sampler.conf", mode='w+') as fp:
    #             json.dump(this_dict, fp)
    #     else:
    #         with open(pathlib.Path(path), mode='w+') as fp:
    #             json.dump(this_dict, fp)

    # def load_json(self, path=None):
    #     if path is None:
    #         my_path = pathlib.Path(self.output_path) / "./Sampler.conf"
    #     else:
    #         my_path = path
    #     parsed_dict, parsed_type_dict = super(GLOBAL_CONF, self).load_json(my_path)
    #     self.__dict__ = parsed_dict
    #     self.parsed_type = copy.deepcopy(parsed_type_dict)
    #     self.parsed_dict = copy.deepcopy(parsed_dict)


class TEST_SURROGATE_MODULE_CONF(CONF):
    def __init__(self, args=None, path=None):

        """
        从args及self中读取训练器所需要的一些参数
        有一些参数不能静态确定，具体如下：
        Trainer还需要读入data_index_to_frequency，y_dim：
          该变量来源于dataset。从self.data_index_to_frequency,self.y_dim读入。需要在初始化数据集后，再读入。
        Trainer还需要读入model_pretrained：
           需要根据iter_n 与train_from_scratch_per_rounds的关系，手动判断model_pretrained取值。
        """

        """
            仿真测评预测器参数配置
        """
        super(TEST_SURROGATE_MODULE_CONF, self).__init__()

        if args is None:
            assert path is not None
            self.load_json(path)
            return
        if not hasattr(args, "test_surrogate_module_method"):
            return
        if hasattr(args, "test_surrogate_module_param"):
            args.test_surrogate_module_param = DefaultMunch.fromDict(args.test_surrogate_module_param)
        self.training_policy = 'StandardNet'
        # todo:完善typing
        self.lr: Union[float, None] = None
        self.epoch: Union[int, None] = None
        self.fragment_shape: Union[List, None] = None
        self.fragment_setting_list: Union[List, None] = None
        self.data_index_to_frequency: Union[List, None] = None
        self.y_dim: Union[int, None] = None
        self.model_pretrained: [str, None, bool] = None
        # todo:界面接入
        self.device: str = "cuda:0"

        if hasattr(args.test_surrogate_module_param, "seed") and args.test_surrogate_module_param.seed is not None:
            self.seed = args.test_surrogate_module_param.seed
        else:
            self.seed = 0
        if args.test_surrogate_module_method == 'LightningSurrogateModule':
            # todo 临时处理
            # self.selected_model = args.test_surrogate_model
            self.train_model_from_scratch_or_not = args.test_surrogate_module_param.train_model_from_scratch_or_not
            self.model_epoch = args.test_surrogate_module_param.model_epoch
            self.batch_size = args.test_surrogate_module_param.batch_size
            self.model_update_epoch = args.test_surrogate_module_param.model_update_epoch
            self.update_batch_size = args.test_surrogate_module_param.update_batch_size
            if args.test_surrogate_module_param.update_method == "每次更新重新训练":
                self.update_method = "continue_train_from_scratch"
            elif args.test_surrogate_module_param.update_method == "持续微调":
                self.update_method = "continue_finetune"
            else:
                self.update_method = 'continue_with_scratch_and_finetune'
            self.train_from_scratch_n = args.test_surrogate_module_param.train_from_scratch_n
            self.check_val_every_n_epoch = args.test_surrogate_module_param.check_val_every_n_epoch
            self.transfer_learning_or_not = False
            self.pretrained_model_path = args.test_surrogate_module_param.pretrained_model_path

            if args.test_surrogate_model == "SurrogateResNet":
                # todo 临时处理
                self.selected_model = "ResNet"
                self.resnet_lr = args.test_surrogate_module_param.test_surrogate_model_param.resnet_lr
                self.fc_lr = args.test_surrogate_module_param.test_surrogate_model_param.fc_lr
                self.upsample_size = int(args.test_surrogate_module_param.test_surrogate_model_param.upsample_size)
                self.pretrained = args.test_surrogate_module_param.test_surrogate_model_param.pretrained
                self.split_mat_as_fragment_shape = args.test_surrogate_module_param.test_surrogate_model_param.split_mat_as_fragment_shape
                self.model_depth = int(args.test_surrogate_module_param.test_surrogate_model_param.model_depth)

            if args.test_surrogate_model == "SurrogateGoogleNet":
                # todo 临时处理
                self.selected_model = "GoogleNet"
                self.googlenet_lr = args.test_surrogate_module_param.test_surrogate_model_param.googlenet_lr
                self.weight_decay = args.test_surrogate_module_param.test_surrogate_model_param.weight_decay
                self.upsample_size = int(args.test_surrogate_module_param.test_surrogate_model_param.upsample_size)
                self.pretrained = args.test_surrogate_module_param.test_surrogate_model_param.pretrained
                self.split_mat_as_fragment_shape = args.test_surrogate_module_param.test_surrogate_model_param.split_mat_as_fragment_shape


        """
        ******控制参数******
        save_model:是否在每轮优化后，保存模型文件;bool;False时train_from_scratch_xxx参数失效，每一轮网络训练按照一般参数(lr,epoch)进行;t0=True
        output_path:每轮优化后，训练完成的模型文件将保存在(output_path / const.MODEL_PATH / args.model_name)，str;t0="./output"
        print_freq：正整数，训练时每隔N轮显示训练集、验证集LOSS情况;int;>0;t0=20
        """
        self.output_path = args.output_path
        self.save_model = args.save_model
        # todo:界面接入
        self.print_freq = args.print_freq

        # todo: 这些参数对应的polydataset（曲线傅里叶变换模块）暂未实现，读取后再次调整为False
        # self.using_polydataset = args.using_polydataset
        # self.using_polydataset = False
        # self.polyfit_dim = None

    # def save_json(self, path=None):
    #     this_dict = super(TRAINER_CONF, self).save_json()
    #     if path is None:
    #         with open(pathlib.Path(self.output_path) / "./trainer.conf", mode='w+') as fp:
    #             json.dump(this_dict, fp)
    #     else:
    #         with open(pathlib.Path(path), mode='w+') as fp:
    #             json.dump(this_dict, fp)
    #
    # def load_json(self, path=None):
    #     if path is None:
    #         my_path = pathlib.Path(self.output_path) / "./trainer.conf"
    #     else:
    #         my_path = path
    #     parsed_dict, parsed_type_dict = super(TRAINER_CONF, self).load_json(my_path)
    #     self.__dict__ = parsed_dict
    #     self.parsed_type = copy.deepcopy(parsed_type_dict)
    #     self.parsed_dict = copy.deepcopy(parsed_dict)

def load_surrogate_method_param(self_param_dict,surrogate_param:object):
    for name,value in surrogate_param.__dict__.items():
        setattr(self_param_dict,name,value)
    return self_param_dict




