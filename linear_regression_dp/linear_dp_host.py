import copy
import numpy as np

from federatedml.model_base import ModelBase
from federatedml.linear_model.logistic_regression.hdp_vfl.batch_data import Host
from federatedml.param.linear_regression_dp_param import LinearDpParam
from federatedml.transfer_variable.transfer_class.hetero_linr_dp_transfer_variable import HeteroLinRDpTransferVariable
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.statistic import data_overview
from federatedml.linear_model.linear_regression.hetero_linear_regression_dp import model_weight
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.protobuf.generated import linr_dp_model_meta_pb2,linr_dp_model_param_pb2

class LinearRDpHost(ModelBase):
    def __init__(self):
        super().__init__()
        self.batch_generator = Host()
        self.model_param = LinearDpParam()
        self.transfer_variable = HeteroLinRDpTransferVariable()
        self.header = None
        self.model = None
        # 以下三个为传输变量
        self.ir_a = None
        self.ir_b = None
        self.host_wx = None
        self.transfer_paillier = None
        # 存取数据的维度特征
        self.data_shape = None
        # 密钥设置
        self.cipher = PaillierEncrypt()
        self.encrypted_calculator = None
        # 用来搞加密计算的
        self.encrypted_calculator = None
        # 用来表示中间的聚合结果(密文状态下)，它的值就是0.25 * wx - 0.5 * y
        self.aggregated_forwards = None

    def fit(self, data_instances):
        LOGGER.info("开始纵向线性回归")
        LOGGER.info("开始纵向线性回归训练")
        self._abnormal_detection(data_instances)
        self.register_gradient_sync(self.transfer_variable)
        self.data_shape = data_overview.get_features_shape(data_instances)
        LOGGER.info("这里打印查看第一个data_instances值的数据类型：{}".format(data_instances.first()[1].features.shape[0]))
        # 有关于密钥的处理
        pub_key = self.transfer_paillier.get(idx=0, suffix=("pub_key",))  # 这里有关于pub_key的类型就是一个对象，不再是列表中存放对象
        self.cipher.set_public_key(pub_key)

        #模型初始化
        self.model = model_weight.LinearDpWeightHost()
        self.model.initialize(data_instances)

        #批处理模块初始化
        self.batch_generator.register_batch_generator(self.transfer_variable)
        batch_suffix = ("batch_info",)
        self.batch_generator.initialize_batch_generator(data_instances, suffix=batch_suffix)

        #加密模块
        self.encrypted_calculator = [EncryptModeCalculator(self.cipher) for _ in range(self.batch_generator.batch_nums)]

        # 开始正式的循环迭代的阶段
        iteration = 0  # 用来记录epoches次数
        suffix_tag = 0  # 用来表示传输变量表示，它的值也是最终的迭代次数
        encrypt_suffix = ("encrypt_suffix",)
        uni_guest_gradient_suffix = ("uni_guest_gradient_suffix",)
        uni_host_gradient_suffix = ("uni_host_gradient_suffix",)
        fore_gradient_suffix = ("fore_gradient",)
        host_to_arbiter_suffix = ("host_to_arbiter_suffix",)
        arbiter_to_host_suffix = ("arbiter_to_host_suffix",)

        while iteration <= self.e:
            for data_inst in self.batch_generator.generator_batch_data():
                LOGGER.info("----------------当前迭代次数:{}-------------------".format(suffix_tag))
                # 以下几个都是数据标签，用来区分传输变量
                suffix_e = encrypt_suffix + (suffix_tag,)  # 初始用来传输加密wx的，host方
                suffix_f = fore_gradient_suffix + (suffix_tag,)  # 用来传输fore_gradient的值
                suffix_ug = uni_guest_gradient_suffix + (suffix_tag,)  # 用来传输guest方的单侧梯度的
                suffix_uh = uni_host_gradient_suffix + (suffix_tag,)  # host方密文梯度
                suffix_ha = host_to_arbiter_suffix + (suffix_tag,)  # host方发往arbiter的加入噪音的梯度（密文）
                suffix_ah = arbiter_to_host_suffix + (suffix_tag,)  # arbiter方发往host解密后自身的梯度

                LOGGER.info("开始计算数据的内积")
                ir_b = self.model.compute_forwards(data_inst, self.model.w)

                LOGGER.info("开始将密文forwards发送给guest方")
                encrypted_ir_b = self.encrypted_calculator[suffix_tag % self.batch_generator.batch_nums].encrypt(ir_b)
                self.encrypted_ir_b.remote(obj=encrypted_ir_b, role=consts.GUEST, idx=0, suffix=suffix_e)

                LOGGER.info("开始从guest方接收：average_unilateral_gradient_guest的结果、fore_gradient的结果")
                average_unilateral_gradient_guest = self.average_unilateral_gradient_guest.get(idx=-1, suffix=suffix_ug)
                fore_gradient = self.fore_gradient.get(idx=-1, suffix=suffix_f)

                LOGGER.info("host方开始计算自身的单侧梯度")
                average_unilateral_gradient_host = self.model.compute_gradient(data_inst,fore_gradient[0],data_inst.count())

                LOGGER.info("开始将average_unilateral_gradient_host发给guest方")
                self.average_unilateral_gradient_host.remote(obj=average_unilateral_gradient_host, idx=-1,
                                                             role=consts.GUEST, suffix=suffix_uh)

                LOGGER.info("开始在average_unilateral_gradient_guest的结果上添加噪音，保护的是host端的wx")
                LOGGER.info("开始生成高斯分布需要的:loc、sigma")
                shape_guest = average_unilateral_gradient_guest[0].shape[0]
                loc, sigma = self.model.gaussian(self.delta, self.epsilon, self.L, self.e,
                                                 int(self.r * self.e), self.learning_rate,
                                                 data_inst.count(), self.k, shape_guest)

                LOGGER.info("开始对guest梯度数据添加噪声")  # 如下结果是一个numpy格式的数据
                average_unilateral_gradient_guest_noise = self.model.sec_intermediate_result(
                    average_unilateral_gradient_guest[0], loc, sigma)
                self.host_to_arbiter.remote(obj=average_unilateral_gradient_guest_noise, idx=-1, role=consts.ARBITER,
                                            suffix=suffix_ha)
                LOGGER.info("开始从arbiter接收解密梯度")
                gradient_host = self.arbiter_to_host.get(idx=-1, suffix=suffix_ah)

                LOGGER.info("开始更新模型参数")
                self.model.update_model(gradient_host[0], self.learning_rate, self.lamb)

                LOGGER.info("开始进行梯度剪切部分")
                self.model.norm_clip(self.k)

                suffix_tag += 1

            iteration += 1

        LOGGER.info("训练正式结束")
        LOGGER.info("host方的模型参数：{}".format(self.model.w))

        self.data_output = self.model.w

    def save_data(self):
        return self.data_output

    def predict(self, data_inst):
        """
        纵向线性回归的预测部分,由于host这里只作为参与方，所以没有输出
        Parameters
        -------------------
        data_inst:Dtable,数据的输入
        """
        LOGGER.info("------------------开始预测阶段----------------------")
        self._abnormal_detection(data_inst)
        self.data_shape = data_overview.get_features_shape(data_inst)
        # 注册传输变量
        self.register_gradient_sync(self.transfer_variable)
        # 预测阶段相当于重新初始化一波，所以这个时候务必注意将用到的东西重新初始化。例如最重要的weight
        # 初始化模型参数
        self.model = model_weight.LinearDpWeightHost()
        self.model.w = self.data_output
        data_instances = data_inst
        LOGGER.info("开始计算host方的内积wx")
        wx_host = data_instances.mapValues(lambda x: np.dot(x.features, self.data_output))
        LOGGER.info("开始将host方的内积wx发送给guest方")
        self.host_wx.remote(wx_host, role=consts.GUEST, idx=-1)
        LOGGER.info("host方完成自己的任务")

    def _init_model(self, params):
        """
        参数的具体含义可以参考params类的说明
        """
        self.epsilon = params.epsilon
        self.delta = params.delta
        self.L = params.L
        self.beta_theta = params.beta_theta
        self.beta_y = params.beta_y
        self.e = params.e
        self.r = params.r
        self.k = params.k
        self.learning_rate = params.learning_rate
        self.lamb = params.lamb
        self.k_y = params.k_y

    def _get_meta(self):
        """
        按照开发文档的说明，这个函数用来保存某次任务的配置
        """
        meta_protobuf_obj = linr_dp_model_meta_pb2.LinRDpModelMeta(epsilon=self.epsilon,
                                                                   delta=self.delta,
                                                                   L=self.L,
                                                                   beta_theta=self.beta_theta,
                                                                   beta_y=self.beta_y,
                                                                   e=self.e,
                                                                   r=self.r,
                                                                   k=self.k,
                                                                   learning_rate=self.learning_rate,
                                                                   lamb=self.lamb,
                                                                   k_y=self.k_y)
        return meta_protobuf_obj

    def _get_param(self):
        """
        这个函数用来保存当前任务的运行结果
        """
        weight_dict = {}
        weight = {}
        LOGGER.info("self.data_output的值是：{}".format(self.data_output))
        for i in range(self.data_shape):
            result = "w" + str(i)
            weight_dict[result] = self.model.w[i]
        weight["weight"] = weight_dict
        param_protobuf_obj = linr_dp_model_param_pb2.LinRDpModelParam(**weight)

        return param_protobuf_obj

    def export_model(self):
        """
        这个函数应当指的是当某次任务结束后，将模型和当前任务的一些参数配置啥玩意的保存起来
        """
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            "LinRDpMeta": meta_obj,
            "LinRDpParam": param_obj
        }
        return result

    def load_model(self, model_dict):
        """
        这个函数指的是当我们进行预测任务的时候，框架估计会调用这个函数。然后载入之前的模型，直接来用
        """
        result_obj = list(model_dict.get('model').values())[0].get("LinRDpParam")
        # 将值取出来，搞成数组的形式，然后传给self.data_output,再实际测试是否赋值给self.data_output
        self.data_output = []
        for i in range(len(result_obj.weight)):
            result = "w" + str(i)
            self.data_output.append(result_obj.weight[result])
        self.data_output = np.array(self.data_output)

    def _abnormal_detection(self,data_instances):
        """
        主要用来检查数据的有效性
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        ModelBase.check_schema_content(data_instances.schema)

    def register_gradient_sync(self,transfer_variable):
        self.ir_a = transfer_variable.ir_a
        self.ir_b = transfer_variable.ir_b
        self.host_wx = transfer_variable.host_wx.disable_auto_clean()
        self.transfer_paillier = transfer_variable.paillier_pubkey
        self.encrypted_ir_b = transfer_variable.encrypted_ir_b
        self.average_unilateral_gradient_guest = transfer_variable.average_unilateral_gradient_guest
        self.average_unilateral_gradient_host = transfer_variable.average_unilateral_gradient_host
        self.fore_gradient = transfer_variable.fore_gradient
        self.host_to_arbiter = transfer_variable.host_to_arbiter
        self.arbiter_to_host = transfer_variable.arbiter_to_host
