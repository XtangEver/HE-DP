import copy
import numpy as np

from federatedml.model_base import ModelBase
from federatedml.linear_model.poisson_regression.hetero_poisson_dp_regression.batch_data import Guest
from federatedml.param.poisson_dp_regression_param import PoissonDpParam
from federatedml.transfer_variable.transfer_class.hetero_poisson_dp_transfer_variable import HeteroPoissonDpTransferVariable
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.statistic import data_overview
from federatedml.linear_model.poisson_regression.hetero_poisson_dp_regression import model_weight
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.protobuf.generated import poisson_dp_model_param_pb2,poisson_dp_model_meta_pb2

class PoissonDpGuest(ModelBase):
    def __init__(self):
        super(PoissonDpGuest, self).__init__()
        self.batch_generator = Guest()
        self.model_param = PoissonDpParam() #初始化参数，将当前算法组件相关的参数载入进来
        self.transfer_variable = HeteroPoissonDpTransferVariable() #传输变量
        self.header = None #取出来头部信息，但是这个是属性貌似后面用不到。
        self.model = None #这个属性用来存最终的模型参数
        #以下几个表示传输变量
        self.ir_a = None
        self.ir_b = None
        self.host_wx = None
        self.transfer_paillier = None #这个属性当时干嘛列出它？
        #存取数据的维度信息
        self.data_shape = None
        #密钥设置，其中第二个属性的意义何在？
        self.cipher = PaillierEncrypt()
        #用来表示中间的聚合结果
        self.aggregated_forwards = None

    def fit(self, data_instances,validate_data = None):
        LOGGER.info("开始纵向线性回归训练")
        #检查数据是否正常
        self._abnormal_detection(data_instances)
        #传输变量初始化
        self.register_gradient_sync(self.transfer_variable)
        #获取数据的维度特征
        self.data_shape = data_overview.get_features_shape(data_instances)
        #？？？特征的值应当不会包含在内，打印看一下吧
        LOGGER.info("这里打印查看第一个data_instances值的数据类型：{}".format(data_instances.first()[1].features.shape[0]))
        #有关于密钥的处理
        pub_key = self.transfer_paillier.get(idx=0, suffix=("pub_key",)) #这里有关于pub_key的类型就是一个对象，不再是列表中存放对象
        self.cipher.set_public_key(pub_key)

        #模型初始化
        self.model =model_weight.PoissonDpWeightGuest()
        self.model.initialize(data_instances)

        #批处理模块初始化
        self.batch_generator.register_batch_generator(self.transfer_variable)
        batch_size = int(data_instances.count() / self.r)
        batch_suffix = ("batch_info",)
        self.batch_generator.initialize_batch_generator(data_instances, batch_size, suffix=batch_suffix)

        #加密模块
        LOGGER.info("这里的batch_nums大小是：{}".format(self.batch_generator.batch_nums))
        self.encrypted_calculator = [EncryptModeCalculator(self.cipher) for _ in range(self.batch_generator.batch_nums)]

        #开始正式的循环迭代训练过程，初始化迭代次数为0
        iteration = 0  # 记录epoches次数
        suffix_tag = 0  # 用来传输变量的标识，同时值也标识最终的传输变量的次数
        encrypt_suffix = ("encrypt_suffix",)
        uni_guest_gradient_suffix = ("uni_guest_gradient_suffix",)
        uni_host_gradient_suffix = ("uni_host_gradient_suffix",)
        fore_gradient_suffix = ("fore_gradient",)
        guest_to_arbiter_suffix = ("guest_to_arbiter_suffix",)
        arbiter_to_guest_suffix = ("arbiter_to_guest_suffix",)

        while iteration <= self.e:
            for data_inst in self.batch_generator.generator_batch_data():
                LOGGER.info("------------------当前迭代次数:{}-------------------".format(suffix_tag))
                LOGGER.info("guest开始从host端接收encrypted_ir_b")
                # 以下都是纯属数据的标签
                suffix_e = encrypt_suffix + (suffix_tag,)
                suffix_f = fore_gradient_suffix + (suffix_tag,)
                suffix_ug = uni_guest_gradient_suffix + (suffix_tag,)  # 用来传输guest方的单侧梯度的
                suffix_uh = uni_host_gradient_suffix + (suffix_tag,)
                suffix_ga = guest_to_arbiter_suffix + (suffix_tag,)
                suffix_ag = arbiter_to_guest_suffix + (suffix_tag,)

                encrypted_ir_bs = self.encrypted_ir_b.get(idx=-1, suffix=suffix_e)
                LOGGER.info("guest开始计算exp_guest_wx")
                exp_guest_wx = data_inst.mapValues(lambda x: np.exp(np.dot(np.append(x.features, 1), self.model.w)))

                # LOGGER.info("开始将guest_wx的值加密")
                # encrypted_guest_wx = self.encrypted_calculator[suffix_tag % self.batch_generator.batch_nums].encrypt(
                #     guest_wx)

                LOGGER.info("开始求解密文下的fore_gradient")
                for encrypted_ir_b in encrypted_ir_bs:
                    self.aggregated_forwards = exp_guest_wx.join(encrypted_ir_b, lambda x, y: x * y)
                fore_gradient = self.model.intermediate_result(data_inst,self.aggregated_forwards)

                LOGGER.info("开始计算guest方的单侧梯度")
                average_unilateral_gradient_guest = self.model.compute_gradient(data_inst,fore_gradient,data_inst.count())

                LOGGER.info("guest开始将average_unilateral_gradient_guest的结果、fore_gradient发送给host方")
                self.average_unilateral_gradient_guest.remote(obj=average_unilateral_gradient_guest, idx=-1,
                                                              role=consts.HOST, suffix=suffix_ug)
                self.fore_gradient.remote(obj=fore_gradient, idx=-1, role=consts.HOST, suffix=suffix_f)

                #---------------------------------------------------------------------------------------------------------
                #第二阶段，开始扰动数据
                LOGGER.info("guest方开始接收host方average_unilateral_gradient_host的结果")
                average_unilateral_gradient_host = self.average_unilateral_gradient_host.get(idx=-1, suffix=suffix_uh)
                LOGGER.info("开始在average_unilateral_gradient_guest的结果上添加噪音，保护的是guest端的h(w,x,y)")

                LOGGER.info("开始计算高斯噪声所需要的loc、sigma")
                shape_host = average_unilateral_gradient_host[0].shape[0]
                shape_guest = len(self.model.w) - 1
                self.beta_theta = np.exp((shape_guest+shape_host) * self.k)
                loc, sigma = self.model.gaussian(self.delta, self.epsilon, self.beta_theta, self.L,
                                                 self.e, int(self.e * self.r), self.learning_rate,
                                                 data_inst.count(), self.k, self.beta_y, self.k_y, shape_host)

                LOGGER.info("开始对host梯度数据添加噪声")
                average_unilateral_gradient_host_noise = self.model.sec_intermediate_result(
                    average_unilateral_gradient_host[0], loc, sigma)
                self.guest_to_arbiter.remote(obj=average_unilateral_gradient_host_noise, role=consts.ARBITER, idx=-1,
                                             suffix=suffix_ga)

                LOGGER.info("开始从arbiter方接收梯度")
                gradient_guest = self.arbiter_to_guest.get(idx=-1, suffix=suffix_ag)

                LOGGER.info("开始更新模型参数w")
                self.model.update_model(gradient_guest[0], self.learning_rate, self.lamb)

                LOGGER.info("开始梯度剪切")
                self.model.norm_clip(self.k)

                suffix_tag += 1

            iteration += 1

        LOGGER.info("训练正式结束")
        LOGGER.info("guest方的模型参数是:{}".format(self.model.w))

        self.data_output = self.model.w

    def save_data(self):
        return self.data_output

    def predict(self, data_inst):
        """
        纵向线性回归的预测模块
        Parameters
        --------------------------
        data_inst:Dtable,数据
        Returns
        --------------------------
        预测的结果
        """
        LOGGER.info("开始纵向泊松回归的预测模块")
        self._abnormal_detection(data_inst)

        #注册传输变量
        self.register_gradient_sync(self.transfer_variable)

        #初始化模型参数
        self.model =model_weight.PoissonDpWeightGuest()
        self.model.w = self.data_output

        data_instances = data_inst
        LOGGER.info("开始计算guest方的wx内积")
        wx_guest = data_instances.mapValues(lambda x: np.exp(np.dot(np.append(x.features, 1), self.data_output)))

        LOGGER.info("开始从host方接收host方的wx")
        wx_host = self.host_wx.get(idx=-1)

        self.data_shape = data_overview.get_features_shape(data_instances)
        # 如下的wx_guest便是完整的wx
        for each_wx_host in wx_host:
            wx_guest = wx_guest.join(each_wx_host, lambda x, y: x * y)

        predict_result = self.predict_score_to_output(data_instances=data_instances, predict_score=wx_guest,
                                                      classes=None)
        LOGGER.info("训练结束")
        return predict_result

    def _init_model(self, params):
        """
        将先前定义的param中的属性一一赋值,相关的定义见param类的说明
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
        meta_protobuf_obj = poisson_dp_model_meta_pb2.PoissonDpModelMeta(epsilon=self.epsilon,
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
        for i in range(self.data_shape):
            result = "w" + str(i)
            weight_dict[result] = self.model.w[i]
        weight_dict["b"] = self.model.w[-1]
        weight["weight"] = weight_dict
        param_protobuf_obj = poisson_dp_model_param_pb2.PoissonDpModelParam(**weight)

        return param_protobuf_obj

    def export_model(self):
        """
        这个函数应当指的是当某次任务结束后，将模型和当前任务的一些参数配置啥玩意的保存起来
        """
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            "PoissonDpMeta": meta_obj,
            "PoissonDpParam": param_obj
        }
        return result

    def load_model(self, model_dict):
        """
        这个函数指的是当我们进行预测任务的时候，框架估计会调用这个函数。然后载入之前的模型，直接来用
        """
        result_obj = list(model_dict.get('model').values())[0].get("PoissonDpParam")
        # 将值取出来，搞成数组的形式，然后传给self.data_output,再实际测试是否赋值给self.data_output
        self.data_output = []
        for i in range(len(result_obj.weight) - 1):
            result = "w" + str(i)
            self.data_output.append(result_obj.weight[result])
        self.data_output.append(result_obj.weight["b"])
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
        self.host_wx = transfer_variable.host_wx
        self.transfer_paillier = transfer_variable.paillier_pubkey
        self.encrypted_ir_b = transfer_variable.encrypted_ir_b
        self.average_unilateral_gradient_guest = transfer_variable.average_unilateral_gradient_guest
        self.average_unilateral_gradient_host = transfer_variable.average_unilateral_gradient_host
        self.fore_gradient = transfer_variable.fore_gradient
        self.guest_to_arbiter = transfer_variable.guest_to_arbiter
        self.arbiter_to_guest = transfer_variable.arbiter_to_guest
