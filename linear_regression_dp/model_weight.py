import numpy as np

from federatedml.util import LOGGER
from federatedml.statistic import data_overview

class LinearDpWeightGuest():
    def __init__(self,w=None):
        self.w = w

    def initialize(self,data_instances):
        """
        初始化模型参数
        """
        data_shape = data_overview.get_features_shape(data_instances)
        LOGGER.info("除去偏置b,数据的维度属性是：{}".format(data_shape))
        # 将偏置b也加进来
        if isinstance(data_shape, int):
            data_shape += 1
        # 初始化模型参数
        self.w = np.random.rand(data_shape)
        LOGGER.info("guest方初始化模型参数self.w是：{}".format(self.w))

    def compute_gradient(self,data_instances,ir_a,b):
        """
        计算梯度
        parameters
        -----------------
        data_instances:Dtable
        ir_a:中间结果
        b:小批量数据量
        """
        result_tables = data_instances.join(ir_a,lambda x,y : np.append(x.features,1) * y)
        result = 0
        for result_table in result_tables.collect():
            result += result_table[1]

        #这里的result就是最终的结果，类型是numpy类型
        gradient_a = result / b

        return gradient_a

    def update_model(self,gradient_a,eta,lamb):
        """
        更新模型参数
        ---------------
        gradient_a:梯度
        eta:学习率
        lamb:正则化系数
        """
        self.w -= eta * (gradient_a + lamb * self.w)

    def gaussian(self,delta,epsilon,beta_theta,L,e,T,eta,b,k,beta_y,k_y,length):
        """
        生成满足差分隐私的高斯机制所对应的loc、sigma
        Parameters
        -----------------------------------
        delta:非严格的DP损失，不同于拉普拉斯机制
        epsilon:隐私保护预算
        beta_theta:smooth parameters,凸优化参数
        L:lipschitz常数，这里默认值是6
        e:epochs，数据集使用次数
        T:e * r,表示总的迭代次数
        eta:learning rate
        b:mini-batch size
        k:梯度剪切参数，默认值为1
        beta_y:smooth parameters,凸优化参数
        k_y:target bound,凸优化参数
        length:host方数据的维度，作为全局敏感度的一个参数
        """
        loc = 0
        partial_1 = np.sqrt(2 * np.log(1.25 / delta))

        partial_2_1 = 4 * np.square(beta_theta) * np.square(L) * np.square(e) * T * np.square(eta) / b
        partial_2_2 = 8 * (beta_theta * k + beta_y * k_y) * beta_theta * L * np.square(e) * eta / b
        partial_2_3 = 4 * np.square(beta_theta * k + beta_y * k_y) * e
        partial_2 = np.sqrt(partial_2_1 + partial_2_2 + partial_2_3) * np.sqrt(length) / b   # 这里使用的是L2敏感度，因为是高斯噪声

        partial_3 = epsilon

        sigma = partial_1 * partial_2 / partial_3
        LOGGER.info("在guest方，全局敏感度值:{}".format(partial_2))
        return loc, sigma

    def sec_intermediate_result(self,average_unilateral_gradient_host,loc,sigma):
        """
        扰动host端发来的密文梯度
        parameters
        -------------------------
        average_unilateral_gradient_host:host端的密文梯度
        loc:高斯噪声的位置参数
        sigma:标准差
        """
        length = average_unilateral_gradient_host.shape[0]
        sec_result_0 = []
        for i in range(length):
            sec_result_0.append(np.random.normal(loc, sigma))
        LOGGER.info("添加的噪音为：{}".format(sec_result_0))
        average_unilateral_gradient_guest_noise = average_unilateral_gradient_host + sec_result_0

        return average_unilateral_gradient_guest_noise

    def intermediate_result(self,data_instances,aggregated_forwards,w):
        """
        这里指的是双方交互时需要计算梯度时，需要的部分参数
        parameters
        -----------------
        data_instances:Dtable,guest方当前批次的数据
        sec_ir_b:Dtable,从host方接受的wx的内积，当然是密文状态
        w:numpy格式guest方的模型参数
        """
        ir_a = data_instances.join(aggregated_forwards,lambda x,y:\
                                   -2 * (x.label - np.dot(np.append(x.features,1),w)-y))
        return ir_a

    def norm_clip(self,k):
        """
        梯度剪切，防止梯度爆炸
        """
        result = np.sqrt(np.sum(np.square(self.w))) / k
        if result > 1:
            self.w /= result


class LinearDpWeightHost():
    def __init__(self,w=None):
        self.w = w

    def initialize(self,data_instances):
        """
        初始化w
        """
        data_shape = data_overview.get_features_shape(data_instances)
        self.w = np.random.rand(data_shape)
        LOGGER.info("Host初始化模型参数self.w是：{}".format(self.w))

    def compute_gradient(self,data_instances,ir_a,b):
        result_tables = data_instances.join(ir_a, lambda x, y: x.features * y)
        result = 0
        for result_table in result_tables.collect():
            result += result_table[1]

        gradient_b = result / b
        return gradient_b

    def compute_forwards(self,data_instances,w):
        """
        计算wx内积
        """
        ir_b = data_instances.mapValues(lambda x: np.dot(x.features, w))
        return ir_b

    def sec_intermediate_result(self,average_unilateral_gradient_guest,loc,sigma):
        """
        扰动guest方梯度
        parameters
        ---------------------
        average_unilateral_gradient_guest:numpy,guest方的梯度
        loc：高斯分布对应的位置参数
        sigma:高斯分布对应的标准差
        """
        length = average_unilateral_gradient_guest.shape[0]
        sec_result_0 = []
        for i in range(length):
            sec_result_0.append(np.random.normal(loc,sigma))
        LOGGER.info("添加的噪音为：{}".format(sec_result_0))
        average_unilateral_gradient_guest_noise = average_unilateral_gradient_guest + sec_result_0

        return average_unilateral_gradient_guest_noise

    def gaussian(self,delta,epsilon,L,e,T,eta,b,k,length):
        """
        生成高斯噪声所需要的loc、sigma
        parameters
        --------------------
        delta:一定程度的允许错误的值，因为高斯机制非严格满足DP机制
        epsion:隐私保护预算
        L:lipschitz 常数，默认值为6
        e:epochs
        T:e * r,表示总的迭代次数
        eta:learning rate
        b:mini-batch size
        k:梯度剪切参数
        length:表述guest方数据维度的长度

        loc,sigma都是高斯分布的俩参数
        """
        loc = 0
        partial_1 = np.sqrt(2 * np.log(1.25 / delta))
        partial_2_1 = (4 * np.square(L) * np.square(e) * T * np.square(eta)) / b
        partial_2_2 = (8 * k * L * np.square(e) * eta) / b
        partial_2_3 = 4 * np.square(k) * e
        partial_2 = 2 * np.sqrt(partial_2_1 + partial_2_2 + partial_2_3) * np.sqrt(length) / b #最大敏感度
        partial_3 = epsilon
        sigma = partial_1 * partial_2 / partial_3

        LOGGER.info("站在host方，全局敏感度值:{}".format(partial_2))
        return loc,sigma

    def norm_clip(self, k):
        """
        parameters
        ------------
        k:这个大小自己来定义，此函数主要用来防止梯度爆炸
        """
        result = np.sqrt(np.sum(np.square(self.w))) / k
        if result > 1:
            self.w /= result

    def update_model(self, gradient_b, eta, lamb):
        """
        更新模型参数
        parameters
        ---------------
        gradient_b:梯度
        eta:学习率
        lamb:正则化参数
        """
        self.w -= eta * (gradient_b + lamb * self.w)

class LinearDpWeightArbiter():
    pass
