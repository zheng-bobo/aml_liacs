import ConfigSpace
import numpy as np
import typing


class RandomSearch(object):

    def __init__(self, config_space: ConfigSpace.ConfigurationSpace):
        self.config_space = config_space

    def initialize(
        self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]
    ) -> None:
        pass

    def select_configuration(self) -> ConfigSpace.Configuration:
        # 这行代码的意思是：从配置空间中随机采样一个配置（超参数组合）。
        # sample_configuration()会返回一个Configuration对象。

        # 配置空间的配置是通过ConfigSpace.ConfigurationSpace对象的sample_configuration方法随机生成的
        # 这里self.config_space就是在__init__方法中传入的配置空间对象
        # sample_configuration()会从这个配置空间随机采样1个配置，返回一个Configuration对象
        return self.config_space.sample_configuration()

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        pass
