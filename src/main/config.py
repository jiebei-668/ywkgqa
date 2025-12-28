import sys
sys.path.append('../')
print(sys.path)

from configs.config_main import config_dict




class hyperparams:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

# Instantiate the class with the config_dict
args = hyperparams(config_dict)


if __name__ == '__main__':
    print(f"当前运行程序参数：{args.__dict__}")
