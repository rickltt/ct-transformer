import json

class CTConfig:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self._config_data = json.load(file)

        # 将字典转化为对象属性
        for key, value in self._config_data.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        if item in self._config_data:
            return self._config_data[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")
    