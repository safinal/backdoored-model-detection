import yaml

class ConfigManager:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = cls.load_config(config_path) if config_path else {}
        return cls._instance

    @staticmethod
    def load_config(config_path):
        """Loads the configuration from a YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get(self, key, default=None):
        """Gets a configuration value, or returns the default if the key is not found."""
        return self._instance.config.get(key, default)