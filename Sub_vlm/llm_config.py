"""
LLM API配置模块 - 简化版
只保留核心的API配置加载功能
"""
import os
import yaml


class LLMConfig:
    """LLM API配置管理器"""
    
    def __init__(self, config_path="llm_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请从 llm_config.yaml.template 复制并填入你的API密钥"
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要字段
        required = ['api_key', 'base_url', 'model']
        for field in required:
            if field not in config or not config[field]:
                raise ValueError(f"配置文件缺少必要字段: {field}")
        
        return config
    
    @property
    def api_key(self):
        return self.config['api_key']
    
    @property
    def base_url(self):
        return self.config['base_url']
    
    @property
    def model(self):
        return self.config['model']
    
    @property
    def temperature(self):
        return self.config.get('temperature', 0.7)
    
    @property
    def max_tokens(self):
        return self.config.get('max_tokens', 2000)
    
    @property
    def timeout(self):
        return self.config.get('timeout', 60)
    
    def get_headers(self):
        """获取API请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
