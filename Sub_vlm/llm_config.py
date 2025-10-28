"""
LLM API配置模块
管理OpenRouter API设置和认证
"""
import os
import yaml


class LLMConfig:
    """LLM API配置管理器"""
    
    def __init__(self, config_path="llm_config.yaml"):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请从 llm_config.yaml.template 创建配置文件"
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要字段
        required_fields = ['api_key', 'base_url', 'model']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件缺少必要字段: {field}")
        
        return config
    
    @property
    def api_key(self):
        """API密钥"""
        return self.config['api_key']
    
    @property
    def base_url(self):
        """API基础URL"""
        return self.config['base_url']
    
    @property
    def model(self):
        """模型名称"""
        return self.config['model']
    
    @property
    def temperature(self):
        """温度参数"""
        return self.config.get('temperature', 0.7)
    
    @property
    def max_tokens(self):
        """最大token数"""
        return self.config.get('max_tokens', 2000)
    
    @property
    def timeout(self):
        """请求超时（秒）"""
        return self.config.get('timeout', 60)
    
    def get_headers(self):
        """获取API请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def __repr__(self):
        return f"LLMConfig(model={self.model}, base_url={self.base_url})"
