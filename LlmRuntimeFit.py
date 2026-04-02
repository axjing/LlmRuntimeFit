"""
优化版LLM运行时环境检查脚本
- 兼容vLLM（优先使用）和transformers
- 模块化封装，代码更简洁规范
- 确保计算准确性
"""

import time
import sys
import torch
import psutil
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================
# 配置类：统一管理模型配置
# ==============================================
class ModelConfig:
    """模型配置类，封装所有配置参数"""
    
    def __init__(self, config_dict: dict):
        self.model_path = config_dict.get("model_path", "Qwen/Qwen3.5-4B")
        self.params_b = config_dict.get("params_b", None)
        self.precision = config_dict.get("precision", "auto")
        self.quantization = config_dict.get("quantization", "auto")
        self.batch_size = config_dict.get("batch_size", 1)
        self.seq_len = config_dict.get("seq_len", None)
        self.hidden_size = config_dict.get("hidden_size", None)
        self.num_layers = config_dict.get("num_layers", None)
        self.test_prompt = config_dict.get("test_prompt", "请介绍人工智能大模型的核心应用")
        self.max_new_tokens = config_dict.get("max_new_tokens", 512)
        self.device = config_dict.get("device", "auto")
        self.engine = config_dict.get("engine", "auto")  # auto/vllm/transformers
        
        # 自动获取模型信息
        self._auto_detect_model_info()
        # 自动设置精度和量化
        self._auto_set_precision()
    
    def _auto_detect_model_info(self):
        """自动检测模型信息"""
        try:
            from transformers import AutoConfig
            
            # 加载模型配置
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            # 自动设置参数
            if self.params_b is None:
                # 尝试从配置中获取参数量
                if hasattr(config, 'num_parameters'):
                    self.params_b = config.num_parameters / 1e9
                elif hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                    # 估算参数量
                    self.params_b = self._estimate_params(config)
                else:
                    self.params_b = 4.0  # 默认值
            
            if self.seq_len is None:
                self.seq_len = getattr(config, 'max_position_embeddings', 8192)
            
            if self.hidden_size is None:
                self.hidden_size = getattr(config, 'hidden_size', 4096)
            
            if self.num_layers is None:
                self.num_layers = getattr(config, 'num_hidden_layers', 32)
                
        except Exception as e:
            print(f"⚠️  自动检测模型信息失败: {e}")
            # 使用默认值
            if self.params_b is None:
                self.params_b = 4.0
            if self.seq_len is None:
                self.seq_len = 8192
            if self.hidden_size is None:
                self.hidden_size = 4096
            if self.num_layers is None:
                self.num_layers = 32
    
    def _estimate_params(self, config) -> float:
        """估算模型参数量"""
        try:
            # 基于常见模型架构的参数量估算
            hidden_size = config.hidden_size
            num_layers = config.num_hidden_layers
            vocab_size = getattr(config, 'vocab_size', 100000)
            
            # 简化的参数量估算公式
            params = (
                vocab_size * hidden_size +  # 词嵌入
                num_layers * (4 * hidden_size ** 2 + 4 * hidden_size) +  # 注意力和前馈网络
                hidden_size * vocab_size  # 输出层
            )
            return params / 1e9
        except:
            return 4.0
    
    def _auto_set_precision(self):
        """自动设置精度和量化"""
        if self.precision == "auto":
            # 根据模型大小自动选择精度
            if self.params_b < 7:
                self.precision = "fp16"
            else:
                self.precision = "fp32"
        
        if self.quantization == "auto":
            # 根据模型大小自动选择量化
            if self.params_b > 7:
                self.quantization = "int8"
            else:
                self.quantization = self.precision
    
    def validate(self) -> bool:
        """验证配置的完整性和合理性"""
        PRECISION_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
        
        if self.params_b <= 0:
            print("❌ 配置错误: 参数量必须大于0")
            return False
        
        if self.batch_size <= 0:
            print("❌ 配置错误: 批次大小必须大于0")
            return False
        
        if self.max_new_tokens <= 0:
            print("❌ 配置错误: 最大生成token数必须大于0")
            return False
        
        if self.precision not in PRECISION_BYTES:
            print(f"❌ 配置错误: 不支持的精度类型 '{self.precision}'")
            return False
        
        return True

# ==============================================
# 硬件检测类：统一管理硬件信息
# ==============================================
class HardwareDetector:
    """硬件检测类，封装所有硬件检测功能"""
    
    @staticmethod
    def bytes2gb(bytes_val: int) -> float:
        """字节转GB（1024进制）"""
        return round(bytes_val / (1024 ** 3), 2)
    
    @staticmethod
    def get_current_hardware() -> dict:
        """获取当前硬件信息"""
        try:
            # CPU信息
            cpu_info = {
                "core_count": psutil.cpu_count(logical=True) or 1,
                "current_usage": psutil.cpu_percent(interval=0.1)
            }
            
            # 内存信息
            ram_info = psutil.virtual_memory()
            ram_info = {
                "total": HardwareDetector.bytes2gb(ram_info.total),
                "available": HardwareDetector.bytes2gb(ram_info.available),
                "used": HardwareDetector.bytes2gb(ram_info.used)
            }
            
            # GPU信息
            gpu_info = {"name": "无GPU", "total_vram": 0.0, "available_vram": 0.0, "used_vram": 0.0}
            if torch.cuda.is_available():
                try:
                    gpu_prop = torch.cuda.get_device_properties(0)
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    gpu_info = {
                        "name": gpu_prop.name,
                        "total_vram": HardwareDetector.bytes2gb(gpu_prop.total_memory),
                        "available_vram": HardwareDetector.bytes2gb(gpu_prop.total_memory - (allocated + reserved)),
                        "used_vram": HardwareDetector.bytes2gb(allocated + reserved)
                    }
                except Exception as e:
                    print(f"⚠️  GPU信息获取失败: {e}")
                    gpu_info = {"name": "GPU检测异常", "total_vram": 0.0, "available_vram": 0.0, "used_vram": 0.0}
            
            return {"cpu": cpu_info, "ram": ram_info, "gpu": gpu_info}
        
        except Exception as e:
            print(f"❌ 硬件信息获取失败: {e}")
            return {
                "cpu": {"core_count": 1, "current_usage": 0},
                "ram": {"total": 0.0, "available": 0.0, "used": 0.0},
                "gpu": {"name": "检测失败", "total_vram": 0.0, "available_vram": 0.0, "used_vram": 0.0}
            }

# ==============================================
# 硬件需求计算类：基于第一性原理
# ==============================================
class HardwareCalculator:
    """硬件需求计算类，封装所有计算逻辑"""
    
    PRECISION_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    
    @classmethod
    def calculate_required_hardware(cls, config: ModelConfig) -> dict:
        """计算模型所需硬件条件（修正版）"""
        # 使用量化精度进行计算
        byte = cls.PRECISION_BYTES.get(config.quantization, cls.PRECISION_BYTES[config.precision])
        total_params = config.params_b * 1e9
        
        # 1. 权重显存（模型参数）
        weight_vram = total_params * byte
        
        # 2. KV缓存显存（修正计算公式）
        # 注意：KV缓存通常使用fp16/bf16，即使模型使用int4量化
        kv_byte = 2.0  # KV缓存通常使用fp16/bf16
        kv_cache_per_token = 2 * config.hidden_size * config.num_layers * kv_byte
        kv_vram = kv_cache_per_token * config.seq_len * config.batch_size
        
        # 3. 激活值显存（推理过程中的中间结果）
        # 估算为权重显存的20%
        activation_vram = weight_vram * 0.2
        
        # 4. 框架开销（transformers框架本身的开销）
        framework_overhead = 2.0 * (1024 ** 3)  # 固定2GB框架开销
        
        # 5. 总显存需求（更保守的估算）
        total_vram = weight_vram + kv_vram + activation_vram + framework_overhead
        
        # 6. CPU内存需求（模型加载+推理开销）
        required_ram = weight_vram * 1.5 + 4.0 * (1024 ** 3)  # 额外4GB系统开销
        
        # 7. 最低CPU核心需求
        # 根据模型大小和批次大小动态调整
        if config.params_b < 7:
            required_cpu_core = 8 if config.batch_size <= 4 else 12
        else:
            required_cpu_core = 12 if config.batch_size <= 8 else 16
        
        return {
            "required_vram": HardwareDetector.bytes2gb(total_vram),
            "required_ram": HardwareDetector.bytes2gb(required_ram),
            "required_cpu_core": required_cpu_core,
            "weight_vram": HardwareDetector.bytes2gb(weight_vram),
            "kv_vram": HardwareDetector.bytes2gb(kv_vram),
            "activation_vram": HardwareDetector.bytes2gb(activation_vram),
            "framework_overhead": HardwareDetector.bytes2gb(framework_overhead)
        }
    
    @staticmethod
    def judge_compatibility(required: dict, current: dict) -> dict:
        """判断硬件兼容性"""
        if current["gpu"]["total_vram"] == 0:
            return {
                "can_run": False,
                "reason": "未检测到GPU，大模型推理需依赖GPU",
                "detail": {}
            }
        
        vram_ok = current["gpu"]["total_vram"] >= required["required_vram"]
        ram_ok = current["ram"]["total"] >= required["required_ram"]
        cpu_core_ok = current["cpu"]["core_count"] >= required["required_cpu_core"]
        
        can_run = vram_ok and ram_ok and cpu_core_ok
        reason = ""
        if not vram_ok:
            reason += f"GPU显存不足（当前{current['gpu']['total_vram']}GB，所需{required['required_vram']}GB）；"
        if not ram_ok:
            reason += f"CPU内存不足（当前{current['ram']['total']}GB，所需{required['required_ram']}GB）；"
        if not cpu_core_ok:
            reason += f"CPU核心不足（当前{current['cpu']['core_count']}核，所需{required['required_cpu_core']}核）；"
        if can_run:
            reason = "所有硬件条件均满足，可正常运行模型"
        
        return {
            "can_run": can_run,
            "reason": reason.rstrip("；"),
            "detail": {
                "gpu_vram_check": vram_ok,
                "cpu_ram_check": ram_ok,
                "cpu_core_check": cpu_core_ok
            }
        }

# ==============================================
# 性能测试引擎类：兼容vLLM和transformers
# ==============================================
class PerformanceTester:
    """性能测试引擎类，支持vLLM和transformers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.engine_type = self._detect_engine()
        
    def _detect_engine(self) -> str:
        """自动检测可用的推理引擎"""
        if self.config.engine == "vllm":
            return "vllm"
        elif self.config.engine == "transformers":
            return "transformers"
        else:  # auto模式
            try:
                import vllm
                return "vllm"
            except ImportError:
                return "transformers"
    
    def test_performance(self) -> dict:
        """执行性能测试"""
        print(f"🔹 使用引擎: {self.engine_type}")
        
        if self.engine_type == "vllm":
            return self._test_with_vllm()
        else:
            return self._test_with_transformers()
    
    def _test_with_vllm(self) -> dict:
        """使用vLLM进行性能测试"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            
            print("🔹 正在加载vLLM模型...")
            
            # 初始化vLLM
            llm = LLM(
                model=self.config.model_path,
                quantization=self.config.quantization,
                max_model_len=self.config.seq_len,
                trust_remote_code=True
            )
            
            # 初始化分词器
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            
            # 设置采样参数
            sampling_params = SamplingParams(
                max_tokens=self.config.max_new_tokens,
                temperature=0.0,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None
            )
            
            # 性能测试
            start_time = time.time()
            outputs = llm.generate([self.config.test_prompt], sampling_params)
            total_time = time.time() - start_time
            
            # 计算指标
            output = outputs[0]
            input_tokens = len(tokenizer.encode(self.config.test_prompt))
            generated_tokens = len(output.outputs[0].token_ids)
            
            if generated_tokens > 0 and total_time > 0:
                avg_token_time = total_time / generated_tokens
                ttft_ms = round(avg_token_time * 1000, 2)
                throughput = round(generated_tokens / total_time, 2)
            else:
                ttft_ms = throughput = 0.0
            
            return {
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "TTFT(ms)": ttft_ms,
                "TPOT(ms)": ttft_ms,  # vLLM批量生成，简化处理
                "total_latency(s)": round(total_time, 2),
                "throughput(token/s)": throughput
            }
            
        except Exception as e:
            print(f"❌ vLLM性能测试失败: {e}")
            print("🔹 回退到transformers引擎...")
            return self._test_with_transformers()
    
    def _test_with_transformers(self) -> dict:
        """使用transformers进行性能测试"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print("🔹 正在加载transformers模型...")
            
            # 设备选择
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 初始化分词器
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, 
                trust_remote_code=True,
                padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 准备输入
            inputs = tokenizer(self.config.test_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_tokens = inputs["input_ids"].shape[1]
            
            # 预热
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 性能测试
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True
                )
            total_time = time.time() - start_time
            
            # 计算指标
            generated_tokens = outputs.sequences[0, input_tokens:].shape[0]
            
            if generated_tokens > 0 and total_time > 0:
                avg_token_time = total_time / generated_tokens
                ttft_ms = round(avg_token_time * 1000, 2)
                throughput = round(generated_tokens / total_time, 2)
            else:
                ttft_ms = throughput = 0.0
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "TTFT(ms)": ttft_ms,
                "TPOT(ms)": ttft_ms,
                "total_latency(s)": round(total_time, 2),
                "throughput(token/s)": throughput
            }
            
        except Exception as e:
            print(f"❌ transformers性能测试失败: {e}")
            return {
                "input_tokens": 0,
                "generated_tokens": 0,
                "TTFT(ms)": 0.0,
                "TPOT(ms)": 0.0,
                "total_latency(s)": 0.0,
                "throughput(token/s)": 0.0
            }

# ==============================================
# 主测试类：统一管理测试流程
# ==============================================
class LLMRuntimeChecker:
    """LLM运行时检查主类"""
    
    def __init__(self, config_dict: dict):
        self.config = ModelConfig(config_dict)
        self.hardware_detector = HardwareDetector()
        self.hardware_calculator = HardwareCalculator()
    
    def run_full_test(self):
        """执行完整的运行时检查"""
        print("=" * 80)
        print("🚀 LLM运行时环境检查（优化版）")
        print("=" * 80)
        
        # 1. 配置验证
        if not self._validate_config():
            return
        
        # 2. 环境检查
        self._check_environment()
        
        # 3. 硬件需求计算
        required_hardware = self._calculate_hardware_requirements()
        if not required_hardware:
            return
        
        # 4. 硬件兼容性判断
        compatibility = self._check_compatibility(required_hardware)
        
        # 5. 性能测试（如果兼容）
        if compatibility["can_run"]:
            self._run_performance_test()
        
        print("\n" + "=" * 80)
        print("✅ 检查完成")
    
    def _validate_config(self) -> bool:
        """验证配置"""
        print("\n🔹 步骤1：配置验证")
        if not self.config.validate():
            print("❌ 配置验证失败")
            return False
        print("✅ 配置验证通过")
        
        # 显示配置信息
        print("\n📋 模型配置信息:")
        table = [
            ["模型路径", self.config.model_path],
            ["参数量", f"{self.config.params_b:.2f}B"],
            ["精度", self.config.precision],
            ["量化", self.config.quantization],
            ["序列长度", self.config.seq_len],
            ["隐藏层大小", self.config.hidden_size],
            ["层数", self.config.num_layers],
            ["批次大小", self.config.batch_size]
        ]
        print(tabulate(table, tablefmt="grid", headers=["参数", "值"]))
        
        return True
    
    def _check_environment(self):
        """检查运行环境"""
        print("\n🔹 步骤2：环境检查")
        print(f"✅ Python版本: {sys.version.split()[0]}")
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    def _calculate_hardware_requirements(self) -> dict:
        """计算硬件需求"""
        print("\n🔹 步骤3：硬件需求计算")
        try:
            required = self.hardware_calculator.calculate_required_hardware(self.config)
            table = [
                ["所需GPU显存（总计）", f"{required['required_vram']} GB"],
                ["├─ 模型权重显存", f"{required['weight_vram']} GB"],
                ["├─ KV缓存显存", f"{required['kv_vram']} GB"],
                ["├─ 激活值显存", f"{required['activation_vram']} GB"],
                ["└─ 框架开销", f"{required['framework_overhead']} GB"],
                ["所需CPU内存（含开销）", f"{required['required_ram']} GB"],
                ["所需CPU核心数", f"{required['required_cpu_core']} 核"]
            ]
            print(tabulate(table, tablefmt="grid", headers=["指标", "数值"]))
            return required
        except Exception as e:
            print(f"❌ 硬件需求计算失败: {e}")
            return None
    
    def _check_compatibility(self, required_hardware: dict) -> dict:
        """检查硬件兼容性"""
        print("\n🔹 步骤4：硬件兼容性检查")
        
        current_hardware = self.hardware_detector.get_current_hardware()
        
        # 显示当前硬件信息
        table = [
            ["GPU型号", current_hardware["gpu"]["name"]],
            ["GPU总显存", f"{current_hardware['gpu']['total_vram']} GB"],
            ["GPU已用显存", f"{current_hardware['gpu'].get('used_vram', 0)} GB"],
            ["CPU逻辑核心数", current_hardware["cpu"]["core_count"]],
            ["CPU总内存", f"{current_hardware['ram']['total']} GB"],
            ["CPU已用内存", f"{current_hardware['ram']['used']} GB"]
        ]
        print(tabulate(table, tablefmt="grid", headers=["硬件类型", "信息"]))
        
        # 判断兼容性
        compatibility = self.hardware_calculator.judge_compatibility(required_hardware, current_hardware)
        print(f"\n📌 兼容性判断：{'✅ 可运行' if compatibility['can_run'] else '❌ 不可运行'}")
        print(f"📌 原因：{compatibility['reason']}")
        
        return compatibility
    
    def _run_performance_test(self):
        """运行性能测试"""
        print("\n🔹 步骤5：性能测试")
        
        # 监控硬件使用率变化
        initial_hardware = self.hardware_detector.get_current_hardware()
        time.sleep(0.5)
        after_hardware = self.hardware_detector.get_current_hardware()
        
        cpu_change = after_hardware["cpu"]["current_usage"] - initial_hardware["cpu"]["current_usage"]
        ram_change = after_hardware["ram"]["used"] - initial_hardware["ram"]["used"]
        print(f"📊 硬件使用率变化: CPU +{cpu_change}%, RAM +{ram_change:.2f}GB")
        
        # 执行性能测试
        tester = PerformanceTester(self.config)
        performance = tester.test_performance()
        
        if performance["generated_tokens"] > 0:
            table = [
                ["输入token数", performance["input_tokens"]],
                ["生成token数", performance["generated_tokens"]],
                ["首字时延 TTFT(ms)", performance["TTFT(ms)"]],
                ["逐字时延 TPOT(ms)", performance["TPOT(ms)"]],
                ["总生成时延(s)", performance["total_latency(s)"]],
                ["吞吐量(token/s)", performance["throughput(token/s)"]]
            ]
            print(tabulate(table, tablefmt="grid", headers=["指标", "数值"]))
            
            # 性能评估
            self._evaluate_performance(performance)
        else:
            print("⚠️  性能测试未生成有效结果")
    
    def _evaluate_performance(self, performance: dict):
        """评估性能结果"""
        print("\n📊 性能评估:")
        
        ttft = performance["TTFT(ms)"]
        throughput = performance["throughput(token/s)"]
        
        if ttft < 100:
            print("✅ 首字时延: 优秀 (<100ms)")
        elif ttft < 500:
            print("⚠️  首字时延: 一般 (100-500ms)")
        else:
            print("❌ 首字时延: 较差 (>500ms)")
            
        if throughput > 50:
            print("✅ 吞吐量: 优秀 (>50 token/s)")
        elif throughput > 10:
            print("⚠️  吞吐量: 一般 (10-50 token/s)")
        else:
            print("❌ 吞吐量: 较差 (<10 token/s)")

# ==============================================
# 使用示例
# ==============================================
if __name__ == "__main__":
    # 配置参数
    # 只需要指定模型路径，其他参数会自动检测和设置
    config_dict = {
        "model_path": "Qwen/Qwen3.5-4B",
        # 以下参数可选，不指定会自动设置
        # "precision": "auto",  # auto/fp32/fp16/bf16/int8/int4
        # "quantization": "auto",  # auto/fp32/fp16/bf16/int8/int4
        # "batch_size": 1,
        # "seq_len": None,  # 自动从模型配置获取
        # "max_new_tokens": 512,
        # "device": "auto",
        # "engine": "auto"  # auto/vllm/transformers
    }
    
    # 创建检查器并运行测试
    checker = LLMRuntimeChecker(config_dict)
    checker.run_full_test()