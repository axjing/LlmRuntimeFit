"""
LLM 运行时环境检查工具
- 兼容 vLLM（优先）和 transformers 推理引擎
- 基于第一性原理的精确硬件需求计算
"""

import time
import sys
import warnings
from typing import Any, ClassVar

import psutil
import torch
from tabulate import tabulate

warnings.filterwarnings("ignore", category=UserWarning)

# vLLM 仅接受特定量化方法
VLLM_QUANT_METHODS = frozenset({"awq", "gptq", "squeezellm", "fp8", "bitsandbytes", "gguf"})


# ==============================================
# 模型配置类
# ==============================================
class ModelConfig:
    """模型配置，支持从 HuggingFace 自动检测"""

    PRECISION_BYTES: ClassVar[dict[str, float]] = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    # 精度名 → torch dtype 映射
    DTYPE_MAP: ClassVar[dict[str, torch.dtype]] = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(self, config_dict: dict[str, Any]) -> None:
        self.model_path: str = config_dict.get("model_path", "Qwen/Qwen3.5-4B")
        self.params_b: float | None = config_dict.get("params_b", None)
        self.precision: str = config_dict.get("precision", "auto")
        self.quantization: str = config_dict.get("quantization", "auto")
        self.batch_size: int = config_dict.get("batch_size", 1)
        self.seq_len: int | None = config_dict.get("seq_len", None)
        self.hidden_size: int | None = config_dict.get("hidden_size", None)
        self.num_layers: int | None = config_dict.get("num_layers", None)
        self.num_attention_heads: int | None = config_dict.get("num_attention_heads", None)
        self.num_kv_heads: int | None = config_dict.get("num_kv_heads", None)
        self.test_prompt: str = config_dict.get(
            "test_prompt", "请介绍人工智能大模型的核心应用"
        )
        self.max_new_tokens: int = config_dict.get("max_new_tokens", 512)
        self.device: str = config_dict.get("device", "auto")
        self.engine: str = config_dict.get("engine", "auto")

        # 自动检测模型信息，再自动设置精度
        self._auto_detect_model_info()
        self._auto_set_precision()

    def _auto_detect_model_info(self) -> None:
        """从 HuggingFace 配置自动检测模型参数"""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # 参数量
            if self.params_b is None:
                num_params_fn = getattr(config, "num_parameters", None)
                if callable(num_params_fn):
                    try:
                        self.params_b = num_params_fn() / 1e9
                    except Exception:
                        pass
                if self.params_b is None and hasattr(config, "hidden_size"):
                    self.params_b = self._estimate_params(config)
                if self.params_b is None:
                    self.params_b = 4.0

            # 序列长度
            if self.seq_len is None:
                self.seq_len = getattr(config, "max_position_embeddings", 8192)

            # 隐藏层维度
            if self.hidden_size is None:
                self.hidden_size = getattr(config, "hidden_size", 4096)

            # 层数
            if self.num_layers is None:
                self.num_layers = getattr(config, "num_hidden_layers", 32)

            # 注意力头数
            if self.num_attention_heads is None:
                self.num_attention_heads = getattr(
                    config, "num_attention_heads", 32
                )

            # KV 头数：GQA 模型使用 num_key_value_heads，MHA 等同于 num_attention_heads
            if self.num_kv_heads is None:
                self.num_kv_heads = getattr(
                    config, "num_key_value_heads", self.num_attention_heads
                )

        except Exception as e:
            print(f"[WARN] 自动检测模型信息失败: {e}")
            defaults = {
                "params_b": 4.0,
                "seq_len": 8192,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_attention_heads": 32,
            }
            for attr, default in defaults.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, default)
            if self.num_kv_heads is None:
                self.num_kv_heads = self.num_attention_heads

    def _estimate_params(self, config: Any) -> float:
        """根据模型架构估算参数量"""
        try:
            h = config.hidden_size
            L = config.num_hidden_layers
            V = getattr(config, "vocab_size", 100000)
            # Embedding + Attention/FFN + Output
            params = V * h + L * (4 * h**2 + 4 * h) + h * V
            return params / 1e9
        except Exception:
            return 4.0

    def _auto_set_precision(self) -> None:
        """根据硬件能力自动设置精度和量化"""
        if self.precision == "auto":
            if torch.cuda.is_available():
                # Ampere+ (compute capability >= 8.0) 支持 bf16，更适合 LLM 训练/推理
                major = torch.cuda.get_device_capability()[0]
                self.precision = "bf16" if major >= 8 else "fp16"
            else:
                self.precision = "fp32"

        if self.quantization == "auto":
            # 默认不额外量化，使用原始精度
            self.quantization = self.precision

    @property
    def kv_head_ratio(self) -> float:
        """KV 头数与注意力头数的比值（MHA=1.0, GQA<1.0）"""
        if self.num_kv_heads and self.num_attention_heads:
            return self.num_kv_heads / self.num_attention_heads
        return 1.0

    @property
    def torch_dtype(self) -> torch.dtype:
        """获取当前精度对应的 torch dtype"""
        return self.DTYPE_MAP.get(self.precision, torch.float16)

    def validate(self) -> bool:
        """校验配置完整性和正确性"""
        if not self.params_b or self.params_b <= 0:
            print("[ERROR] params_b 必须 > 0")
            return False
        if self.batch_size <= 0:
            print("[ERROR] batch_size 必须 > 0")
            return False
        if self.max_new_tokens <= 0:
            print("[ERROR] max_new_tokens 必须 > 0")
            return False
        if self.precision not in self.PRECISION_BYTES:
            print(f"[ERROR] 不支持的精度 '{self.precision}'")
            return False
        if self.quantization not in self.PRECISION_BYTES:
            print(f"[ERROR] 不支持的量化 '{self.quantization}'")
            return False
        if not self.num_attention_heads or self.num_attention_heads <= 0:
            print("[ERROR] num_attention_heads 必须 > 0")
            return False
        if not self.num_kv_heads or self.num_kv_heads <= 0:
            print("[ERROR] num_kv_heads 必须 > 0")
            return False
        return True


# ==============================================
# 类型定义
# ==============================================
HardwareInfo = dict[str, dict[str, Any]]
Requirements = dict[str, float | int]
CompatibilityResult = dict[str, bool | str | dict[str, bool]]
PerformanceResult = dict[str, int | float | None]


# ==============================================
# 硬件检测类
# ==============================================
class HardwareDetector:
    """硬件检测工具"""

    @staticmethod
    def bytes2gb(bytes_val: int | float) -> float:
        """字节转 GiB（1024 进制）"""
        return round(bytes_val / (1024**3), 2)

    @staticmethod
    def get_current_hardware() -> HardwareInfo:
        """获取当前硬件信息"""
        try:
            cpu_info = {
                "core_count": psutil.cpu_count(logical=True) or 1,
                "current_usage": psutil.cpu_percent(interval=0.1),
            }

            ram = psutil.virtual_memory()
            ram_info = {
                "total": HardwareDetector.bytes2gb(ram.total),
                "available": HardwareDetector.bytes2gb(ram.available),
                "used": HardwareDetector.bytes2gb(ram.used),
            }

            gpu_info: dict[str, Any] = {
                "name": "无 GPU",
                "total_vram": 0.0,
                "available_vram": 0.0,
                "used_vram": 0.0,
            }
            if torch.cuda.is_available():
                try:
                    prop = torch.cuda.get_device_properties(0)
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    gpu_info = {
                        "name": prop.name,
                        "total_vram": HardwareDetector.bytes2gb(prop.total_memory),
                        "available_vram": HardwareDetector.bytes2gb(
                            prop.total_memory - allocated - reserved
                        ),
                        "used_vram": HardwareDetector.bytes2gb(allocated + reserved),
                    }
                except Exception as e:
                    print(f"[WARN] GPU 信息获取失败: {e}")

            return {"cpu": cpu_info, "ram": ram_info, "gpu": gpu_info}

        except Exception as e:
            print(f"[ERROR] 硬件信息获取失败: {e}")
            return {
                "cpu": {"core_count": 1, "current_usage": 0},
                "ram": {"total": 0.0, "available": 0.0, "used": 0.0},
                "gpu": {
                    "name": "检测失败",
                    "total_vram": 0.0,
                    "available_vram": 0.0,
                    "used_vram": 0.0,
                },
            }


# ==============================================
# 硬件需求计算类
# ==============================================
class HardwareCalculator:
    """基于第一性原理的硬件需求计算"""

    @classmethod
    def calculate_required_hardware(
        cls, config: ModelConfig, input_tokens: int | None = None
    ) -> dict[str, Requirements]:
        """计算模型推理所需硬件资源

        返回两组需求：
        - 'min': 基于短输入（默认 256 tokens）
        - 'max': 基于完整上下文窗口（seq_len * batch_size）

        内存模型：
        VRAM = 权重 + KV Cache + 激活 + 框架开销
        """
        precision_bytes = ModelConfig.PRECISION_BYTES

        # 断言：validate() 应在调用前执行
        assert config.params_b is not None and config.params_b > 0
        assert config.hidden_size is not None
        assert config.num_layers is not None
        assert config.seq_len is not None

        # 1. 权重显存：参数量 × 每参数字节数（使用量化精度）
        weight_byte = precision_bytes.get(
            config.quantization, precision_bytes[config.precision]
        )
        weight_mem = config.params_b * 1e9 * weight_byte

        # 2. 激活显存：约为权重的 5%
        activation_mem = weight_mem * 0.05

        # 3. 框架开销：约 1 GiB
        framework_mem = 1.0 * (1024**3)

        # 4. KV Cache（使用推理精度，非量化精度）
        #    每token: 2(K+V) × kv_head_ratio × hidden_size × num_layers × byte
        #    GQA 模型 kv_head_ratio < 1，KV Cache 更小
        kv_byte = precision_bytes.get(config.precision, 2)
        kv_per_token = (
            2 * config.kv_head_ratio * config.hidden_size * config.num_layers * kv_byte
        )

        # MIN：短输入模式
        min_tokens = input_tokens or 256
        kv_min = kv_per_token * min_tokens
        vram_min = weight_mem + kv_min + activation_mem + framework_mem

        # MAX：完整上下文模式
        max_tokens = config.seq_len * config.batch_size
        kv_max = kv_per_token * max_tokens
        vram_max = weight_mem + kv_max + activation_mem + framework_mem

        # RAM：至少需要加载模型权重 + 开销
        ram = weight_mem * 1.1 + framework_mem

        # CPU 核心数（推理对 CPU 要求较低）
        if config.params_b < 7:
            cpu_cores = 2 if config.batch_size <= 2 else 4
        else:
            cpu_cores = 4 if config.batch_size <= 4 else 6

        b2gb = HardwareDetector.bytes2gb
        return {
            "min": {
                "required_vram": b2gb(vram_min),
                "required_ram": b2gb(ram),
                "required_cpu_core": cpu_cores,
                "weight_vram": b2gb(weight_mem),
                "kv_vram": b2gb(kv_min),
                "activation_vram": b2gb(activation_mem),
                "framework_overhead": b2gb(framework_mem),
                "input_tokens": min_tokens,
            },
            "max": {
                "required_vram": b2gb(vram_max),
                "required_ram": b2gb(ram),
                "required_cpu_core": cpu_cores,
                "weight_vram": b2gb(weight_mem),
                "kv_vram": b2gb(kv_max),
                "activation_vram": b2gb(activation_mem),
                "framework_overhead": b2gb(framework_mem),
                "input_tokens": max_tokens,
            },
        }

    @staticmethod
    def judge_compatibility(
        required: Requirements, current: HardwareInfo
    ) -> CompatibilityResult:
        """判断硬件兼容性"""
        if current["gpu"]["total_vram"] == 0:
            return {
                "can_run": False,
                "reason": "未检测到 GPU，LLM 推理需要 GPU",
                "detail": {},
            }

        vram_ok = current["gpu"]["total_vram"] >= required["required_vram"]
        ram_ok = current["ram"]["total"] >= required["required_ram"]
        cpu_ok = current["cpu"]["core_count"] >= required["required_cpu_core"]

        can_run = vram_ok and ram_ok and cpu_ok
        reasons = []
        if not vram_ok:
            reasons.append(
                f"GPU 显存不足 (有 {current['gpu']['total_vram']:.1f}GB, "
                f"需 {required['required_vram']:.1f}GB)"
            )
        if not ram_ok:
            reasons.append(
                f"内存不足 (有 {current['ram']['total']:.1f}GB, "
                f"需 {required['required_ram']:.1f}GB)"
            )
        if not cpu_ok:
            reasons.append(
                f"CPU 核心不足 (有 {current['cpu']['core_count']}, "
                f"需 {required['required_cpu_core']})"
            )

        return {
            "can_run": can_run,
            "reason": "; ".join(reasons) if reasons else "硬件条件全部满足，模型可正常运行",
            "detail": {
                "gpu_vram_check": vram_ok,
                "cpu_ram_check": ram_ok,
                "cpu_core_check": cpu_ok,
            },
        }


# ==============================================
# 性能测试引擎类
# ==============================================
class PerformanceTester:
    """性能测试引擎，支持 vLLM 和 transformers"""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.engine_type = self._detect_engine()

    def _detect_engine(self) -> str:
        """自动检测可用的推理引擎"""
        if self.config.engine == "vllm":
            return "vllm"
        if self.config.engine == "transformers":
            return "transformers"
        # auto: 优先 vLLM
        try:
            import vllm  # noqa: F401

            return "vllm"
        except ImportError:
            return "transformers"

    def test_performance(self) -> PerformanceResult:
        """执行性能测试"""
        print(f"[INFO] 使用引擎: {self.engine_type}")
        if self.engine_type == "vllm":
            return self._test_with_vllm()
        return self._test_with_transformers()

    # ---- vLLM 引擎 ----

    def _test_with_vllm(self) -> PerformanceResult:
        """使用 vLLM 进行性能测试"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            print("[INFO] 正在加载 vLLM 模型...")

            # vLLM 的 quantization 参数仅接受特定量化方法
            # 精度名（fp16/bf16/fp32）不是量化方法，应传 None
            vllm_quant = (
                self.config.quantization
                if self.config.quantization in VLLM_QUANT_METHODS
                else None
            )

            llm = LLM(
                model=self.config.model_path,
                quantization=vllm_quant,  # type: ignore[arg-type]
                max_model_len=self.config.seq_len,
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=True
            )
            input_tokens = len(tokenizer.encode(self.config.test_prompt))

            # 测量 TTFT（3 次取均值，每次生成 1 个 token）
            ttft_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                stop_token_ids=(
                    [tokenizer.eos_token_id] if tokenizer.eos_token_id else None
                ),
            )
            ttft_times = []
            for _ in range(3):
                t0 = time.time()
                llm.generate([self.config.test_prompt], ttft_params)
                ttft_times.append(time.time() - t0)

            avg_ttft = sum(ttft_times) / len(ttft_times)
            ttft_ms = round(avg_ttft * 1000, 2)

            # 完整生成吞吐量
            full_params = SamplingParams(
                max_tokens=self.config.max_new_tokens,
                temperature=0.0,
                stop_token_ids=(
                    [tokenizer.eos_token_id] if tokenizer.eos_token_id else None
                ),
            )

            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            outputs = llm.generate([self.config.test_prompt], full_params)
            total_time = time.time() - t0

            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            cur_mem = torch.cuda.memory_allocated() / (1024**3)

            generated_tokens = len(outputs[0].outputs[0].token_ids)
            tpot_ms, throughput = self._calc_decode_metrics(
                generated_tokens, total_time, avg_ttft
            )

            return {
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "TTFT(ms)": ttft_ms,
                "TPOT(ms)": tpot_ms,
                "total_latency(s)": round(total_time, 2),
                "throughput(token/s)": throughput,
                "gpu_peak_memory_gb": round(peak_mem, 2),
                "gpu_current_memory_gb": round(cur_mem, 2),
            }

        except Exception as e:
            print(f"[ERROR] vLLM 测试失败: {e}")
            print("[INFO] 回退到 transformers 引擎...")
            return self._test_with_transformers()

    # ---- Transformers 引擎 ----

    def _test_with_transformers(self) -> PerformanceResult:
        """使用 transformers 进行性能测试"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print("[INFO] 正在加载 transformers 模型...")

            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=True, padding_side="left"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 使用配置的精度，而非硬编码
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=self.config.torch_dtype,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            inputs = tokenizer(
                self.config.test_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_tokens = inputs["input_ids"].shape[1]

            # 预热
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 测量 TTFT（3 次取均值，每次生成 1 个 token）
            ttft_times = []
            for _ in range(3):
                t0 = time.time()
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                ttft_times.append(time.time() - t0)

            avg_ttft = sum(ttft_times) / len(ttft_times)
            ttft_ms = round(avg_ttft * 1000, 2)

            # 完整生成吞吐量
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            total_time = time.time() - t0

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                cur_mem = torch.cuda.memory_allocated() / (1024**3)
                torch.cuda.empty_cache()
            else:
                peak_mem = cur_mem = 0.0

            generated_tokens = outputs.sequences[0, input_tokens:].shape[0]  # type: ignore[union-attr]
            tpot_ms, throughput = self._calc_decode_metrics(
                generated_tokens, total_time, avg_ttft
            )

            return {
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "TTFT(ms)": ttft_ms,
                "TPOT(ms)": tpot_ms,
                "total_latency(s)": round(total_time, 2),
                "throughput(token/s)": throughput,
                "gpu_peak_memory_gb": round(peak_mem, 2),
                "gpu_current_memory_gb": round(cur_mem, 2),
            }

        except Exception as e:
            print(f"[ERROR] transformers 测试失败: {e}")
            return self._empty_result()

    # ---- 公共工具方法 ----

    @staticmethod
    def _calc_decode_metrics(
        generated_tokens: int, total_time: float, avg_ttft: float
    ) -> tuple[float, float]:
        """从原始计时计算 TPOT 和吞吐量

        TPOT = (总时间 - TTFT) / (生成token数 - 1)
        吞吐量 = 生成token数 / 总时间
        """
        if generated_tokens > 0 and total_time > 0:
            n_decode = max(generated_tokens - 1, 1)
            tpot_ms = round((total_time - avg_ttft) / n_decode * 1000, 2)
            throughput = round(generated_tokens / total_time, 2)
        else:
            tpot_ms = throughput = 0.0
        return tpot_ms, throughput

    @staticmethod
    def _empty_result() -> PerformanceResult:
        """返回失败的空结果"""
        return {
            "input_tokens": 0,
            "generated_tokens": 0,
            "TTFT(ms)": 0.0,
            "TPOT(ms)": 0.0,
            "total_latency(s)": 0.0,
            "throughput(token/s)": 0.0,
            "gpu_peak_memory_gb": None,
            "gpu_current_memory_gb": None,
        }


# ==============================================
# 主测试类：统一管理测试流程
# ==============================================
class LLMRuntimeChecker:
    """LLM 运行时环境检查主类"""

    def __init__(self, config_dict: dict[str, Any]) -> None:
        self.config = ModelConfig(config_dict)
        self.hardware_detector = HardwareDetector()
        self.hardware_calculator = HardwareCalculator()

    def run_full_test(self) -> None:
        """执行完整的运行时检查"""
        print("=" * 80)
        print("LLM 运行时环境检查")
        print("=" * 80)

        # 1. 配置验证
        if not self._validate_config():
            return

        # 2. 环境检查
        self._check_environment()

        # 3. 硬件需求计算
        required = self._calculate_hardware_requirements()
        if not required:
            return

        # 4. 硬件兼容性判断
        compatibility = self._check_compatibility(required)

        # 5. 性能测试（兼容时执行）
        if compatibility["can_run"]:
            self._run_performance_test()

        print("\n" + "=" * 80)
        print("[OK] 检查完成")

    def _validate_config(self) -> bool:
        """验证配置"""
        print("\n[步骤1] 配置验证")
        if not self.config.validate():
            print("[ERROR] 配置验证失败")
            return False
        print("[OK] 配置验证通过")

        table = [
            ["模型路径", self.config.model_path],
            ["参数量", f"{self.config.params_b:.2f}B"],
            ["精度", self.config.precision],
            ["量化", self.config.quantization],
            ["序列长度", self.config.seq_len],
            ["隐藏层维度", self.config.hidden_size],
            ["层数", self.config.num_layers],
            ["注意力头数", self.config.num_attention_heads],
            ["KV 头数", self.config.num_kv_heads],
            ["GQA 比值", f"{self.config.kv_head_ratio:.2f}"],
            ["批次大小", self.config.batch_size],
        ]
        print("\n模型配置信息:")
        print(tabulate(table, tablefmt="grid", headers=["参数", "值"]))
        return True

    def _check_environment(self) -> None:
        """检查运行环境"""
        print("\n[步骤2] 环境检查")
        print(f"Python 版本: {sys.version.split()[0]}")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")

    def _calculate_hardware_requirements(self) -> dict[str, Requirements] | None:
        """计算硬件需求（min/max 两组）"""
        print("\n[步骤3] 硬件需求计算")
        try:
            required = self.hardware_calculator.calculate_required_hardware(
                self.config
            )

            # 最小需求（短输入模式）
            print("\n[最小需求 - 短输入模式]")
            min_req = required["min"]
            table_min = [
                ["VRAM 需求", f"{min_req['required_vram']:.2f} GB"],
                ["|-- 权重", f"{min_req['weight_vram']:.2f} GB"],
                ["|-- KV Cache (256 tokens)", f"{min_req['kv_vram']:.2f} GB"],
                ["|-- 激活", f"{min_req['activation_vram']:.2f} GB"],
                ["|-- 框架开销", f"{min_req['framework_overhead']:.2f} GB"],
                ["RAM 需求", f"{min_req['required_ram']:.2f} GB"],
                ["CPU 核心数", f"{min_req['required_cpu_core']}"],
            ]
            print(tabulate(table_min, tablefmt="grid"))

            # 最大需求（完整上下文模式）
            print("\n[最大需求 - 完整上下文模式]")
            max_req = required["max"]
            table_max = [
                ["VRAM 需求", f"{max_req['required_vram']:.2f} GB"],
                ["|-- 权重", f"{max_req['weight_vram']:.2f} GB"],
                ["|-- KV Cache (full)", f"{max_req['kv_vram']:.2f} GB"],
                ["|-- 激活", f"{max_req['activation_vram']:.2f} GB"],
                ["|-- 框架开销", f"{max_req['framework_overhead']:.2f} GB"],
                ["RAM 需求", f"{max_req['required_ram']:.2f} GB"],
                ["CPU 核心数", f"{max_req['required_cpu_core']}"],
            ]
            print(tabulate(table_max, tablefmt="grid"))

            return required
        except Exception as e:
            print(f"[ERROR] 硬件需求计算失败: {e}")
            return None

    def _check_compatibility(
        self, required_hardware: dict[str, Requirements]
    ) -> CompatibilityResult:
        """硬件兼容性判断"""
        print("\n[步骤4] 硬件兼容性判断")

        current = self.hardware_detector.get_current_hardware()

        table = [
            ["GPU 型号", current["gpu"]["name"]],
            ["GPU 总显存", f"{current['gpu']['total_vram']:.2f} GB"],
            ["GPU 已用显存", f"{current['gpu'].get('used_vram', 0):.2f} GB"],
            ["CPU 逻辑核心数", f"{current['cpu']['core_count']}"],
            ["内存总量", f"{current['ram']['total']:.2f} GB"],
            ["内存已用", f"{current['ram']['used']:.2f} GB"],
        ]
        print(tabulate(table, tablefmt="grid"))

        # 短输入模式兼容性
        compat_min = self.hardware_calculator.judge_compatibility(
            required_hardware["min"], current
        )
        print(f"\n[兼容性 - 短输入模式]")
        print(
            f"状态: {'[OK] 可运行' if compat_min['can_run'] else '[ERROR] 无法运行'}"
        )
        print(f"原因: {compat_min['reason']}")

        # 完整上下文模式兼容性
        compat_max = self.hardware_calculator.judge_compatibility(
            required_hardware["max"], current
        )
        print(f"\n[兼容性 - 完整上下文模式]")
        print(
            f"状态: {'[OK] 可运行' if compat_max['can_run'] else '[WARN] 可能无法运行'}"
        )
        print(f"原因: {compat_max['reason']}")

        return compat_min

    def _run_performance_test(self) -> None:
        """运行性能测试"""
        print("\n[步骤5] 性能测试")

        # 监测测试前硬件变化
        hw_before = self.hardware_detector.get_current_hardware()
        time.sleep(0.5)
        hw_after = self.hardware_detector.get_current_hardware()

        cpu_delta = hw_after["cpu"]["current_usage"] - hw_before["cpu"]["current_usage"]
        ram_delta = hw_after["ram"]["used"] - hw_before["ram"]["used"]
        print(f"CPU 变化: {cpu_delta:+.1f}%, RAM 变化: {ram_delta:+.2f}GB")

        # 执行性能测试
        tester = PerformanceTester(self.config)
        perf = tester.test_performance()

        if perf["generated_tokens"] is not None and perf["generated_tokens"] > 0:
            # 性能指标
            perf_table = [
                ["输入 tokens", perf["input_tokens"]],
                ["生成 tokens", perf["generated_tokens"]],
                ["TTFT(ms)", perf["TTFT(ms)"]],
                ["TPOT(ms)", perf["TPOT(ms)"]],
                ["总延迟(s)", perf["total_latency(s)"]],
                ["吞吐量(token/s)", perf["throughput(token/s)"]],
            ]
            print(tabulate(perf_table, tablefmt="grid"))

            # 显存对比
            if perf.get("gpu_peak_memory_gb") is not None:
                self._show_memory_comparison(perf)

            # 性能评估
            self._evaluate_performance(perf)
        else:
            print("[WARN] 性能测试未产生有效结果")

    def _show_memory_comparison(self, perf: PerformanceResult) -> None:
        """展示实际显存与计算值对比"""
        print("\n显存使用对比:")

        actual_peak = perf["gpu_peak_memory_gb"]
        assert actual_peak is not None  # 仅在非 None 时调用此方法
        input_tok = int(perf["input_tokens"]) if perf["input_tokens"] else None
        required = self.hardware_calculator.calculate_required_hardware(
            self.config, input_tokens=input_tok
        )

        min_vram = required["min"]["required_vram"]
        max_vram = required["max"]["required_vram"]

        mem_table = [
            ["实际峰值显存", f"{actual_peak:.2f} GB"],
            ["计算值: 最小 (256 tokens)", f"{min_vram:.2f} GB"],
            ["计算值: 最大 (full)", f"{max_vram:.2f} GB"],
            ["与最小值偏差", f"{actual_peak - min_vram:+.2f} GB"],
            [
                "状态",
                "[OK] 在最小值内" if actual_peak <= min_vram else "[WARN] 超出最小值",
            ],
        ]
        print(tabulate(mem_table, tablefmt="grid"))

        print("\n显存明细 (最小模式):")
        min_req = required["min"]
        detail_table = [
            ["权重", f"{min_req['weight_vram']:.2f} GB"],
            ["KV Cache", f"{min_req['kv_vram']:.2f} GB"],
            ["激活", f"{min_req['activation_vram']:.2f} GB"],
            ["框架开销", f"{min_req['framework_overhead']:.2f} GB"],
        ]
        print(tabulate(detail_table, tablefmt="grid"))

    def _evaluate_performance(self, perf: PerformanceResult) -> None:
        """评估性能结果"""
        print("\n性能评估:")

        ttft = perf["TTFT(ms)"]
        throughput = perf["throughput(token/s)"]
        assert ttft is not None and throughput is not None

        if ttft < 100:
            print("[OK] 首字时延: 优秀 (<100ms)")
        elif ttft < 500:
            print("[WARN] 首字时延: 一般 (100-500ms)")
        else:
            print("[ERROR] 首字时延: 较差 (>500ms)")

        if throughput > 50:
            print("[OK] 吞吐量: 优秀 (>50 token/s)")
        elif throughput > 10:
            print("[WARN] 吞吐量: 一般 (10-50 token/s)")
        else:
            print("[ERROR] 吞吐量: 较差 (<10 token/s)")


# ==============================================
# 入口
# ==============================================
if __name__ == "__main__":
    config_dict = {
        "model_path": "Qwen/Qwen3.5-4B",
        # 以下参数可选，不指定会自动设置
        # "precision": "auto",       # auto/fp32/fp16/bf16
        # "quantization": "auto",    # auto/fp32/fp16/bf16/int8/int4
        # "batch_size": 1,
        # "seq_len": None,           # 自动从模型配置获取
        # "max_new_tokens": 512,
        # "device": "auto",
        # "engine": "auto",          # auto/vllm/transformers
    }

    checker = LLMRuntimeChecker(config_dict)
    checker.run_full_test()
