import json
import requests
from typing import Dict, Any, List


# =========================================================
# 1️⃣ 任务分类 Prompt
# =========================================================

INTENT_PROMPT = """
你是测井智能解释系统的任务分类器。

根据用户输入判断任务类型。

仅返回JSON，不要解释，不要输出推理过程：

{
  "task_type": "image_analysis | param_analysis | visualization",
  "need_report": true | false
}

规则：
- 如果包含“分析图像”“识别裂缝”“检测”“重新分析” → image_analysis
- 如果包含“根据参数”“已有数据”“已有裂缝数据” → param_analysis
- 如果包含“叠加”“绘制”“可视化” → visualization
- 默认 image_analysis
- 如果出现“生成报告”“输出报告”“报告””分析“ → need_report = true
- 默认 need_report = true
"""


# =========================================================
# 2️⃣ LLM 调用函数（Ollama deepseek-r1）
# =========================================================

MODEL_NAME = "deepseek-r1:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def extract_json(text: str) -> str:
    """
    从模型输出中抽取 JSON
    防止 deepseek 输出思维链
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return text[start:end]
    except ValueError:
        return text


def call_llm(prompt: str) -> str:
    """
    调用本地 Ollama
    必须返回纯 JSON 字符串
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "").strip()

        # 清洗 JSON
        text = extract_json(text)

        return text

    except Exception as e:
        print(f"⚠ LLM 调用失败: {e}")

        # 稳定兜底
        return json.dumps({
            "task_type": "image_analysis",
            "need_report": True
        })


# =========================================================
# 3️⃣ 语义分类
# =========================================================

def classify_intent(user_input: str) -> Dict[str, Any]:
    prompt = INTENT_PROMPT + "\n\n用户输入:\n" + user_input

    response = call_llm(prompt)

    try:
        result = json.loads(response)
    except Exception:
        result = {
            "task_type": "image_analysis",
            "need_report": True
        }

    # 安全兜底
    if "task_type" not in result:
        result["task_type"] = "image_analysis"

    if "need_report" not in result:
        result["need_report"] = True

    return result


# =========================================================
# 4️⃣ 预定义模板步骤
# =========================================================

FULL_PIPELINE = [
    {"tool": "call_yolo_api"},
    {"tool": "deepseek_decide_models"},
    {"tool": "sliding_window_analysis"},
    {"tool": "extract_metrics_and_xpoints"},
    {"tool": "deepseek_filter_curves_safe"},
    {"tool": "rebuild_unet_results"},
    {"tool": "visualize_pipeline"},
    {"tool": "build_final_result_json"}
]

VISUALIZE_ONLY = [
    {"tool": "visualize_pipeline"},
    {"tool": "build_final_result_json"}
]

PARSE_STEP = {"tool": "parse_deepseek_json"}
REPORT_STEP = {"tool": "generate_comprehensive_report"}


# =========================================================
# 5️⃣ 构建 Plan
# =========================================================

def build_plan(intent: Dict[str, Any]) -> Dict[str, Any]:
    task_type = intent["task_type"]
    need_report = intent["need_report"]

    steps: List[Dict[str, Any]] = []

    if task_type == "image_analysis":
        steps.extend(FULL_PIPELINE)

    elif task_type == "param_analysis":
        steps.append({"tool": "build_final_result_json"})

    elif task_type == "visualization":
        steps.extend(VISUALIZE_ONLY)

    else:
        steps.extend(FULL_PIPELINE)

    # 统一后处理
    steps.append(PARSE_STEP)

    if need_report:
        steps.append(REPORT_STEP)

    # 自动编号
    for idx, step in enumerate(steps):
        step["step"] = idx + 1

    return {
        "intent": task_type,
        "plan": steps
    }
from tool_registry import TOOL_REGISTRY

def build_plan_executable(intent: Dict[str, Any], initial_context: dict) -> Dict[str, Any]:
    """
    生成完整可执行 plan，自动填充 inputs/outputs
    """
    task_type = intent["task_type"]
    need_report = intent["need_report"]

    # 1️⃣ 根据原模板生成初步工具顺序
    steps: List[Dict[str, Any]] = []

    if task_type == "image_analysis":
        steps.extend(FULL_PIPELINE)
    elif task_type == "param_analysis":
        steps.append({"tool": "build_final_result_json"})
    elif task_type == "visualization":
        steps.extend(VISUALIZE_ONLY)
    else:
        steps.extend(FULL_PIPELINE)

    # 后处理步骤
    steps.append(PARSE_STEP)
    if need_report:
        steps.append(REPORT_STEP)

    # 2️⃣ 初始化 context
    context = initial_context.copy()
    # 🔹 将执行器必需变量放入 context
    if "user_input" not in context:
        context["user_input"] = intent.get("raw_input", "用户输入")
    if "timeout" not in context:
        context["timeout"] = 120
    if "params" not in context:
        context["params"] = None

    # 3️⃣ 遍历步骤，自动填充 inputs/outputs
    for idx, step in enumerate(steps):
        tool_name = step["tool"]
        inputs = {}
        outputs = []

        if tool_name in TOOL_REGISTRY:
            tool_def = TOOL_REGISTRY[tool_name]

            # 填充 inputs
            for inp in tool_def.get("inputs", []):
                if inp in context:
                    # 🔹 统一使用变量引用
                    inputs[inp] = f"${inp}"
                else:
                    # 占位符
                    inputs[inp] = f"${inp}"

            # 填充 outputs
            for out in tool_def.get("outputs", []):
                full_name = f"{out}"
                context[out] = full_name
                outputs.append(full_name)
        else:
            step["inputs"] = {}
            step["outputs"] = []

        step["inputs"] = inputs
        step["outputs"] = outputs
        step["step"] = idx + 1

    return {
        "intent": task_type,
        "plan": steps
    }
# =========================================================
# 6️⃣ Plan 校验器
# =========================================================

ALLOWED_TOOLS = {
    "call_yolo_api",
    "deepseek_decide_models",
    "sliding_window_analysis",
    "extract_metrics_and_xpoints",
    "deepseek_filter_curves_safe",
    "rebuild_unet_results",
    "visualize_pipeline",
    "build_final_result_json",
    "parse_deepseek_json",
    "generate_comprehensive_report"
}


def validate_plan(plan: Dict[str, Any]) -> bool:
    if "plan" not in plan:
        raise ValueError("Plan 结构错误：缺少 plan 字段")

    steps = plan["plan"]

    for i, step in enumerate(steps):
        if step["tool"] not in ALLOWED_TOOLS:
            raise ValueError(f"非法工具: {step['tool']}")

        if step["step"] != i + 1:
            raise ValueError("step 编号不连续")

    return True


# =========================================================
# 7️⃣ Planner 对外接口
# =========================================================

def planner(user_input: str) -> Dict[str, Any]:
    intent = classify_intent(user_input)
    plan = build_plan(intent)
    validate_plan(plan)
    return plan

def planner_executable(user_input: str, initial_context: dict) -> Dict[str, Any]:
    """
    调用原 classify_intent + build_plan_executable
    """
    intent = classify_intent(user_input)
    plan = build_plan_executable(intent, initial_context)
    validate_plan(plan)
    return plan

# =========================================================
# 8️⃣ 测试入口
# =========================================================

if __name__ == "__main__":
    user_query = "请分析提供的成像并生成解释报告"

    #result_plan = planner(user_query)

    #print(json.dumps(result_plan, indent=2, ensure_ascii=False))


    initial_context = {
            "image_path": "test.jpg",
            "log_fn": None,
            "image_width_px": 1024,
            "image_height_px": 2048,
            "px_to_m": 0.001,
            "start_depth_m": 4000,
            "strategy": "hard-constraint",
            "flags": {"enable_reflection": True},
            "min_points": 100,
            "max_retries": 3,
            "model_parameters": None,
            "sam2_results": None,
            "params_used": None,
            "report_api_url": None,
            "log": None
        }

    plan = planner_executable(user_query, initial_context)
    print(json.dumps(plan, indent=2, ensure_ascii=False))