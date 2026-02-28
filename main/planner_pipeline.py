import os
import json

import numpy as np

from planner import planner_executable
from tool_registry import TOOL_REGISTRY
from executor import execute_plan, ExecutionError

def main():

    user_input = "请分析裂缝并生成报告"

    # 🔹 图片路径处理（跨平台安全）

    image_path = "test.jpg"

    # 🔹 初始上下文
    initial_context = {
        "image_path": image_path,
        "image_width_px": 472,
        "image_height_px": 1475,
        "min_points": 100,
        "max_retries": 3,
        "model_parameters": None,
        "sam2_results": None,
        "params_used": None,
        "report_api_url": None,
        "log": None,
        "user_input": user_input,  # ✅ 必须加
        "strategy": "hard-constraint",  # ✅ plan 会引用
        "px_to_m": 0.001,
        "start_depth_m": 4000,
        "timeout": 300,
        "params": None,
        "flags": {"enable_reflection": True},
        "log_fn": None
    }

    # 🔹 用户上下文（Executor 可见）
    USER_CONTEXT = {
        "user_input": user_input,
        "strategy": "hard-constraint",
        "min_points": 100,
        "flags": {"enable_reflection": True},
        "max_retries": 5,
        "log_fn": None,
        "log": None,
        "timeout": 300,
        "user_prompt": "全流程裂缝解释、自反思复核与报告生成",
        "sam2_results": "null",
        "params_used": "null",
        "model_parameters": {
            "image_height_mm": 2500.0,
            "image_width_mm": 215.0,
            "window_height_mm": 50.0
        },
        "image_width_px": 472,
        "image_height_px": 1475,
        "start_depth_m": 4000,
        "px_to_m": 0.001
    }

    # =====================================================
    # 1️⃣ 生成完整可执行 Plan
    # =====================================================
    plan = planner_executable(user_input, initial_context)

    print("\n📋 生成的完整可执行 Plan:\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    # 🔹 打印每步 inputs/outputs 检查
    print("\n🔹 每步 Inputs/Outputs:")
    for step in plan['plan']:
        print(f"Step {step['step']} -> {step['tool']}")
        print(f"  Inputs: {step.get('inputs', {})}")
        print(f"  Outputs: {step.get('outputs', [])}")

    # =====================================================
    # 2️⃣ 执行 Plan
    # =====================================================
    try:
        result = execute_plan(plan, TOOL_REGISTRY, image_path=image_path, user_context=USER_CONTEXT)
    except ExecutionError as e:
        print(f"\n❌ Execution failed: {e}")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return

    # =====================================================
    # 3️⃣ 输出执行结果
    # =====================================================

    print("\n================ Execution Result ================\n")

    # 清理 context 中不可序列化的对象
    if "context" in result:
        # 移除 log_fn 函数
        result["context"].pop("log_fn", None)
        # 也可以移除其他 callable 对象
        result["context"] = {k: v for k, v in result["context"].items() if not callable(v)}

    print(json.dumps(result, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    # 🔹 输出 context keys
    print("\n================ Context Keys ====================\n")
    for k in result.get("context", {}):
        print(f"- {k}")

    # 🔹 输出执行日志
    print("\n================ Execution Log ===================\n")
    for log in result.get("log", []):
        print(log)


if __name__ == "__main__":
    main()
