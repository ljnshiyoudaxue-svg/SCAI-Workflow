# tool_registry2.py
from Agent_tools import call_unet_api,save_base64_mask,call_sam2_box,preprocess_mask_for_analysis,split_mask_to_contours,call_crack_api,parse_deepseek_json,sliding_window_sam2_analysis,sliding_window_unet_analysis,overlay_masks,draw_final_results,sliding_window_analysis,generate_x_points,overwrite_metrics,resolve_masks_from_sliding_results,build_final_result_json,extract_metrics_and_xpoints
from Agent_tools import call_vug_api,overlay_masks,sliding_window_unet_analysis,sliding_window_vug_analysis,visualize_pipeline,draw_final_results,deepseek_decide_models,generate_prompt,deepseek_filter_curves_safe,generate_comprehensive_report,record_execution_state,extract_fracture_metrics,draw_final_results,extract_overlay_masks,rebuild_unet_results,call_yolo_api
# ✅ 导入核心 API
# ======================================
# ✅ Executor-Compatible TOOL_REGISTRY
# ======================================

TOOL_REGISTRY = {

    # ===== 一、DeepSeek / LLM =====
    "parse_deepseek_json": {
        "func": parse_deepseek_json,
        "inputs": ["deepseek_json", "px_to_m", "start_depth_m"],
        "outputs": ["result"]
    },

"deepseek_filter_curves_safe": {
    "func": deepseek_filter_curves_safe,
    "inputs": [
        "curves_metrics",
        "x_points_list",
        "image_height_px",
        "image_width_px",
        "min_points",
        "max_retries",
        "flags",
        "log_fn",
        "strategy"
    ],
    "outputs": [
        "fracture_metrics_list",   # 覆盖后的 metrics，用于后续步骤
        "reflection.curves_filtered",      # DeepSeek 过滤后的曲线
        "reflection.analysis_log",         # DeepSeek 分析日志
        "reflection.pre_filtered"          # 预筛选曲线
    ]
},
"rebuild_unet_results": {
    "func": rebuild_unet_results,
    "inputs": [
        "sliding_results",
        "fracture_metrics_list",
        "log_fn"
    ],
    "outputs": [
        "unet_results"
    ]
},
    "deepseek_decide_models": {
        "func": deepseek_decide_models,
        "inputs": ["user_input", "yolo_results", "strategy"],
        "outputs": ["model_ids"]
    },

    # ===== 二、检测 / 分割模型 =====
    "call_yolo_api": {
        "func": call_yolo_api,
        "inputs": ["image_path"],
        "outputs": ["yolo_results"]
    },

    "call_unet_api": {
        "func": call_unet_api,
        "inputs": ["model_id", "image_path"],
        "outputs": ["mask"]
    },

    "call_sam2_box": {
        "func": call_sam2_box,
        "inputs": ["image_path", "box_coords"],
        "outputs": ["mask"]
    },

    # ===== 三、Mask 处理 =====
    "save_base64_mask": {
        "func": save_base64_mask,
        "inputs": ["mask_b64", "save_path"],
        "outputs": ["save_path"]
    },

    "preprocess_mask_for_analysis": {
        "func": preprocess_mask_for_analysis,
        "inputs": ["mask_path", "log_fn"],
        "outputs": ["temp_mask_path"]
    },

    "split_mask_to_contours": {
        "func": split_mask_to_contours,
        "inputs": ["mask_path"],
        "outputs": ["single_masks"]
    },

    "overlay_masks": {
        "func": overlay_masks,
        "inputs": ["masks", "model_ids", "base_image_path","log_fn"],
        "outputs": ["out_path"]
    },
"draw_final_results": {
    "func": draw_final_results,
    "inputs": [
        "base_image_path",   # 底图路径
        "unet_results",      # U-Net 分析结果（列表，每项包含 class, mask_result, metrics_list）
        "yolo_results",      # YOLO 检测结果（字典，包含 "detections"）
        "H"                  # 图像高度，用于绘制曲线时裁剪
    ],
    "outputs": [
        "out_path"           # 绘制完成的最终图路径
    ]
},
"overwrite_metrics": {
        "func": overwrite_metrics,
        "inputs": [
            "target",
            "source",
            "flags",
            "log_fn"
        ],
        "outputs": ["derived.fracture.metrics_list"]
    },
    # ===== 四、参数分析 =====
    "call_crack_api": {
        "func": call_crack_api,
        "inputs": ["mask_path", "image_height_mm", "image_width_mm"],
        "outputs": ["result"]
    },

    "call_vug_api": {
        "func": call_vug_api,
        "inputs": [
            "image_path",
            "image_height_mm",
            "image_width_mm",
            "window_height_mm"
        ],
        "outputs": ["result"]
    },

    # ===== 五、滑窗分析（统一调度） =====
    "sliding_window_analysis": {
        "func": sliding_window_analysis,
        "inputs": ["image_path", "model_ids", "model_parameters", "log_fn"],
        "outputs": ["sliding_results"]
    },

    "sliding_window_unet_analysis": {
        "func": sliding_window_unet_analysis,
        "inputs": [
            "image_path",
            "model_id",
            "image_height_mm",
            "image_width_mm",
            "log_fn"
        ],
        "outputs": ["fracture_mask", "fracture_metrics"]
    },
"resolve_masks_from_sliding_results": {
    "func": resolve_masks_from_sliding_results,
    "inputs": [
        "sliding_results",
        "log_fn"
    ],
    "outputs": [
        "derived.overlay.masks"
    ]
},
"extract_overlay_masks": {
    "func": extract_overlay_masks,
    "inputs": [
        "sliding_results",
        "log_fn"
    ],
    "outputs": [
        "derived.overlay.fracture_mask",
        "derived.overlay.vug_mask"
    ]
},"visualize_pipeline": {
    "func": visualize_pipeline,
    "inputs": [
        "image_path",
        "sliding_results",
        "unet_results",
        "yolo_results",
        "model_ids",
        "image_height_px",
        "log_fn"
    ],
    "outputs": [
        "overlay_image",
        "final_image"
    ]
},
    "sliding_window_vug_analysis": {
        "func": sliding_window_vug_analysis,
        "inputs": [
            "image_path",
            "model_id",
            "image_height_mm",
            "image_width_mm",
            "window_height_mm",
            "window_px",
            "log_fn"
        ],
        "outputs": ["vug_mask", "window_metrics", "summary"]
    },
# ===== 二、Recorder =====
    "record_execution_state": {
        "func": record_execution_state,
        "inputs": [
            "intent",
            "planner",
            "flags",
            "raw_sliding",
            "refined_curves",
            "analysis_log",
            "extra",
            "output_dir",
            "prefix"
        ],
        "outputs": [
            "record_path",
            "record_id",
            "timestamp"
        ]
    },
"build_final_result_json": {
    "func": build_final_result_json,
    "inputs": [
        "user_prompt",
        "yolo_results",
        "sam2_results",
        "unet_results",
        "sliding_results",
        "params_used",
        "log_fn"
    ],
    "outputs": [
        "deepseek_json"
    ]
},
"generate_x_points": {
    "func": generate_x_points,
    "inputs": ["curves_metrics", "image_width_px"],
    "outputs": ["x_points_list"]
},
"extract_fracture_metrics": {
        "func": extract_fracture_metrics,
        "inputs": [
            "sliding_results"
        ],
        "outputs": [
            "metrics_list"
        ]
    },
"extract_metrics_and_xpoints": {
    "func": extract_metrics_and_xpoints,
    "inputs": ["sliding_results", "image_width_px"],
    "outputs": ["curves_metrics", "x_points_list"]
},
    "generate_comprehensive_report": {
        "func": generate_comprehensive_report,
        "inputs": [
            "result",
            "final_image",
            "timeout"
        ],
        "outputs": [
            "report.report_path",
            "report.report_preview",
            "report.status"
        ]
    }
}
