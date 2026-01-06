"""
Model Loading Utilities

Functions for loading and configuring ONNX models.
"""

import onnxruntime as ort


def load_model(model_name, device):
    """
    Load ONNX model with specified device (CUDA or CPU).
    
    Parameters:
    -----------
    model_name : str
        Path to the ONNX model file
    device : str
        Device to use: 'cuda' or 'cpu'
    
    Returns:
    --------
    ort.InferenceSession
        ONNX Runtime inference session
    """
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    
    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'})]
    elif device == "cpu":
        providers = ['CPUExecutionProvider']
        options.intra_op_num_threads = 24
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(
        model_name,
        sess_options=options,
        providers=providers
    )
    return session
