import torch
import numpy as np
import time

# Configuration Constants
LATENT_CHANNELS = 4
LATENT_HEIGHT = 90
LATENT_WIDTH = 160
ACTION_DIM = 768
WARMUP_STEPS = 50
NUM_RUNS = 200
BATCH_SIZE = 1

class MockMAGModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(LATENT_CHANNELS, 128, 3, padding=1).cuda()
        self.transformer_block = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512).cuda()

    def forward(self, x, action_prompt):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)
        x = self.transformer_block(x)
        return x

def load_model():
    model = MockMAGModel()
    model.eval()
    return model.cuda()

def prepare_inputs():
    dummy_video_latent = torch.randn(BATCH_SIZE, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH).cuda()
    dummy_action = torch.randn(BATCH_SIZE, ACTION_DIM).cuda()
    return dummy_video_latent, dummy_action

def benchmark_fps(model, inputs):
    video_input, action_input = inputs
    
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(video_input, action_input)
    
    torch.cuda.synchronize()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    
    timings = []
    
    with torch.no_grad():
        for i in range(NUM_RUNS):
            start_events[i].record()
            _ = model(video_input, action_input)
            end_events[i].record()

    torch.cuda.synchronize()
    
    for i in range(NUM_RUNS):
        timings.append(start_events[i].elapsed_time(end_events[i]))
    
    return np.array(timings)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_NOT_AVAILABLE")
        
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    model = load_model()
    inputs = prepare_inputs()
    
    latencies_ms = benchmark_fps(model, inputs)
    
    avg_latency_ms = np.mean(latencies_ms)
    std_latency_ms = np.std(latencies_ms)
    avg_fps = 1000.0 / avg_latency_ms
    
    print("="*40)
    print("PERFORMANCE REPORT")
    print("="*40)
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Avg Latency     : {avg_latency_ms:.2f} ms")
    print(f"Std Deviation   : {std_latency_ms:.2f} ms")
    print("-" * 40)
    print(f"Inference FPS   : {avg_fps:.2f}")
    print("="*40)
