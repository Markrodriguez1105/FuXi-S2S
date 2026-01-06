import argparse
import shutil
import os
import subprocess

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_outputs():
    folders = [
        'output',
        'compare/result',
        'compare/result_improved',
        'compare/plots'
    ]
    files = ['bias_correction_params.pkl']
    for folder in folders:
        path = os.path.join(WORKING_DIR, folder)
        if os.path.exists(path):
            print(f"Removing {path}")
            shutil.rmtree(path)
    for file in files:
        path = os.path.join(WORKING_DIR, file)
        if os.path.exists(path):
            print(f"Removing {path}")
            os.remove(path)
    print("✅ Cleaned all outputs.")


def run_inference(lite=False):
    cmd = [
        'python', 'inference.py',
        '--model', 'model/fuxi_s2s.onnx',
        '--input', 'data/sample',
        '--save_dir', 'output',
        '--total_step', '42',
        '--total_member', '11',
        '--device', 'cuda' if has_cuda() else 'cpu'
    ]
    if lite:
        cmd.append('--lite')
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ Inference completed.")
    else:
        print(f"❌ Inference failed with code {result.returncode}")


def run_all_comparisons():
    for member in range(11):
        cmd = [
            'python', 'compare_pagasa.py',
            '--fuxi_output', 'output',
            '--member', str(member)
        ]
        print(f"Processing member {member}...")
        subprocess.run(cmd)
    print("✅ All comparisons complete.")


def train_bias_correction():
    cmd = ['python', '-c',
           'from utils.bias_correction import BiasCorrector, load_training_data; d=load_training_data(\'compare/result\'); c=BiasCorrector(); c.fit(d); c.save(\'bias_correction_params.pkl\')']
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ Bias correction trained and saved.")
    else:
        print(f"❌ Bias correction training failed with code {result.returncode}")


def run_improved_comparison():
    cmd = [
        'python', 'compare_improved.py',
        '--members', '11',
        '--pagasa', 'data/pagasa/CBSUA Pili, Camarines Sur Daily data.xlsx',
        '--output_dir', 'compare/result_improved'
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ Improved comparison complete.")
    else:
        print(f"❌ Improved comparison failed with code {result.returncode}")


def analyze_results():
    cmd = ['python', 'analyze_results.py']
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ Analysis complete.")
    else:
        print(f"❌ Analysis failed with code {result.returncode}")


def visualize_results():
    cmd = ['python', 'visualize_results.py']
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("✅ All plots saved to compare/plots/")
    else:
        print(f"❌ Visualization failed with code {result.returncode}")


def has_cuda():
    try:
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description='FuXi-S2S CLI Workflow')
    parser.add_argument('--clean', action='store_true', help='Clean all outputs and intermediate files')
    parser.add_argument('--inference', action='store_true', help='Run model inference')
    parser.add_argument('--lite', action='store_true', help='Run inference in lite mode (quick test)')
    parser.add_argument('--compare', action='store_true', help='Run comparison for all members')
    parser.add_argument('--train-bias', action='store_true', help='Train and save bias correction')
    parser.add_argument('--improved-compare', action='store_true', help='Run improved comparison (ensemble + correction)')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    parser.add_argument('--visualize', action='store_true', help='Generate all plots')
    parser.add_argument('--all', action='store_true', help='Run full workflow (clean, inference, compare, train, improved compare, analyze, visualize)')
    args = parser.parse_args()

    if args.all:
        clean_outputs()
        run_inference(lite=args.lite)
        run_all_comparisons()
        train_bias_correction()
        run_improved_comparison()
        analyze_results()
        visualize_results()
        return
    if args.clean:
        clean_outputs()
    if args.inference:
        run_inference(lite=args.lite)
    if args.compare:
        run_all_comparisons()
    if args.train_bias:
        train_bias_correction()
    if args.improved_compare:
        run_improved_comparison()
    if args.analyze:
        analyze_results()
    if args.visualize:
        visualize_results()

if __name__ == '__main__':
    main()
