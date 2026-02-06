#!/usr/bin/env python
"""
LoraRetriever 综合评测脚本

对所有 14 个 app LoRA 和 5 个 category LoRA 的测试数据分别运行推理和评测，
计算每个数据集的 step-level 和 episode-level 准确率，以及 Average 和 OVERALL 指标，
生成 summary.md 报告。
"""

import os
import sys
import json
import argparse
import subprocess
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据集定义
APP_DATASETS = [
    'adidas', 'amazon', 'calendar', 'clock', 'decathlon',
    'ebay', 'etsy', 'flipkart', 'gmail', 'google_drive',
    'google_maps', 'kitchen_stories', 'reminder', 'youtube'
]

CATEGORY_DATASETS = [
    'entertainment', 'lives', 'office', 'shopping', 'traveling'
]


def parse_args():
    parser = argparse.ArgumentParser(description='LoraRetriever 综合评测')

    # 推理参数
    parser.add_argument('--model_type', type=str, default='internvl2-2b',
                        help='模型类型 (默认: internvl2-2b)')
    parser.add_argument('--gpu_id', type=str, default='5',
                        help='GPU ID (默认: 5)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='选择 top-k 个 LoRA (默认: 3)')
    parser.add_argument('--merge_method', type=str, default='mixture',
                        choices=['mixture', 'fusion'],
                        help='合并方法 (默认: mixture)')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='最大生成 token 数 (默认: 512)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='温度参数 (默认: 0.0)')

    # 输出目录
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: 自动生成带时间戳)')

    # 评测范围
    parser.add_argument('--app_only', action='store_true',
                        help='只评测 app 级别')
    parser.add_argument('--category_only', action='store_true',
                        help='只评测 category 级别')

    # 控制参数
    parser.add_argument('--skip_inference', action='store_true',
                        help='跳过推理，使用已有结果')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='每个数据集只处理 N 个样本')
    parser.add_argument('--show_similarities', action='store_true',
                        help='显示 LoRA 相似度分数')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印命令，不实际执行')

    args = parser.parse_args()

    if args.app_only and args.category_only:
        parser.error('--app_only 和 --category_only 不能同时使用')

    return args


def get_output_dir(args) -> Path:
    """生成或使用指定的输出目录"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_suffix = 'internvl2' if 'internvl2' in args.model_type else 'qwen2vl'
        output_dir = PROJECT_ROOT / 'output' / f'retriever_eval_{model_suffix}_{args.merge_method}_k{args.top_k}_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_inference(dataset_name: str, test_data_path: str, lora_type: str,
                  result_dir: Path, args) -> Optional[Path]:
    """运行单个数据集的推理，返回结果文件路径"""
    result_file = result_dir / f'{dataset_name}_results.jsonl'

    # 如果跳过推理，检查已有结果
    if args.skip_inference:
        if result_file.exists():
            print(f'  [跳过推理] 使用已有结果: {result_file}')
            return result_file
        else:
            print(f'  [警告] 结果文件不存在，跳过: {result_file}')
            return None

    # 使用临时输出目录，避免与其他运行冲突
    temp_output_dir = result_dir / f'_temp_{dataset_name}'
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # 构建推理命令
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'infer_lora_retriever.py'),
        '--model_type', args.model_type,
        '--test_data', test_data_path,
        '--output_dir', str(temp_output_dir),
        '--top_k', str(args.top_k),
        '--merge_method', args.merge_method,
        '--lora_type', lora_type,
        '--max_new_tokens', str(args.max_new_tokens),
        '--temperature', str(args.temperature),
        '--gpu_id', args.gpu_id,
    ]

    if args.debug:
        cmd.append('--debug')
    if args.num_samples is not None:
        cmd.extend(['--num_samples', str(args.num_samples)])
    if args.show_similarities:
        cmd.append('--show_similarities')

    print(f'  命令: {" ".join(cmd)}')

    if args.dry_run:
        return None

    # 执行推理
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    env.setdefault('MAX_PIXELS', '100000')
    env.setdefault('MAX_NUM', '12')
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['TRITON_PTXAS_PATH'] = ''
    env.setdefault('JINA_MAX_PIXELS', '100000')
    env.setdefault('JINA_IMAGE_BATCH_SIZE', '1')

    try:
        proc = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT),
                              capture_output=False, text=True)
        if proc.returncode != 0:
            print(f'  [错误] 推理失败 (返回码: {proc.returncode})')
            return None
    except Exception as e:
        print(f'  [错误] 推理异常: {e}')
        return None

    # 查找生成的结果文件（按修改时间排序取最新）
    result_files = sorted(temp_output_dir.glob('retriever_results_*.jsonl'),
                          key=lambda f: f.stat().st_mtime, reverse=True)
    if not result_files:
        print(f'  [错误] 未找到推理结果文件')
        return None

    # 移动到目标路径
    shutil.move(str(result_files[0]), str(result_file))

    # 清理临时目录
    shutil.rmtree(str(temp_output_dir), ignore_errors=True)

    print(f'  结果保存: {result_file}')
    return result_file


def run_evaluation(result_file: Path) -> Tuple[Optional[float], Optional[float]]:
    """运行评测，返回 (step_accuracy, episode_accuracy)，均为百分比值"""
    cmd = [sys.executable, str(PROJECT_ROOT / 'evaluation' / 'test_swift.py'),
           '--data_path', str(result_file)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(PROJECT_ROOT))
        output = proc.stdout + '\n' + proc.stderr
    except Exception as e:
        print(f'  [错误] 评测异常: {e}')
        return None, None

    step_acc = None
    episode_acc = None

    for line in output.split('\n'):
        if 'Step-level accuracy' in line:
            # 格式: "Step-level accuracy: XX.XX%"
            match = re.search(r'([\d.]+)%', line)
            if match:
                step_acc = float(match.group(1))
        elif 'Episode-level accuracy' in line:
            # 格式: "Episode-level accuracy: XX.XX" (无 % 号)
            match = re.search(r':\s*([\d.]+)', line)
            if match:
                episode_acc = float(match.group(1))

    return step_acc, episode_acc


def merge_jsonl_files(input_files: List[Path], output_file: Path):
    """合并多个 JSONL 文件"""
    with open(output_file, 'w', encoding='utf-8') as out:
        for f in input_files:
            with open(f, 'r', encoding='utf-8') as inp:
                for line in inp:
                    line = line.strip()
                    if line:
                        out.write(line + '\n')


def evaluate_level(level_name: str, datasets: List[str], test_data_dir: str,
                   lora_type: str, result_dir: Path, args) -> Dict:
    """评测某个级别（app 或 category）的所有数据集"""
    result_dir.mkdir(parents=True, exist_ok=True)

    per_dataset_results = {}
    successful_result_files = []

    for i, dataset in enumerate(datasets):
        test_data_path = str(PROJECT_ROOT / test_data_dir / f'{dataset}_train.jsonl')
        print(f'\n[{level_name}] ({i+1}/{len(datasets)}) 评测 {dataset}')
        print(f'  测试数据: {test_data_path}')

        # 推理
        result_file = run_inference(dataset, test_data_path, lora_type,
                                    result_dir, args)
        if result_file is None:
            per_dataset_results[dataset] = {'step_acc': None, 'episode_acc': None}
            continue

        # 评测
        step_acc, episode_acc = run_evaluation(result_file)
        per_dataset_results[dataset] = {
            'step_acc': step_acc,
            'episode_acc': episode_acc,
        }
        successful_result_files.append(result_file)

        if step_acc is not None:
            print(f'  Step-level: {step_acc:.2f}%  Episode-level: {episode_acc:.2f}%')

    # 计算 Average（各数据集准确率的算术平均）
    valid_step = [r['step_acc'] for r in per_dataset_results.values() if r['step_acc'] is not None]
    valid_episode = [r['episode_acc'] for r in per_dataset_results.values() if r['episode_acc'] is not None]
    avg_step = sum(valid_step) / len(valid_step) if valid_step else None
    avg_episode = sum(valid_episode) / len(valid_episode) if valid_episode else None

    # 计算 OVERALL（合并所有数据后评测）
    overall_step = None
    overall_episode = None
    if successful_result_files and not args.dry_run:
        merged_file = result_dir / f'{lora_type}_overall_merged.jsonl'
        merge_jsonl_files(successful_result_files, merged_file)
        overall_step, overall_episode = run_evaluation(merged_file)
        print(f'\n[{level_name}] OVERALL - Step: {overall_step}%  Episode: {overall_episode}%')

    return {
        'per_dataset': per_dataset_results,
        'average': {'step_acc': avg_step, 'episode_acc': avg_episode},
        'overall': {'step_acc': overall_step, 'episode_acc': overall_episode},
    }


def fmt_acc(val: Optional[float]) -> str:
    """格式化准确率值"""
    if val is None:
        return 'N/A'
    return f'{val:.2f}%'


def generate_summary(app_results: Optional[Dict], category_results: Optional[Dict],
                     args, output_dir: Path):
    """生成 summary.md 报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    lines.append('# LoraRetriever Evaluation Summary')
    lines.append('')
    lines.append(f'**Date:** {now}')
    lines.append('')
    lines.append('**Configuration:**')
    lines.append(f'- GPU ID: {args.gpu_id}')
    lines.append(f'- Top-K: {args.top_k}')
    lines.append(f'- Merge Method: {args.merge_method}')
    lines.append(f'- Model Type: {args.model_type}')
    lines.append('')
    lines.append('---')

    # App-level 结果
    if app_results is not None:
        lines.append('')
        lines.append(f'## App-level Results ({len(APP_DATASETS)} App LoRAs)')
        lines.append('')
        lines.append('| App | Step-level Accuracy | Episode-level Accuracy |')
        lines.append('|-----|---------------------|------------------------|')
        for dataset in APP_DATASETS:
            r = app_results['per_dataset'].get(dataset, {})
            lines.append(f'| {dataset} | {fmt_acc(r.get("step_acc"))} | {fmt_acc(r.get("episode_acc"))} |')
        avg = app_results['average']
        lines.append(f'| **Average** | **{fmt_acc(avg["step_acc"])}** | **{fmt_acc(avg["episode_acc"])}** |')
        overall = app_results['overall']
        lines.append(f'| **OVERALL** | **{fmt_acc(overall["step_acc"])}** | **{fmt_acc(overall["episode_acc"])}** |')
        lines.append('')
        lines.append('---')

    # Category-level 结果
    if category_results is not None:
        lines.append('')
        lines.append(f'## Category-level Results ({len(CATEGORY_DATASETS)} Category LoRAs)')
        lines.append('')
        lines.append('| Category | Step-level Accuracy | Episode-level Accuracy |')
        lines.append('|----------|---------------------|------------------------|')
        for dataset in CATEGORY_DATASETS:
            r = category_results['per_dataset'].get(dataset, {})
            lines.append(f'| {dataset} | {fmt_acc(r.get("step_acc"))} | {fmt_acc(r.get("episode_acc"))} |')
        avg = category_results['average']
        lines.append(f'| **Average** | **{fmt_acc(avg["step_acc"])}** | **{fmt_acc(avg["episode_acc"])}** |')
        overall = category_results['overall']
        lines.append(f'| **OVERALL** | **{fmt_acc(overall["step_acc"])}** | **{fmt_acc(overall["episode_acc"])}** |')
        lines.append('')
        lines.append('---')

    # Overall Summary
    if app_results is not None and category_results is not None:
        lines.append('')
        lines.append('## Overall Summary')
        lines.append('')
        lines.append('| Level | Step-level Accuracy | Episode-level Accuracy |')
        lines.append('|-------|---------------------|------------------------|')
        app_o = app_results['overall']
        cat_o = category_results['overall']
        lines.append(f'| App-level OVERALL | {fmt_acc(app_o["step_acc"])} | {fmt_acc(app_o["episode_acc"])} |')
        lines.append(f'| Category-level OVERALL | {fmt_acc(cat_o["step_acc"])} | {fmt_acc(cat_o["episode_acc"])} |')
        lines.append('')

    summary_path = output_dir / 'summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n报告已保存: {summary_path}')

    # 同时保存 JSON 格式结果
    results_json = {
        'date': now,
        'config': {
            'gpu_id': args.gpu_id,
            'top_k': args.top_k,
            'merge_method': args.merge_method,
            'model_type': args.model_type,
        },
    }
    if app_results is not None:
        results_json['app_results'] = app_results
    if category_results is not None:
        results_json['category_results'] = category_results

    results_json_path = output_dir / 'results.json'
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f'JSON 结果已保存: {results_json_path}')


def main():
    args = parse_args()
    output_dir = get_output_dir(args)

    print('=' * 60)
    print('LoraRetriever 综合评测')
    print('=' * 60)
    print(f'模型类型: {args.model_type}')
    print(f'GPU: {args.gpu_id}')
    print(f'Top-K: {args.top_k}')
    print(f'合并方法: {args.merge_method}')
    print(f'输出目录: {output_dir}')
    if args.skip_inference:
        print('模式: 跳过推理，使用已有结果')
    if args.dry_run:
        print('模式: Dry Run（只打印命令）')
    if args.num_samples:
        print(f'样本限制: {args.num_samples}')
    print('=' * 60)

    app_results = None
    category_results = None

    # App-level 评测
    if not args.category_only:
        print('\n' + '=' * 60)
        print(f'开始 App-level 评测 ({len(APP_DATASETS)} 个数据集)')
        print('=' * 60)

        app_result_dir = output_dir / 'app_results'
        app_results = evaluate_level(
            'App', APP_DATASETS, 'data/test_data_by_app',
            'app', app_result_dir, args
        )

    # Category-level 评测
    if not args.app_only:
        print('\n' + '=' * 60)
        print(f'开始 Category-level 评测 ({len(CATEGORY_DATASETS)} 个数据集)')
        print('=' * 60)

        category_result_dir = output_dir / 'category_results'
        category_results = evaluate_level(
            'Category', CATEGORY_DATASETS, 'data/test_data_by_category',
            'category', category_result_dir, args
        )

    # 生成报告
    if not args.dry_run:
        generate_summary(app_results, category_results, args, output_dir)

    # 打印最终摘要
    print('\n' + '=' * 60)
    print('评测完成！')
    print('=' * 60)
    if app_results and app_results['overall']['step_acc'] is not None:
        print(f'App OVERALL     - Step: {app_results["overall"]["step_acc"]:.2f}%  '
              f'Episode: {app_results["overall"]["episode_acc"]:.2f}%')
    if category_results and category_results['overall']['step_acc'] is not None:
        print(f'Category OVERALL - Step: {category_results["overall"]["step_acc"]:.2f}%  '
              f'Episode: {category_results["overall"]["episode_acc"]:.2f}%')
    print(f'输出目录: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
