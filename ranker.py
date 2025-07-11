"""
Calculate level scores based on Excel files.
"""

import pandas as pd
import numpy as np
from utils import special_metrix
import logging
import sys
import argparse
import math

def setup_logging(model_name):
    """Configure logging with model name in filename"""
    log_filename = f'outcome/score_calculation_{model_name.lower()}.log'
    
    # 创建一个handler，用UTF-8编码写入文件
    handler = logging.FileHandler(log_filename, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 移除所有已存在的handler
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    
    # 添加新的handler
    root_logger.addHandler(handler)
    
    return log_filename

def normalize_special_metrics(metrics_list, scores_list):
    """Normalize special metrics"""
    special_metrics = set([k.upper() for k in special_metrix.special_metric_dict.keys()])
    logging.info(f'Special metrics: {special_metrics}')

    normalized_scores = []
    for metric, score in zip(metrics_list, scores_list):
        metric_upper = metric.upper() if isinstance(metric, str) else metric
        if metric_upper in special_metrics:
            logging.info('-'*25)
            logging.info(f'>>> Metric: {metric} | Original: {score}')
            if pd.isna(score) or score == float('inf') or score == 0.0:
                normalized_score = 0.0
            else:
                normalized_score = special_metrix.map_function_for_special(metric_upper, score)
            logging.info(f'>>> Metric: {metric} | Normalized: {normalized_score}')
            normalized_scores.append(normalized_score)
        else:
            normalized_scores.append(score)

    return normalized_scores

def get_level_2_mono(scores):
    """Calculate level-2 score for a single modality"""
    valid_scores = [s for s in scores if not pd.isna(s) and s != float('inf')]
    if not valid_scores:
        return 0.0
    avg = sum(valid_scores) / len(scores)
    logging.info(f"Valid scores: {valid_scores}")
    logging.info(f"Average: {avg}")
    logging.info(f"Total scores: {len(scores)}")
    logging.info(f"Valid scores count: {len(valid_scores)}")
    logging.info(f"Invalid scores count: {len(scores) - len(valid_scores)}")
    return avg

def get_level_2(comprehension_scores, generation_scores):
    """Calculate level-2 score for a single modality"""
    score_c = get_level_2_mono(comprehension_scores)
    score_g = get_level_2_mono(generation_scores)
    return (score_c + score_g) / 2

def get_level_mono(sota_scores, model_scores, level, task_type="Comprehension"):
    """Calculate level score for a single modality (Level-3 and Level-4 use the same logic)"""
    valid_pairs = [(s, m) for s, m in zip(sota_scores, model_scores) 
                  if not pd.isna(s) and not pd.isna(m) and s != float('inf') and m != float('inf')]
    if not valid_pairs:
        return 0.0
    
    logging.info(f"\nLevel-{level} scoring details ({task_type}):")
    logging.info(f"Valid score pairs: {len(valid_pairs)}")
    
    scores = [m if m >= s else 0.0 for s, m in valid_pairs]
    avg_score = sum(scores) / len(sota_scores)
    logging.info(f"Final Level-{level} score: {avg_score:.2f}")
    return avg_score

def get_level_3(sota_c, score_c, sota_g, score_g):
    """
    计算单个模态的level-3分数
    """
    score_c = get_level_mono(sota_c, score_c, 3, "Comprehension")
    score_g = get_level_mono(sota_g, score_g, 3, "Generation")
    return (score_c + score_g) / 2

def get_level_4(sota_c, score_c, sota_g, score_g, epsilon=1e-6):
    """
    计算单个模态的level-4分数
    """
    score_c = get_level_mono(sota_c, score_c, 4, "Comprehension")
    score_g = get_level_mono(sota_g, score_g, 4, "Generation")
    
    if score_c == 0 or score_g == 0:
        return 0.0
    
    return 2 * (score_c * score_g) / (score_c + score_g + epsilon)

def process_sheet(sota_df, model_df, model_name):
    """
    处理单个sheet的数据
    """
    # 提取需要的列
    metrics = sota_df['Metrics'].tolist()
    sota = sota_df['SoTA Performance'].tolist()
    
    # 查找模型名称（大小写不敏感）
    model_columns = model_df.columns
    model_col = next((col for col in model_columns if col.lower() == model_name.lower()), None)
    if model_col is None:
        raise ValueError(f"在Excel文件中找不到模型列: {model_name}")
    
    model_scores = model_df[model_col].tolist()
    
    def to_float_inf(x):
        if pd.isna(x):
            return float('inf')
        if isinstance(x, str) and (x.strip() == '∞' or x.strip().lower() == 'inf'):
            return float('inf')
        try:
            return float(x)
        except Exception:
            return float('inf')
    
    # 转换为float类型
    sota = [to_float_inf(x) for x in sota]
    model_scores = [to_float_inf(x) for x in model_scores]
    
    # 归一化特殊指标
    sota = normalize_special_metrics(metrics, sota)
    model_scores = normalize_special_metrics(metrics, model_scores)
    
    return metrics, sota, model_scores

def get_modality_scores(comprehension_metrics, comprehension_sota, comprehension_scores,
                       generation_metrics, generation_sota, generation_scores):
    """
    计算单个模态的各个level分数
    """
    # Level-2: 理解和生成的平均分
    score_level_2 = get_level_2(comprehension_scores, generation_scores)
    
    # Level-3: 相对于SoTA的表现
    score_level_3 = get_level_3(comprehension_sota, comprehension_scores,
                               generation_sota, generation_scores)
    
    # Level-4: 理解和生成的综合表现
    score_level_4 = get_level_4(comprehension_sota, comprehension_scores,
                               generation_sota, generation_scores)
    
    return score_level_2, score_level_3, score_level_4

def sigmoid_adjust(x):
    """
    对RMSE指标进行sigmoid调整
    """
    T = 5
    return 2 / (1 + math.exp(-T / x)) - 1

def get_level_5(l4_score, sota_df, model_df, model_name):
    """
    计算Level-5分数
    """
    # 从Excel中读取NLP分数
    metrics = sota_df['Metrics'].tolist()
    sota_scores = sota_df['SoTA Performance'].tolist()
    
    # 查找模型名称（大小写不敏感）
    model_columns = model_df.columns
    model_col = next((col for col in model_columns if col.lower() == model_name.lower()), None)
    if model_col is None:
        raise ValueError(f"在Excel文件中找不到模型列: {model_name}")
    
    model_scores = model_df[model_col].tolist()
    
    def to_float_inf(x):
        if pd.isna(x):
            return float('inf')
        if isinstance(x, str) and (x.strip() == '∞' or x.strip().lower() == 'inf'):
            return float('inf')
        try:
            return float(x)
        except Exception:
            return float('inf')
    
    # 转换为float类型
    sota_scores = [to_float_inf(x) for x in sota_scores]
    model_scores = [to_float_inf(x) for x in model_scores]
    
    # 对RMSE指标进行特殊处理
    rmse_index = next((i for i, m in enumerate(metrics) if m.upper() == 'RMSE'), None)
    if rmse_index is not None:
        model_scores[rmse_index] = sigmoid_adjust(model_scores[rmse_index]) * 100
    
    # 计算获胜任务的平均分
    valid_pairs = [(s, m) for s, m in zip(sota_scores, model_scores) 
                  if not pd.isna(s) and not pd.isna(m) and s != float('inf') and m != float('inf')]
    if not valid_pairs:
        return 0.0
    
    T = len(valid_pairs)
    # 计算获胜任务数
    wins = sum(1 for s, m in valid_pairs if m >= s)
    
    s_l = [m if m >= s else 0 for s, m in valid_pairs]
    s_l = sum(s_l) / len(sota_scores)
    
    # 计算权重
    w_l = s_l / 100
    # 计算Level-5分数
    l5_score = l4_score * w_l
    
    # 打印详细信息
    logging.info(f"\nLevel-5 scoring details:")
    logging.info(f"NLP task statistics: Supporting {T}/{len(metrics)} tasks, Wins {wins}")
    logging.info(f"Task comparison:")
    for i, (metric, sota, model) in enumerate(zip(metrics, sota_scores, model_scores)):
        if not pd.isna(sota) and not pd.isna(model) and sota != float('inf') and model != float('inf'):
            status = "✓" if model >= sota else "✗"
            logging.info(f"Task {i+1:2d}: {metric:10s} | SoTA: {sota:6.2f} | Model: {model:6.2f} | {status}")
    logging.info(f"\nWinning task average score: {s_l:.4f}")
    logging.info(f"Weight (w_l): {w_l:.4f}")
    logging.info(f"Level-4 score: {l4_score:.4f}")
    logging.info(f"Final Level-5 score: {l5_score:.4f}")
    
    return l5_score

def main(model_name, sota_file, pred_result_file):
    # Set up logging
    log_filename = setup_logging(model_name)
    print(f"Results will be saved to log file: {log_filename}")
    
    logging.info(f'Reading files: {sota_file} and {pred_result_file}')
    
    # Get all sheet names
    sota_sheets = pd.ExcelFile(sota_file).sheet_names
    model_sheets = pd.ExcelFile(pred_result_file).sheet_names
    
    logging.info(f'SoTA file sheets: {sota_sheets}')
    logging.info(f'Model file sheets: {model_sheets}')
    
    # Skip level-scores sheet
    sota_sheets = [s for s in sota_sheets if s.lower() != 'level-scores']
    model_sheets = [s for s in model_sheets if s.lower() != 'level-scores']
    
    # Ensure both files have matching sheets
    assert set(sota_sheets) == set(model_sheets), "Sheets in both Excel files must match"
    
    # Organize data by modality
    modality_data = {}
    
    # Save NLP data for Level-5 calculation
    nlp_sota_df = None
    nlp_model_df = None
    
    for sheet in sota_sheets:
        # Save NLP data
        if sheet == 'NLP':
            nlp_sota_df = pd.read_excel(sota_file, sheet_name=sheet)
            nlp_model_df = pd.read_excel(pred_result_file, sheet_name=sheet)
            logging.info(f'NLP data loaded for Level-5 calculation')
            continue

        # Parse sheet name
        try:
            modality, task = sheet.split('-')
        except ValueError:
            raise ValueError(f'Invalid sheet name format: {sheet}')
            
        # Verify modality
        if modality not in ['Image', 'Audio', 'Video', '3D']:
            logging.info(f'Unknown modality: {modality}')
            continue
            
        logging.info(f'Processing {modality} modality {task} task: {sheet}')
        
        # Initialize modality data
        if modality not in modality_data:
            modality_data[modality] = {
                'comprehension': {'metrics': [], 'sota': [], 'scores': []},
                'generation': {'metrics': [], 'sota': [], 'scores': []}
            }
        
        # Read data
        sota_df = pd.read_excel(sota_file, sheet_name=sheet)
        model_df = pd.read_excel(pred_result_file, sheet_name=sheet)
        
        # Process data
        metrics, sota, scores = process_sheet(sota_df, model_df, model_name)
        
        # Categorize by task type
        if task == 'Comprehension':
            modality_data[modality]['comprehension']['metrics'].extend(metrics)
            modality_data[modality]['comprehension']['sota'].extend(sota)
            modality_data[modality]['comprehension']['scores'].extend(scores)
        elif task == 'Generation':
            modality_data[modality]['generation']['metrics'].extend(metrics)
            modality_data[modality]['generation']['sota'].extend(sota)
            modality_data[modality]['generation']['scores'].extend(scores)
    
    if not modality_data:
        raise ValueError("No valid modality data found")
    
    # Calculate scores for each modality
    modality_scores = {}
    for modality, data in modality_data.items():
        logging.info(f'\nCalculating scores for {modality} modality...')
        scores = get_modality_scores(
            data['comprehension']['metrics'],
            data['comprehension']['sota'],
            data['comprehension']['scores'],
            data['generation']['metrics'],
            data['generation']['sota'],
            data['generation']['scores']
        )
        modality_scores[modality] = scores
    
    # Calculate final scores (average across modalities)
    final_scores = {
        'Level-2': sum(s[0] for s in modality_scores.values()) / len(modality_scores),
        'Level-3': sum(s[1] for s in modality_scores.values()) / len(modality_scores),
        'Level-4': sum(s[2] for s in modality_scores.values()) / len(modality_scores)
    }
    
    # Calculate Level-5 score
    if nlp_sota_df is not None and nlp_model_df is not None:
        final_scores['Level-5'] = get_level_5(final_scores['Level-4'], nlp_sota_df, nlp_model_df, model_name)
    else:
        raise ValueError("NLP data not found, cannot calculate Level-5 score")
    
    # Prepare result string
    result_str = '\n' + '='*50 + '\n'
    result_str += f'Evaluation Results for Model {model_name}:\n\n'
    result_str += 'Results by Modality:\n'
    for modality, data in modality_data.items():
        # Calculate total and valid tasks
        comp_tasks = len(data['comprehension']['metrics'])
        gen_tasks = len(data['generation']['metrics'])
        total_tasks = comp_tasks + gen_tasks
        
        def count_valid_wins(sota_list, score_list):
            valid_count = sum(1 for s, m in zip(sota_list, score_list) 
                            if not pd.isna(s) and not pd.isna(m) and 
                            s != float('inf') and m != float('inf') and
                            s != 0.0 and m != 0.0)
            wins = sum(1 for s, m in zip(sota_list, score_list)
                      if not pd.isna(s) and not pd.isna(m) and
                      s != float('inf') and m != float('inf') and
                      m >= s)
            return valid_count, wins
        
        comp_valid, comp_wins = count_valid_wins(data['comprehension']['sota'], 
                                               data['comprehension']['scores'])
        gen_valid, gen_wins = count_valid_wins(data['generation']['sota'],
                                            data['generation']['scores'])
        total_valid = comp_valid + gen_valid
        total_wins = comp_wins + gen_wins
        
        result_str += f'\n{modality} Modality (Supporting {total_valid}/{total_tasks} tasks, Wins: {total_wins}):\n'
        scores = modality_scores[modality]
        result_str += f'>>> Level-2 Score: {scores[0]:.2f}\n'
        result_str += f'>>> Level-3 Score: {scores[1]:.2f}\n'
        result_str += f'>>> Level-4 Score: {scores[2]:.2f}\n'
    
    # Add NLP results if available
    if nlp_sota_df is not None and nlp_model_df is not None:
        metrics = nlp_sota_df['Metrics'].tolist()
        sota_scores = nlp_sota_df['SoTA Performance'].tolist()
        model_col = next((col for col in nlp_model_df.columns if col.lower() == model_name.lower()), None)
        if model_col:
            model_scores = nlp_model_df[model_col].tolist()
            valid_pairs = [(s, m) for s, m in zip(sota_scores, model_scores) 
                          if not pd.isna(s) and not pd.isna(m) and 
                          s != float('inf') and m != float('inf')]
            wins = sum(1 for s, m in valid_pairs if m >= s)
            result_str += f'\nNLP Modality (Supporting {len(valid_pairs)}/{len(metrics)} tasks, Wins: {wins})\n'
    
    result_str += '\n' + '='*50 + '\n'
    result_str += 'Final Scores:\n'
    result_str += f'>>> Level-2 Score: {final_scores["Level-2"]:.2f}\n'
    result_str += f'>>> Level-3 Score: {final_scores["Level-3"]:.2f}\n'
    result_str += f'>>> Level-4 Score: {final_scores["Level-4"]:.2f}\n'
    result_str += f'>>> Level-5 Score: {final_scores["Level-5"]:.2f}\n'
    result_str += '='*50 + '\n'
    result_str += 'Notes:\n'
    result_str += '1. NLP modality is not included in Level-2 to Level-4 scoring\n'
    result_str += '2. Each modality calculates both comprehension and generation scores\n'
    result_str += '3. Final scores are averaged across all participating modalities\n'
    result_str += '4. All scores are converted to percentages\n'
    result_str += '5. Level-5 score is based on Level-4 score and NLP task weights\n'
    
    # Write to log
    logging.info("\nFinal Evaluation Results:")
    logging.info(result_str)
    
    # Print to console
    print(result_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate model scores')
    parser.add_argument('-s', '--sota_file', type=str, default='references/sota_result.xlsx', help='SoTA score file')
    parser.add_argument('-p', '--pred_result_file', type=str, default='outcome/emu2_result.xlsx', help='Model prediction Excel file')
    parser.add_argument('-m', '--model_name', type=str, default='Emu2-32B', help='Model name (matching Excel column name)')
    args = parser.parse_args()
    
    main(args.model_name, args.sota_file, args.pred_result_file) 