import os
from typing import List, Dict, Optional
import pandas as pd
from utils.data_types import ModalityType, TaskType, TaskResult, ModalityResults

# Import modality processors
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from processors.audio_processor import AudioProcessor
from processors.nlp_processor import NLPProcessor
from processors.three_d_processor import ThreeDProcessor
# from processors.pseudo_audio_processor import PseudoAudioProcessor

def process_all_modalities(dataset_dir: str, pred_json_file: str) -> ModalityResults:
    """Process data for all modalities
    
    Args:
        dataset_dir: Dataset directory path
        pred_json_file: Prediction JSON filename
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    print(f"Using dataset directory: {dataset_dir}")
    
    # Available processors
    processors = [
        ImageProcessor(ModalityType.IMAGE, dataset_dir, pred_json_file),
        VideoProcessor(ModalityType.VIDEO, dataset_dir, pred_json_file),
        AudioProcessor(ModalityType.AUDIO, dataset_dir, pred_json_file),
        NLPProcessor(ModalityType.NLP, dataset_dir, pred_json_file),
        ThreeDProcessor(ModalityType.THREE_D, dataset_dir, pred_json_file)
    ]
    
    # Collect results
    results: ModalityResults = {}
    for processor in processors:
        if processor.modality == ModalityType.NLP:
            # NLP doesn't distinguish between comprehension and generation
            nlp_results = processor.process()
            if nlp_results:
                results[processor.modality] = {
                    TaskType.COMPREHENSION: nlp_results
                }
        else:
            # Other modalities have both comprehension and generation
            comp_results = processor.process_comprehension()
            gen_results = processor.process_generation()
            
            if comp_results or gen_results:
                results[processor.modality] = {}
                if comp_results:
                    results[processor.modality][TaskType.COMPREHENSION] = comp_results
                if gen_results:
                    results[processor.modality][TaskType.GENERATION] = gen_results
    
    print(f"Implemented modalities: {[m.value for m in results.keys()]}")
    
    return results

def save_to_excel(results: ModalityResults, template_excel: str, output_dir: str, model_name: str):
    """Save results to Excel file, keeping all sheets and empty columns for unimplemented modalities"""
    # Read template Excel sheets
    template_dfs = pd.read_excel(template_excel, sheet_name=None)
    
    # Create new Excel writer
    output_file = os.path.join(output_dir, f"{model_name}_result.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, template_df in template_dfs.items():
            new_df = template_df.copy()
            new_df[model_name] = None
            found = False
            for modality, task_results in results.items():
                for task_type, results_list in task_results.items():
                    expected_sheet = f"{modality.value}-{task_type.value.capitalize()}" if modality != ModalityType.NLP else modality.value
                    if expected_sheet == sheet_name:
                        found = True
                        for task_result in results_list:
                            mask = (new_df['Task Name'] == task_result.task_name) & \
                                   (new_df['Metrics'] == task_result.metric)
                            if mask.any():
                                new_df.loc[mask, model_name] = task_result.score
            new_df.to_excel(writer, sheet_name=sheet_name, index=False)
            if found:
                print(f"Updated {sheet_name} sheet")
            else:
                print(f"{sheet_name} sheet empty, column preserved")
    print(f"Results saved to {output_file}")

def main():
    """Main function to process command line args and execute workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process multimodal evaluation data and generate Excel report')
    parser.add_argument('-d', '--dataset_dir', type=str, default='General-Bench-Openset',
                      help='Dataset directory path (default: General-Bench-Openset)')
    parser.add_argument('-t', '--template', type=str, default='references/template_result.xlsx',
                      help='Template Excel file path (default: references/template_result.xlsx)')
    parser.add_argument('-p', '--pred_json_file', type=str, default='prediction.json',
                      help='Prediction JSON file name(default: prediction.json)')
    parser.add_argument('-o', '--output_dir', type=str, default='outcome',
                      help='Output directory path (default: outcome)')
    parser.add_argument('-m', '--model_name', type=str, default='test', help='Model name')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing evaluation data for {args.model_name}...")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Template file: {args.template}")
    print(f"Output directory: {args.output_dir}")
    
    results = process_all_modalities(args.dataset_dir, args.pred_json_file)
    save_to_excel(results, args.template, args.output_dir, args.model_name)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 