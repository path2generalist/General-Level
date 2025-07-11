export CUDA_VISIBLE_DEVICES=0

DATASET_DIR=General-Bench-Openset
NLP_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
AUDIO_MODEL_NAME=Qwen/Qwen2-Audio-7B-Instruct
VIDEO_MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
IMAGE_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
3D_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

# 解析 step 参数
STEP="123"
for arg in "$@"; do
case $arg in
  --step=*)
    STEP="${arg#*=}"
    ;;
  --step)
    shift
    STEP="$1"
    ;;
esac
done

contains_step() {
case "$STEP" in
  *$1*) return 0 ;;
  *) return 1 ;;
esac
}

# Step1: Generate predictions for NLP, Image, Audio, Video, 3D tasks
if contains_step 1; then
  # NLP
  python predictors/nlp_predictor.py --dataset_dir ${DATASET_DIR}/nlp --model_name ${NLP_MODEL_NAME}

  # Audio
  python predictors/audio_predict_comprehension.py -m Qwen/Qwen2-Audio-7B-Instruct -d ${DATASET_DIR}/audio/comprehension/ -o ${DATASET_DIR}/audio/predictions/comprehension/ -t AccentClassification AccentSexClassification
  python predictors/audio_predict_generation.py -m SpeechGPT -d ${DATASET_DIR}/audio/generation/ -o ${DATASET_DIR}/audio/predictions/generation/ -t SingleCaptionToAudio VideoToAudio ImageToSpeech 

  # Video
  python predictors/video_comprehension_tasks.py
  python predictors/video_comprehension_flow_matching_tracking.py
  python predictors/video_comprehension_qa_caption.py
  python predictors/video_translation_restoration_superresolution_objectdetection.py
  python predictors/video_generation_evaluate_kit.py
fi

MODEL_NAME=Qwen2.5-7B-Instruct
# Step2: Obtain the score for each task
if contains_step 2; then
  python register.py -d ${DATASET_DIR} -t references/template_result.xlsx -o outcome -m ${MODEL_NAME} -p prediction.json
fi

MODEL_NAME=Qwen2.5-7B-Instruct
# Step3: Obtain the Level score 
if contains_step 3; then
  python ranker.py -p outcome/${MODEL_NAME}_result.xlsx -m ${MODEL_NAME}
fi