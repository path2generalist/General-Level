from email.mime import audio
import json
import os
from pyexpat import model
from regex import B, D
import tqdm
from typing import List, Dict, Any
import nltk
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import time
from urllib.request import urlopen
import librosa
import torch
from torch import nn
import numpy as np
from encodec import EncodecModel
import laion_clap
import resampy
import soundfile as sf
from scipy import linalg
from multiprocessing.dummy import Pool as ThreadPool
import copy
import pickle
from collections import defaultdict



def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# ================================================ FAD related functions ================================================
# These functions are used to calculate the FAD score


def load_audio_task(fname, sample_rate, channels, dtype="float32"):
    if dtype not in ['float64', 'float32', 'int32', 'int16']:
        raise ValueError(f"dtype not supported: {dtype}")

    wav_data, sr = sf.read(fname, dtype=dtype)
    # For integer type PCM input, convert to [-1.0, +1.0]
    if dtype == 'int16':
        wav_data = wav_data / 32768.0
    elif dtype == 'int32':
        wav_data = wav_data / float(2**31)

    # Convert to mono
    assert channels in [1, 2], "channels must be 1 or 2"
    if len(wav_data.shape) > channels:
        wav_data = np.mean(wav_data, axis=1)

    if sr != sample_rate:
        wav_data = resampy.resample(wav_data, sr, sample_rate)

    return wav_data


class FrechetAudioDistance:
    def __init__(
        self,
        ckpt_dir=None,
        model_name="clap",
        submodel_name="630k-audioset",  # only for CLAP
        sample_rate=16000,
        channels=1,
        use_pca=False,  # only for VGGish
        use_activation=False,  # only for VGGish
        verbose=False,
        audio_load_worker=8,
        enable_fusion=False,  # only for CLAP
    ):
        """
        Initialize FAD

        -- ckpt_dir: folder where the downloaded checkpoints are stored
        -- model_name: one between vggish, pann, clap or encodec
        -- submodel_name: only for clap models - determines which checkpoint to use. 
                          options: ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        -- sample_rate: one between [8000, 16000, 32000, 48000]. depending on the model set the sample rate to use
        -- channels: number of channels in an audio track
        -- use_pca: whether to apply PCA to the vggish embeddings
        -- use_activation: whether to use the output activation in vggish
        -- enable_fusion: whether to use fusion for clap models (valid depending on the specific submodel used)
        """
        assert model_name in ["vggish", "clap", "encodec"], "model_name must be either 'vggish', 'pann', 'clap' or 'encodec'"
        if model_name == "vggish":
            assert sample_rate == 16000, "sample_rate must be 16000"
        elif model_name == "clap":
            assert sample_rate == 48000, "sample_rate must be 48000"
            assert submodel_name in ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        elif model_name == "encodec":
            assert sample_rate in [24000, 48000], "sample_rate must be 24000 or 48000"
            if sample_rate == 48000:
                assert channels == 2, "channels must be 2 for 48khz encodec model"
        self.model_name = model_name
        self.submodel_name = submodel_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.verbose = verbose
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        if self.device == torch.device('mps') and self.model_name == "clap":
            if self.verbose:
                print("[Frechet Audio Distance] CLAP does not support MPS device yet, because:")
                print("[Frechet Audio Distance] The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device.")
                print("[Frechet Audio Distance] Using CPU device instead.")
            self.device = torch.device('cpu')
        if self.verbose:
            print("[Frechet Audio Distance] Using device: {}".format(self.device))
        self.audio_load_worker = audio_load_worker
        self.enable_fusion = enable_fusion
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.hub.set_dir(ckpt_dir)
            self.ckpt_dir = ckpt_dir
        else:
            # by default `ckpt_dir` is `torch.hub.get_dir()`
            self.ckpt_dir = torch.hub.get_dir()
        self.__get_model(model_name=model_name, use_pca=use_pca, use_activation=use_activation)

    def __get_model(self, model_name="vggish", use_pca=False, use_activation=False):
        """
        Get ckpt and set model for the specified model_name

        Params:
        -- model_name: one between vggish, pann or clap
        -- use_pca: whether to apply PCA to the vggish embeddings
        -- use_activation: whether to use the output activation in vggish
        """
        # vggish
        if model_name == "vggish":
            # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
            self.model = torch.hub.load(repo_or_dir='harritaylor/torchvggish', model='vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
            self.model.device = self.device
        # clap
        elif model_name == "clap":
            # choose the right checkpoint and model
            if self.submodel_name == "630k-audioset":
                if self.enable_fusion:
                    download_name = "630k-audioset-fusion-best.pt"
                else:
                    download_name = "630k-audioset-best.pt"
            elif self.submodel_name == "630k":
                if self.enable_fusion:
                    download_name = "630k-fusion-best.pt"
                else:
                    download_name = "630k-best.pt"
            elif self.submodel_name == "music_audioset":
                download_name = "music_audioset_epoch_15_esc_90.14.pt"
            elif self.submodel_name == "music_speech":
                download_name = "music_speech_epoch_15_esc_89.25.pt"
            elif self.submodel_name == "music_speech_audioset":
                download_name = "music_speech_audioset_epoch_15_esc_89.98.pt"

            model_path = os.path.join(self.ckpt_dir, download_name)

            # download checkpoint
            if not (os.path.exists(model_path)):
                if self.verbose:
                    print("[Frechet Audio Distance] Downloading {}...".format(model_path))
                torch.hub.download_url_to_file(
                    url=f"https://huggingface.co/lukewys/laion_clap/resolve/main/{download_name}",
                    dst=model_path
                )
            # init model and load checkpoint
            if self.submodel_name in ["630k-audioset", "630k"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    device=self.device)
            elif self.submodel_name in ["music_audioset", "music_speech", "music_speech_audioset"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    amodel='HTSAT-base',
                                                    device=self.device)
            self.model.load_ckpt(model_path)

            # init model and load checkpoint
            if self.submodel_name in ["630k-audioset", "630k"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    device=self.device)
            elif self.submodel_name in ["music_audioset", "music_speech", "music_speech_audioset"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    amodel='HTSAT-base',
                                                    device=self.device)
            self.model.load_ckpt(model_path)

        # encodec
        elif model_name == "encodec":
            # choose the right model based on sample_rate
            # weights are loaded from the encodec repo: https://github.com/facebookresearch/encodec/
            if self.sample_rate == 24000:
                self.model = EncodecModel.encodec_model_24khz()
            elif self.sample_rate == 48000:
                self.model = EncodecModel.encodec_model_48khz()
            # 24kbps is the max bandwidth supported by both versions
            # these models use 32 residual quantizers
            self.model.set_target_bandwidth(24.0)

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, x, sr):
        """
        Get embeddings using VGGish, PANN, CLAP or EnCodec models.
        Params:
        -- x    : a list of np.ndarray audio samples
        -- sr   : sampling rate.
        """
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):
                if self.model_name == "vggish":
                    embd = self.model.forward(audio, sr)
                elif self.model_name == "clap":
                    audio = torch.tensor(audio).float().unsqueeze(0)
                    embd = self.model.get_audio_embedding_from_data(audio, use_tensor=True)
                elif self.model_name == "encodec":
                    # add two dimensions
                    audio = torch.tensor(
                        audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    # if SAMPLE_RATE is 48000, we need to make audio stereo
                    if self.model.sample_rate == 48000:
                        if audio.shape[-1] != 2:
                            if self.verbose:
                                print(
                                    "[Frechet Audio Distance] Audio is mono, converting to stereo for 48khz model..."
                                )
                            audio = torch.cat((audio, audio), dim=1)
                        else:
                            # transpose to (batch, channels, samples)
                            audio = audio[:, 0].transpose(1, 2)

                    if self.verbose:
                        print(
                            "[Frechet Audio Distance] Audio shape: {}".format(
                                audio.shape
                            )
                        )

                    with torch.no_grad():
                        # encodec embedding (before quantization)
                        embd = self.model.encoder(audio)
                        embd = embd.squeeze(0)

                if self.verbose:
                    print(
                        "[Frechet Audio Distance] Embedding shape: {}".format(
                            embd.shape
                        )
                    )
                
                if embd.device != torch.device("cpu"):
                    embd = embd.cpu()
                
                if torch.is_tensor(embd):
                    embd = embd.detach().numpy()
                
                embd_lst.append(embd)
        except Exception as e:
            print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def __load_audio_files(self, dir, dtype="float32"):
        task_results = []

        pool = ThreadPool(self.audio_load_worker)
        pbar = tqdm(total=len(os.listdir(dir)), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print("[Frechet Audio Distance] Loading audio from {}...".format(dir))
        for fname in os.listdir(dir):
            res = pool.apply_async(
                load_audio_task,
                args=(os.path.join(dir, fname), self.sample_rate, self.channels, dtype),
                callback=update,
            )
            task_results.append(res)
        pool.close()
        pool.join()

        return [k.get() for k in task_results]

    def score(self,
              background_dir,
              eval_dir,
              background_embds_path=None,
              eval_embds_path=None,
              dtype="float32"
              ):
        """
        Computes the Frechet Audio Distance (FAD) between two directories of audio files.

        Parameters:
        - background_dir (str): Path to the directory containing background audio files.
        - eval_dir (str): Path to the directory containing evaluation audio files.
        - background_embds_path (str, optional): Path to save/load background audio embeddings (e.g., /folder/bkg_embs.npy). If None, embeddings won't be saved.
        - eval_embds_path (str, optional): Path to save/load evaluation audio embeddings (e.g., /folder/test_embs.npy). If None, embeddings won't be saved.
        - dtype (str, optional): Data type for loading audio. Default is "float32".

        Returns:
        - float: The Frechet Audio Distance (FAD) score between the two directories of audio files.
        """
        try:
            # Load or compute background embeddings
            if background_embds_path is not None and os.path.exists(background_embds_path):
                if self.verbose:
                    print(f"[Frechet Audio Distance] Loading embeddings from {background_embds_path}...")
                embds_background = np.load(background_embds_path)
            else:
                audio_background = self.__load_audio_files(background_dir, dtype=dtype)
                embds_background = self.get_embeddings(audio_background, sr=self.sample_rate)
                if background_embds_path:
                    os.makedirs(os.path.dirname(background_embds_path), exist_ok=True)
                    np.save(background_embds_path, embds_background)

            # Load or compute eval embeddings
            if eval_embds_path is not None and os.path.exists(eval_embds_path):
                if self.verbose:
                    print(f"[Frechet Audio Distance] Loading embeddings from {eval_embds_path}...")
                embds_eval = np.load(eval_embds_path)
            else:
                audio_eval = self.__load_audio_files(eval_dir, dtype=dtype)
                embds_eval = self.get_embeddings(audio_eval, sr=self.sample_rate)
                if eval_embds_path:
                    os.makedirs(os.path.dirname(eval_embds_path), exist_ok=True)
                    np.save(eval_embds_path, embds_eval)

            # Check if embeddings are empty
            if len(embds_background) == 0:
                print("[Frechet Audio Distance] background set dir is empty, exiting...")
                return -1
            if len(embds_eval) == 0:
                print("[Frechet Audio Distance] eval set dir is empty, exiting...")
                return -1

            # Compute statistics and FAD score
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background,
                sigma_background,
                mu_eval,
                sigma_eval
            )

            return fad_score
        except Exception as e:
            print(f"[Frechet Audio Distance] An error occurred: {e}")
            return -1


def calculate_fad_score(background_dir, eval_dir, background_embds_path=None, eval_embds_path=None, dtype="float32", ckpt_dir=None, model_name="clap", submodel_name="630k-audioset", sample_rate=16000, channels=1, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8, enable_fusion=False):
    """
    Calculate the Frechet Audio Distance (FAD) score between two directories of audio files.

    Parameters:
    - background_dir: Directory containing background audio files.
    - eval_dir: Directory containing evaluation audio files.
    - background_embds_path: Path to save/load background audio embeddings.
    - eval_embds_path: Path to save/load evaluation audio embeddings.
    - dtype: Data type for loading audio files (default is "float32").
    - ckpt_dir: Directory where the model checkpoints are stored.
    - model_name: Name of the model to use (default is "clap").
    - submodel_name: Submodel name for CLAP (default is "630k-audioset").
    - sample_rate: Sample rate for audio files (default is 16000).
    - channels: Number of channels in the audio files (default is 1).
    - use_pca: Whether to apply PCA to VGGish embeddings (default is False).
    - use_activation: Whether to use output activation in VGGish (default is False).
    - verbose: Whether to print verbose output (default is False).
    - audio_load_worker: Number of workers for loading audio files (default is 8).
    - enable_fusion: Whether to enable fusion for CLAP models (default is False).

    Returns:
    - FAD score as a float.
    """
    
    fad = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name=model_name,
        submodel_name=submodel_name,
        sample_rate=sample_rate,
        channels=channels,
        use_pca=use_pca,
        use_activation=use_activation,
        verbose=verbose,
        audio_load_worker=audio_load_worker,
        enable_fusion=enable_fusion
    )
    
    return {
        "FAD_score": fad.score(background_dir, eval_dir, background_embds_path, eval_embds_path, dtype)
    }





# ================================================ CLAP related functions ================================================
# These functions are used to calculate the CLAP score


# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')


def calculate_cosine_similarity(embeddings1, embeddings2):
    dot_product = np.dot(embeddings1, embeddings2)
    norm1 = np.linalg.norm(embeddings1)
    norm2 = np.linalg.norm(embeddings2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def calculate_clap_score(clap_checkpoint=None, model_id=-1, verbose=True, audio_file_list=None, text_file_list=None):
    """Load the pretrained checkpoint of CLAP model

    Parameters
    ----------
    ckpt: str
        if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. \n 
        For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
    model_id:
        if model_id is specified, you can download our best ckpt, as:
            id = 0 --> 630k non-fusion ckpt \n
            id = 1 --> 630k+audioset non-fusion ckpt \n
            id = 2 --> 630k fusion ckpt \n
            id = 3 --> 630k+audioset fusion ckpt \n
        Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
    """
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt(ckpt = clap_checkpoint, model_id = model_id, verbose=verbose) # download the default pretrained checkpoint.
    audio_embeddings = []
    for file in audio_file_list:
        audio, sr = librosa.load(file, sr=16000)
        audio = int16_to_float32(audio)
        embeddings = laion_clap.get_audio_embedding(audio)
        audio_embeddings.append(embeddings)

    text_embeddings = []
    for file in text_file_list:
        if os.path.exists(file):
            with open(file, 'r') as f:
                text = f.read()
        else:
            text = file
        embeddings = laion_clap.get_text_embedding(text)
        text_embeddings.append(embeddings)

    # Compute similarity scores
    scores = []
    for audio_emb, text_emb in zip(audio_embeddings, text_embeddings):
        score = calculate_cosine_similarity(audio_emb, text_emb)
        scores.append(score)
    
    # compute the average score
    if len(scores) > 0:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0.0
    
    return {"CLAP_score": average_score, "scores": scores}


# ================================================ CIDEr (Consensus-based Image Description Evaluation) related functions ================================================
# These functions are used to calculate the CIDEr score


import whisper  # a tool from OpenAI for speech recognition


def speech_to_text(model_name="turbo", audio_file="audio.mp3"):
    """
    Convert speech to text using a speech recognition model.
    """
    model = whisper.load_model(model_name)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return result.text


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)


# https://github.com/ramavedantam/cider/blob/master/pyciderevalcap/cider/cider_scorer.py
class CiderScorer(object):
    """CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self
    
    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self, df_mode="corpus"):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.iteritems():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for
                # computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].iteritems():
                    val[n] += vec_hyp[n][ngram] * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
            return val

        # compute log reference length
        if df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))
        elif df_mode == "coco-val-df":
            # if coco option selected, use length of coco-val set
            self.ref_len = np.log(float(40504))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, df_mode, option=None, verbose=0):
        # compute idf
        if df_mode == "corpus":
            self.compute_doc_freq()
            # assert to check document frequency
            assert(len(self.ctest) >= max(self.document_frequency.values()))
            # import json for now and write the corresponding files
        else:
            self.document_frequency = pickle.load(open(os.path.join('data', df_mode + '.p'),'r'))
        # compute cider score
        score = self.compute_cider(df_mode)
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)


# https://github.com/ramavedantam/cider/blob/master/pyciderevalcap/cider/cider.py
class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=4, df="corpus"):
        """
        Initialize the CIDEr scoring function
        : param n (int): n-gram size
        : param df (string): specifies where to get the IDF values from
                    takes values 'corpus', 'coco-train'
        : return: None
        """
        # set cider to sum over 1 to 4-grams
        self._n = n
        self._df = df

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        : param  gts (dict) : {image:tokenized reference sentence}
        : param res (dict)  : {image:tokenized candidate sentence}
        : return: cider (float) : computed CIDEr score for the corpus
        """

        cider_scorer = CiderScorer(n=self._n)

        for res_id in res:

            hypo = res_id['caption']
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score(self._df)

        return score, scores

    def method(self):
        return "CIDEr"


def calculate_CIDEr_score(audio_file_list=None, text_file_list=None):
    # convert audio files to text using speech-to-text
    if audio_file_list is None or text_file_list is None:
        raise ValueError("Both audio_file_list and text_file_list must be provided.")
    if len(audio_file_list) != len(text_file_list):
        raise ValueError("audio_file_list and text_file_list must have the same length.")
    # Load the CIDEr scorer
    cider_scorer = Cider(n=4, df="corpus")
    # Prepare the ground truth and results
    gts = {}
    res = []
    from spacy.tokenizer import Tokenizer
    from spacy.lang.en import English
    nlp = English()
    # Create a blank Tokenizer with just the English vocab
    tokenizer = Tokenizer(nlp.vocab)

    for audio_file, text_file in zip(audio_file_list, text_file_list):
        # Convert audio to text
        text = speech_to_text(audio_file=audio_file)
        
        gts[audio_file] = [tokenizer(text).words]  # Tokenize the text

        with open(text_file, 'r') as f:
            reference_text = f.read().strip()
        # Tokenize the reference text
        text = tokenizer(reference_text).words
        res.append({
            'image_id': audio_file,
            'caption': [text]
        })
    # Compute the CIDEr score
    score, scores = cider_scorer.compute_score(gts, res)
    return {
        "CIDEr_score": score,
        "scores": scores
    }







# ================================================ WER (Word Error Rate) related functions ================================================
# These functions are used to calculate the WER

# pip install werpy

import werpy
def calculate_wer(audio_file_list: list, text_file_list: list) -> float:
    """Calculate the Word Error Rate (WER) between a reference and a hypothesis.
    Args:
        audio_file_list (list): List of audio files to be transcribed.
        text_file_list (list): List of text files containing the reference transcriptions.
    """
    if len(audio_file_list) != len(text_file_list):
        raise ValueError("audio_file_list and text_file_list must have the same length.")
    
    total_wer = 0.0
    for audio_file, text_file in zip(audio_file_list, text_file_list):
        # Convert audio to text using speech-to-text
        transcribed_text = speech_to_text(audio_file=audio_file)
        
        # Read the reference text from the file
        with open(text_file, 'r') as f:
            reference_text = f.read().strip()

        # Calculate WER
        wer_score = werpy.wer(reference_text, transcribed_text)
        total_wer += wer_score
    
    average_wer = total_wer / len(audio_file_list)
    return {"WER_score": average_wer}




# ================================================ MCD (Mel Cepstral Distortion ) related functions ================================================
# These functions are used to calculate the MCD

# pip install -U pymcd
from pymcd.mcd import Calculate_MCD

def calculate_mcd(reference_audio_list: str, generated_audio_list: str) -> float:
    """Calculate the Mel Cepstral Distortion (MCD) between two audio files.
    
    Args:
        reference_audio (str): Path to the reference audio file.
        generated_audio (str): Path to the generated audio file.
    
    Returns:
        float: The MCD score.
    """
    # instance of MCD class
    # three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics 
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")

    # two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively
    mcd_scores = []
    for ref_audio, gen_audio in zip(reference_audio_list, generated_audio_list):
        # calculate MCD score
        mcd_score = mcd_toolbox.calculate_mcd(ref_audio, gen_audio)
        mcd_scores.append(mcd_score)
    # calculate average MCD score
    mcd_score = sum(mcd_scores) / len(mcd_scores)
    if mcd_score is None:
        raise ValueError("MCD score could not be calculated. Please check the audio files.")
    
    return {"MCD_score": mcd_score, "mcd_scores": mcd_scores}



class AudioGenerationModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        # Placeholder for loading the model
        pass

    def generate(self, input_text: str) -> np.ndarray:
        # Placeholder for audio generation logic
        # This should return the generated audio as a numpy array or a file path
        pass



@dataclass
class Instance:
    input: Dict[str, Any]
    output: Dict[str, Any]
    id: str


class BaseTask(ABC):
    def __init__(self, task_data: Dict[str, Any], model: AudioGenerationModel, audio_dir: str = None, output_dir: str = None, task_name: str = None):
        self.task_data = read_json(task_data)
        self.model = model
        self.audio_dir = audio_dir  # should include the audios files
        self.data = self._parse_data(self.task_data)
        self.task_name = os.path.dirname(task_data).split("/")[-1] if task_name is None else task_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True) if self.output_dir else None

        self.references = []
        self.predictions = []

    def save_predictions(self, audio_paths):
        results = []
        for gt, response, audio_path in zip(self.references, self.predictions, audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(self.output_dir, f'{self.task_name }_{time_prefix}.json') if self.output_dir else f'{self.task_name }_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))

    @abstractmethod
    def _get_choice_candidate(self):
        pass

    @abstractmethod
    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def run_inference(self):
        pass


class SingleCaptionToAudio(BaseTask):
    def __init__(self, task_data: Dict[str, Any], model: AudioGenerationModel, audio_dir: str = None, output_dir: str = None, task_name: str = None):
        super().__init__(task_data, model, audio_dir, output_dir, task_name)
        self._get_choice_candidate()

    def _get_choice_candidate(self):
        # Placeholder for getting choice candidates
        pass

    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]
    
    def save_predictions(self, audio_paths):
        results = []
        for gt, response, audio_path in zip(self.references, self.predictions, audio_paths):
            results.append({
                'gt': gt,
                'response': response,
                'audio_path': audio_path,
            })
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = os.path.join(self.output_dir, f'{self.task_name }_{time_prefix}.json') if self.output_dir else f'{self.task_name }_{time_prefix}.json'
        json.dump(results, open(results_file, 'w'))


    def evaluate(self) -> Dict[str, float]:
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            audio_path = os.path.join(self.audio_dir, inst.input["audio_file"])
            prompt = inst.input["prompt"]
            try:
                response = self.model.generate(prompt, audio_path=audio_path)
            except:
                print("error audio {}".format(inst.input["audio_file"]))
                continue
            # response is the generated audio file path
            self.predictions.append(response)
            self.references.append(prompt)
        # self.save_predictions(audio_paths)

    def run_inference(self):
        clap_score = calculate_clap_score(self.predictions, self.references)
        return clap_score
    

class VideoToAudio(BaseTask):
    def __init__(self, task_data: Dict[str, Any], model: AudioGenerationModel, audio_dir: str = None, output_dir: str = None, task_name: str = None):
        super().__init__(task_data, model, audio_dir, output_dir, task_name)
        self._get_choice_candidate()

    def _get_choice_candidate(self):
        # Placeholder for getting choice candidates
        pass

    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def evaluate(self) -> Dict[str, float]:
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            video_path = os.path.join(self.audio_dir, inst.input["video_file"])
            prompt = inst.input["prompt"]
            try:
                response = self.model.generate(prompt, video_path=video_path)
            except:
                print("error video {}".format(inst.input["video_file"]))
                continue
            # response is the generated audio file path
            self.predictions.append(response)
            self.references.append(prompt)

    def run_inference(self):
        fad_score = calculate_fad_score(
            background_dir=self.audio_dir,
            eval_dir=self.output_dir
        )
        return fad_score


class ImageToSpeech(BaseTask):
    def __init__(self, task_data: Dict[str, Any], model: AudioGenerationModel, audio_dir: str = None, output_dir: str = None, task_name: str = None):
        super().__init__(task_data, model, audio_dir, output_dir, task_name)
        self._get_choice_candidate()

    def _get_choice_candidate(self):
        # Placeholder for getting choice candidates
        pass

    def _parse_data(self, task_data: Dict[str, Any]) -> List[Instance]:
        return [Instance(input=d["input"], output=d["output"], id=d["id"]) 
                for d in task_data["data"]]

    def evaluate(self) -> Dict[str, float]:
        # Placeholder for evaluation logic
        self.predictions = []
        self.references = []
        for inst in tqdm.tqdm(self.data):
            image_path = os.path.join(self.audio_dir, inst.input["image_file"])
            prompt = inst.input["prompt"]
            try:
                response = self.model.generate(prompt, image_path=image_path)
            except:
                print("error image {}".format(inst.input["image_file"]))
                continue
            # response is the generated audio file path
            self.predictions.append(response)
            self.references.append(prompt)

    def run_inference(self):
        CIDEr_score = calculate_CIDEr_score(
            audio_file_list=self.predictions,
            text_file_list=self.references
        )
        return CIDEr_score


def log_performance_csv(model_name, task_name, metric, score, root_path, output_file='prediction.json'):
    import csv
    file_exists = os.path.isfile(os.path.join(root_path, output_file))

    row_data = {
        'model': model_name,
        'task': task_name,
        'metric': metric,
        'score': str(score),
    }

    with open(os.path.join(root_path, output_file), mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def log_performance_json(model_name, task_name, metric, score, root_path, output_file='prediction.json'):
    import json
    log_data = {
        'model': model_name,
        'task': task_name,
        'metric': metric,
        'score': str(score),
    }
    
    log_file_path = os.path.join(root_path, output_file)
    
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(log_data)

    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)
    



if __name__ == "__main__":
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run audio generation tasks")
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the audio generation model to use')
    parser.add_argument('-d', '--data_dir', type=str, default='./audio/generation/', help='Directory containing task data')
    parser.add_argument('-o', '--output_dir', type=str, default='./audio/predictions/generation/', help='Directory to save predictions for each task')
    parser.add_argument('-r', '--root_path', type=str, default='./', help='Root path for logging performance')
    parser.add_argument('-t', '--task_names', type=str, nargs='+',
                        help='List of task names to run (for example: SingleCaptionToAudio VideoToAudio ImageToSpeech)')
    args = parser.parse_args()

    # Initialize the model
    model = AudioGenerationModel(model_name=args.model_name)
    # data_dir = './generation/'
    # output_dir = f'./predictions/generation/{args.model_name}'
    # root_path = './'

    task_name_list = [
        'SingleCaptionToAudio', 'VideoToAudio', 'ImageToSpeech',
        # Add more task names as needed
    ]
    
    if args.task_names is None or len(args.task_names) == 0:
        args.task_names = task_name_list
    
    for task_name in args.task_names: # os.listdir(data_dir):

        # Dynamically get the class by its name
        if task_name in globals():  # Ensure the class is defined in the current scope
            task_class = globals()[task_name]
        else:
            # Optionally, handle cases where the class is not found
            print(f"Task {task_name} is not defined in the current scope.")
            continue

        # Initialize the task class
        import glob
        json_file_list = glob.glob(os.path.join(args.data_dir, task_name, "*.json"))
        if len(json_file_list) == 0:
            print(f"No JSON files found for task: {task_name}")
            continue
        elif len(json_file_list) > 1:
            print(f"Multiple JSON files found for task: {task_name}, using the first one: {json_file_list[0]}")
            task_annotation_data = json_file_list[0]
        else:
            task_annotation_data = json_file_list[0]
        print(f"Using task annotation data: {task_annotation_data}")
        task = task_class(
            task_data=task_annotation_data,
            model=model,
            audio_dir=os.path.join(args.data_dir, task_name, 'audios'),
            output_dir=args.output_dir
        )
        
        # Run inference for the task
        # This should generate audio files based on the task's data
        print(f"Running inference for task: {task_name}")
        task.run_inference()
        # if you want to save the predictions, you need to rewrite the save_predictions() in each Task class depending on your need, and call task.save_predictions() after task.run_inference() or inside the run_inference method.


        # Evaluate the task, return a dictionary of metrics
        # For example, {'FAD_score': 0.123}
        eval_results = task.evaluate()   
        print("Task name: ", task_name, "Evaluation results:", eval_results)
        log_performance_json(
            model_name=args.model_name, 
            task_name=task_name, 
            metric=list(eval_results.keys())[0].split('_')[0],   # FAD_score
            score=eval_results[list(eval_results.keys())[0]],  # e.g., 0.123
            root_path=args.data_dir)

    # or you can run the tasks one by one like below:
    # task_name = 'SingleCaptionToAudio'
    # task = SingleCaptionToAudio(
    #     task_data=os.path.join(data_dir, f"{task_name}/annotation.json"),
    #     model=model,
    #     audio_dir=os.path.join(data_dir, f"{task_name}/audios"),
    #     output_dir=output_dir)
    # task.run_inference()
    # print(task.evaluate())


