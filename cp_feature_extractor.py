import textgrid
import os
import pickle
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
import torchaudio
import torch
import librosa
# Read a TextGrid object from a file.
# tg = textgrid.TextGrid.fromFile('CP/en/grids/common_voice_en_292.TextGrid')
def model_output(audio_path,codebook,feature_extractor,model):
    wav,sample = torchaudio.load(audio_path)
    input_values = feature_extractor(wav.squeeze(), return_tensors="pt",sampling_rate = 16000).input_values
    with torch.no_grad():
        codevector_probs,codevectors,codevector_idx = model(input_values)
    output = codevector_idx.view(-1,2)[:,codebook-1].tolist()
    return output
def get_phn_mapping(textgrid_path):
    low = []
    high = []
    phns = []
    for phn in  textgrid.TextGrid.fromFile(textgrid_path)[2]:
        low.append(librosa.time_to_samples(phn.minTime,sr = 16000))
        high.append(librosa.time_to_samples(phn.maxTime,sr = 16000))
        phns.append(phn.mark)
    return low,high,phns
def codes_low_high(audio_path,xlsr_codes):
    y, sr = librosa.load(audio_path)
    tot_samp = librosa.time_to_samples(librosa.get_duration(y=y, sr=sr),sr=16000)
    low = 0,
    high = 0
    time = 0
    time_1 = 0.025
    low_lst =[]
    high_lst =[]
    # print(len(xlsr_codes))
    for i,k in enumerate(xlsr_codes):
        low,high = librosa.time_to_samples(time,sr = 16000),librosa.time_to_samples(time_1,sr = 16000)
        low_lst.append(int(low))
        high_lst.append(int(high))
        # print(f'{low}-{high}:{i}')
        # if i ==0:
        #     time += 0.025
        #     time_1 += 0.025
        # else:
        time += 0.020
        time_1 += 0.020
        # print(low_lst)
    if high<tot_samp:
        # low_lst.append(tot_samp-librosa.time_to_samples(0.025,sr=16000))
        low_lst.append(tot_samp)
        high_lst.append(tot_samp)
    return low_lst,high_lst
def val_ret(lst,val):
    return lst[next(x[0]-1 for x in enumerate(lst) if x[1] > val)]
def phn_ind(min,max,phn,low_lst,high_lst,mdl_out,pkl_path,phn_dict ={}):

    for val,p in enumerate(phn):
        l = val_ret(low_lst,int(min[val]))
        if val < len(phn)-1:
            h = val_ret(high_lst,int(max[val]))
        else:
            h = high_lst[-1]
        
        if p in phn_dict:
            phn_dict[p] += mdl_out[low_lst.index(l):high_lst.index(h)+1]
        else:
            phn_dict[p] = mdl_out[low_lst.index(l):high_lst.index(h)+1]
        print(f'{min[val]}-{max[val]}_{p}:{mdl_out[low_lst.index(l):high_lst.index(h)+1]}')
    if not os.path.exists('/'.join(pkl_path.split('/')[:-1])):
        os.makedirs('/'.join(pkl_path.split('/')[:-1]))
    with open(pkl_path, "wb") as f:
        pickle.dump(phn_dict, f)
def phn_dict_generator(save_path,dataset_path,codebook,feature_extractor,model):
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if (".wav" in file):
                mdl_out = model_output(audio_path = subdir+'/'+file,codebook = codebook,feature_extractor = feature_extractor,model=model)
                low_lst,high_lst = codes_low_high(subdir+'/'+file,mdl_out)
                pth_var = "/".join(list(subdir.split('/')[:-1])+['grids'])+"/"+file.split('.')[0]
                text_grid_path = pth_var+'.TextGrid'
                pkl_path = save_path+"/".join(pth_var.split("/")[1:])+'.pkl'
                low,high,phns=get_phn_mapping(text_grid_path)
                phn_ind(low,high,phns,low_lst,high_lst,mdl_out,pkl_path)
if __name__ == "__main__":
    dataset_path = "CP"
    codebook  =1
    save_path = f"CP_wav2vec2_pkl/codebook_{codebook}/"
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
    phn_dict_generator(save_path,dataset_path,codebook,feature_extractor,model)
    print("end")