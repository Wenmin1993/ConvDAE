from os.path import getsize
import wfdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from wfdb.processing import resample_sig,resample_singlechan, normalize_bound
from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,UpSampling1D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

def cr(y_true,y_pred):
    cr = len(y_true.encode())/len(y_pred.encode())
    return cr

def prd(y_true,y_pred):
    squared_differnece = k.sum(k.square(y_true -y_pred))
    squared_d = k.sum(k.square(y_true))
    prd = k.sqrt(squared_differnece/squared_d)
    return prd*100

def prdn(y_true,y_pred):
    y_true_mean = k.mean(y_true)
    squared_differnece = k.sum(k.square(y_true-y_pred))
    squared_d_m =k.sum(k.square(y_true-y_true_mean))
    prdn = k.sqrt(squared_differnece/squared_d_m)
    return prdn*100

def snr(y_true,y_pred):
    y_true_mean = k.mean(y_true)
    squared_differnece = k.sum(k.square(y_true-y_pred))
    squared_d_m =k.sum(k.square(y_true-y_true_mean))
    snr = 10*k.log(squared_d_m/squared_differnece)
    return snr

def qs(y_true,y_pred):
    qs = cr(y_true,y_pred)/prd(y_true,y_pred)

def serialization(name, content):
    f = open(r'pickle\s_' + name, 'wb')
    pickle.dump(content, f)
    f.close()
def deserialization(name):
    f = open(r'pickle\s_' + name, 'rb')
    x = pickle.load(f)
    f.close()
    return x
def get_ecg_records(database, channel):
    signals = []
    #beat_locations = []
    #beat_types = []
    useless_afrecord = ['00735', '03665', '04043', '08405', '08434']

    record_files = wfdb.get_record_list(database)
    print('record_files:', record_files)

    for record in record_files:
        if record in useless_afrecord:
            continue
        else:
            print('processing record:', record)
            s, f = wfdb.rdsamp(record, pn_dir=database)
            print(f)
            annotation = wfdb.rdann('{}/{}'.format(database, record), extension='atr')
            signal, annotation = resample_singlechan(s[:, channel], annotation, fs=f['fs'], fs_target=128)

            print(signal)
            print(signal.shape)
            #beat_loc, beat_type = get_beats(annotation)
            signals.append(signal)
            #beat_locations.append(beat_loc)
            #beat_types.append(beat_type)
            print('size of signal list: ', len(signals))
            print('--------')
    print('---------record processed!---')

    serialization(database + '_signal', signals)
    #serialization(database + '_beat_loc', beat_locations)
    #serialization(database + '_beat_types', beat_types)
    #return signals, beat_locations, beat_types
    return signals
def load_ecg_records(database):
    print('-----loading----- ')
    signals = deserialization(database + '_signal')
    print('singnals.shape:', np.asarray(signals).shape)
    print('-------ecg record from ' + database + 'loaded!------')
    return signals
def get_noise_record(noise_type, database):
    print('processing noise:', noise_type)
    s, f = wfdb.rdsamp(noise_type, channels=[0], pn_dir=database)

    print(s)
    print(f)
    signal, _ = resample_sig(s[:, 0], fs=360, fs_target=128)
    print(signal)
    signal_size = f['sig_len']
    print('-----noise processed!-----')

    # serialization the data
    serialization(database + '_' + noise_type + '_signal', signal)
    #serialization(database + '_' + noise_type + '_field', f)
    #serialization(database + '_' + noise_type + '_size', signal_size)

    return signal
    #return signal, f, signal_size
def load_noise_signal(database, noise_type):
    # deserialization the data
    signal = deserialization(database + '_' + noise_type + '_signal')
    #field = deserialization(database + '_' + noise_type + '_field')
    #signal_size = deserialization(database + '_' + noise_type + '_size')
    print(noise_type + ' singnals.shape:', np.asarray(signal).shape)
    print('-------' + noise_type + ' noise signal loaded!------')
    #return signal, field, signal_size

    return signal
def get_white_Gaussian_Noise(noise_type, signal, snr):
    snr = 10 ** (snr / 10.0)
    print(len(signal))
    power_signal = np.sum(signal ** 2) / len(signal)
    power_noise = power_signal / snr
    wgn = np.random.randn(len(signal)) * np.sqrt(power_noise)
    serialization(noise_type + '_wgn_noise', wgn)
    return wgn
def load_wgn_noise(noise_type):
    wgn = deserialization(noise_type + '_wgn_noise')
    print(noise_type + '_wgn.shape:', wgn.shape)
    return wgn
def creat_sine(sampling_frequency, time_s, sine_frequency):
    """
       Create sine wave.

       Function creates sine wave of wanted frequency and duration on a
       given sampling frequency.

       Parameters
       ----------
       sampling_frequency : float
           Sampling frequency used to sample the sine wave
       time_s : float
           Lenght of sine wave in seconds
       sine_frequency : float
           Frequency of sine wave

       Returns
       -------
       sine : array
           Sine wave

       """
    sine = np.sin(2 * sine_frequency * np.pi * np.arange(time_s * sampling_frequency) / sampling_frequency)
    '''
    n=np.arange(0,time_s,1/sampling_frequency)
    plt.plot(n,sine)
    plt.title('sine')
    plt.show()
    '''
    return sine
def get_noise(wgn, ma, bw, win_size):
    # Get the slice of data
    beg = np.random.randint(ma.shape[0] - win_size)
    end = beg + win_size
    beg2 = np.random.randint(bw.shape[0] - win_size)
    end2 = beg2 + win_size
    beg3 = np.random.randint(wgn.shape[0] - win_size)
    end3 = beg3 + win_size

    mains = creat_sine(128, int(win_size / 128), 50) * uniform(0, 0.5)

    mode = np.random.randint(6)
    # mode = 0
    ma_multip = uniform(0, 5)
    bw_multip = uniform(0, 10)
    #print('noise mode for ' + name + ':', mode)

    '''
    #test for noise
    noise_0 = ma[beg:end]*ma_multip
    noise_1 = bw[beg2:end2] * bw_multip
    noise_2 = wgn[beg3:end3]
    noise_3 = ma[beg:end] * ma_multip + wgn[beg3:end3]
    noise_4 = bw[beg2:end2] * bw_multip + wgn[beg3:end3]
    noise_5 = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip
    noise_6 = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip + wgn[beg3:end3]


    plt.subplot(711)
    plt.plot(noise_0)
    plt.title('noise_0')
    plt.subplot(712)
    plt.plot(noise_1)
    plt.title('noise_1')
    plt.subplot(713)
    plt.plot(noise_2)
    plt.title('noise_2')
    plt.subplot(714)
    plt.plot(noise_3)
    plt.title('noise_3')
    plt.subplot(715)
    plt.plot(noise_4)
    plt.title('noise_4')
    plt.subplot(716)
    plt.plot(noise_5)
    plt.title('noise_5')
    plt.subplot(717)
    plt.plot(noise_6)
    plt.title('noise_6')
    plt.show()
    '''

    # Add noise
    if mode == 0:
        noise = ma[beg:end] * ma_multip
    elif mode == 1:
        noise = bw[beg2:end2] * bw_multip
    elif mode == 2:
        noise = wgn[beg3:end3]
    elif mode == 3:
        noise = ma[beg:end] * ma_multip + wgn[beg3:end3]
    elif mode == 4:
        noise = bw[beg2:end2] * bw_multip + wgn[beg3:end3]
    # else:
    elif mode == 5:
        noise = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip
    else:
        noise = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip + wgn[beg3:end3]

    noise_final = noise + mains
    # print('shape: ma:',ma.shape,',bw:',bw.shape,',wgn:',wgn.shape,',mains:',mains.shape,',fianl:',noise_final.shape)
    return noise_final
def ecg_generator(name, signals, wgn, ma, bw, win_size, batch_size):

    print('processing')
    while True:
        x = []
        section = []
        noise_list = []
        while len(x) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))
            random_sig = signals[random_sig_idx]
            #print(random_sig_idx)

            # Select one window
            beg = np.random.randint(random_sig.shape[0] - win_size)
            end = beg + win_size
            #section = normalize_bound(section, lb=-1, ub=1)

            # Select data for window and normalize it (-1, 1)
            data_win = normalize_bound(random_sig[beg:end],lb=-1, ub=1)
            section.append(random_sig[beg:end])
            # Add noise into data window and normalize it again
            added_noise = get_noise(wgn,ma,bw,win_size)
            added_noise = normalize_bound(added_noise,lb=-1, ub=1)
            noise_list.append(added_noise)
            data_win = data_win + added_noise
            data_win = normalize_bound(data_win, lb=-1, ub=1)
            x.append(data_win)

        section = np.asarray(section)
        section = section.reshape(section.shape[0],section.shape[1],1)
        #print(section, section.shape, type(section))
        x = np.asarray(x)
        print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1],1)
        #print(x)

        id = np.random.randint(0,len(x))
        plt.subplot(311)
        plt.plot(section[id])
        plt.title('original '+name+' ecg')
        plt.subplot(312)
        plt.plot(x[id])
        plt.title('noised '+name+' ecg')
        plt.subplot(313)
        plt.title('added noise')
        plt.plot(noise_list[id])
        #plt.savefig(name+'.png')
        plt.show()

        print('shape:',x.shape)
        serialization(name+'_original_ecg',section)
        serialization(name+'_noised_ecg',x)
        return x,section
#serialization of nsr and af ecg records
'''
nsr,nsr_bls,nsr_labels = get_ecg_records('nsrdb', 0)
af,af_bls,af_labels = get_ecg_records('afdb', 0)

#serialization of ma and bw noise
ma,ma_field,ma_size = get_noise_record('ma','nstdb')
bw,bw_field,bw_size= get_noise_record('bw','nstdb')
'''
'''
nsr = get_ecg_records('nsrdb', 0)
af = get_ecg_records('afdb', 0)

#serialization of ma and bw noise
ma = get_noise_record('ma','nstdb')
bw = get_noise_record('bw','nstdb')
'''

#deserialization of ecg records
nsr = load_ecg_records('nsrdb')
af = load_ecg_records('afdb')

#deserialization of noise
ma = load_noise_signal('nstdb','ma')
bw = load_noise_signal('nstdb','bw')

'''
#serialization of noise signal
id_nsr = np.random.randint(len(nsr))
id_af = np.random.randint(len(af))
wgn_nsr = get_white_Gaussian_Noise('nsr',nsr[id_nsr],6)
wgn_af = get_white_Gaussian_Noise('af',af[id_af],6)
'''

#deserialization of noise_signal
wgn_nsr = load_wgn_noise('nsr')
wgn_af = load_wgn_noise('af')
'''
#generate noised ecg signals and serialization
noised_nsr,original_nsr = ecg_generator('nsr',nsr,wgn_nsr,ma,bw,win_size=1280,batch_size=256)
noised_af,original_af = ecg_generator('af',af,wgn_af,ma,bw,win_size=1280,batch_size=256)
'''
#deserialization noised ecg signals
noised_nsr = deserialization('nsr_noised_ecg')
noised_af = deserialization('af_noised_ecg')
original_nsr = deserialization('nsr_original_ecg')
original_af = deserialization('af_original_ecg')

#data sets
'''
X_noisy = np.concatenate((noised_nsr,noised_af), axis=0)
X_original = np.concatenate((original_nsr,original_af),axis=0)
print('noised X:',X_noisy)
print('original X:',X_original)
serialization('whole noised dataset',X_noisy)
serialization('whole original dataset',X_original)
'''
X_noisy = deserialization('whole noised dataset')
X_original = deserialization('whole original dataset')

#CNN
input_signal = Input(shape =(1280,1))
'''
x = Conv1D(64,27,strides=1,activation='relu', padding='same')(input_signal)
x = MaxPooling1D(3,strides=2,padding='same')(x)
x = Conv1D(32,25,strides=1,activation='relu',padding='same')(x)
x = MaxPooling1D(3,strides=2,padding='same')(x)
x = Conv1D(16,23,strides=1,activation='relu',padding='same')(x)
x = MaxPooling1D(3,strides=2,padding='same')(x)
x = Conv1D(4,21,strides=1,activation='relu',padding='same')(x)
encoded = MaxPooling1D(3,strides=2,padding='same')(x)

x = Conv1D(4, 21,strides=1, activation='relu', padding='same')(encoded)
x = UpSampling1D(3)(x)
x = Conv1D(16, 23,strides=1, activation='relu', padding='same')(x)
x = UpSampling1D(4)(x)
x = Conv1D(32, 25,strides=1, activation='relu', padding='same')(x)
x = UpSampling1D(4)(x)
x = Conv1D(64, 27,strides=1, activation='relu', padding='same')(x)
x = UpSampling1D(7)(x)
decoded = Conv1D(1,27,strides=1, activation='sigmoid', padding='same')(x)
'''

x = Conv1D(64, (3), activation='relu', padding='same')(input_signal)
x = MaxPooling1D((4),padding='same')(x)
x = Conv1D(32,(3),activation='relu',padding='same')(x)
x = MaxPooling1D((4),padding='same')(x)
x = Conv1D(16,(3),activation='relu',padding='same')(x)
x = MaxPooling1D((4),padding='same')(x)
x = Conv1D(4,(3),activation='relu',padding='same')(x)
encoded = MaxPooling1D((4),padding='same')(x)

x = Conv1D(8, (3), activation='relu', padding='same')(encoded)
x = UpSampling1D((4))(x)
x = Conv1D(16, (3), activation='relu', padding='same')(x)
x = UpSampling1D((4))(x)
x = Conv1D(32, (3), activation='relu', padding='same')(x)
x = UpSampling1D((4))(x)
x = Conv1D(64, (3), activation='relu', padding='same')(x)
x = UpSampling1D((4))(x)
decoded = Conv1D(1, (3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_signal,decoded)
opt = optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'
)

autoencoder.compile(optimizer='sgd',loss='mse',metrics=[metrics.RootMeanSquaredError(),prd,prdn,snr])
#autoencoder.compile(optimizer=opt,loss='mse',metrics=[metrics.RootMeanSquaredError(),percentage_rms_difference])

'''
print(X_noisy)
print(X_noisy.shape)
print(X_original)
print(X_original.shape)
print(len(X_noisy))
'''
#test data sets
RATIO = 0.3
shuffle_index = np.random.permutation(len(X_noisy))
#print('shuffle_index:',shuffle_index)
test_length = int(RATIO * len(shuffle_index)) # RATIO = 0.3
#print('test dataset length:',test_length)
test_index = shuffle_index[:test_length]
#print('test index:',test_index)
train_index = shuffle_index[test_length:]
#print('train index:',train_index)

X_test_noisy = X_noisy[test_index]
#print('x test noisy:',X_test_noisy)
X_test_original = X_original[test_index]
#print('x test origin:',X_test_original)
X_train_noisy = X_noisy[train_index]
#print('x train origin:',X_train_noisy)
X_train_original = X_original[train_index]
#print('x train origin:',X_train_original)

autoencoder.fit(X_train_noisy,X_train_original,epochs=250,batch_size=32, shuffle=True,
                validation_data=(X_test_noisy,X_test_original),
                callbacks=[TensorBoard(log_dir='noisy', histogram_freq=0, write_grads=False)])
autoencoder.summary()

'''
autoencoder.fit(X_train_noisy,X_train_original,epochs=100, batch_size=32, shuffle=True,
                validation_data = (X_test_noisy,X_test_original),
                validation_split=RATIO, verbose=2,
                callbacks=[TensorBoard(log_dir='noisy', histogram_freq=0, write_grads=False)])
'''
encoder_signals = autoencoder.predict(X_test_noisy)
decoded_signals = autoencoder.predict(encoder_signals)
serialization('signal_pred',decoded_signals)
serialization('signal_true',X_test_original)

plt.subplot(211)
plt.plot(decoded_signals[1])
plt.title('decoded ecg')
plt.subplot(212)
plt.plot(X_test_original[1])
plt.title('original ecg')
# plt.savefig(name+'.png')
plt.show()


CR = getsize(r'pickle\s_signal_true')/getsize(r'pickle\s_signal_pred')
print('CR:',CR)

k.clear_session()

#Model.save(filepath = 'model')
#history = autoencoder.fit(normal_train_data,normal_train_data,epochs=100,batchsize=32,validation_data=(test_data,test_data),shuffle=True)


