import os
import json
import librosa
import soundfile
import itertools
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pprint import pprint as pp
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts

# Print a loading bar with name where ending char can be changed
def PBR(count, total, name = "", end = "\n"):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    perc = round(100.0 * count / float(total), 2)
    if filled_len > 0:
        char = ">"
        if filled_len == bar_len:
            char = "■"
        bar = "■" * (filled_len - 1) + char + '.' * (bar_len - filled_len)
    else:
        bar = '.' * (bar_len - filled_len)

    if len(name) != 0:
        name += " "

    print("%s[%s] %s/%s --> %s%s" % (name, bar, count, total ,perc, "%"), end = end)

# Print a loading bar with name
def PBP(count, total, name = ""):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    perc = round(100.0 * count / float(total), 1)
    if filled_len > 0:
        char = ">"
        if filled_len == bar_len:
            char = "■"
        bar = "■" * (filled_len - 1) + char + '.' * (bar_len - filled_len)
    else:
        bar = '.' * (bar_len - filled_len)

    if len(name) != 0:
        name += " "

    print("%s[%s] %s/%s --> %s%s" % (name, bar, count, total ,perc, "%"))

# Returns a loading bar
def PBS(count, total, name = ""):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    perc = round(100.0 * count / float(total), 1)
    if filled_len > 0:
        char = ">"
        if filled_len == bar_len:
            char = "■"
        bar = "■" * (filled_len - 1) + char + '.' * (bar_len - filled_len)
    else:
        bar = '.' * (bar_len - filled_len)

    if len(name) != 0:
        name += " "

    return ("%s[%s] %s/%s --> %s%s" % (name, bar, count, total ,perc, "%"))

# Directory_DB is a class that creates a json DB of names of all files
# and sub directories and saves it for quick access and expansion
class Directory_DB:
    # Initiation function that makes sure that Location exists
    def __init__(self, loc_name = ""):
        self.db = {}
        if not self.__Check_Location(loc_name):
            print("Use Create_DB function to make DB")
            self.start = ""
            return

        else:
            self.start = loc_name
            self.db = {"name":loc_name,
                       "location":loc_name,
                       "type": "dir",
                       "content": self.Create_DB(loc_name)}
            #print(self.db)
            print("DB Created\n")

    # Function that is in charge of creating the main database if location
    # exists
    def Create_DB(self, loc_name):
        if not self.__Check_Location(loc_name):
            self.start = ""
            return False
        else:
            self.start = loc_name
            return self.__Create_DB_Helper(loc_name)

    # Recursive function that retrieves all files from location and makes
    # a database
    def __Create_DB_Helper(self, location):
        fs = []
        if location[-1] == "/":
            location = location[-1]

        files = sorted(os.listdir(location))
        for file in files:
            loc = location+"/"+file
            if os.path.isdir(loc):
                fs.append({"name": file,
                           "location": loc,
                           "type" : "dir",
                           "content": self.__Create_DB_Helper(loc)})
            else:
                fs.append({"name": file,
                           "location": loc,
                           "type" : "file",
                           "content": []})

        return fs

    # Function that displays a rough filesystem from database using its
    # helper function
    def Print_DB(self):
        if self.db == {}:
            print("No Filesystem to display!")
            return
        print()
        self.__Print_DB_Helper(self.db, "", True, True)

    def __Print_DB_Helper(self, db, ind, e, f):
        new_char = ""
        if e == False:
            print(ind + "├── " + db["name"])
            new_char = "|   "
        else:
            if f == True:
                print(ind + "─── " + db["name"])
            else:
                print(ind + "└── " + db["name"])

            if db["type"] == "file":
                print(ind)
            new_char = "    "

        if db["type"] == "dir":
            last_pos = len(db["content"]) - 1
            for i, item in enumerate(db["content"]):
                if i < last_pos:
                    self.__Print_DB_Helper(item, ind + new_char, False, False)
                else:
                    self.__Print_DB_Helper(item, ind + new_char, True, False)

    # Private Helper Function in charge of checking of the existance of a
    # location
    def __Check_Location(self, loc_name):
        if not os.path.exists(loc_name) or loc_name == "":
            print("Location hasn't been entered or doesnt exist...")
            return False
        else:
            return True

# Function in charge of creating Unique ID from string
def ID(s):
    from hashlib import md5
    return md5(s.encode("utf-8")).hexdigest()

# Function in charge of turning the initial File Database
# to a usable and appropriate version for Machine Learning
def File_DB_to_Data_DB(f_db):
    db = DB_Helper(f_db)
    return db

# Recursive Helper function to make new db
def DB_Helper(db):
    out = []
    for item in db["content"]:
        if item["type"] == "file":
            actor = item["location"].split("/")[-2]
            out.append({"_id": ID(item["location"]), "name": item["name"], "location": item["location"], "actor":actor})
        else:
            out.extend(DB_Helper(item))
    return out

# Class in charge of taking a database and turning it into usable data
# for Machine Learning
class ML_Data_Prep:
    def __init__(self, db = []):
        print("Initialising DB for ML")
        self.ID_INDX = {}
        self.INDX_ID = {}
        self.data = db
        self.ML_data = []
        self.VPU = Voice_Processing_Unit()
        self.emotions = { '01':'neutral', '02':'calm', '03':'happy',
                            '04':'sad', '05':'angry', '06':'fearful',
                            '07':'disgust', '08':'surprised'}
        self.observed_emotion = ['calm', 'happy', 'fearful', 'disgust']

        for i, item in enumerate(self.data):
            self.INDX_ID[i] = item["_id"]
            self.ID_INDX[item["_id"]] = i
        print("Init Complete\n")

    # Processes all voices and saves their data for later creation of datasets
    def Process_Voices(self):
        if os.path.exists("db.json"):
            print("DB Exists, Loading DB...", end = "")
            openF = open("db.json")
            self.data = json.load(openF)
            openF.close()
            print("Done")
        else:
            print("DB Doesn't Exist, Creating db...")
            print("Processing Voices...")
            tot = len(self.INDX_ID)
            for count, i in enumerate(self.INDX_ID.keys()):
                PBR(count, tot, name = "\nVoices", end = "\n")
                print("{:8}ID: {}\n\tNAME: {}\n".format(str(i)+":", self.INDX_ID[i], self.data[i]["name"]))
                self.VPU.Process_File(self.INDX_ID[i], self.data[i]["location"])
                name = self.data[i]["name"].split("-")
                self.data[i]["str_lbl"]         = name[2]
                self.data[i]["lbl"]             = self.emotions[name[2]]
                self.data[i]["int_lbl"]         = int(name[2])
                self.data[i]["arr_lbl"]         = [0]*8
                self.data[i]["arr_lbl"][self.data[i]["int_lbl"] - 1] = 1
                #print(self.data[i])
                #input()
                self.data[i]["mfcc"]            = np.array(self.VPU.mfcc).tolist()
                self.data[i]["chroma_stft"]     = np.array(self.VPU.chroma_stft).tolist()
                self.data[i]["mel_spect"]       = np.array(self.VPU.mel_spect).tolist()
                self.data[i]["rms"]             = np.array(self.VPU.rms).tolist()
                self.data[i]["chroma_cqt"]      = np.array(self.VPU.chroma_cqt).tolist()
                self.data[i]["chroma_cens"]     = np.array(self.VPU.chroma_cens).tolist()
                self.data[i]["spectrl_contr"]   = np.array(self.VPU.spectrl_contr).tolist()
                self.data[i]["tonnetz"]         = np.array(self.VPU.tonnetz).tolist()
                #print(self.data[i])
            print("Processing Complete!")
            openF = open("db.json", "w")
            json.dump(self.data, openF, indent = 4)
            openF.close()
        return

    # Make Data Set with split for training and testing
    def Make_Set(self, choice, split):
        X     = []
        ysing = []
        yarr  = []
        tot = len(self.INDX_ID)
        #print()
        for count, i in enumerate(self.INDX_ID.keys()):
            #PBR(count, tot, name = "Voices", end = "\r")
            dat = []
            if choice[0] == 1:
                dat.extend(self.data[i]["mfcc"])
            elif choice[1] == 1:
                dat.extend(self.data[i]["chroma_stft"])
            elif choice[2] == 1:
                dat.extend(self.data[i]["mel_spect"])
            elif choice[3] == 1:
                dat.extend(self.data[i]["rms"])
            elif choice[4] == 1:
                dat.extend(self.data[i]["chroma_cqt"])
            elif choice[5] == 1:
                dat.extend(self.data[i]["chroma_cens"])
            elif choice[6] == 1:
                dat.extend(self.data[i]["spectrl_contr"])
            elif choice[7] == 1:
                dat.extend(self.data[i]["tonnetz"])
            X.append(dat)
            ysing.append(self.data[i]["int_lbl"] - 1)
            yarr.append(self.data[i]["arr_lbl"])
        #print()
        #print(ysing)
        #X_train, ysing_train, yarr_train, X_test, ysing_test, yarr_test = tts(X, ysing, yarr, test_size=split, random_state=0)
        #X_train, ysing_train, X_test, ysing_test = tts(X, ysing, test_size=0.25, random_state=0)
        #print(ysing_train)
        X, ysing, yarr = shuffle(X, ysing, yarr, random_state = 0)
        split = int(len(X)*split)
        #X = MinMaxScaler().fit_transform(X)
        X_train, X_test = X[:split], X[split:]
        ys_train, ys_test = ysing[:split], ysing[split:]
        ya_train, ya_test = yarr[:split], yarr[split:]

        return X_train, ys_train, ya_train, X_test, ys_test, ya_test

# Class that is the center of the algorithm, within this class, voice
# files are analysed and turned into a series of fetures for Machine
# Learning
class Voice_Processing_Unit:
    def __init__(self, mean = False, max = False, std = False):
        self.last_ID = ""
        self.last_location = ""
        self.last_dat      = []
        self.mfcc          = []
        self.chroma_stft   = []
        self.mel_spect     = []
        self.rms           = []
        self.chroma_cqt    = []
        self.chroma_cens   = []
        self.spectrl_contr = []
        self.tonnetz       = []
        self.mean          = mean
        self.max           = max
        self.std           = std

    def Process_File(self, id, loc):
        self.last_ID = id
        self.last_location = loc
        self.last_dat = []
        print("Loading file...")

        f = soundfile.SoundFile(loc)

        dat = f.read(dtype="float32")
        sr = f.samplerate

        # Data Extraction
        stft = np.abs(librosa.stft(dat))
        S, phase = librosa.magphase(librosa.stft(dat))

        # Each of these will have a mean, max and std found for each stretch
        self.mfcc           = self.Get_Min_Mean_Max(librosa.feature.mfcc(y=dat, sr=sr, n_mfcc=40).T)
        self.chroma_stft    = self.Get_Min_Mean_Max(librosa.feature.chroma_stft(S=stft, sr=sr).T)
        self.mel_spect      = self.Get_Min_Mean_Max(librosa.feature.melspectrogram(dat, sr=sr).T)
        self.rms            = self.Get_Min_Mean_Max(librosa.feature.rms(S=S), rms = True)
        self.chroma_cqt     = self.Get_Min_Mean_Max(librosa.feature.chroma_cqt(y=dat, sr=sr).T)
        self.chroma_cens    = self.Get_Min_Mean_Max(librosa.feature.chroma_cens(y=dat, sr=sr).T)
        self.spectrl_contr  = self.Get_Min_Mean_Max(librosa.feature.spectral_contrast(S=S, sr=sr).T)
        self.tonnetz        = self.Get_Min_Mean_Max(librosa.feature.tonnetz(y=dat, sr=sr).T)

        f.close()
        #print(self.last_dat)
        return self.last_dat

    # Can be used to get Min, Mean and Min
    # Will most likeyl be adjusted to find max and mins within the mean
    def Get_Min_Mean_Max(self, data, rms = False):
        dat_mean = np.mean(data, axis = 0)
        before = len(dat_mean)
        if rms:
            if len(dat_mean) > 100:
                dat_mean = dat_mean[:100]
            elif len(dat_mean) < 100:
                pad_num = 100 - len(dat_mean)
                dat_mean = np.concatenate((dat_mean, np.zeros(pad_num)), axis=None)
        #print(dat_mean.shape)
        out = []
        #plt.plot(dat_mean)
        #plt.show()
        out.extend(dat_mean)
        if self.mean:
            out.append(np.mean(dat_mean, axis=0))
        if self.max:
            out.append(np.max(dat_mean, axis=0))
        if self.std:    
            out.append(np.std(dat_mean, axis=0))
        #print(out)
        #if rms and before < 100:
        #    print(out)
        #    input()
        return out

# Reverse ArgSort
def RAS(arr, sz):
    ranked = np.argsort(arr)
    return ranked[::-1][:sz]


from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.tree import ExtraTreeClassifier as ETC
from sklearn.svm import LinearSVC as LSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.ensemble import RandomTreesEmbedding as RTE

from sklearn.neural_network import MLPClassifier as MLP

# Function that attempts all combinations of feature to determine the most significant
def Find_Optimal_Features(ML_DATA):
    # Creates all combinations of features
    lst = list(itertools.product([0,1], repeat = 8))
    lst.remove((0,0,0,0,0,0,0,0))
    for i in range(len(lst)):
        lst[i] = list(lst[i])
    
    outs = []
    for i in range(8):
        outs.append([])

    # Clasifiers put into array for easier access and expandability
    clfs = [DTC(), KNN(), BC(ETC()), LSVC(max_iter = 10_000), RFC(),
           SVC(), SGDC(), MLP(max_iter = 10_000)]
    tot = len(lst)
    y = []

    # Trains all models on all feature sets and gets scores
    for i, item in enumerate(lst[:15]):
        X_train, ys_train, ya_train, X_test, ys_test, ya_test = ML_DATA.Make_Set(item, 0.25)
        for item in clfs:
            item.fit(X_train, ys_train)
        
        scores = []

        for j, item in enumerate(clfs):
            scores.append(item.score(X_test, ys_test))
            outs[j].append(scores[j])

        PBR(i+1, tot, name = "Test Set", end = "\r")
        y = ys_test
    print()

    temp = Counter(y)
    total = len(y)
    rand_ratio = 0
    for item in temp.values():
        rand_ratio += item/total
    rand_ratio = (rand_ratio/8)*100
    print("Random Choice :", rand_ratio)

    final = np.zeros(8)
    final_top = np.zeros(8)
    final_btop = np.zeros(8)

    names = ["Decision Tree", "KNN", "Bag Extra Tree", "Linear SVC",
             "Random Forest", "SVC", "SGDC", "MLP"]

    # Prints all results
    for k, name in enumerate(names):
        o = RAS(outs[k], 10)
        final_top += np.asarray(lst[o[0]])
        f = np.zeros(8)
        count = 0
        best = outs[k][0]
        print(name + ":")
        for i, item in enumerate(o):
            if i == 0:
                best = outs[k][item]
            print("\t\t{:2}) {:7.4f}% {}".format(i, outs[k][item]*100, lst[item]))
            if outs[k][item] == best:
                count +=1
                f += np.asarray(lst[item])
            final += np.asarray(lst[item])
        final_btop += f/count
        print()

    print(final)
    print(final_top)
    print(final_btop)

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout as DO
#from matplotlib.backends.backend_pdf import PdfPages
from pyfiglet import figlet_format as ff

def tf_dense_sequential(data):
    x_train, ys_train, ya_train, x_test, ys_test, ya_test = data

    in_len = len(x_train[0])
    in_shp = np.shape(x_train[0])
    print(in_len, in_shp)

    l_sz = [10, 25, 50, 100, 250, 500, 1000]
    d_r = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]

    pdf_outs = PdfPages("Keras_Models.pdf")

    count = 0

    for layers in range(1, 11):
        for l in l_sz:
            for d in d_r:
                model = Sequential()

                model.add(Dense(in_len, activation = 'relu', input_shape = in_shp))
                for i in range(layers - 1):
                    model.add(Dense(l, activation = 'relu'))
                    if d > 0:
                        model.add(DO(d))
                model.add(Dense(l, activation = 'relu'))

                model.add(Dense(8, activation = 'sigmoid'))

                model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

                out_str = "Layers: " + str(layers) + " --> Layer Size: " + str(l) + " --> Dropout Rate: " + str(d)
                
                print(ff("Layers: " + str(layers)))
                print(ff("Layer Size: " + str(l)))
                print(ff("Dropout Rate: " + str(d)))

                model.summary()

                history = model.fit(np.array(x_train), np.array(ya_train), epochs = 150,
                            validation_data = (np.array(x_test), np.array(ya_test)))

                # Plots History of model
                fig = plt.figure()
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title(out_str)
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                
                count += 1
                pdf_outs.savefig(fig)
                plt.clf()

        #        if count > 3:
        #            break
        #    if count > 3:
        #        break
        #if count > 3:
        #    break

    pdf_outs.close()
    return

if __name__ == "__main__":
    voice_loc = "speech-emotion-recognition-ravdess-data"
    #voice_loc = "test"
    #voice_loc = "t"
    dbc = Directory_DB(voice_loc)
    #dbc.Print_DB()

    data = dbc.db
    data = File_DB_to_Data_DB(data)
    print("Number of files:", len(data))
    ML_DATA = ML_Data_Prep(data)
    ML_DATA.Process_Voices()

    # Find_Optimal_Features(ML_DATA)
    #features = [1,0,0,1,1,0,1,1]
    features = [1,1,1,1,1,1,1,1]
    X_train, ys_train, ya_train, X_test, ys_test, ya_test = ML_DATA.Make_Set(features, 0.25)
    split_data = (X_train, ys_train, ya_train, X_test, ys_test, ya_test)

    clf = RFC(1000).fit(X_train, ys_train)
    print("Optimal Feature Score: {:.2f}%".format(clf.score(X_test, ys_test)*100))

    #tf_dense_sequential(split_data)