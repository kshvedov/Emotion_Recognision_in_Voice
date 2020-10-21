import os
from pprint import pprint as pp

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
            print("DB Created")

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
    def __init__(self, db):
        self.ML_data = []

# Class that is the center of the algorithm, within this class, voice
# files are analysed and turned into a series of fetures for Machine
# Learning
class Voice_Processing_Unit:
    def __init__(self):
        pass

if __name__ == "__main__":
    voice_loc = "speech-emotion-recognition-ravdess-data"
    #voice_loc = "test"
    #voice_loc = "t"
    dbc = Directory_DB(voice_loc)
    #dbc.Print_DB()

    data = dbc.db
    data = File_DB_to_Data_DB(data)
    print(len(data))
    pass
