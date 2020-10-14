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
        db = {}
        temp_dirs = []
        temp_files = []
        if location[-1] == "/":
            location = location[-1]
        for file in os.listdir(location):
            if os.path.isdir(location+"/"+file):
                temp_dirs.append({"name": file,
                                  "content": self.__Create_DB_Helper(location+"/"+file)})
            else:
                temp_files.append(file)

        return {"dirs":temp_dirs, "files": temp_files}

    # Function that displays a rough filesystem from database using its
    # helper function
    def Print_DB(self):
        if self.db == {}:
            print("No Filesystem to display!")
            return
        self.__Print_DB_Helper(self.db, 0)

    def __Print_DB_Helper(self, db, lvl):
        name_indent = ""
        file_indent = ""
        last_file_indent = ""
        spacer = "|   " + "|   "*lvl
        if lvl == 0:
            file_indent = "├── "
            last_file_indent = "└── "
        if lvl >= 1:
            file_indent = "|   "*lvl + "├── "
            last_file_indent = "|   "*lvl + "└── "
            name_indent = "|   "*(lvl-1) + "├── "

        print(name_indent + db["name"])

        for item in db["content"]["dirs"]:
            self.__Print_DB_Helper(item, lvl+1)
            print(spacer)

        last = len(db["content"]["files"]) - 1
        for i, item in enumerate(db["content"]["files"]):
            if i == last:
                print(last_file_indent + item)
            else:
                print(file_indent + item)


    # Private Helper Function in charge of checking of the existance of a
    # location
    def __Check_Location(self, loc_name):
        if not os.path.exists(loc_name) or loc_name == "":
            print("Location hasn't been entered or doesnt exist...")
            return False
        else:
            return True

if __name__ == "__main__":
    voice_loc = "speech-emotion-recognition-ravdess-data"
    #voice_loc = "test"
    #voice_loc = "t"
    dbc = Directory_DB(voice_loc)
    dbc.Print_DB()
    pass
