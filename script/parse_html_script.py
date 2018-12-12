import bs4
import spacy
import csv
import os
import numpy


class Character:

    def __init__(self, name, dialog=""):
        self.name = name
        self.text = dialog
        self.id = 0

    def __str__(self):
        return "Char: " + self.name + " " + "Dialouge: " + self.text


class Setting:

    def __init__(self, text):
        self.text = text
        self.id = 1

    def __str__(self):
        return "Setting: " + self.text


class StageDir:

    def __init__(self, text):
        self.text = text
        self.id = 2

    def __str__(self):
        return "StageDir " + self.text


class Script:

    def __init__(self, filepath):
        self.name = filepath
        self.script = self.get_script(filepath)
        self.chars = self.get_chars()
        self.nlp = spacy.load('en_core_web_sm')

    def get_chars(self):
        """Get all the characters in the script"""
        chars = {}
        for el in self.script:
            if type(el) == Character:
                chars[el.name] = True
        return chars

    def script_to_txt(self):
        seq = [el.id for el in self.script]
        numpy.savetxt("script_arrays/" + filename + ".csv", numpy.reshape(seq[:200], (20,10)), delimiter=",")

    def get_char_lines(self):
        chars = []
        for el in self.script:
            if type(el) == Character:
                chars.append(el)
        return chars

    def get_sd_lines(self):
        sds = []
        for el in self.script:
            if type(el) == StageDir:
                sds.append(el)
        return sds

    def get_set_lines(self):
        sds = []
        for el in self.script:
            if type(el) == Setting:
                sds.append(el)
        return sds

    def get_script(self, filepath):
        html_file = open(filepath, 'r').read()
        soup = bs4.BeautifulSoup(html_file, 'html.parser')
        tags = soup.find("root").find_all(recursive=False)[1:]
        script = []

        # load each tag into the correct object
        for tag in tags:
            if tag.name == "char":
                script.append(Character(tag["name"], tag["dialog"]))
            elif tag.name == "sd":
                script.append(StageDir(tag.text))
            elif tag.name == "set":
                script.append(Setting(tag.text))

        # correct some mislabeled dialogue cases
        for i in range(len(script)):
            if i >= len(script) - 2:
                break
            if type(script[i]) == Character and type(script[i + 1]) == StageDir and type(script[i + 2]) == Character:
                script[i].text += " " + script[i + 1].text
                script.remove(script[i + 1])

        return script

    def get_script_training_data(self):
        # convert script to numerical seq
        seq = [el.id for el in self.script]
        data = []

        # pass window over seq and extract a 1D convolution
        window_size = 100
        for i in range(len(seq)-window_size):
            data.append(seq[i:i+window_size])

        return data

    def debug(self):
        for el in self.get_script_training_data():
            print(el)


if __name__ == "__main__":

    with open("data/script_seq_data.csv","w+", newline="") as csvfile:
        writer = csv.writer(csvfile,delimiter=",")
        avg = 0

        #files for individual line types
        charFile = open("data/char_lines.csv", "w+", newline="")
        charWriter = csv.writer(charFile, delimiter=",", quotechar="|")
        sdFile = open("data/sd_lines.csv", "w+", newline="")
        sdWriter = csv.writer(sdFile, delimiter=",", quotechar="|")
        setFile = open("data/set_lines.csv", "w+", newline="")
        setWriter = csv.writer(setFile, delimiter=",", quotechar="|")

        # iterate through directory and process each script, adding training data to same csv
        for filename in os.listdir("processed_html_scripts"):
            print("processing: " + filename + "...")
            s = Script("processed_html_scripts/" + filename)
            for data in s.get_script_training_data():
                writer.writerow(data)

            # get line data
            for el in s.get_char_lines():
                charWriter.writerow([el.name, el.text])
            for el in s.get_sd_lines():
                sdWriter.writerow([el.text])
            for el in s.get_set_lines():
                setWriter.writerow([el.text])

        # clean up
        charFile.close()
        sdFile.close()
        setFile.close()


